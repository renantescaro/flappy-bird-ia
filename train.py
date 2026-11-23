"""Treinamento visual da IA do Flappy (neuroevolução).

Uso:
    python train.py                 # continua de onde parou (se existir checkpoint)
    python train.py --novo          # começa do zero
    python train.py --populacao 200 --score-max 1000

Controles na janela:
    1..6   velocidade da simulação (1x, 2x, 5x, 10x, 50x, máxima)
    N      encerra a geração atual e evolui
    ESC    salva e sai
"""

import argparse
import random
import sys
import time

import numpy as np
import pygame

import main as game
from brain import (
    BEST_MODEL,
    CHECKPOINT,
    FLAP_THRESHOLD,
    INPUT_LABELS,
    Population,
    load_checkpoint,
    next_pipe,
    save_brain,
    save_checkpoint,
    sense,
)
from main import GROUND_HEIGHT, HEIGHT, PIPE_WIDTH, WIDTH, Bird

PANEL_W = 380
SPEEDS = [1, 2, 5, 10, 50, 10_000]
SPEED_NAMES = ["1x", "2x", "5x", "10x", "50x", "MÁX"]

BG = (30, 33, 40)
FG = (230, 230, 230)
MUTED = (140, 145, 155)
ACCENT = (250, 210, 40)
GREEN = (90, 200, 110)
RED = (230, 80, 80)
BLUE = (90, 160, 240)


# ---------------------------------------------------------------------------
# Desenho compartilhado (também usado pelo play_ai.py)
# ---------------------------------------------------------------------------
def draw_network(surf, font, genome_view, x_in, hidden, out, rect):
    """Desenha a rede: linhas = pesos (verde +, vermelho -), nós = ativação."""
    from brain import N_HIDDEN, N_INPUTS, unpack

    w1, _, w2, _ = (a[0] for a in unpack(genome_view.reshape(1, -1)))
    left, top, width, height = rect
    col_x = [left + 135, left + width // 2 + 60, left + width - 45]

    def col_y(n, i):
        step = height / n
        return int(top + step * (i + 0.5))

    inputs = [(col_x[0], col_y(N_INPUTS, i)) for i in range(N_INPUTS)]
    hiddens = [(col_x[1], col_y(N_HIDDEN, i)) for i in range(N_HIDDEN)]
    output = (col_x[2], top + height // 2)

    def line(a, b, w):
        color = GREEN if w > 0 else RED
        pygame.draw.line(surf, color, a, b, max(1, min(4, int(abs(w) * 1.5))))

    for h, hp in enumerate(hiddens):
        for i, ip in enumerate(inputs):
            if abs(w1[h, i]) > 0.25:
                line(ip, hp, w1[h, i])
        line(hp, output, w2[0, h])

    def node(pos, value, radius=10):
        v = max(-1.0, min(1.0, float(value)))
        color = (
            int(60 + 190 * max(v, 0)),
            int(60 + 150 * max(v, 0)),
            int(60 + 180 * max(-v, 0)),
        )
        pygame.draw.circle(surf, color, pos, radius)
        pygame.draw.circle(surf, FG, pos, radius, 2)

    for i, pos in enumerate(inputs):
        node(pos, x_in[i])
        label = font.render(INPUT_LABELS[i], True, MUTED)
        surf.blit(label, label.get_rect(midright=(pos[0] - 15, pos[1])))
    for h, pos in enumerate(hiddens):
        node(pos, hidden[h], 8)
    flap = out > FLAP_THRESHOLD
    node(output, 1 if flap else -0.3, 13)
    label = font.render(
        f"PULA {out:.2f}" if flap else f"{out:.2f}", True, ACCENT if flap else MUTED
    )
    surf.blit(label, label.get_rect(midtop=(output[0], output[1] + 18)))


def draw_world(surf, pipes, ground_offset):
    surf.fill(game.SKY)
    for pipe in pipes:
        pipe.draw(surf)
    top = HEIGHT - GROUND_HEIGHT
    pygame.draw.rect(surf, game.GROUND, (0, top, WIDTH, GROUND_HEIGHT))
    pygame.draw.rect(surf, game.GRASS, (0, top, WIDTH, 14))
    for i in range(-1, WIDTH // 24 + 2):
        x = i * 24 - int(ground_offset)
        pygame.draw.polygon(
            surf, (95, 160, 35), [(x, top + 14), (x + 12, top + 14), (x + 6, top)]
        )
    pygame.draw.line(surf, game.GROUND_DARK, (0, top), (WIDTH, top), 3)


# ---------------------------------------------------------------------------
# Treinador
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, args):
        pygame.init()
        pygame.display.set_caption("Flappy - Treinamento da IA")
        self.screen = pygame.display.set_mode((WIDTH + PANEL_W, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.font_b = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_title = pygame.font.SysFont("arial", 24, bold=True)
        self.font_score = pygame.font.SysFont("arial", 48, bold=True)

        self.max_score = args.score_max
        self.speed_idx = 0
        self.ground_offset = 0.0

        self.ghost = pygame.Surface((Bird.RADIUS * 2, Bird.RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            self.ghost,
            (*game.BIRD_YELLOW, 110),
            (Bird.RADIUS, Bird.RADIUS),
            Bird.RADIUS,
        )
        pygame.draw.circle(
            self.ghost, (0, 0, 0, 110), (Bird.RADIUS, Bird.RADIUS), Bird.RADIUS, 2
        )

        if CHECKPOINT.exists() and not args.novo:
            self.pop, self.generation, self.best_score, self.history = load_checkpoint()
            print(
                f"Continuando da geração {self.generation} (recorde {self.best_score})."
            )
        else:
            self.pop = Population.random(
                args.populacao, np.random.default_rng(args.seed)
            )
            self.generation, self.best_score, self.history = 1, -1, []
            print(f"Novo treinamento com {self.pop.size} pássaros por geração.")
        self.best_gen = None
        self.new_generation()

    # -- simulação ----------------------------------------------------------
    def new_generation(self):
        size = self.pop.size
        self.birds = [Bird() for _ in range(size)]
        self.alive = np.ones(size, dtype=bool)
        self.fitness = np.zeros(size)
        self.scores = np.zeros(size, dtype=int)
        self.pipes = []
        self.score = 0
        self.frame = 0
        self.last = None  # (índice, entradas, ocultos, saída) do pássaro em destaque

    def step(self):
        self.frame += 1
        speed, _, spawn = game.difficulty(self.score)
        self.ground_offset = (self.ground_offset + speed) % 24

        if not self.pipes or self.pipes[-1].x < WIDTH - spawn:
            self.pipes.append(game.spawn_pipe(self.pipes, self.score))
        bird_x = self.birds[0].x
        for pipe in self.pipes:
            pipe.update(speed)
            if not pipe.passed and pipe.x + PIPE_WIDTH < bird_x:
                pipe.passed = True
                self.score += 1
        self.pipes = [p for p in self.pipes if p.x > -PIPE_WIDTH - 10]

        idx = np.flatnonzero(self.alive)
        pipe = next_pipe(self.pipes, bird_x)
        x = sense(
            [self.birds[i].y for i in idx],
            [self.birds[i].vel for i in idx],
            self.pipes,
            speed,
            bird_x,
        )
        hidden, out = Population(self.pop.genomes[idx]).forward(x)
        self.last = (idx[0], x[0], hidden[0], out[0])

        for k, i in enumerate(idx):
            bird = self.birds[i]
            if out[k] > FLAP_THRESHOLD:
                bird.flap()
            bird.update()
            dead = (
                bird.y + Bird.RADIUS >= HEIGHT - GROUND_HEIGHT
                or bird.y - Bird.RADIUS <= 0
                or any(p.collides(bird.rect) for p in self.pipes)
            )
            if dead:
                self.kill(i, pipe)

        if not self.alive.any() or self.score >= self.max_score:
            self.end_generation()

    def kill(self, i, pipe):
        self.alive[i] = False
        self.scores[i] = self.score
        # fitness: tempo vivo + bônus por cano - distância do centro do vão ao morrer
        dist = abs(self.birds[i].y - (pipe.gap_y + pipe.gap / 2)) if pipe else 0
        self.fitness[i] = self.frame + self.score * 100 - dist * 0.5

    def end_generation(self):
        for i in np.flatnonzero(self.alive):
            self.kill(i, next_pipe(self.pipes, self.birds[i].x))
        best = int(np.argmax(self.fitness))
        gen_best = int(self.scores[best])
        self.history.append((gen_best, float(self.scores.mean())))

        if gen_best > self.best_score:
            self.best_score = gen_best
            self.best_gen = self.generation
            save_brain(self.pop.genomes[best], self.generation, gen_best)
            print(
                f"Geração {self.generation}: novo recorde {gen_best} -> salvo em {BEST_MODEL}"
            )
        else:
            print(
                f"Geração {self.generation}: melhor {gen_best} (recorde {self.best_score})"
            )

        self.pop = self.pop.evolve(self.fitness)
        self.generation += 1
        save_checkpoint(self.pop, self.generation, self.best_score, self.history)
        self.new_generation()

    # -- loop ---------------------------------------------------------------
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.quit()
                    elif event.key == pygame.K_n:
                        self.end_generation()
                    elif pygame.K_1 <= event.key <= pygame.K_6:
                        self.speed_idx = event.key - pygame.K_1

            # roda N passos por quadro, sem travar a janela (limite de ~30 ms)
            deadline = time.perf_counter() + 0.03
            for _ in range(SPEEDS[self.speed_idx]):
                self.step()
                if time.perf_counter() > deadline:
                    break
            self.draw()
            pygame.display.flip()
            self.clock.tick(game.FPS)

    def quit(self):
        save_checkpoint(self.pop, self.generation, self.best_score, self.history)
        print(f"Checkpoint salvo em {CHECKPOINT}")
        pygame.quit()
        sys.exit()

    # -- desenho ------------------------------------------------------------
    def draw(self):
        world = self.screen.subsurface((0, 0, WIDTH, HEIGHT))
        draw_world(world, self.pipes, self.ground_offset)

        highlight = self.last[0] if self.last else None
        for i in np.flatnonzero(self.alive):
            if i != highlight:
                b = self.birds[i]
                world.blit(self.ghost, (b.x - Bird.RADIUS, int(b.y) - Bird.RADIUS))
        if highlight is not None and self.alive[highlight]:
            self.birds[highlight].draw(world)

        score = self.font_score.render(str(self.score), True, game.WHITE)
        world.blit(score, score.get_rect(center=(WIDTH // 2, 60)))
        self.draw_panel()

    def draw_panel(self):
        px = WIDTH
        pygame.draw.rect(self.screen, BG, (px, 0, PANEL_W, HEIGHT))
        y = 14
        self.screen.blit(
            self.font_title.render("TREINAMENTO DA IA", True, ACCENT), (px + 16, y)
        )
        y += 38
        alive = int(self.alive.sum())
        rows = [
            ("Geração", str(self.generation)),
            ("Vivos", f"{alive}/{self.pop.size}"),
            (
                "Score atual",
                f"{self.score}  (nível {self.score // game.PIPES_PER_LEVEL + 1})",
            ),
            (
                "Recorde",
                f"{max(self.best_score, 0)}"
                + (f"  (geração {self.best_gen})" if self.best_gen else ""),
            ),
            ("Velocidade", SPEED_NAMES[self.speed_idx]),
        ]
        for label, value in rows:
            self.screen.blit(self.font.render(label, True, MUTED), (px + 16, y))
            self.screen.blit(self.font_b.render(value, True, FG), (px + 140, y - 1))
            y += 22

        y += 8
        self.screen.blit(
            self.font.render("Cérebro do pássaro em destaque:", True, MUTED),
            (px + 16, y),
        )
        if self.last:
            i, x_in, hidden, out = self.last
            draw_network(
                self.screen,
                self.font,
                self.pop.genomes[i],
                x_in,
                hidden,
                out,
                (px, y + 22, PANEL_W, 190),
            )
        self.draw_chart((px + 16, 418, PANEL_W - 32, 120))

        help_lines = ["1-6: velocidade   N: pular geração", "ESC: salvar e sair"]
        for k, line in enumerate(help_lines):
            self.screen.blit(
                self.font.render(line, True, MUTED), (px + 16, HEIGHT - 46 + k * 20)
            )

    def draw_chart(self, rect):
        x0, y0, w, h = rect
        self.screen.blit(
            self.font.render("Score por geração", True, MUTED), (x0, y0 - 2)
        )
        self.screen.blit(self.font.render("— melhor", True, ACCENT), (x0 + 175, y0 - 2))
        self.screen.blit(self.font.render("— média", True, BLUE), (x0 + 265, y0 - 2))
        area = pygame.Rect(x0, y0 + 20, w, h - 20)
        pygame.draw.rect(self.screen, (45, 49, 58), area, border_radius=4)
        data = self.history[-80:]
        if len(data) < 2:
            msg = self.font.render("aguardando gerações...", True, MUTED)
            self.screen.blit(msg, msg.get_rect(center=area.center))
            return
        top = max(1.0, max(b for b, _ in data))
        label = self.font.render(f"{top:.0f}", True, MUTED)
        self.screen.blit(label, (area.right - label.get_width() - 4, area.top + 2))

        def points(col):
            return [
                (
                    area.left + 4 + (area.width - 8) * k / (len(data) - 1),
                    area.bottom - 4 - (area.height - 8) * row[col] / top,
                )
                for k, row in enumerate(data)
            ]

        pygame.draw.lines(self.screen, BLUE, False, points(1), 2)
        pygame.draw.lines(self.screen, ACCENT, False, points(0), 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Treina uma IA para jogar o Flappy.")
    parser.add_argument(
        "--novo", action="store_true", help="ignora o checkpoint e começa do zero"
    )
    parser.add_argument(
        "--populacao", type=int, default=150, help="pássaros por geração"
    )
    parser.add_argument(
        "--score-max",
        type=int,
        default=500,
        help="encerra a geração se a IA chegar nesse score",
    )
    parser.add_argument(
        "--seed", type=int, help="semente aleatória (reprodutibilidade)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    Trainer(args).run()
