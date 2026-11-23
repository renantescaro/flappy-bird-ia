"""Flappy - jogo estilo Flappy Bird feito com pygame.

Controles:
    ESPAÇO  -> bate as asas (e inicia / reinicia o jogo)
    ESC     -> sai do jogo
"""

import math
import random
import sys
from pathlib import Path

import pygame

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 400, 600
FPS = 60
GROUND_HEIGHT = 80

GRAVITY = 0.45
FLAP_STRENGTH = -8.0
MAX_FALL_SPEED = 10

PIPE_WIDTH = 70

# Dificuldade inicial e limites
BASE_PIPE_SPEED = 3.0
BASE_GAP = 170
BASE_SPAWN_DISTANCE = 230  # distância horizontal entre canos
SPEED_STEP = 0.25
GAP_STEP = 6
SPAWN_STEP = 3
MAX_PIPE_SPEED = 5.5
MIN_GAP = 115
MIN_SPAWN_DISTANCE = 200
PIPES_PER_LEVEL = 10

NAME_LENGTH = 5
TOP_N = 5
SCORES_FILE = Path(__file__).with_name("scores.txt")

# Cores
SKY = (112, 197, 206)
GROUND = (222, 216, 149)
GROUND_DARK = (84, 56, 71)
GRASS = (115, 191, 46)
PIPE_GREEN = (94, 168, 52)
PIPE_DARK = (55, 110, 30)
BIRD_YELLOW = (250, 210, 40)
BIRD_ORANGE = (240, 120, 30)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CLOUD = (235, 248, 250)
PANEL = (222, 216, 149)
PANEL_BORDER = (84, 56, 71)
GOLD = (230, 180, 20)


def difficulty(score):
    """Retorna (velocidade dos canos, tamanho do vão, distância entre canos) para o score."""
    level = score // PIPES_PER_LEVEL
    speed = min(BASE_PIPE_SPEED + level * SPEED_STEP, MAX_PIPE_SPEED)
    gap = max(BASE_GAP - level * GAP_STEP, MIN_GAP)
    spawn = max(BASE_SPAWN_DISTANCE - level * SPAWN_STEP, MIN_SPAWN_DISTANCE)
    return speed, gap, spawn


def spawn_pipe(pipes, score):
    """Cria o próximo cano aleatório, respeitando o que o pássaro consegue alcançar."""
    speed, gap, spawn = difficulty(score)
    # quadros "livres" entre sair de um cano e entrar no próximo
    free_frames = max(0.0, (spawn - PIPE_WIDTH - 2 * Bird.RADIUS) / speed)
    # pior caso: sai do cano parado no topo do vão e precisa cair até o próximo (½·g·t²),
    # mais uma pequena parte da folga do vão
    max_shift = (gap - 2 * Bird.RADIUS) / 4 + 0.5 * GRAVITY * free_frames**2
    return Pipe(WIDTH + 10, gap, pipes[-1] if pipes else None, max_shift)


# ---------------------------------------------------------------------------
# Placar (arquivo txt)
# ---------------------------------------------------------------------------
def load_scores():
    """Lê scores.txt no formato NOME;PONTOS (um por linha)."""
    scores = []
    if not SCORES_FILE.exists():
        return scores
    for line in SCORES_FILE.read_text(encoding="utf-8").splitlines():
        if ";" not in line:
            continue
        name, _, value = line.partition(";")
        try:
            scores.append((name.strip(), int(value)))
        except ValueError:
            continue
    return scores


def save_score(name, score):
    with SCORES_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{name};{score}\n")


def top_scores(n=TOP_N):
    return sorted(load_scores(), key=lambda s: s[1], reverse=True)[:n]


# ---------------------------------------------------------------------------
# Entidades
# ---------------------------------------------------------------------------
class Bird:
    RADIUS = 15

    def __init__(self):
        self.x = WIDTH // 3
        self.y = HEIGHT // 2 - 40
        self.vel = 0.0
        self.anim = 0.0

    def flap(self):
        self.vel = FLAP_STRENGTH

    def update(self):
        self.vel = min(self.vel + GRAVITY, MAX_FALL_SPEED)
        self.y += self.vel
        self.anim += 0.3

    def idle(self):
        """Flutua parado na tela inicial."""
        self.anim += 0.15
        self.y = HEIGHT // 2 - 40 + 8 * math.sin(self.anim)

    @property
    def rect(self):
        r = self.RADIUS - 3  # hitbox um pouco menor, mais justo para o jogador
        return pygame.Rect(self.x - r, self.y - r, r * 2, r * 2)

    def draw(self, surf):
        angle = max(-25, min(70, self.vel * 6))
        size = self.RADIUS * 2 + 12
        img = pygame.Surface((size, size), pygame.SRCALPHA)
        c = size // 2
        # corpo
        pygame.draw.circle(img, BIRD_YELLOW, (c, c), self.RADIUS)
        pygame.draw.circle(img, BLACK, (c, c), self.RADIUS, 2)
        # asa (bate)
        wing_y = c + (4 if int(self.anim) % 2 else -2)
        pygame.draw.ellipse(img, WHITE, (c - 14, wing_y - 4, 14, 9))
        pygame.draw.ellipse(img, BLACK, (c - 14, wing_y - 4, 14, 9), 1)
        # olho
        pygame.draw.circle(img, WHITE, (c + 6, c - 5), 6)
        pygame.draw.circle(img, BLACK, (c + 8, c - 5), 2)
        # bico
        pygame.draw.polygon(
            img, BIRD_ORANGE, [(c + 10, c + 1), (c + 22, c + 4), (c + 10, c + 8)]
        )
        rotated = pygame.transform.rotate(img, -angle)
        surf.blit(rotated, rotated.get_rect(center=(self.x, int(self.y))))


class Pipe:
    MARGIN = 60

    def __init__(self, x, gap, prev=None, max_shift=None):
        """prev: cano anterior. max_shift: quanto o vão pode mudar de altura em relação a ele,
        para que nunca surja uma sequência fisicamente impossível de passar."""
        self.x = float(x)
        self.gap = gap
        low, high = self.MARGIN, HEIGHT - GROUND_HEIGHT - self.MARGIN - gap
        if prev is not None and max_shift is not None:
            center = prev.gap_y + prev.gap / 2 - gap / 2
            low = max(low, int(center - max_shift))
            high = min(high, int(center + max_shift))
        self.gap_y = random.randint(low, max(low, high))
        self.passed = False

    def update(self, speed):
        self.x -= speed

    @property
    def top_rect(self):
        return pygame.Rect(int(self.x), 0, PIPE_WIDTH, self.gap_y)

    @property
    def bottom_rect(self):
        top = self.gap_y + self.gap
        return pygame.Rect(int(self.x), top, PIPE_WIDTH, HEIGHT - GROUND_HEIGHT - top)

    def collides(self, rect):
        return rect.colliderect(self.top_rect) or rect.colliderect(self.bottom_rect)

    def draw(self, surf):
        for body, cap_y in (
            (self.top_rect, self.top_rect.bottom - 26),
            (self.bottom_rect, self.bottom_rect.top),
        ):
            pygame.draw.rect(surf, PIPE_GREEN, body)
            pygame.draw.rect(surf, PIPE_DARK, body, 3)
            pygame.draw.rect(surf, (140, 210, 90), (body.x + 8, body.y, 8, body.height))
            cap = pygame.Rect(body.x - 5, cap_y, PIPE_WIDTH + 10, 26)
            pygame.draw.rect(surf, PIPE_GREEN, cap)
            pygame.draw.rect(surf, PIPE_DARK, cap, 3)


# ---------------------------------------------------------------------------
# Jogo
# ---------------------------------------------------------------------------
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont("arial", 56, bold=True)
        self.font_med = pygame.font.SysFont("arial", 30, bold=True)
        self.font_small = pygame.font.SysFont("arial", 20, bold=True)
        self.clouds = [
            [random.randint(0, WIDTH), random.randint(30, 250), random.randint(30, 60)]
            for _ in range(5)
        ]
        self.ground_offset = 0.0
        self.reset()

    # -- estado -------------------------------------------------------------
    def reset(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.state = "ready"  # ready | playing | dying | name | over
        self.name = ""
        self.death_timer = 0
        self.top = top_scores()

    @property
    def level(self):
        return self.score // PIPES_PER_LEVEL

    @property
    def pipe_speed(self):
        return difficulty(self.score)[0]

    @property
    def gap(self):
        return difficulty(self.score)[1]

    @property
    def spawn_distance(self):
        return difficulty(self.score)[2]

    # -- loop ---------------------------------------------------------------
    def run(self):
        while True:
            for event in pygame.event.get():
                self.handle_event(event)
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.quit()
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.quit()

        if self.state == "ready" and event.key == pygame.K_SPACE:
            self.state = "playing"
            self.bird.flap()
        elif self.state == "playing" and event.key == pygame.K_SPACE:
            self.bird.flap()
        elif self.state == "name":
            if event.key == pygame.K_BACKSPACE:
                self.name = self.name[:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if len(self.name) == NAME_LENGTH:
                    save_score(self.name, self.score)
                    self.top = top_scores()
                    self.state = "over"
            elif (
                len(self.name) < NAME_LENGTH
                and event.unicode
                and event.unicode.isalnum()
                and event.unicode.isascii()
            ):
                self.name += event.unicode.upper()
        elif self.state == "over" and event.key == pygame.K_SPACE:
            self.reset()

    def update(self):
        # nuvens sempre se movem (cenário infinito)
        for cloud in self.clouds:
            cloud[0] -= 0.4
            if cloud[0] < -cloud[2] * 2:
                cloud[0] = WIDTH + cloud[2]
                cloud[1] = random.randint(30, 250)

        if self.state in ("ready", "playing"):
            speed = self.pipe_speed
            self.ground_offset = (self.ground_offset + speed) % 24

        if self.state == "ready":
            self.bird.idle()
            return

        if self.state == "dying":
            # pássaro cai até o chão antes de pedir o nome
            if self.bird.y < HEIGHT - GROUND_HEIGHT - Bird.RADIUS:
                self.bird.update()
            self.death_timer += 1
            if self.death_timer > 45:
                self.state = "name"
            return

        if self.state != "playing":
            return

        self.bird.update()

        # gera canos aleatórios infinitamente
        if not self.pipes or self.pipes[-1].x < WIDTH - self.spawn_distance:
            self.pipes.append(spawn_pipe(self.pipes, self.score))

        for pipe in self.pipes:
            pipe.update(self.pipe_speed)
            if not pipe.passed and pipe.x + PIPE_WIDTH < self.bird.x:
                pipe.passed = True
                self.score += 1
        self.pipes = [p for p in self.pipes if p.x > -PIPE_WIDTH - 10]

        # colisões
        bird_rect = self.bird.rect
        hit_pipe = any(p.collides(bird_rect) for p in self.pipes)
        hit_ground = self.bird.y + Bird.RADIUS >= HEIGHT - GROUND_HEIGHT
        hit_ceiling = self.bird.y - Bird.RADIUS <= 0
        if hit_pipe or hit_ground or hit_ceiling:
            self.state = "dying"
            self.bird.vel = max(self.bird.vel, 0)

    # -- desenho ------------------------------------------------------------
    def text(self, txt, font, color, center, outline=True):
        if outline:
            shadow = font.render(txt, True, BLACK)
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                self.screen.blit(
                    shadow, shadow.get_rect(center=(center[0] + dx, center[1] + dy))
                )
        surf = font.render(txt, True, color)
        self.screen.blit(surf, surf.get_rect(center=center))

    def draw_background(self):
        self.screen.fill(SKY)
        for x, y, r in self.clouds:
            pygame.draw.circle(self.screen, CLOUD, (int(x), y), r)
            pygame.draw.circle(
                self.screen, CLOUD, (int(x + r * 0.9), y + 8), int(r * 0.8)
            )
            pygame.draw.circle(
                self.screen, CLOUD, (int(x - r * 0.9), y + 10), int(r * 0.7)
            )

    def draw_ground(self):
        top = HEIGHT - GROUND_HEIGHT
        pygame.draw.rect(self.screen, GROUND, (0, top, WIDTH, GROUND_HEIGHT))
        pygame.draw.rect(self.screen, GRASS, (0, top, WIDTH, 14))
        for i in range(-1, WIDTH // 24 + 2):
            x = i * 24 - int(self.ground_offset)
            pygame.draw.polygon(
                self.screen,
                (95, 160, 35),
                [(x, top + 14), (x + 12, top + 14), (x + 6, top)],
            )
        pygame.draw.line(self.screen, GROUND_DARK, (0, top), (WIDTH, top), 3)

    def draw_panel(self, rect):
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=12)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, 4, border_radius=12)

    def draw_ready(self):
        self.text("FLAPPY", self.font_big, WHITE, (WIDTH // 2, 120))
        self.text(
            "Pressione ESPAÇO", self.font_small, WHITE, (WIDTH // 2, HEIGHT // 2 + 40)
        )
        self.text("para voar", self.font_small, WHITE, (WIDTH // 2, HEIGHT // 2 + 66))

    def draw_name_entry(self):
        self.text("GAME OVER", self.font_big, BIRD_ORANGE, (WIDTH // 2, 100))
        panel = pygame.Rect(40, 160, WIDTH - 80, 260)
        self.draw_panel(panel)
        self.text(
            "SCORE", self.font_small, BIRD_ORANGE, (WIDTH // 2, 190), outline=False
        )
        self.text(str(self.score), self.font_big, WHITE, (WIDTH // 2, 240))
        self.text(
            "Digite seu nome (5 letras):",
            self.font_small,
            GROUND_DARK,
            (WIDTH // 2, 295),
            outline=False,
        )

        # caixas de caracteres
        box_w, spacing = 40, 8
        total = NAME_LENGTH * box_w + (NAME_LENGTH - 1) * spacing
        start_x = WIDTH // 2 - total // 2
        blink = (pygame.time.get_ticks() // 400) % 2 == 0
        for i in range(NAME_LENGTH):
            box = pygame.Rect(start_x + i * (box_w + spacing), 320, box_w, 48)
            pygame.draw.rect(self.screen, WHITE, box, border_radius=6)
            active = i == len(self.name)
            pygame.draw.rect(
                self.screen,
                BIRD_ORANGE if active else GROUND_DARK,
                box,
                3,
                border_radius=6,
            )
            if i < len(self.name):
                self.text(self.name[i], self.font_med, BLACK, box.center, outline=False)
            elif active and blink:
                pygame.draw.line(
                    self.screen,
                    BLACK,
                    (box.centerx - 10, box.bottom - 10),
                    (box.centerx + 10, box.bottom - 10),
                    3,
                )

        hint = (
            "ENTER para salvar"
            if len(self.name) == NAME_LENGTH
            else f"faltam {NAME_LENGTH - len(self.name)} caractere(s)"
        )
        self.text(hint, self.font_small, GROUND_DARK, (WIDTH // 2, 395), outline=False)

    def draw_game_over(self):
        self.text("GAME OVER", self.font_big, BIRD_ORANGE, (WIDTH // 2, 80))
        panel = pygame.Rect(30, 130, WIDTH - 60, 340)
        self.draw_panel(panel)
        self.text(f"Seu score: {self.score}", self.font_med, WHITE, (WIDTH // 2, 165))
        self.text("TOP 5", self.font_med, GOLD, (WIDTH // 2, 215))

        medal = [GOLD, (192, 192, 192), (205, 127, 50)]
        for i in range(TOP_N):
            y = 260 + i * 40
            if i < len(self.top):
                name, pts = self.top[i]
                highlight = name == self.name and pts == self.score
                color = BIRD_ORANGE if highlight else GROUND_DARK
                pos_color = medal[i] if i < 3 else GROUND_DARK
                pygame.draw.circle(self.screen, pos_color, (75, y), 14)
                self.text(str(i + 1), self.font_small, WHITE, (75, y), outline=False)
                self.text(name, self.font_med, color, (170, y), outline=False)
                self.text(
                    str(pts), self.font_med, color, (WIDTH - 85, y), outline=False
                )
            else:
                self.text(
                    "-----", self.font_med, (170, 160, 110), (170, y), outline=False
                )
                self.text(
                    "-", self.font_med, (170, 160, 110), (WIDTH - 85, y), outline=False
                )

        self.text(
            "ESPAÇO para jogar de novo", self.font_small, WHITE, (WIDTH // 2, 500)
        )

    def draw(self):
        self.draw_background()
        for pipe in self.pipes:
            pipe.draw(self.screen)
        self.draw_ground()
        self.bird.draw(self.screen)

        if self.state == "ready":
            self.draw_ready()
        elif self.state in ("playing", "dying"):
            self.text(str(self.score), self.font_big, WHITE, (WIDTH // 2, 70))
            self.text(
                f"Nível {self.level + 1}",
                self.font_small,
                WHITE,
                (WIDTH // 2, HEIGHT - GROUND_HEIGHT // 2 + 5),
            )
        elif self.state == "name":
            self.draw_name_entry()
        elif self.state == "over":
            self.draw_game_over()

    @staticmethod
    def quit():
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Game().run()
