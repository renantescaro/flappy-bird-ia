"""Coloca o melhor cérebro treinado para jogar o Flappy sozinho.

Uso:
    python play_ai.py
    python play_ai.py --modelo models/melhor_cerebro.npz

Controles na janela:
    1..4   velocidade (1x, 2x, 5x, 10x)
    R      mostra/esconde a rede neural
    ESC    sai
"""

import argparse
import sys
from pathlib import Path

import pygame

import main as game
from brain import BEST_MODEL, load_brain
from main import HEIGHT, WIDTH
from train import BG, FG, MUTED, PANEL_W, draw_network

SPEEDS = [1, 2, 5, 10]


class AIGame(game.Game):
    def __init__(self, brain):
        super().__init__()
        self.brain = brain
        self.speed_idx = 0
        self.show_net = True
        self.attempts = 0
        self.session_best = 0
        self.last = None
        self.window = pygame.display.set_mode((WIDTH + PANEL_W, HEIGHT))
        self.screen = self.window.subsurface((0, 0, WIDTH, HEIGHT))
        pygame.display.set_caption("Flappy - IA jogando")
        self.font_ui = pygame.font.SysFont("consolas", 16)
        self.font_ui_b = pygame.font.SysFont("consolas", 18, bold=True)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.quit()
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.quit()
        elif pygame.K_1 <= event.key <= pygame.K_4:
            self.speed_idx = event.key - pygame.K_1
        elif event.key == pygame.K_r:
            self.show_net = not self.show_net

    def update(self):
        if self.state == "ready":
            self.state = "playing"
            self.attempts += 1
        if self.state == "playing":
            flap, *self.last = self.brain.decide(self.bird, self.pipes, self.pipe_speed)
            if flap:
                self.bird.flap()
        super().update()
        if self.state == "name":  # morreu: guarda o resultado e recomeça sozinho
            self.session_best = max(self.session_best, self.score)
            self.reset()

    def run(self):
        while True:
            for event in pygame.event.get():
                self.handle_event(event)
            for _ in range(SPEEDS[self.speed_idx]):
                self.update()
            self.draw()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(game.FPS)

    def draw_panel(self):
        px = WIDTH
        pygame.draw.rect(self.window, BG, (px, 0, PANEL_W, HEIGHT))
        title = self.font_small.render("IA JOGANDO", True, game.BIRD_YELLOW)
        self.window.blit(title, (px + 16, 14))
        rows = [
            ("Modelo", f"geração {self.brain.generation}"),
            ("Recorde treino", str(self.brain.score)),
            ("Tentativa", str(self.attempts)),
            ("Score atual", str(self.score)),
            ("Melhor sessão", str(max(self.session_best, self.score))),
            ("Velocidade", f"{SPEEDS[self.speed_idx]}x"),
        ]
        y = 52
        for label, value in rows:
            self.window.blit(self.font_ui.render(label, True, MUTED), (px + 16, y))
            self.window.blit(self.font_ui_b.render(value, True, FG), (px + 170, y - 1))
            y += 22
        if self.show_net and self.last:
            x_in, hidden, out = self.last
            draw_network(
                self.window,
                self.font_ui,
                self.brain.genome[0],
                x_in,
                hidden,
                out,
                (px, y + 20, PANEL_W, 240),
            )
        help_text = self.font_ui.render(
            "1-4: velocidade  R: rede  ESC: sair", True, MUTED
        )
        self.window.blit(help_text, (px + 16, HEIGHT - 30))


def main():
    parser = argparse.ArgumentParser(description="A IA treinada joga o Flappy sozinha.")
    parser.add_argument(
        "--modelo", type=Path, default=BEST_MODEL, help="arquivo .npz do cérebro"
    )
    args = parser.parse_args()
    if not args.modelo.exists():
        print(f"Modelo não encontrado em {args.modelo}. Rode primeiro: python train.py")
        sys.exit(1)
    brain = load_brain(args.modelo)
    print(f"Carregado {args.modelo} (geração {brain.generation}, score {brain.score}).")
    AIGame(brain).run()


if __name__ == "__main__":
    main()
