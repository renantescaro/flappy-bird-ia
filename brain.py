"""Rede neural + algoritmo genético (neuroevolução) para o Flappy.

Cada pássaro tem um "cérebro": uma rede neural pequena (7 entradas -> 8 ocultos -> 1 saída).
Os pesos da rede formam o "genoma". A cada geração os melhores pássaros são
selecionados, cruzados e sofrem mutações para formar a próxima geração.
"""

from pathlib import Path

import numpy as np

from main import HEIGHT, MAX_FALL_SPEED, MAX_PIPE_SPEED, PIPE_WIDTH, WIDTH, Bird

N_INPUTS = 7
N_HIDDEN = 8
N_OUTPUTS = 1
INPUT_LABELS = (
    "altura",
    "velocidade",
    "dist. cano",
    "topo vão",
    "base vão",
    "vão seguinte",
    "vel. canos",
)
GENOME_SIZE = N_HIDDEN * N_INPUTS + N_HIDDEN + N_OUTPUTS * N_HIDDEN + N_OUTPUTS
FLAP_THRESHOLD = 0.5

MODELS_DIR = Path(__file__).with_name("models")
BEST_MODEL = MODELS_DIR / "melhor_cerebro.npz"
CHECKPOINT = MODELS_DIR / "populacao.npz"


# ---------------------------------------------------------------------------
# Rede neural (vetorizada: processa a população inteira de uma vez)
# ---------------------------------------------------------------------------
def unpack(genomes):
    """Converte genomas (P, G) em matrizes de pesos de cada camada."""
    p = genomes.shape[0]
    idx = 0

    def take(shape):
        nonlocal idx
        n = int(np.prod(shape))
        arr = genomes[:, idx : idx + n].reshape(p, *shape)
        idx += n
        return arr

    w1 = take((N_HIDDEN, N_INPUTS))
    b1 = take((N_HIDDEN,))
    w2 = take((N_OUTPUTS, N_HIDDEN))
    b2 = take((N_OUTPUTS,))
    return w1, b1, w2, b2


def forward(genomes, x):
    """x: (P, N_INPUTS). Retorna (ativações ocultas (P, H), saída (P,))."""
    w1, b1, w2, b2 = unpack(genomes)
    hidden = np.tanh(np.einsum("phi,pi->ph", w1, x) + b1)
    z = np.einsum("poh,ph->po", w2, hidden) + b2
    out = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    return hidden, out[:, 0]


# ---------------------------------------------------------------------------
# Sensores: o que o pássaro "enxerga"
# ---------------------------------------------------------------------------
def next_pipe(pipes, bird_x):
    for pipe in pipes:
        if pipe.x + PIPE_WIDTH > bird_x - Bird.RADIUS:
            return pipe
    return None


def sense(ys, vels, pipes, pipe_speed, bird_x):
    """Monta as entradas da rede para vários pássaros (mesmo x, alturas diferentes)."""
    ys = np.asarray(ys, dtype=float)
    vels = np.asarray(vels, dtype=float)
    ahead = [p for p in pipes if p.x + PIPE_WIDTH > bird_x - Bird.RADIUS]
    default_center = HEIGHT / 2 - 40
    if ahead:
        pipe = ahead[0]
        dx = (pipe.x + PIPE_WIDTH - bird_x) / WIDTH
        gap_top, gap_bottom = pipe.gap_y, pipe.gap_y + pipe.gap
    else:
        dx = 1.0
        gap_top, gap_bottom = default_center - 80, default_center + 80
    # centro do vão do cano seguinte (permite planejar a transição entre canos)
    next_center = (
        (ahead[1].gap_y + ahead[1].gap / 2)
        if len(ahead) > 1
        else (gap_top + gap_bottom) / 2
    )
    x = np.empty((len(ys), N_INPUTS))
    x[:, 0] = ys / HEIGHT
    x[:, 1] = vels / MAX_FALL_SPEED
    x[:, 2] = dx
    x[:, 3] = (gap_top - ys) / HEIGHT * 2
    x[:, 4] = (gap_bottom - ys) / HEIGHT * 2
    x[:, 5] = (next_center - ys) / HEIGHT * 2
    x[:, 6] = pipe_speed / MAX_PIPE_SPEED
    return x


# ---------------------------------------------------------------------------
# Algoritmo genético
# ---------------------------------------------------------------------------
class Population:
    def __init__(self, genomes, rng=None):
        self.genomes = np.asarray(genomes, dtype=float)
        self.rng = rng or np.random.default_rng()

    @property
    def size(self):
        return self.genomes.shape[0]

    @classmethod
    def random(cls, size, rng=None):
        rng = rng or np.random.default_rng()
        return cls(rng.normal(0, 1, (size, GENOME_SIZE)), rng)

    def forward(self, x):
        return forward(self.genomes, x)

    def evolve(self, fitness, mutation_rate=0.15, mutation_strength=0.5):
        """Gera a próxima população. O índice 0 sempre é o melhor da geração anterior."""
        rng = self.rng
        size = self.size
        ranked = self.genomes[np.argsort(fitness)[::-1]]

        n_elite = max(2, size // 20)  # 5% passam intactos
        n_random = max(1, size // 20)  # 5% totalmente novos (diversidade)
        n_children = size - n_elite - n_random

        # seleção: os 25% melhores podem ser pais, com chance maior para os primeiros
        pool = ranked[: max(4, size // 4)]
        weights = np.arange(len(pool), 0, -1, dtype=float)
        weights /= weights.sum()
        a = pool[rng.choice(len(pool), n_children, p=weights)]
        b = pool[rng.choice(len(pool), n_children, p=weights)]

        # cruzamento uniforme + mutação gaussiana
        children = np.where(rng.random(a.shape) < 0.5, a, b)
        mutate = rng.random(children.shape) < mutation_rate
        children = children + mutate * rng.normal(0, mutation_strength, children.shape)

        newcomers = rng.normal(0, 1, (n_random, GENOME_SIZE))
        return Population(np.concatenate([ranked[:n_elite], children, newcomers]), rng)


# ---------------------------------------------------------------------------
# Salvar / carregar
# ---------------------------------------------------------------------------
def _check_layers(data, path):
    layers = tuple(int(v) for v in data["layers"])
    if layers != (N_INPUTS, N_HIDDEN, N_OUTPUTS):
        raise ValueError(f"{path} foi salvo com outra arquitetura de rede: {layers}")


def save_brain(genome, generation, score, path=BEST_MODEL):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        genome=genome,
        generation=generation,
        score=score,
        layers=np.array([N_INPUTS, N_HIDDEN, N_OUTPUTS]),
    )


def load_brain(path=BEST_MODEL):
    with np.load(path) as data:
        _check_layers(data, path)
        return Brain(data["genome"].copy(), int(data["generation"]), int(data["score"]))


def save_checkpoint(population, generation, best_score, history, path=CHECKPOINT):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        genomes=population.genomes,
        generation=generation,
        best_score=best_score,
        history=np.array(history, dtype=float).reshape(-1, 2),
        layers=np.array([N_INPUTS, N_HIDDEN, N_OUTPUTS]),
    )


def load_checkpoint(path=CHECKPOINT):
    with np.load(path) as data:
        _check_layers(data, path)
        history = [tuple(row) for row in data["history"]]
        return (
            Population(data["genomes"].copy()),
            int(data["generation"]),
            int(data["best_score"]),
            history,
        )


class Brain:
    """Um único cérebro já treinado (usado para jogar sozinho)."""

    def __init__(self, genome, generation=0, score=0):
        self.genome = np.asarray(genome, dtype=float).reshape(1, -1)
        self.generation = generation
        self.score = score

    def decide(self, bird, pipes, pipe_speed):
        x = sense([bird.y], [bird.vel], pipes, pipe_speed, bird.x)
        hidden, out = forward(self.genome, x)
        return out[0] > FLAP_THRESHOLD, x[0], hidden[0], out[0]
