# Flappy + IA

Jogo estilo Flappy Bird em Python (pygame), com uma rede neural que aprende a jogar sozinha por neuroevolução.

<img width="781" height="629" alt="Sleekshot 2026-11-23 14-12-28" src="https://github.com/user-attachments/assets/dc66f60c-656b-45f7-9d9e-90a04d855d3f" />



## Instalação

```powershell
python -m venv venv
.\venv\Scripts\python -m pip install -r requirements.txt
```

## Jogar

```powershell
.\venv\Scripts\python main.py
```

- **Espaço**: bate as asas (também inicia e reinicia o jogo)
- **Esc**: sai
- Cada cano vale 1 ponto, e a dificuldade sobe a cada 10 canos.
- Ao morrer, digite um nome de 5 caracteres e aperte **Enter**. O score vai para o `scores.txt` e aparece o Top 5.

## Treinar a IA

```powershell
.\venv\Scripts\python train.py
```

Uma janela mostra todos os pássaros de cada geração jogando juntos. Ao lado aparecem estatísticas, a rede neural do pássaro em destaque e um gráfico da evolução do score.

- **1–6**: velocidade da simulação (1x até a máxima)
- **N**: encerra a geração atual
- **Esc**: salva e sai

O treino continua de onde parou na próxima execução. Opções:

| Opção | Descrição |
|---|---|
| `--novo` | começa o treino do zero |
| `--populacao 200` | pássaros por geração (padrão 150) |
| `--score-max 1000` | encerra a geração quando a IA chega nesse score (padrão 500) |
| `--seed 42` | semente aleatória, para repetir um treino |

## Ver a IA jogando

```powershell
.\venv\Scripts\python play_ai.py
```

Carrega o melhor cérebro salvo e joga sozinho, recomeçando a cada morte.

- **1–4**: velocidade
- **R**: mostra ou esconde a rede neural
- **Esc**: sai

Para usar outro modelo: `--modelo caminho\do\arquivo.npz`.

## Como funciona

Cada pássaro tem uma rede neural com **7 entradas, 8 neurônios ocultos e 1 saída** (pular ou não). As entradas são: altura, velocidade, distância até o cano, topo e base da abertura, abertura do cano seguinte e velocidade dos canos.

A cada geração, os pássaros que foram mais longe passam direto para a próxima ou são cruzados e sofrem mutações nos pesos. Com o tempo, a população aprende a jogar.

Explicação detalhada, com contas feitas à mão: [ia.md](ia.md).

Arquivos gerados:

- `models/melhor_cerebro.npz`: o cérebro do recorde, usado pelo `play_ai.py`
- `models/populacao.npz`: o progresso do treino, para continuar depois
- `scores.txt`: o ranking dos jogadores

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `main.py` | o jogo |
| `brain.py` | a rede neural e o algoritmo genético |
| `train.py` | o treinamento visual |
| `play_ai.py` | a IA jogando com o melhor modelo salvo |
