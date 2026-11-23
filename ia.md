# Como a IA funciona

A IA do Flappy é uma **rede neural** com funções de ativação, treinada por **neuroevolução**, isto é, por um **algoritmo genético**. **Não há backpropagation**: os pesos não são ajustados por gradiente, e sim por seleção, cruzamento e mutação entre gerações de pássaros.

Código: `brain.py` (rede e algoritmo genético) e `train.py` (simulação e treino).

---

## 1. Arquitetura da rede

```
 7 entradas  ──►  8 neurônios ocultos (tanh)  ──►  1 saída (sigmoide)  ──►  pula se > 0,5
```

| Camada | Pesos | Biases | Total |
|---|---|---|---|
| Entrada → oculta | 8 × 7 = 56 | 8 | 64 |
| Oculta → saída | 1 × 8 = 8 | 1 | 9 |
| **Total** | | | **73** |

Esses 73 números formam o **genoma** do pássaro. Eles ficam guardados num vetor, nesta ordem: `W1` (56), `b1` (8), `W2` (8), `b2` (1).

O cálculo da rede a cada quadro do jogo é:

```
z_oculto = W1 · x + b1           (8 somas ponderadas)
h        = tanh(z_oculto)        (ativação da camada oculta)
z_saida  = W2 · h + b2           (1 soma ponderada)
saída    = sigmoide(z_saida)     (valor entre 0 e 1)
pula     = saída > 0,5
```

---

## 2. As entradas

A cada quadro, a rede recebe 7 números calculados a partir do estado do jogo. Todos são normalizados, para ficarem em escalas parecidas (perto de -1 a 1):

| # | Nome | Fórmula | Significado |
|---|---|---|---|
| x₁ | altura | `y / 600` | 0 = topo da tela, 1 = fundo |
| x₂ | velocidade | `vel / 10` | negativo = subindo, positivo = caindo |
| x₃ | dist. cano | `(cano.x + 70 − pássaro.x) / 400` | quanto falta para passar o cano |
| x₄ | topo vão | `(topo_vão − y) / 600 × 2` | negativo = o topo do vão está acima do pássaro |
| x₅ | base vão | `(base_vão − y) / 600 × 2` | positivo = a base do vão está abaixo do pássaro |
| x₆ | vão seguinte | `(centro_vão_seguinte − y) / 600 × 2` | onde está a abertura do cano depois desse |
| x₇ | vel. canos | `velocidade / 5,5` | o quão rápido está o jogo (nível) |

---

## 3. Funções de ativação

### tanh (camada oculta)

```
tanh(z) = (e^z − e^(−z)) / (e^z + e^(−z))       resultado entre −1 e 1
```

| z | −3 | −1 | 0 | 1 | 3 |
|---|---|---|---|---|---|
| tanh(z) | −0,995 | −0,762 | 0 | 0,762 | 0,995 |

Ela é **não linear**, e é isso que permite à rede aprender regras combinadas, como "se estou caindo **e** abaixo do vão, então pula". Sem ativação, a rede inteira se reduziria a uma única conta linear.

### sigmoide (saída)

```
sigmoide(z) = 1 / (1 + e^(−z))                   resultado entre 0 e 1
```

| z | −4 | −2 | 0 | 2 | 4 |
|---|---|---|---|---|---|
| sigmoide(z) | 0,018 | 0,119 | 0,5 | 0,881 | 0,982 |

O resultado funciona como uma "vontade de pular". Acima de **0,5**, o pássaro bate as asas.

---

## 4. Exemplo manual completo

Os pesos abaixo são **ilustrativos**: foram arredondados para 1 casa decimal, para facilitar as contas à mão. Uma rede treinada tem pesos parecidos, só que com mais casas decimais.

### Pesos usados

**W1** (entrada → oculta) e **b1**:

| | altura | veloc. | dist. | topo | base | seguinte | vel. canos | **b1** |
|---|---|---|---|---|---|---|---|---|
| h₁ | −0,4 | 0,0 | −0,2 | −0,2 | 0,1 | −0,2 | 0,3 | **1,8** |
| h₂ | 0,0 | 0,6 | −0,7 | −0,4 | −0,9 | −1,7 | −2,2 | **1,0** |
| h₃ | 0,1 | −0,8 | −0,8 | −0,6 | 0,4 | 0,6 | 0,7 | **0,6** |
| h₄ | 0,9 | 3,0 | 0,3 | 0,2 | −2,2 | −1,7 | 2,2 | **0,7** |
| h₅ | 2,7 | 0,4 | −0,1 | 0,6 | −2,0 | −0,5 | −0,4 | **0,9** |
| h₆ | 1,7 | −1,0 | 0,4 | 0,7 | 1,1 | −0,4 | −0,7 | **−1,3** |
| h₇ | −2,2 | 1,1 | 1,2 | −0,5 | 0,2 | −1,4 | −0,2 | **1,1** |
| h₈ | 0,7 | −0,5 | 0,1 | 1,1 | 0,6 | −0,6 | 1,6 | **−0,8** |

**W2** (oculta → saída) e **b2**:

| h₁ | h₂ | h₃ | h₄ | h₅ | h₆ | h₇ | h₈ | **b2** |
|---|---|---|---|---|---|---|---|---|
| 0,2 | −0,1 | 0,5 | 1,8 | 0,5 | −0,1 | 0,7 | 0,1 | **−1,7** |

---

### Cenário A: pássaro abaixo do vão e caindo → deve PULAR

Estado do jogo (nível 1):
- pássaro em `x = 133`, `y = 300`, caindo com velocidade `3,0`
- próximo cano em `x = 250`, com vão de `y = 180` até `y = 344`
- centro do vão seguinte em `y = 302`
- velocidade dos canos: `3,25`

**Passo 1: calcular as entradas**

| Entrada | Conta | Valor |
|---|---|---|
| x₁ altura | 300 / 600 | **0,50** |
| x₂ velocidade | 3,0 / 10 | **0,30** |
| x₃ dist. cano | (250 + 70 − 133) / 400 = 187 / 400 | **0,47** |
| x₄ topo vão | (180 − 300) / 600 × 2 | **−0,40** |
| x₅ base vão | (344 − 300) / 600 × 2 | **0,15** |
| x₆ vão seguinte | (302 − 300) / 600 × 2 | **0,01** |
| x₇ vel. canos | 3,25 / 5,5 | **0,59** |

**Passo 2: um neurônio oculto passo a passo (h₄)**

Cada neurônio multiplica cada entrada pelo seu peso, soma tudo, soma o bias e aplica a tanh:

```
z₄ = (0,9 × 0,50) + (3,0 × 0,30) + (0,3 × 0,47) + (0,2 × −0,40)
   + (−2,2 × 0,15) + (−1,7 × 0,01) + (2,2 × 0,59) + 0,7

   =  0,450 + 0,900 + 0,141 − 0,080 − 0,330 − 0,017 + 1,298 + 0,7

   =  2,362 + 0,7  =  3,062

h₄ = tanh(3,062)
   = (e^3,062 − e^−3,062) / (e^3,062 + e^−3,062)
   = (21,37 − 0,047) / (21,37 + 0,047)
   = 0,996
```

O h₄ tem peso **+3,0** na velocidade: quanto mais rápido o pássaro cai, mais esse neurônio "acende". Aqui ele está quase no máximo (0,996).

**Passo 3: todos os neurônios ocultos**

| Neurônio | Produtos (peso × entrada) | Soma | + bias | z | **h = tanh(z)** |
|---|---|---|---|---|---|
| h₁ | −0,200 + 0 − 0,094 + 0,080 + 0,015 − 0,002 + 0,177 | −0,024 | +1,8 | 1,776 | **0,944** |
| h₂ | 0 + 0,180 − 0,329 + 0,160 − 0,135 − 0,017 − 1,298 | −1,439 | +1,0 | −0,439 | **−0,413** |
| h₃ | 0,050 − 0,240 − 0,376 + 0,240 + 0,060 + 0,006 + 0,413 | 0,153 | +0,6 | 0,753 | **0,637** |
| h₄ | 0,450 + 0,900 + 0,141 − 0,080 − 0,330 − 0,017 + 1,298 | 2,362 | +0,7 | 3,062 | **0,996** |
| h₅ | 1,350 + 0,120 − 0,047 − 0,240 − 0,300 − 0,005 − 0,236 | 0,642 | +0,9 | 1,542 | **0,912** |
| h₆ | 0,850 − 0,300 + 0,188 − 0,280 + 0,165 − 0,004 − 0,413 | 0,206 | −1,3 | −1,094 | **−0,798** |
| h₇ | −1,100 + 0,330 + 0,564 + 0,200 + 0,030 − 0,014 − 0,118 | −0,108 | +1,1 | 0,992 | **0,758** |
| h₈ | 0,350 − 0,150 + 0,047 − 0,440 + 0,090 − 0,006 + 0,944 | 0,835 | −0,8 | 0,035 | **0,035** |

**Passo 4: a saída**

```
z_saida = (0,2 × 0,944) + (−0,1 × −0,413) + (0,5 × 0,637) + (1,8 × 0,996)
        + (0,5 × 0,912) + (−0,1 × −0,798) + (0,7 × 0,758) + (0,1 × 0,035) + (−1,7)

        = 0,189 + 0,041 + 0,318 + 1,793 + 0,456 + 0,080 + 0,531 + 0,004 − 1,7

        = 3,411 − 1,7  =  1,711

saída = sigmoide(1,711) = 1 / (1 + e^−1,711) = 1 / (1 + 0,181) = 0,847
```

**Resultado: 0,847 > 0,5 → PULA** ✅

O maior responsável é o h₄: `1,8 × 0,996 = 1,793`, mais da metade da soma.

---

### Cenário B: pássaro no vão e subindo rápido → NÃO deve pular

Mesmo cano, mas agora o pássaro está em `y = 200`, **subindo** com velocidade `−5,0`.

**Entradas**

| x₁ | x₂ | x₃ | x₄ | x₅ | x₆ | x₇ |
|---|---|---|---|---|---|---|
| 200/600 = **0,33** | −5/10 = **−0,50** | **0,47** | (180−200)/300 = **−0,07** | (344−200)/300 = **0,48** | (302−200)/300 = **0,34** | **0,59** |

**Camada oculta**

| Neurônio | Soma dos produtos | + bias | z | **h** |
|---|---|---|---|---|
| h₁ | −0,055 | +1,8 | 1,745 | **0,941** |
| h₂ | −2,909 | +1,0 | −1,909 | **−0,957** |
| h₃ | 0,908 | +0,6 | 1,508 | **0,907** |
| h₄ | −1,412 | +0,7 | −0,712 | **−0,612** |
| h₅ | −0,764 | +0,9 | 0,136 | **0,135** |
| h₆ | 1,179 | −1,3 | −0,121 | **−0,120** |
| h₇ | −1,175 | +1,1 | −0,075 | **−0,075** |
| h₈ | 1,479 | −0,8 | 0,679 | **0,591** |

Veja o h₄: a soma dele inclui `3,0 × −0,50 = −1,500` (subindo) e `−2,2 × 0,48 = −1,056` (longe da base do vão). Por isso ele fica **negativo**, ao contrário do cenário A.

**Saída**

```
z_saida = 0,188 + 0,096 + 0,453 − 1,101 + 0,068 + 0,012 − 0,052 + 0,059 − 1,7
        = −0,277 − 1,7  =  −1,977

saída = sigmoide(−1,977) = 1 / (1 + e^1,977) = 1 / (1 + 7,22) = 0,122
```

**Resultado: 0,122 < 0,5 → NÃO pula** ✅

O mesmo h₄, que antes empurrava para pular (+1,793), agora empurra para não pular (−1,101).

---

## 5. Como os pesos são aprendidos (algoritmo genético)

No Flappy não existe uma "resposta certa" para cada quadro. Só se sabe, no fim da partida, se o pássaro foi bem ou mal. Por isso não dá para usar backpropagation diretamente. Em vez disso, o treino imita a seleção natural.

### 5.1 Fitness (nota de cada pássaro)

```
fitness = quadros vivo + 100 × canos passados − 0,5 × distância do centro do vão ao morrer
```

| Pássaro | Quadros | Canos | Distância ao morrer | Conta | Fitness |
|---|---|---|---|---|---|
| A | 1240 | 15 | 30 px | 1240 + 1500 − 15 | **2725** |
| B | 200 | 0 | 20 px | 200 + 0 − 10 | **190** |
| C | 180 | 0 | 120 px | 180 + 0 − 60 | **120** |

B e C não passaram nenhum cano, mas B morreu mais perto do vão e por isso leva nota maior. No começo do treino é isso que diferencia os pássaros.

### 5.2 Montagem da próxima geração (população de 150)

| Parte | Quantidade | Como é criada |
|---|---|---|
| Elite | 7 (5%) | os 7 melhores, copiados **sem alteração** |
| Filhos | 136 (90%) | cruzamento + mutação de pais sorteados entre os 37 melhores (25%) |
| Novatos | 7 (5%) | pesos totalmente aleatórios (diversidade) |

O sorteio dos pais é **ponderado pelo ranking**: o 1º colocado tem peso 37, o 2º tem 36 e assim por diante, até o 37º, que tem peso 1.

### 5.3 Exemplo de cruzamento (6 dos 73 genes)

Cada gene vem do pai A ou do pai B, com 50% de chance cada (cruzamento uniforme):

| Gene | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Pai A | 0,9 | 3,0 | 0,3 | 0,2 | −2,2 | −1,7 |
| Pai B | 1,4 | 2,1 | −0,5 | 0,8 | −1,9 | 0,4 |
| Sorteio | A | B | A | A | B | B |
| **Filho** | **0,9** | **2,1** | **0,3** | **0,2** | **−1,9** | **0,4** |

### 5.4 Exemplo de mutação

Cada gene tem **15%** de chance de receber um ruído gaussiano (média 0, desvio padrão 0,5):

| Gene | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Filho | 0,9 | 2,1 | 0,3 | 0,2 | −1,9 | 0,4 |
| Muta? (15%) | não | **sim** | não | não | não | **sim** |
| Ruído | — | +0,37 | — | — | — | −0,62 |
| **Resultado** | **0,9** | **2,47** | **0,3** | **0,2** | **−1,9** | **−0,22** |

### 5.5 O ciclo

```
geração 1: 150 redes aleatórias
     │
     ▼
todos jogam no MESMO cenário ──► calcula a fitness de cada um
     │
     ▼
salva o melhor se bateu o recorde (models/melhor_cerebro.npz)
     │
     ▼
elite + filhos (cruzamento + mutação) + novatos = nova geração
     │
     └──► repete
```

Ao longo das gerações, as combinações de pesos que funcionam se espalham pela população, e as mutações testam variações em torno delas. O resultado é parecido com o "neurônio h₄" do exemplo: neurônios que se especializam em detectar situações como "caindo e abaixo do vão".

---

## 6. Backpropagation × Neuroevolução

| | Backpropagation | Neuroevolução (este projeto) |
|---|---|---|
| Precisa de | erro por decisão (gabarito) ou recompensa por passo | só uma nota no fim da partida |
| Ajuste dos pesos | gradiente (derivadas) | seleção + cruzamento + mutação |
| Avalia | uma rede, muitas vezes | muitas redes ao mesmo tempo |
| Bom para | redes grandes, muitos dados | redes pequenas, objetivo fácil de medir |

Com 73 parâmetros e um objetivo simples ("sobreviver"), a neuroevolução é simples de implementar e converge rápido.
