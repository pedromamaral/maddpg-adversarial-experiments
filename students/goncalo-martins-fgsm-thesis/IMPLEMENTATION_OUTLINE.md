# Capítulo 7 — Implementação: estrutura revista

Esta é a tua estrutura, com as alterações marcadas. A ordem mantém-se: vítima (as variações MADDPG) → ambiente →
MDP → treino → ataques → controlos → protocolo. O que mudou vem das experiências desta semana.

Legenda: **[MANTÉM]** está bem como está · **[AJUSTAR]** muda o conteúdo · **[NOVO]** falta
acrescentar.

---

## 0. Organização da Implementação — [MANTÉM]

Bibliotecas, simulador próprio, classes e programa de execução.


## 1. Sistema-Vítima — [MANTÉM]

- Proveniência dos pesos (estudo comparativo das variantes).
- As sete variantes avaliadas, com remissão para a Tabela das Variantes.
- Fase de avaliação: pesos congelados, sem exploração nem atualizações, decisão pelo
  `argmax`; só o ator e o codificador intervêm.
- **[NOVO]** Uma frase sobre o **carregamento dos pesos**: a avaliação verifica que os pesos
  treinados foram de facto carregados e recusa continuar se não os encontrar. Isto não é um
  pormenor: numa das execuções antigas as pastas de modelos estavam vazias, e a linha da
  política da antiga T7 veio de uma rede sem pesos treinados (Secção 8.4). É também por isso
  que a linha da política da T7 vem da varredura de carga da fase 2.

## 2. Ambiente de Simulação — [MANTÉM], com três acrescentos

- Topologia: rede de um *service provider*, tipos de nó (núcleo, distribuição, acesso),
  parâmetros das ligações.
- Caminhos pré-calculados: como são obtidos os K = 3 caminhos por par, com o critério de
  diferença de saltos.
  - **[NOVO]** Diz o que acontece quando um par tem **menos de 3 caminhos distintos**: a lista
    é completada **repetindo o caminho mais curto**. Acontece em 37 % dos pares (agente,
    destino). É isto que explica as inversões "só no papel" da Secção 8.3 (8–22 % das
    inversões trocam entre duas cópias do mesmo caminho).
- Ciclo do passo: o `step()`, com a ação vinda do ator, a aplicação ao ambiente, e o estado
  seguinte e a recompensa que saem.
- Fluxos: como são criados, quanto tempo ocupam a largura de banda, quantos pacotes injetam.
- Regime de carga: *load factor* e *hotspot* (75 % dos fluxos das oito origens BS1–BS4 e
  MECS1–MECS4 para CS1 e CS2).
- Falhas de ligações: como são sorteadas, os valores testados, e a consequência do grafo fixo
  do codificador.
- Sementes: a de tráfego, fixa em todas as execuções, e a de treino, que gera vítimas
  diferentes por variante.
  - **[NOVO]** Acrescenta a **terceira semente**: a do sorteio dos agentes comprometidos
    (Secção 5.4), independente da de tráfego.
  - **[NOVO]** Diz que as variantes GNN estão a ser treinadas com mais duas sementes
    (resultados pendentes). A T3b ainda as mostra com uma só.

## 3. Estado, Ação e Recompensa — [AJUSTAR]

- Valores concretos das dimensões.
- **[AJUSTAR]** "Truncamento 97/96" está incompleto. São **três números**:
  - o ambiente monta **97** características;
  - a eq. 4.3 dá **96**;
  - o ator consome **94** (`state[:94]`).

  O que se perde: o sinal de *mean-hops* e as **duas últimas utilizações de caminho do último
  destino**. Remete daqui para a Secção 5.1 ("61 de 63 posições"), que é a consequência disto.
- Normalizações.
- **[NOVO]** A ação como cadeia completa, porque a Secção 5 precisa dela: o ator produz
  **logits** z (63 valores), a saída é a = σ(z) com **sigmoides independentes** (não é um
  *softmax*; as saídas não somam 1), e a rota é o `argmax` de cada bloco de K = 3, que é o
  mesmo em z e em a.
- Pesos da recompensa e o `backlog_penalty` a zero.

## 4. Configuração do Treino — [MANTÉM]

- Hiperparâmetros a partir da Tabela VI do Paper 1.
- *Early stopping* e escolha do *best checkpoint*.
- Divergências nas redes (sigmoide, *Duelling* degenerado, soma na GNN).
  - **[NOVO]** No codificador GNN, diz explicitamente que ele lê as observações **dos 14
    agentes** (mais os nós sem agente, com características a zero). Logo, a decisão de um
    agente depende das observações de todos. A Secção 5.2 assenta nisto.

## 5. Implementação dos Ataques — [AJUSTAR], é a secção que mais muda

Sugiro dividi-la em subsecções:

### 5.1 FGSM — [MANTÉM], com uma divergência explicada

- ε testados, c = 0.5 e λ = 10, com λ **dentro** da sigmoide: σ(λ(u − c)). Confirma que a
  eq. 6.2 fica igual.
- O u é lido da observação limpa e tratado como constante: não há gradiente através dele.
- Divergências:
  - o *clamp* só reprojeta as **4 primeiras** componentes (quatro das seis larguras de
    banda); o U_i **nunca** é reprojetado em [0, 1];
  - o objetivo só alcança **61 das 63** posições (consequência do truncamento, Secção 3).

### 5.2 O gradiente nas variantes GNN — **[NOVO]**, a mais importante

- A vítima GNN decide a partir do codificador aplicado às observações de todos os agentes.
  Para ser *white-box*, o ataque tem de derivar **esse** caminho: o codificador, com as
  observações limpas dos outros agentes como contexto e a do agente atacado a variar.
- **Nota de honestidade:** o código original derivava só o ator aplicado à observação do
  agente, sem passar pelo codificador. Os números GNN antigos (a antiga Secção 8.7) vinham
  daí. Todos os números GNN da tese usam o ataque corrigido.
- Consequência para o modelo de ameaça (liga ao 6.1): contra uma vítima GNN, o atacante
  precisa das observações limpas de **todos** os agentes.

### 5.3 PGD e MI-FGSM — [MANTÉM]

- Configurações e varrimento (tamanho do passo, iterações, momento, reinício aleatório).
- Nota sobre a correção do objetivo (o alvo passou a ser fixado na observação limpa; o FGSM
  não é afetado, verificado por *checksum*).
- **[NOVO]** Diz como o diagnóstico da T8 avalia as variantes GNN (codificador com as
  observações dos outros agentes a zero), porque isso limita a leitura das linhas GNN da T8.

### 5.4 Ataque sobre os logits — **[NOVO]**

- Os dois objetivos, sobre z e não sobre σ(z):
  - "congestionamento": Σ_d (z_{d,pior} − z_{d,escolhido});
  - "margem": Σ_d (max_{k≠escolhido} z_{d,k} − z_{d,escolhido}).
- O caminho "escolhido" vem da observação limpa. Confirma no código
  (`_logit_margin_objective`) que blocos em que a política já está no pior caminho, ou sem
  utilização disponível (as 2 posições truncadas), não contribuem.
- Tudo o resto igual ao FGSM: mesmo ε, um passo com sinal, mesmo *clamp*.
- Porquê: uma frase sobre a saturação da sigmoide, com remissão para a Secção 8.9.

### 5.5 Compromisso parcial — **[NOVO]**

- Fração de agentes comprometidos (1, 4, 7 e 14 de 14).
- O conjunto é **sorteado de novo em cada episódio**, com uma semente própria, separada da
  de tráfego.
- Quatro sorteios independentes por célula, com o tráfego fixo (Secção 8.10).

## 6. Execuções de Controlo e Referência — [AJUSTAR]

- **[AJUSTAR]** Controlo aleatório: diz **qual**. Cada componente é perturbada em ±ε, com
  sinal aleatório, e passa pelo mesmo *clamp*. É o mesmo orçamento que o ataque, sem direção.
  (É diferente do do Miguel, que sorteia uniformemente na bola; a comparação entre as duas
  teses depende disto.)
- Regras de encaminhamento de referência, que não usam a política:
  - *greedy*: o caminho menos congestionado dos três (não é um ótimo);
  - **[NOVO]** *aleatória*: um caminho ao acaso em cada passo (entra agora na T7);
  - *worst*: o caminho mais congestionado dos três.
- **[NOVO]** Teto de dano = entrega da política limpa − entrega com a regra *worst*, medido
  por vítima (Secção 8.4).

## 7. Protocolo de Avaliação e Métricas — [AJUSTAR]

- Emparelhamento (remissão para as sementes da Secção 2), 15 episódios por condição.
- **[NOVO]** Intervalos de confiança a 95 % com a distribuição *t* de Student, 14 graus de
  liberdade (t = 2.145), sobre as diferenças por episódio.
- Métricas principais: entrega (PDR), taxa de inversões por (agente, destino), efeito
  adversarial (queda com ataque − queda com aleatório).
- Métricas derivadas: fração do teto de dano.
- **[NOVO]** Métricas de diagnóstico, que a Secção 8 cita:
  - **saturação**: fração de decisões em que a saída do caminho escolhido passa de 0.99
    (Secção 8.9);
  - **inversões vazias**: inversões entre duas cópias do mesmo caminho (Secção 8.3);
  - **inversões para pior**: fração das inversões reais que levam o tráfego para um caminho
    mais congestionado (Secção 8.3);
  - **orçamento gasto**: média de |δ| / ε (Secção 8.8).

---

## Resumo das alterações

| Secção | O quê | Porquê |
|---|---|---|
| 1 | verificação do carregamento dos pesos | explica a correção da T7 |
| 2 | preenchimento dos caminhos; semente do compromisso; sementes GNN | Secções 8.3, 8.10, 8.5 |
| 3 | 97 / 96 / 94; cadeia z → σ(z) → `argmax` | base das Secções 5.1 e 5.4 |
| 4 | o codificador lê os 14 agentes | base da Secção 5.2 |
| 5.2 | gradiente pelo codificador + nota de honestidade | Secção 8.7 |
| 5.4 | ataque sobre os logits | Secção 8.9 |
| 5.5 | compromisso parcial | Secção 8.10 |
| 6 | tipo de controlo aleatório; regra aleatória; teto de dano | Secções 8.4, 8.5 |
| 7 | intervalos *t*; métricas de diagnóstico | toda a Secção 8 |
