# Capítulo 8 — Resultados: estrutura, figuras e explicações

Guia para escreveres o capítulo de resultados. Cada secção tem: **a figura**, **os números**,
**como explicar** (para escreveres por palavras tuas) e **o que NÃO podes dizer**.

> **Nada precisa de ser corrido outra vez.** Todas as figuras que já tens (8.1–8.7) continuam
> válidas: foram produzidas com FGSM de um passo, e verificou-se que a correcção recente ao
> código de ataque deixa esse caminho **bit a bit idêntico** (*digest* `41587b6c4aae1d10` antes
> e depois). Só há **uma figura nova** para produzir, na Secção 8.8.

---

## Fio condutor do capítulo

A história tem de se ler nesta ordem, porque cada secção responde à pergunta que a anterior levanta:

1. O ataque **muda decisões**? → sim (8.2)
2. Isso **muda a entrega**? → quase nada (8.3)
3. Então **quanto dano havia para fazer**? → muito mais do que o ataque consegue (8.4)
4. O pouco efeito que há é **mesmo adversarial** ou é ruído? → é pequeno em todas as
   arquiteturas, e a diferença entre elas não se reproduz entre sementes (8.5)
5. E **sob falhas**, onde a rede é frágil? → a fragilidade é ruído, não ataque (8.6)
6. O **GNN** muda alguma coisa? → suprime inversões, com uma excepção (8.7)
7. O ataque não estaria simplesmente **mal afinado**? → não, o FGSM é o melhor ataque míope (8.8)

8. (opcional) Quantos agentes precisa o atacante de controlar? → o dano cresce com o número
   deles, e um só agente quase não chega (8.9)

O ponto 7 é o que protege todo o capítulo. Sem ele, qualquer arguente pergunta
"o seu resultado negativo não será só porque o ataque é fraco?".

---

## 8.1 Condições de Avaliação

Mantém o que já tens. Corrige apenas as referências `??` (pesos pré-treinados → Cap. 5;
emparelhamento de episódios → Cap. 7).

Deves deixar explícito, porque tudo o resto depende disto:

- Pesos **congelados**, sem qualquer atualização durante a avaliação.
- Política **determinística** na avaliação: `argmax` por destino, **sem exploração**.
  (A exploração ε-greedy é só de treino. É isto que torna a métrica de inversões bem definida.)
- Regime **hotspot a 2× a carga nominal**, salvo indicação em contrário.
- Nas figuras de uma só arquitetura, a variante é **CC-Simple**; nas de um só orçamento, **ε = 0.3**.
- Comparações limpo / ataque-gradiente / controlo-aleatório em **episódios emparelhados**
  (mesma semente de tráfego e mesmo sorteio de falhas) — é isso que sustenta os intervalos de confiança.

---

## 8.2 O ataque inverte decisões — **Figura T1** (a tua 8.1)

**Números** (CC-Simple, hotspot 2×):

| ε | inversões (gradiente) | inversões (aleatório) |
|---|---|---|
| 0.05 | 8.6 % | — |
| 0.10 | 14.3 % | ~3 % |
| 0.20 | 22.0 % | — |
| 0.30 | **25.9 %** | **8.6 %** |

**Como explicar.** A taxa de inversões cresce de forma monótona com o orçamento: quanto mais
perturbação, mais decisões de encaminhamento mudam. Aos 0.3, o gradiente inverte cerca de
**3× mais decisões** do que ruído aleatório com o mesmo orçamento. A conclusão desta secção é
uma só: **ao nível da decisão, o ataque funciona e não é fraco**. Isto é importante porque todo
o resto do capítulo vai mostrar que o efeito na entrega é pequeno — e é preciso deixar claro,
logo aqui, que isso não se deve a um ataque incompetente.

**O que NÃO podes dizer.** Não digas que a linha do controlo aleatório é uma curva completa —
só tem **2 pontos** (ε = 0.1 e 0.3), porque vem do `fgsm_full`. Ou corres os pontos que faltam,
ou descreve-a como "pontos de referência".

---

## 8.3 A topologia absorve as inversões — **Figura T2** (a tua 8.2)

**Números** (nominal, ε = 0.3, 7 variantes): inversões entre **1.4 % e 25.7 %**;
entrega perdida **no máximo 2.8 pp** (LC-Simple), várias variantes em ~0.

**Como explicar.** Esta é **a figura central da tese**. Mostra as duas métricas lado a lado e o
contraste é o resultado principal: muitas decisões mudam, quase nenhum pacote se perde.

O mecanismo é o seguinte: uma decisão invertida move um fluxo do caminho *k* para outro caminho
*k'* do mesmo conjunto de K caminhos pré-calculados. Esses caminhos alternativos existem porque a
topologia é redundante e, na maior parte dos casos, **também eles conseguem entregar o tráfego**.
Mudar de caminho não é o mesmo que perder o pacote. É por isso que a redundância de caminhos
funciona, sem querer, como uma defesa: absorve a perturbação ao nível da decisão antes que ela
se traduza em perda.

A frase que deves deixar assente aqui: **mudar decisões e mudar resultados são coisas diferentes**,
e a literatura confunde frequentemente as duas.

---

## 8.4 Quanto dano estava disponível — **Figura T7** (a tua 8.3)

**Como explicar.** A secção anterior mostra que se perde pouco. Mas "pouco" em relação a quê?
Esta figura responde: traça, em função da carga, (i) a regra *greedy* benigna (o melhor caso),
(ii) a política limpa, e (iii) o encaminhamento por pior caminho (o dano máximo que um atacante
que controlasse totalmente as decisões conseguiria provocar). A área sombreada entre a política
limpa e o pior caminho é **o dano que estava ao alcance**.

No ponto de operação usado (2×), essa margem é da ordem de **~21 pp** de entrega. O FGSM extrai
entre ~0 e 2.8 pp. Ou seja, o ataque consegue **uma fracção quase nula do dano disponível**.

Isto é mais forte do que dizer "o ataque perde só 2.8 pp": mostra que havia muito mais para
perder e que o ataque não lá chegou.

---

## 8.5 O efeito adversarial é pequeno em todas as arquiteturas — **Figura T3b** (nova)

> **Usa a `T3b_gap_across_seeds`, não a T3.** A T3 (a tua 8.6) ordena as arquiteturas a
> partir de **uma só** vítima treinada por variante, e essa ordenação **não se reproduz**
> quando se treina outra vez. A T3b mostra um marcador por vítima treinada: vê-se que o
> efeito é pequeno em todas, que a dispersão entre sementes engole as diferenças entre
> variantes, e quais as variantes que ainda só têm uma semente (marcador vazio). Tira a T3
> do capítulo — se quiseres manter a versão com intervalos de confiança, põe-na em anexo.

**Legenda sugerida para a T3b.** *Efeito adversarial específico (gradiente − aleatório) no
ponto nominal, ε = 0.30, com um marcador por vítima treinada independentemente. As quatro
variantes sem GNN foram treinadas com três sementes; as três variantes GNN têm apenas uma
(marcador vazio). A banda é ±1 desvio-padrão entre sementes. A dispersão entre sementes é
maior do que a diferença entre arquiteturas, pelo que os dados não sustentam uma ordenação
de robustez entre variantes.*

**Números da vítima canónica** (a que está na T3) — efeito = queda com gradiente − queda com
aleatório, nominal. Serve para referência; a leitura por variante está corrigida logo a seguir:

| Variante | efeito (pp) | leitura |
|---|---|---|
| LC-Simple | **+3.3 ± 1.3** | o maior efeito real |
| CC-Duelling | **+2.5 ± 1.4** | real, moderado |
| LC-Duelling | **+1.2 ± 0.6** | real, pequeno |
| CC-Duelling-GNN | +1.0 ± 1.0 | toca o zero |
| CC-Simple | +0.4 ± 0.6 | **não distinguível de ruído** |
| CC-Simple-GNN | −0.0 ± 0.4 | nulo |
| LC-Duelling-GNN | −0.1 ± 0.1 | nulo |

**Como explicar.** Comparar o ataque com o *limpo* não chega: qualquer perturbação, mesmo
aleatória, mexe na rede. Para isolar o que é **mesmo adversarial**, subtrai-se a queda provocada
por ruído aleatório com o mesmo orçamento. O que sobra é o efeito atribuível à **direcção do
gradiente**, e não à mera presença de perturbação.

Só as variantes cujo intervalo de confiança **não cruza o zero** têm um efeito real. Pela tabela,
isso são **LC-Simple, CC-Duelling e LC-Duelling**. As restantes — incluindo **CC-Simple** — não se
distinguem estatisticamente de ruído.

**⚠️⚠️ ATENÇÃO — esta secção mudou (23/09/2026). Lê isto antes de escrever.**

Repetimos a medição em **três vítimas treinadas de forma independente** (a canónica mais duas
sementes, `s1042` e `s2042`). O resultado desfaz a leitura por variante:

| Variante | canónica | s1042 | s2042 | média | desvio |
|---|---|---|---|---|---|
| CC-Simple | +0.37 | +2.70 | **+5.11** | +2.72 | 2.37 |
| CC-Duelling | +2.53 | +2.57 | +5.81 | +3.63 | 1.88 |
| LC-Simple | +3.27 | +3.49 | +2.00 | +2.92 | **0.81** |
| LC-Duelling | +1.24 | +1.19 | +4.66 | +2.36 | 1.99 |

O CC-Simple varia de **+0.37 a +5.11** entre sementes. A semente canónica é a excepção: nas
outras duas, o CC-Simple parece *explorável*. As bandas de ±1 desvio de todas as variantes
sobrepõem-se, e a diferença entre médias (1.3 pp) é **menor** do que a variação entre sementes
(0.8–2.4 pp).

**Conclusão: não podes afirmar que uma arquitetura é mais robusta do que outra.** Isso é uma
propriedade daquele treino, não do desenho. A única afirmação por variante que resiste é que o
**LC-Simple é o mais estável** (desvio 0.81) e positivo nas três sementes.

**O que escrever em vez disso.** Apresenta a figura como "o efeito adversarial é pequeno em todas
as arquiteturas", não como um ranking. E acrescenta a limitação: com uma só semente por
arquitetura, as diferenças entre variantes não são atribuíveis ao desenho. (Se quiseres, inclui a
tabela acima — é um resultado teu e mostra rigor.)

**O que continua a valer.** A média dos ganhos (+2.4 a +3.6 pp) continua a ser uma fracção pequena
do tecto de dano (~21 pp), ou seja, **~11–17 %**. O resultado principal do capítulo não só se
mantém como fica mais forte: passa a estar apoiado em três vítimas em vez de uma.

---

## 8.6 A fragilidade sob falhas é ruído, não ataque — **Figuras T4 e T5** (as tuas 8.4 e 8.5)

**Números** (CC-Simple, carga 2×):

| falhas | PDR limpo | queda gradiente | queda aleatória | efeito líquido |
|---|---|---|---|---|
| 0 | 87.7 % | +0.4 pp | −0.0 pp | +0.4 ± 0.6 |
| 2 | 86.2 % | +0.3 pp | **+6.5 pp** | **−6.1 ± 4.5** |
| 4 | 77.6 % | +32.9 pp | +29.9 pp | +3.0 ± **18.7** |
| 6 | **9.4 %** | −0.8 pp | +1.9 pp | −2.7 ± 1.3 |

**Como explicar.** Com falhas, a queda de entrega dispara — aos 4 links falhados chega a ~33 pp.
À primeira vista parece que o ataque se tornou devastador. Mas o controlo aleatório, com o mesmo
orçamento, provoca **29.9 pp**, praticamente o mesmo. E aos 2 links falhados o ruído aleatório
chega a ser **pior** que o gradiente (+6.5 contra +0.3).

A interpretação correcta é: **sob falhas, a rede fica frágil a qualquer perturbação**, venha ela
de um adversário ou de ruído. O que estás a medir não é um ataque bem sucedido — é uma rede
já perto do limite, que qualquer empurrão desequilibra. A direcção do gradiente deixa de
acrescentar seja o que for.

**O que NÃO podes dizer — dois pontos importantes:**

1. **Aos 4 links, o efeito líquido não é significativo.** É +3.0 pp com um intervalo de **±18.7 pp**.
   O intervalo engole completamente o valor. Tens de escrever explicitamente que **não há
   diferença estatisticamente significativa** entre gradiente e aleatório neste ponto. Não vale a
   pena correr mais episódios: para reduzir o intervalo a ±4 pp precisarias de ~20× mais episódios
   e a conclusão não mudaria.

2. **Os 6 links são uma rede morta.** Com PDR limpo a **9.4 %**, a rede já colapsou por efeito das
   falhas, antes de qualquer ataque. A prova: nesse ponto as **sete arquiteturas dão exactamente o
   mesmo número** (−0.8 / +1.9 / −2.7 ± 1.3) — quando todas as arquiteturas dão o mesmo valor, a
   política é irrelevante e o que estás a medir é a falha, não o ataque.

   ⚠️ **A tua Figura 8.5 tem de ser corrigida.** O título diz *"The attack flips MORE decisions
   under failure"* e a afirmação assenta no pico de **38.9 %** de inversões aos 6 links — numa rede
   que entrega 9.4 % dos pacotes. Isso não se sustenta. Ou **cortas o ponto dos 6 links**, ou
   mantém-lo **explicitamente marcado como regime de colapso** e muda o título. A Figura 8.4 já
   tem essa anotação ("network self-collapsed"); a 8.5 não tem.

---

## 8.7 O GNN suprime inversões — **Figura T6** (a tua 8.7)

**Números** (inversões, nominal, ε = 0.3):

| sem GNN | | com GNN | |
|---|---|---|---|
| CC-Simple | 25.7 % | CC-Simple-GNN | **1.4 %** |
| CC-Duelling | 20.1 % | CC-Duelling-GNN | **12.6 %** ← excepção |
| LC-Simple | 19.4 % | LC-Duelling-GNN | **2.6 %** |
| LC-Duelling | 18.4 % | | |

**Como explicar.** O codificador GNN reduz drasticamente as inversões — de ~20–26 % para 1.4 % e
2.6 %. A excepção é **CC-Duelling-GNN**, que mantém 12.6 %.

**Mecanismo provável (novo, e vale a pena incluir).** Medimos a norma do gradiente do ataque em
cada variante GNN, e ela ordena-se exactamente como as inversões:

| Variante | norma do gradiente | inversões |
|---|---|---|
| CC-Simple-GNN | ~0.0000 (< 1e-4) | 1.4 % |
| LC-Duelling-GNN | 0.3065 | 2.6 % |
| CC-Duelling-GNN | **6.3060** | 12.6 % |

Ou seja: as variantes GNN que resistem são precisamente aquelas em que o codificador **faz
desvanecer o gradiente do ataque** — no CC-Simple-GNN é inferior a 1e-4, o que significa que a
direcção da perturbação é essencialmente ruído numérico. O CC-Duelling-GNN, a tua excepção, é a
única variante GNN que continua a deixar passar um gradiente grande (~20× maior). Isto dá-te uma
**explicação** para a excepção, em vez de apenas a constatares.

**O que NÃO podes dizer.** Esta relação vale **dentro** das variantes GNN, não entre todas as sete:
o CC-Simple tem o menor gradiente das não-GNN (0.27) e mesmo assim a maior taxa de inversões
(25.7 %). As normas de gradiente não são comparáveis entre arquiteturas com escalas de pesos
diferentes. Apresenta isto como **indício**, não como lei.

---

## 8.8 O ataque iterado não é mais forte que o FGSM — **FIGURA NOVA**

Esta secção substitui o "não tenho nada relacionado a PGD". Passas de um buraco no capítulo
para uma contribuição.

**Números** (ε = 0.3, todas as 7 variantes):

Inversões de decisão, nas três configurações que a Figura T8 mostra (o PGD é o do
código, com α = κ·ε/n; o MI-FGSM é o melhor iterado, n = 20, α = ε/8, μ = 1):

| Variante | FGSM | PGD | MI-FGSM | melhor iterado − FGSM |
|---|---|---|---|---|
| CC-Simple | **17.1 %** | 13.7 % | 13.4 % | −2.2 pp |
| CC-Duelling | **17.1 %** | 13.2 % | 13.2 % | −2.5 pp |
| LC-Simple | 16.6 % | 16.1 % | 15.2 % | **+0.9 pp** |
| LC-Duelling | 14.3 % | 13.7 % | 14.4 % | **+1.1 pp** |
| CC-Simple-GNN | **15.8 %** | 5.9 % | 6.0 % | −9.5 pp |
| CC-Duelling-GNN | 32.3 % | 30.7 % | 30.5 % | −0.1 pp (empate) |
| LC-Duelling-GNN | **13.0 %** | 8.3 % | 9.5 % | −3.1 pp |

A última coluna compara o FGSM com o **melhor** de todas as configurações iteradas
testadas, não apenas com as duas da tabela.

**Orçamento efetivamente gasto** (`média|δ| / ε`): FGSM **95–100 %**; o PGD do código
**40–54 %**; com momento **82–90 %**. Ao longo de todo o varrimento de passos, o PGD
simples desce até 20 % e sobe no máximo a 68 %, mas nunca se aproxima do FGSM.

**Como explicar.** Um resultado negativo só vale o que valer o ataque que o sustenta. Por isso
afinámos o ataque iterado a sério: varrimento do tamanho do passo (ε, ε/2, ε/4, ε/8), número de
iterações (10 e 20), momento (MI-FGSM) e reinício aleatório.

Duas descobertas:

1. **O PGD simples desperdiça o orçamento.** O sinal do gradiente **oscila** entre iterações, os
   passos assinados cancelam-se parcialmente, e a perturbação acaba **dentro** da bola ε em vez de
   na fronteira: gasta cerca de metade do orçamento, contra ~98 % do FGSM. Acrescentar momento elimina a
   oscilação e repõe o gasto em 82–90 %.

2. **Mesmo depois dessa correcção, nenhuma configuração iterada supera o passo único de forma
   relevante.** O FGSM fica à frente em **5 das 7** arquiteturas (por 2.2 a 9.5 pp em quatro
   delas; no CC-Duelling-GNN a diferença é de 0.1 pp, ou seja, um empate). O iterado só ganha nas
   duas variantes LC sem GNN, e por **≤1.1 pp** — margem muito abaixo do que seria preciso para mexer
   na entrega, dado o desfasamento entre inversões e entrega que mostraste em 8.3.

**Conclusão a escrever:** o FGSM de passo único é um **limite superior justo** do atacante míope
neste modelo de ameaça. O resultado negativo do capítulo não é artefacto de um ataque
mal afinado.

**Nota de honestidade a incluir.** Durante esta análise descobriu-se que a implementação do
objectivo relia os pesos de congestionamento do **estado já perturbado** a cada iteração, o que
tornava o objectivo não-estacionário e fazia o PGD **descer** a própria função que devia subir.
Foi corrigido (o alvo passa a ser fixado na observação limpa). **O FGSM não é afectado** — ao passo
0 o estado perturbado é igual ao original, e isso foi verificado por checksum. Vale a pena
mencionar: mostra rigor, e explica porque é que resultados PGD anteriores não eram utilizáveis.

**Figura — `T8_iterated_vs_fgsm.pdf`** (já produzida, em `figures/`, com o mesmo estilo das
outras). Tem dois painéis:

- **Esquerda — orçamento efetivamente gasto.** O FGSM encosta à linha tracejada de 1.0 em todas
  as variantes; o PGD fica por ~0.5; o momento recupera para ~0.85. É este painel que explica
  *porquê* o PGD rende menos — não é a rede que resiste, é o ataque que não gasta o que tem.
- **Direita — inversões em diferença face ao FGSM.** A linha vermelha vertical é o FGSM (zero).
  Praticamente todas as barras ficam à esquerda dela, ou seja, os ataques iterados invertem
  menos decisões do que o passo único.

Gerada por `tools/plot_thesis.py` (função `t8`), a partir de
`host_data/results/pgd_diagnostic/<variante>.json` — sete ficheiros, um por variante, com todas
as configurações medidas. Para regenerar: `PAPER_MODE=1 python tools/plot_thesis.py`.

**Legenda sugerida:** *ε = 0.30, regime hotspot 2×, 42 observações por variante; MI-FGSM com
n = 20, α = ε/8, μ = 1. O painel da direita está em diferença face ao FGSM, porque só a
comparação dentro de cada variante é válida (ver aviso abaixo).*

**⚠️ Aviso metodológico obrigatório.**

Regra, em uma frase: **os valores de inversões da Figura T8 servem para comparar ataques entre
si dentro da mesma linha da tabela, e mais nada.** Nunca os compares com os das Figuras 8.2–8.7.

Porquê? Porque são medidos de outra maneira:

- Nas **Figuras 8.2–8.7**, as inversões vêm de **15 episódios completos** (256 passos cada), com
  a rede inteira a funcionar normalmente.
- Na **Figura T8** vêm de **42 observações** recolhidas numa única execução curta. E nas variantes
  GNN há ainda uma segunda diferença: a decisão é calculada como **o ataque** a calcula, isto é,
  só com a observação do agente atacado e pondo a zero as dos outros agentes.

Como são duas contas diferentes, dão números diferentes — em qualquer variante, não só nas GNN:

| | Figura T8 | Figura 8.7 |
|---|---|---|
| CC-Duelling-GNN | 32.3 % | 12.6 % |
| CC-Simple | 17.1 % | 25.7 % |

Nenhum dos dois está errado: medem coisas ligeiramente diferentes. O que se mantém válido na T8 é
**a diferença entre FGSM, PGD e MI-FGSM dentro de cada variante** — e é exactamente essa
diferença que a figura serve para mostrar.

---

## 8.9 (opcional) Quantos agentes tem o atacante de comprometer?

Resultado novo, ainda não está na tese. Mede-se o mesmo efeito adversarial, mas variando a
**fracção de agentes comprometidos** (1, 4, 7 e os 14). Cada célula foi repetida com **quatro
sorteios independentes** do conjunto comprometido, com o tráfego fixo, para separar "*quais*
os agentes" de "*quais* os episódios".

| Variante | 1 agente | 4 agentes | 7 agentes | 14 agentes |
|---|---|---|---|---|
| CC-Simple | +0.15 ± 0.16 | +0.09 ± 0.12 | +0.20 ± 0.11 | +0.37 |
| CC-Duelling | +0.11 ± 0.20 | +0.56 ± 0.22 | +1.26 ± 0.43 | +2.53 |
| LC-Simple | +0.34 ± 0.17 | +0.67 ± 0.25 | +1.93 ± 0.33 | +3.27 |

(± é o desvio-padrão entre os quatro sorteios do conjunto comprometido.)

**Como explicar.** O dano cresce com o número de agentes comprometidos, de forma regular e
reprodutível. E o desvio entre sorteios (0.11–0.43 pp) é pequeno face ao efeito (até 3.3 pp),
o que quer dizer que, nas fracções maiores, importa muito mais **quantos** agentes o atacante
controla do que **quais**.

**O que NÃO podes dizer.** ⚠️ Não escrevas que "basta comprometer um agente". Com um único
sorteio parecia haver um efeito significativo com 1 agente (+1.02 pp no LC-Simple); repetindo
com quatro sorteios, a média é **+0.34 ± 0.17 pp**, ou seja, aquele valor era o extremo de
uma distribuição. Com 1 agente o efeito é pequeno e pouco distinguível de zero em todas as
variantes. Também não digas que há "retornos decrescentes": o dano por agente é
aproximadamente constante (no CC-Duelling até **sobe**, de +0.11 para +0.18 pp por agente).

---

## Lista de correcções às figuras existentes

| Figura | Correcção |
|---|---|
| todas (8.1–8.7) | Legendas dizem **"Enter Caption"** — escrever legendas a sério |
| 8.5 | Título insustentável; tratar o ponto dos 6 links (cortar ou marcar como colapso) |
| 8.6 | **Promover** para a Secção 8.5, antes das falhas |
| 8.1 | Controlo aleatório só tem 2 pontos — descrever como tal, ou correr ε = 0.05 e 0.2 |
| corpo do cap. | Remover notas de trabalho: "T1 teste teste", "uso depois do teste", "T2 e T7", "não tenho nada relacionado a PGD" |
| refs | Corrigir todos os `??` |
