# Capítulo 8 — Resultados: estrutura, figuras e explicações

Guia para escreveres o capítulo de resultados. Cada secção tem: **a figura**, **os números**,
**como explicar** (para escreveres por palavras tuas) e **o que NÃO podes dizer**.

> **⚠️ Versão de 24/09/2026 — lê esta caixa antes de tudo.**
>
> Quatro descobertas mudam partes do capítulo:
>
> 1. **Nas variantes GNN, o ataque original estava mal apontado.** O código calculava o
>    gradiente só através do ator, sem passar pelo codificador GNN que a vítima usa para
>    decidir. Refizemos o ataque pelo caminho de decisão real. A conclusão "o GNN suprime
>    inversões" cai (Secção 8.7, reescrita).
> 2. **O objetivo do FGSM tem o gradiente mascarado.** Um ataque de um só passo sobre os
>    *logits* do ator faz mais dano do que o FGSM em três das sete variantes (sobretudo
>    CC-Duelling e CC-Duelling-GNN). A
>    conclusão "o FGSM é o melhor ataque míope" cai (Secções 8.8 e 8.9, nova).
> 3. **O teto de dano é de cada vítima: entre 7 e 21 pp.** Os ~21 pp eram só do
>    CC-Simple (Secções 8.4 e 8.5).
> 4. **A linha da política na antiga T7 estava errada.** Vinha de uma execução em que os pesos
>    treinados não foram carregados (as pastas de modelos estavam vazias). Corrigida.
>
> **Não precisas de correr nada:** todos os dados existem e as figuras já foram regeneradas.
>
> - Novas: `T6_gnn_noise_vs_attack` (substitui a antiga `T6_gnn_flip_robustness`, que foi
>   apagada) e `T9_logit_vs_fgsm`.
> - Regeneradas: T2, T3, T3b, T4 e T7.
> - Sem alterações: T1, T5 e T8.
>
> Faz `git pull`.
>
> **Intervalos de confiança:** passam a usar a distribuição *t* de Student, que é a correta
> para 15 episódios emparelhados (t₀.₉₇₅,₁₄ = 2.145 em vez de 1.96). Ficam ~9 % mais largos.
> Os números deste guia já estão assim, e coincidem com os que o Miguel usa na tese dele.

---

## Fio condutor do capítulo

A história tem de se ler nesta ordem, porque cada secção responde à pergunta que a anterior levanta:

1. O ataque **muda decisões**? → sim, ~3× mais do que ruído aleatório (8.2)
2. Isso **muda a entrega**? → pouco (8.3)
3. **Quanto dano havia para fazer**? → entre 7 e 21 pp, conforme a vítima; o FGSM extrai
   entre ~0 e ~30 % disso (8.4)
4. O efeito é **mesmo adversarial**? → é pequeno em todas as arquiteturas, e a diferença entre
   elas não se reproduz entre sementes (8.5)
5. E **sob falhas**, onde a rede é frágil? → a fragilidade é da rede, não da direção do FGSM (8.6)
6. O **GNN** protege? → contra ruído sim, contra um ataque que passe pelo codificador não (8.7)
7. **Iterar** o ataque ajuda? → não (8.8)
8. Então o FGSM é o **melhor ataque de gradiente**? → não: tem o gradiente mascarado. Mudar o
   objetivo aumenta o dano em três das sete variantes, mas nenhum ataque de gradiente testado
   passa de ~35 % do teto (8.9)
9. (opcional) Quantos agentes precisa o atacante de controlar? (8.10)

Os pontos 7 e 8 respondem à pergunta que qualquer arguente faz: *"o seu resultado negativo não
será só porque o ataque é fraco?"*. A resposta honesta passa a ser: **em parte, sim, e medimos
quanto.** Iterar não ajuda. Mudar o objetivo ajuda: no CC-Duelling, o dano mais do que
duplica. Mesmo assim, o melhor ataque de gradiente que testámos fica por uma
**minoria** do dano disponível. A conclusão do capítulo mantém-se, numa forma mais fraca e
mais defensável.

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
  (mesma semente de tráfego e mesmo sorteio de falhas). É isso que sustenta os intervalos de confiança.
- Intervalos de confiança a 95 % com a distribuição ***t*** (15 episódios → t₀.₉₇₅,₁₄ = 2.145).

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
**3× mais decisões** do que ruído aleatório com o mesmo orçamento. A conclusão desta secção:
**ao nível da decisão, o ataque é claramente mais eficaz do que ruído.**

**O que NÃO podes dizer.**
- Não digas que o ataque "não é fraco" ou que é "o mais forte possível". Na Secção 8.9
  mostras que um ataque de um passo com outro objetivo inverte **45 %** das decisões do
  CC-Simple, contra 26 % do FGSM. Deixa aqui uma frase a apontar para lá.
- Não digas que a linha do controlo aleatório é uma curva completa: só tem **2 pontos**
  (ε = 0.1 e 0.3), porque vem do `fgsm_full`. Descreve-a como "pontos de referência".

---

## 8.3 A topologia absorve as inversões — **Figura T2** (a tua 8.2)

**Números** (nominal, ε = 0.3, 7 variantes; **regenerada**, com as variantes GNN atacadas
pelo caminho de decisão real):
- inversões entre **7.2 % e 25.7 %**;
- entrega perdida **no máximo 2.8 pp** (LC-Simple), várias variantes perto de 0;
- no CC-Simple-GNN a barra da entrega é **negativa** (−1.4 pp): o ataque inverte 18.6 % das
  decisões e a vítima **entrega mais**.

**Como explicar.** Esta é **a figura central da tese**. Mostra as duas métricas lado a lado, e o
contraste é o resultado principal: muitas decisões mudam, poucos pacotes se perdem.

O mecanismo: uma decisão invertida move um fluxo do caminho *k* para outro caminho *k'* do mesmo
conjunto de K caminhos pré-calculados. Na maior parte dos casos, **esse outro caminho também
entrega o tráfego**. Medimos isto diretamente em três variantes (CC-Simple, CC-Duelling e
LC-Simple), e há três razões concretas:

1. **Parte das inversões só existe no papel.** Em 37 % dos pares (agente, destino) há menos de
   3 caminhos realmente diferentes, e a lista de K = 3 é completada repetindo o caminho mais
   curto. Trocar entre duas cópias do mesmo caminho conta como inversão, mas o pacote segue
   exatamente os mesmos saltos. Entre **8 % e 22 %** das inversões do FGSM são deste tipo.
2. **Das inversões reais, só 34–47 % levam o tráfego para um caminho mais congestionado.** As
   restantes levam-no para um caminho igual ou menos carregado.
3. **Muitas vezes não há "pior" para onde ir.** Em **36–48 %** das decisões com mais de um
   caminho distinto, a política já está no caminho mais congestionado.

A frase que deves deixar assente aqui: **mudar decisões e mudar resultados são coisas diferentes**,
e a literatura confunde frequentemente as duas.

---

## 8.4 Quanto dano estava disponível — **Figura T7** (a tua 8.3)

> ⚠️ **A T7 foi corrigida.** A versão anterior mostrava a política a ~78 % aos 2×. Essa linha
> vinha de uma execução em que as pastas de modelos estavam vazias, ou seja, **sem os pesos
> treinados**. A nova linha vem da varredura de carga da fase 2, com a rede que foi de facto
> atacada. Se já escreveste algo a partir da figura antiga (por exemplo, "uma margem de ~12 pp
> aos 2×"), corrige.

**Como explicar.** A secção anterior mostra que se perde pouco. Mas "pouco" em relação a quê?
A figura traça, em função da carga (CC-Simple):

- a regra *greedy*: escolhe o caminho menos congestionado dos três. É uma boa regra, mas não
  é um ótimo;
- a regra **aleatória**: escolhe um caminho ao acaso em cada passo (**nova na figura**);
- a política limpa;
- o **pior caminho**: todas as decisões mandadas para o caminho mais congestionado. É o dano
  máximo que um atacante que controlasse totalmente as decisões conseguiria provocar.

A área sombreada, entre a política e o pior caminho, é **o dano que estava ao alcance**. Aos
2×, na figura, é de ~20 pp. Medido nos próprios episódios do ataque, é de 21.4 pp. É a mesma
rede, com episódios diferentes (30 contra 20).

**Duas observações novas que tens de fazer:**

1. **O teto é de cada vítima, não é o mesmo para todas.** O pior caminho entrega o mesmo em
   todas (65.9 % aos 2×); o que muda é de onde cada vítima parte:

   | Variante | teto (pp) | efeito FGSM (pp) | fração do teto |
   |---|---|---|---|
   | CC-Simple | 21.4 | +0.37 | ~2 % |
   | CC-Duelling | 16.5 | +2.53 | 15 % |
   | CC-Simple-GNN | 14.7 | −1.39 | 0 (melhora a entrega) |
   | CC-Duelling-GNN | 14.4 | +1.94 | 13 % |
   | LC-Simple | 11.3 | +3.27 | 29 % |
   | LC-Duelling | 10.8 | +1.24 | 11 % |
   | LC-Duelling-GNN | 7.3 | +0.12 | ~2 % (nulo) |

   Teto = entrega limpa − entrega com todas as decisões no pior caminho (20 episódios, 2×).
   Efeito FGSM = o da Secção 8.5.

2. **A política entrega menos do que escolher um caminho ao acaso**, em todas as cargas (3–4 pp
   abaixo da linha aleatória). Isto é um resultado do Paper 1, e deves citá-lo. É importante
   para ler o resto do capítulo. Uma vítima que já encaminha pior do que o acaso tem menos a
   perder. E baralhar-lhe as decisões pode até **ajudar**, porque a aproxima do encaminhamento
   aleatório. É por isso que alguns ataques melhoram a entrega (Secções 8.7 e 8.9).

**O que NÃO podes dizer.**
- Não digas que o FGSM extrai "uma fração quase nula do dano disponível". Isso vale para o
  CC-Simple (~2 %), não para o LC-Simple (29 %). Diz: **entre ~0 e ~30 %, conforme a vítima**.
- Não uses ~21 pp como teto de todas as variantes.
- A fração é aproximada: o teto foi medido em 20 episódios e o efeito em 15, contra o controlo
  aleatório. Serve para dar a ordem de grandeza, não para comparar décimas.

---

## 8.5 O efeito adversarial é pequeno em todas as arquiteturas — **Figura T3b**

> **Usa a `T3b_gap_across_seeds`, não a T3.** A T3 ordena as arquiteturas a partir de **uma
> só** vítima treinada por variante, e essa ordenação **não se reproduz** quando se treina outra
> vez. A T3b mostra um marcador por vítima treinada: vê-se que o efeito é pequeno em todas, que a
> dispersão entre sementes engole as diferenças entre variantes, e quais as variantes que ainda
> só têm uma semente (marcador vazio). Tira a T3 do capítulo; se quiseres mantê-la, põe-na
> em anexo (também foi regenerada).

**Legenda sugerida para a T3b.** *Efeito adversarial específico (gradiente − aleatório) no
ponto nominal, ε = 0.30, com um marcador por vítima treinada independentemente. As quatro
variantes sem GNN foram treinadas com três sementes; as três variantes GNN têm apenas uma
(marcador vazio). A banda é ±1 desvio-padrão entre sementes. A dispersão entre sementes é
maior do que a diferença entre arquiteturas, pelo que os dados não sustentam uma ordenação
de robustez entre variantes.*

**Números da vítima canónica** (efeito = queda com gradiente − queda com aleatório, nominal,
IC 95 % com *t*). As linhas GNN já usam o ataque corrigido:

| Variante | efeito (pp) | IC 95 % | leitura |
|---|---|---|---|
| LC-Simple | **+3.27** | ±1.45 | real, o maior |
| CC-Duelling | **+2.53** | ±1.54 | real, moderado |
| CC-Duelling-GNN | **+1.94** | ±0.82 | real (era +1.0 ± 1.0 com o ataque antigo) |
| LC-Duelling | **+1.24** | ±0.71 | real, pequeno |
| CC-Simple | +0.37 | ±0.71 | **não distinguível de ruído** |
| LC-Duelling-GNN | +0.12 | ±0.57 | nulo |
| CC-Simple-GNN | **−1.39** | ±0.77 | **negativo**: o ataque *melhora* a entrega |

**Como explicar.** Comparar o ataque com o *limpo* não chega: qualquer perturbação, mesmo
aleatória, mexe na rede. Para isolar o que é **mesmo adversarial**, subtrai-se a queda provocada
por ruído aleatório com o mesmo orçamento. O que sobra é o efeito atribuível à **direção do
gradiente**, e não à mera presença de perturbação.

Só as variantes cujo intervalo de confiança **não cruza o zero** têm um efeito real. Pela tabela,
isso são **LC-Simple, CC-Duelling, CC-Duelling-GNN e LC-Duelling**. O CC-Simple não se distingue
de ruído. O CC-Simple-GNN separa-se do zero, mas **no sentido contrário**.

**⚠️ A leitura por variante não se reproduz entre sementes.** Repetimos a medição em **três
vítimas treinadas de forma independente** (a canónica mais duas sementes, `s1042` e `s2042`):

| Variante | canónica | s1042 | s2042 | média | desvio |
|---|---|---|---|---|---|
| CC-Simple | +0.37 | +2.70 | **+5.11** | +2.72 | 2.37 |
| CC-Duelling | +2.53 | +2.57 | +5.81 | +3.63 | 1.88 |
| LC-Simple | +3.27 | +3.49 | +2.00 | +2.92 | **0.81** |
| LC-Duelling | +1.24 | +1.19 | +4.66 | +2.36 | 1.99 |

O CC-Simple varia de **+0.37 a +5.11** entre sementes. A semente canónica é a exceção: nas
outras duas, o CC-Simple parece *explorável*. As bandas de ±1 desvio de todas as variantes
sobrepõem-se, e a diferença entre médias (1.3 pp) é **menor** do que a variação entre sementes
(0.8–2.4 pp).

**Conclusão: não podes afirmar que uma arquitetura é mais robusta do que outra.** Isso é uma
propriedade daquele treino, não do desenho. A única afirmação por variante que resiste é que o
**LC-Simple é o mais estável** (desvio 0.81) e positivo nas três sementes.

**As variantes GNN ainda só têm uma semente.** Estão a ser treinadas mais duas por variante (4
das 6 já terminaram). Vão ser atacadas com o ataque corrigido, e a T3b passa a ter três
marcadores nessas linhas. Até lá, escreve que as linhas GNN assentam numa só vítima.

**O que continua a valer.** A média dos efeitos, em cada variante, é uma fração pequena do teto
**dessa** variante: CC-Simple 13 %, CC-Duelling 22 %, LC-Simple 26 %, LC-Duelling 22 %. Ou seja,
o FGSM extrai **uma minoria (13–26 %) do dano disponível**, agora apoiado em três vítimas em vez
de uma. (O teto foi medido só na vítima canónica; as frações são aproximadas.)

---

## 8.6 A fragilidade sob falhas é da rede, não do ataque — **Figuras T4 e T5** (as tuas 8.4 e 8.5)

**Números** (CC-Simple, carga 2×, IC com *t*; a T4 foi regenerada só por causa das bandas):

| falhas | PDR limpo | queda gradiente | queda aleatória | efeito líquido |
|---|---|---|---|---|
| 0 | 87.7 % | +0.4 pp | −0.0 pp | +0.4 ± 0.7 |
| 2 | 86.2 % | +0.3 pp | **+6.5 pp** | **−6.1 ± 4.9** |
| 4 | 77.6 % | +32.9 pp | +29.9 pp | +3.0 ± **20.5** |
| 6 | **9.4 %** | −0.8 pp | +1.9 pp | −2.7 ± 1.4 |

**Como explicar.** Com falhas, a queda de entrega dispara: aos 4 links falhados chega a ~33 pp.
À primeira vista parece que o ataque se tornou devastador. Mas o controlo aleatório, com o mesmo
orçamento, provoca **29.9 pp**, praticamente o mesmo. E aos 2 links falhados o ruído aleatório
chega a ser **pior** do que o gradiente (+6.5 contra +0.3).

A interpretação correta é: **sob falhas, a rede fica frágil a qualquer perturbação**, venha ela
de um adversário ou de ruído. O que estás a medir não é um ataque bem-sucedido: é uma rede
já perto do limite, que qualquer empurrão desequilibra. A direção do gradiente **do FGSM**
deixa de acrescentar seja o que for.

**O que NÃO podes dizer — três pontos:**

1. **Aos 4 links, o efeito líquido não é significativo.** É +3.0 pp com um intervalo de
   **±20.5 pp**. Tens de escrever explicitamente que **não há diferença estatisticamente
   significativa** entre gradiente e aleatório neste ponto. Não vale a pena correr mais
   episódios: para reduzir o intervalo a ±4 pp precisarias de ~25× mais, e a conclusão não
   mudaria.

2. **Os 6 links são uma rede morta.** Com PDR limpo a **9.4 %**, a rede já colapsou por efeito das
   falhas, antes de qualquer ataque. A prova: nesse ponto as **sete arquiteturas dão exatamente o
   mesmo número**. Quando todas as arquiteturas dão o mesmo valor, a política é irrelevante e o
   que estás a medir é a falha, não o ataque.

   ⚠️ **A tua Figura 8.5 (T5) tem de ser corrigida.** O título diz *"The attack flips MORE
   decisions under failure"*, e a afirmação assenta no pico de **38.9 %** de inversões aos 6
   links, numa rede que entrega 9.4 % dos pacotes. Isso não se sustenta. Ou **cortas o ponto
   dos 6 links**, ou manténs o ponto **explicitamente marcado como regime de colapso** e mudas
   o título. A Figura 8.4 já tem essa anotação ("network self-collapsed"); a 8.5 não tem.

3. **Não generalizes para "qualquer ataque".** Isto foi medido com o FGSM. O ataque sobre os
   logits da Secção 8.9 não foi testado com falhas.

---

## 8.7 O GNN amortece o ruído, não um ataque apontado a ele — **Figura T6 (nova)**

> **Esta secção foi reescrita.** A conclusão antiga, "o GNN suprime inversões", resultava de um
> ataque mal apontado. Usa a nova figura `T6_gnn_noise_vs_attack`. A antiga
> `T6_gnn_flip_robustness` foi apagada: atualiza o `\includegraphics`.

**O que aconteceu, em termos simples.** Numa variante GNN, a vítima não decide a partir da sua
observação, mas do resultado do codificador GNN aplicado às observações **de todos os agentes**.
O ataque original calculava o gradiente do **ator aplicado só à observação do agente**, uma
função que a vítima GNN nunca avalia. Era como apontar a uma cópia do alvo à qual faltava o
codificador. Refizemos o ataque pelo caminho de decisão real: codificador incluído, com as
observações limpas dos outros agentes como contexto.

**Números** (inversões, nominal, ε = 0.3):

| Variante | aleatório | FGSM sem codificador (original) | FGSM pelo caminho real |
|---|---|---|---|
| CC-Simple-GNN | **0.6 %** | 1.4 % | **18.6 %** |
| CC-Duelling-GNN | 13.5 % | 12.6 % | **19.0 %** |
| LC-Duelling-GNN | **1.9 %** | 2.6 % | **7.2 %** |
| *sem GNN (referência)* | 8–15 % | 18–26 % | igual ao original |

Entrega (efeito adversarial, ataque pelo caminho real): CC-Duelling-GNN **+1.94 ± 0.82 pp**
(real); LC-Duelling-GNN +0.12 ± 0.57 (nulo); CC-Simple-GNN **−1.39 ± 0.77** (o ataque
*melhora* a entrega).

**Como explicar.** A figura separa três coisas que a versão antiga misturava:

1. **O que o GNN faz ao ruído.** Em duas das três variantes, o ruído aleatório quase não muda
   decisões: 0.6 % e 1.9 %, contra 8–15 % sem GNN. O codificador amortece perturbações sem
   direção. A exceção é o CC-Duelling-GNN (13.5 %), ao nível das variantes sem GNN.
2. **O que o GNN faz a um ataque que o ignora.** Quase nada passa (1.4 % e 2.6 %). Foi daqui
   que saiu a conclusão antiga. A "exceção" CC-Duelling-GNN (12.6 %) invertia até **menos**
   decisões do que o próprio ruído (13.5 %).
3. **O que o GNN faz a um ataque que o atravessa.** 18.6 % e 19.0 % de inversões: ao nível das
   variantes sem GNN (18–26 %). Só o LC-Duelling-GNN fica mais baixo (7.2 %).

A frase a deixar assente: **o GNN protege contra ruído, não contra um atacante que conheça o
codificador.** Na entrega, o resultado é misto: dano real no CC-Duelling-GNN, nulo no
LC-Duelling-GNN, e o CC-Simple-GNN entrega mais sob ataque. Isto é coerente com a Secção 8.4:
baralhar decisões a uma política que encaminha pior do que o acaso pode ajudá-la.

**Nota de honestidade a incluir** (como a do PGD, na Secção 8.8). Diz que o erro foi detetado
durante a análise, que o ataque foi refeito pelo caminho de decisão real e que os números das
variantes GNN do capítulo são os do ataque corrigido. Podes manter os números antigos na tabela
como "ataque sem codificador": mostram que um atacante que ignore o codificador quase não passa.

**O que NÃO podes dizer.**
- "O GNN torna a vítima robusta" ou "o GNN suprime inversões". Só vale para o ruído.
- **Retira o "mecanismo" da norma do gradiente** que estava nesta secção. As normas vinham do
  diagnóstico, que avalia o GNN com as observações dos outros agentes a zero. As taxas de
  inversão vinham do ataque original, sem codificador. São duas funções diferentes, e
  correlacioná-las não tem significado. O erro foi nosso.
- Não tires conclusões por variante GNN: cada uma tem uma só vítima treinada (Secção 8.5).

---

## 8.8 Iterar o ataque não ajuda — **Figura T8**

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
   na fronteira: gasta cerca de metade do orçamento, contra ~98 % do FGSM. Acrescentar momento
   elimina a oscilação e repõe o gasto em 82–90 %.

2. **Mesmo depois dessa correção, nenhuma configuração iterada supera o passo único de forma
   relevante.** O FGSM fica à frente em **5 das 7** arquiteturas (por 2.2 a 9.5 pp em quatro
   delas; no CC-Duelling-GNN a diferença é de 0.1 pp, ou seja, um empate). O iterado só ganha nas
   duas variantes LC sem GNN, e por **≤1.1 pp**.

**Conclusão a escrever (mudou):** iterar **o mesmo objetivo** não ajuda. O que limita o FGSM
não é ter um só passo. A Secção 8.9 mostra que é o **objetivo**. **Já não escrevas** que o FGSM
é "um limite superior justo do atacante míope". Isso deixou de ser verdade.

**Nota de honestidade a incluir.** Durante esta análise descobriu-se que a implementação do
objetivo relia os pesos de congestionamento do **estado já perturbado** a cada iteração, o que
tornava o objetivo não-estacionário e fazia o PGD **descer** a própria função que devia subir.
Foi corrigido: o alvo passa a ser fixado na observação limpa. **O FGSM não é afetado**, porque
ao passo 0 o estado perturbado é igual ao original, e isso foi verificado por *checksum*.

**Figura — `T8_iterated_vs_fgsm.pdf`** (sem alterações). Tem dois painéis:

- **Esquerda — orçamento efetivamente gasto.** O FGSM encosta à linha tracejada de 1.0 em todas
  as variantes; o PGD fica por ~0.5; o momento recupera para ~0.85. É este painel que explica
  *porquê* o PGD rende menos: não é a rede que resiste, é o ataque que não gasta o que tem.
- **Direita — inversões em diferença face ao FGSM.** A linha vermelha vertical é o FGSM (zero).
  Praticamente todas as barras ficam à esquerda dela: os ataques iterados invertem menos
  decisões do que o passo único.

**⚠️ Aviso metodológico obrigatório (reescrito).** Os valores da T8 servem **só para comparar
ataques entre si dentro da mesma linha**. Nunca os compares com os das outras figuras, por duas
razões:

- Vêm de **42 observações** recolhidas numa única execução curta, e não de 15 episódios completos.
- Nas **linhas GNN**, o diagnóstico avalia o GNN com as observações **dos outros agentes a
  zero**. Não é o ataque original (sem codificador) nem a decisão real da vítima (com as
  observações de todos). É uma terceira função. Exemplo, CC-Duelling-GNN:

  | medição | inversões |
  |---|---|
  | T8 (diagnóstico, outros agentes a zero) | 32.3 % |
  | ataque original, sem codificador | 12.6 % |
  | ataque pelo caminho real (T6) | 19.0 % |

Dentro de cada linha, a comparação FGSM / PGD / MI-FGSM continua válida: são otimizadores
diferentes sobre a mesma função. Mas nas linhas GNN essa função não é a vítima. **Escreve isso
na legenda.** Estamos a decidir se refazemos as linhas GNN pelo caminho real; se isso
acontecer, a figura é substituída.

**Legenda sugerida:** *ε = 0.30, regime hotspot 2×, 42 observações por variante; MI-FGSM com
n = 20, α = ε/8, μ = 1. O painel da direita está em diferença face ao FGSM, porque só a
comparação dentro de cada variante é válida. Nas variantes GNN, o diagnóstico avalia o
codificador com as observações dos restantes agentes a zero.*

---

## 8.9 O objetivo do FGSM está mascarado — **Figura T9 (nova)**

**O problema, em termos simples.** O ator produz 63 números (*logits*, z), e cada um passa
por uma sigmoide: π = σ(z). O objetivo do FGSM mede o ataque em π, **depois** da sigmoide. Mas a
política treinada é muito "confiante": em **82–96 %** das decisões, a saída do caminho escolhido
está acima de 0.99. (Medido no CC-Simple, 84.3 %; no CC-Duelling, 81.9 %; no LC-Simple, 96.0 %.)
Aí a sigmoide está achatada: a sua derivada, σ(1 − σ), é inferior a 0.01. Por isso o gradiente
que chega à observação é quase nulo, precisamente nos termos que decidem a escolha. Na
literatura isto chama-se **mascaramento do gradiente**.

**O teste.** Mantivemos tudo igual ao FGSM (um passo, ε = 0.3, sinal do gradiente, mesma
projeção, os mesmos 15 episódios). Mudámos só **onde** o objetivo é medido: nos logits z,
**antes** da sigmoide. Testámos duas versões:

- **logits, "congestionamento":** empurra para cima o caminho mais congestionado e para baixo o
  escolhido, Σ_d (z_{d,pior} − z_{d,escolhido});
- **logits, "margem":** empurra para cima a melhor alternativa, qualquer que seja,
  Σ_d (max_{k≠escolhido} z_{d,k} − z_{d,escolhido}).

O caminho "escolhido" é o da observação limpa. Nas variantes GNN, o ataque passa pelo codificador.

**Números** (dano extra face ao FGSM = entrega do braço FGSM − entrega do braço logits, nos
mesmos episódios; IC 95 % com *t*; positivo = os logits fizeram mais dano):

| Variante | inversões FGSM | inversões logits (cong. / margem) | dano extra: cong. | dano extra: margem |
|---|---|---|---|---|
| CC-Simple | 25.7 % | 42.4 / 45.5 % | +1.18 [−0.45, +2.80] | +1.81 [−0.01, +3.63] |
| CC-Duelling | 20.1 % | 25.6 / 27.2 % | **+3.28 [+1.76, +4.80]** | +1.22 [−0.48, +2.91] |
| CC-Simple-GNN | 18.6 % | 21.6 / 24.0 % | +0.30 [−0.07, +0.68] | +0.33 [−0.12, +0.78] |
| CC-Duelling-GNN | 19.0 % | 23.7 / 25.3 % | **+2.06 [+1.03, +3.10]** | **+1.76 [+1.01, +2.51]** |
| LC-Simple | 19.4 % | 27.0 / 30.2 % | **−3.48 [−4.76, −2.20]** | **−3.57 [−4.74, −2.41]** |
| LC-Duelling | 18.4 % | 27.6 / 31.1 % | **−2.38 [−3.55, −1.21]** | **−4.78 [−6.58, −2.97]** |
| LC-Duelling-GNN | 7.2 % | 10.9 / 11.5 % | **+0.91 [+0.32, +1.51]** | +0.33 [−0.13, +0.79] |

**Como explicar.**

1. **Ao nível da decisão, o mascaramento é real.** Os ataques nos logits invertem mais decisões
   do que o FGSM **em todas as 7 variantes** (no CC-Simple, 45 % contra 26 %). O FGSM deixava
   alavanca por usar.
2. **Na entrega, o ganho depende da vítima.** No CC-Duelling, o ataque nos logits faz **3.3 pp**
   mais dano do que o FGSM (o efeito adversarial passa de +2.5 para +5.8 pp). No
   CC-Duelling-GNN faz 2.1 pp mais, e no LC-Duelling-GNN 0.9 pp mais. No CC-Simple e no
   CC-Simple-GNN a diferença não é significativa. No **LC-Simple e no LC-Duelling**, os logits
   invertem mais decisões, mas a vítima **entrega mais** do que sob FGSM.
3. **Porquê no LC-Simple e no LC-Duelling?** Não temos uma explicação verificada. É coerente
   com a Secção 8.4: estas vítimas encaminham pior do que o acaso, e mudar-lhes mais decisões
   aproxima-as do encaminhamento aleatório. Apresenta isto como hipótese, não como mecanismo.
4. **Quanto chega a extrair o melhor ataque de gradiente?** No máximo **~35 %** do teto da
   vítima (CC-Duelling: 5.8 de 16.5 pp). Nos restantes casos, 29 % ou menos.

**Conclusão a escrever.** O FGSM **não** é um limite superior justo do atacante de um passo,
porque sofre de mascaramento do gradiente. O que o limita é o objetivo, não o número de passos
(Secção 8.8). Mesmo o melhor ataque de gradiente que testámos extrai **uma minoria** do dano
disponível. O resultado negativo do capítulo mantém-se, mas mais fraco e mais honesto: os
ataques de observação por gradiente extraem uma minoria do dano disponível, e não "quase nada".

**Legenda sugerida:** *Ataque de um passo sobre os logits do ator, comparado com o FGSM: ε = 0.30,
regime hotspot 2×, os mesmos 15 episódios emparelhados; nas variantes GNN o gradiente atravessa o
codificador. Esquerda: decisões invertidas por cada ataque e pelo controlo aleatório. Direita:
entrega perdida para além da do FGSM, com IC 95 %; barras vazias têm intervalo que inclui o zero.*

**O que NÃO podes dizer.**
- "O FGSM é o melhor ataque míope." Já não é verdade.
- "O ataque nos logits é sempre mais forte." Nas variantes LC faz **menos** dano do que o FGSM.
- "A vítima é robusta." O que podes dizer é que nenhum ataque de gradiente testado extraiu mais
  do que ~35 % do dano disponível.
- Estes números são da vítima canónica. As sementes adicionais ainda não foram atacadas com os logits.

---

## 8.10 (opcional) Quantos agentes tem o atacante de comprometer?

Resultado novo, ainda não está na tese. Mede-se o mesmo efeito adversarial, mas variando a
**fração de agentes comprometidos** (1, 4, 7 e os 14). Cada célula foi repetida com **quatro
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
o que quer dizer que, nas frações maiores, importa muito mais **quantos** agentes o atacante
controla do que **quais**.

**O que NÃO podes dizer.** ⚠️ Não escrevas que "basta comprometer um agente". Com um único
sorteio parecia haver um efeito significativo com 1 agente (+1.02 pp no LC-Simple); repetindo
com quatro sorteios, a média é **+0.34 ± 0.17 pp**, ou seja, aquele valor era o extremo de
uma distribuição. Com 1 agente o efeito é pequeno e pouco distinguível de zero em todas as
variantes. Também não digas que há "retornos decrescentes": o dano por agente é
aproximadamente constante (no CC-Duelling até **sobe**, de +0.11 para +0.18 pp por agente).

---

## Lista de correções às figuras

| Figura | Estado / correção |
|---|---|
| T2 (a tua 8.2) | **Regenerada**: linhas GNN com o ataque corrigido |
| T3 (anexo) | **Regenerada**: linhas GNN corrigidas, IC com *t* |
| T3b | **Regenerada**: marcadores GNN corrigidos |
| T4 (a tua 8.4) | **Regenerada**: bandas de confiança com *t* (ligeiramente mais largas) |
| T5 (a tua 8.5) | Título insustentável; tratar o ponto dos 6 links (cortar ou marcar como colapso) |
| T6 (a tua 8.7) | **Substituída** por `T6_gnn_noise_vs_attack`; atualiza o `\includegraphics` |
| T7 (a tua 8.3) | **Regenerada**: linha da política corrigida, linha aleatória acrescentada |
| T8 | Inalterada; a legenda tem de dizer como são calculadas as linhas GNN |
| T9 | **Nova** (Secção 8.9) |
| todas | Legendas dizem **"Enter Caption"**: escrever legendas a sério |
| corpo do cap. | Remover notas de trabalho: "T1 teste teste", "uso depois do teste", "T2 e T7", "não tenho nada relacionado a PGD" |
| refs | Corrigir todos os `??` |

---

## O que isto muda nos Capítulos 4–6 (descrição do modelo)

As Secções 8.7 e 8.9 só se percebem se o modelo estiver descrito com estes pormenores:

- **4.3 / Cap. 5 — a ação.** Define a cadeia completa: o ator produz **logits** z_i ∈ ℝ^{|Dacc|·K};
  a saída é a_i = σ(z_i) ∈ [0,1]^{|Dacc|·K}, com **sigmoides independentes** (não é um *softmax*,
  e as saídas não são probabilidades nem somam 1); a rota para cada destino é o `argmax` do
  bloco correspondente, que é o mesmo em a_i e em z_i. Isto responde à tua dúvida do 4.3: a
  ação que o crítico vê no treino é a_i; o encaminhamento usa o seu `argmax`.
- **Cap. 5 — o ator e o GNN.** Ator: 94 → 256 → 128 → 63, com ReLU nas camadas escondidas e
  sigmoide na saída. Nas variantes GNN, o codificador corre **antes** do ator e lê as
  observações **dos 14 agentes** (mais 72 nós sem agente, com características a zero). Logo, a
  decisão do agente i depende das observações de todos os agentes.
- **6.1 — conhecimento do atacante.** Nas variantes GNN, um ataque *white-box* precisa do
  codificador e das observações limpas **de todos os agentes**. Acrescenta a nota de
  honestidade: o código original calculava o gradiente só através do ator.
- **6.2 — a equação 6.2 está errada.** O λ está fora da sigmoide. O código usa
  L = Σ_{d,k} π_{d,k} · σ(λ(u_{d,k} − c)), com c = 0.5 e λ = 10. O u_{d,k} é lido da observação
  limpa e tratado como constante (não há gradiente através dele), e π = σ(z). Acrescenta uma
  frase: o gradiente passa por σ′(z) = σ(1 − σ), que se anula quando a saída satura. É a
  origem do mascaramento (Secção 8.9).
- **6.5 (nova) — ataque sobre os logits.** Define os dois objetivos da Secção 8.9. Diz que o
  ataque mantém o ε, o passo único com sinal e a projeção do FGSM, e que o caminho "escolhido"
  vem da observação limpa. Acrescenta uma coluna à Tabela 6.1.
