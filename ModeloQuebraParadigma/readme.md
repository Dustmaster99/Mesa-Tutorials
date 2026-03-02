📌 Descrição dos Parâmetros do Gerador de Rede com Três Comunidades

Este modelo gera uma rede complexa composta por três comunidades A, B e C, com estrutura interna independente e conexões intercomunitárias restritas apenas a nós de interface.

A rede final é representada por um objeto G do tipo networkx.Graph.

🔢 Parâmetros Estruturais
🧮 Tamanho das Comunidades

nA – Número de agentes (nós) da comunidade A

nB – Número de agentes da comunidade B

nC – Número de agentes da comunidade C

Esses parâmetros definem o tamanho de cada subgrafo interno.

🧩 Densidade Interna das Comunidades

dA – Probabilidade de conexão entre dois nós dentro da comunidade A

dB – Probabilidade de conexão dentro de B

dC – Probabilidade de conexão dentro de C

Cada comunidade é gerada como um grafo de Erdős–Rényi:

GX∼G(nX,dX)
G
X
	​

∼G(n
X
	​

,d
X
	​

)

onde:

nX
n
X
	​

 é o número de nós

dX
d
X
	​

 é a probabilidade de criação de aresta entre quaisquer dois nós

Valores maiores implicam comunidades mais densas.

🔗 Parâmetros de Interface (Interseção Estrutural)
📌 Tamanho das Interfaces

rho_AB – Fração de nós de A que pertencem à interface A–B

rho_AC – Fração de nós de A que pertencem à interface A–C

rho_BC – Fração de nós de B que pertencem à interface B–C

Formalmente:

∣IXYX∣=⌊ρXY⋅nX⌋
∣I
XY
X
	​

∣=⌊ρ
XY
	​

⋅n
X
	​

⌋

Esses parâmetros controlam quantos nós de cada comunidade estão autorizados a se conectar à outra comunidade específica.

🔎 Observações Importantes

As interfaces são independentes.

Um nó pode pertencer a múltiplas interfaces.

Nós que não pertencem a uma interface não podem formar arestas intercomunitárias.

🌉 Densidade das Conexões Entre Interfaces

delta_AB – Densidade das conexões entre interface A–B

delta_AC – Densidade entre A–C

delta_BC – Densidade entre B–C

Se:

IABA⊂A,IABB⊂B
I
AB
A
	​

⊂A,I
AB
B
	​

⊂B

O número esperado de arestas entre A e B é:

E[EAB]=(ρABnA)(ρABnB)δAB
E[E
AB
	​

]=(ρ
AB
	​

n
A
	​

)(ρ
AB
	​

n
B
	​

)δ
AB
	​

📖 Interpretação

rho controla o tamanho da fronteira

delta controla a intensidade de conexão dentro da fronteira

🧱 Estrutura Gerada

A rede final possui:

Três subgrafos internos independentes

Conexões intercomunitárias restritas apenas a subconjuntos de interface

Controle separado sobre:

Modularidade

Permeabilidade entre comunidades

Grau de mistura

🧠 Objeto de Saída
📦 G

O objeto retornado é:

G : networkx.Graph
Características

Grafo não direcionado

Nós nomeados como:

"A_i"

"B_j"

"C_k"

Cada nó possui o atributo:

G.nodes[node]["community"]

que pode assumir valores:

"A"

"B"

"C"

🧪 Propriedades Emergentes

Dependendo dos parâmetros escolhidos, o modelo pode gerar:

Alta modularidade

Comunidades quase isoladas

Redes quase homogêneas

Estruturas núcleo-periferia

Regimes críticos de conectividade

🎯 Interpretação Teórica

O modelo pode ser visto como:

Um SBM (Stochastic Block Model) restrito por interfaces

Um modelo de mistura controlada

Um gerador de redes para simulação de:

Dinâmica de opinião

Difusão

Consenso

Polarização

Processos multiestados