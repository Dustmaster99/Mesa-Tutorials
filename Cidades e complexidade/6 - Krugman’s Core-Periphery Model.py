import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parâmetros da simulação
# ----------------------------

''' 
#parãmetros para centralização forte
# Forças de dispersão < Forças de aglomeração
N = 24               # número de células
steps = 200       # número de etapas
K1, K2 = 1.0, 0.7    # escalas
alpha, beta = 1.4, 0.25  # decaimentos
'''
'''
#parãmetros para descentralização forte
# Forças de dispersão > Forças de aglomeração
N = 24               # número de células
steps = 200       # número de etapas
K1, K2 = 0.5, 1.2    # escalas
alpha, beta = 0.3, 1.8   # decaimentos
'''
#parãmetros para pequenas flutuações
# Leve tendência à dispersão
N = 24               # número de células
steps = 500       # número de etapas
K1, K2 = 0.8, 1.0    # K2 ligeiramente maior
alpha, beta = 0.6, 1.2  # Dispersão moderada

'''
# Equilíbrio quase perfeito
N = 24               # número de células
steps = 200       # número de etapas
K1, K2 = 1.0, 1.0    # Forças balanceadas
alpha, beta = 0.8, 0.8  # Decaimentos iguais
'''

# Distribuição inicial JÁ NORMALIZADA
P = np.ones(N) + 0.2 * np.random.rand(N)
P = P / np.sum(P)  # ⬅️ NORMALIZAÇÃO INICIAL

# Função para calcular distância circular mínima
def circular_distance(i, j, N):
    diff = abs(i - j)
    return min(diff, N - diff)

# Pré-computar matriz de distâncias
d = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        d[i, j] = circular_distance(i, j, N)

# Guardar histórico
history = [P.copy()]

for t in range(steps-1):
    # Calcular V1 e V2
    V1 = K1 * np.array([np.sum(P * np.exp(-alpha * d[i, :])) for i in range(N)])
    V2 = K2 * np.array([np.sum(P * np.exp(-beta * d[i, :])) for i in range(N)])
    
    V = V1 - V2
    V_mean = V.mean()
    
    # Growth rate
    growth_rate = 1 + (V - V_mean)
    P = P * growth_rate
    
    # ⬇️ NORMALIZAÇÃO A CADA PASSO ⬇️
    P = P / np.sum(P)
    
    history.append(P.copy())

# ⬇️ REMOVA esta linha - já está normalizado durante a simulação
# history = [dist / np.sum(dist) for dist in history]

# ----------------------------
# Visualização
# ----------------------------
def plot_evolution_steps(history, n, m):
    theta = np.linspace(0, 2*np.pi, len(history[0]), endpoint=False)
    x, y = np.cos(theta), np.sin(theta)
    
    total_steps = len(history)
    num_graphs = int(np.ceil(total_steps / n))
    
    # Escala de cores consistente
    all_values = np.concatenate(history)
    vmin, vmax = all_values.min(), all_values.max()
    
    for graph_num in range(num_graphs):
        start_step = graph_num * n
        end_step = min((graph_num + 1) * n, total_steps)
        steps_in_this_graph = end_step - start_step
        
        rows = int(np.ceil(steps_in_this_graph / m))
        cols = min(steps_in_this_graph, m)
        
        # ⬇️ CORREÇÃO: Verifica se há subplots para criar
        if steps_in_this_graph > 0:
            fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
            
            # Garante que axes seja sempre um array 2D
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            axes = axes.flatten()
            
            for i in range(steps_in_this_graph):
                step_index = start_step + i
                sc = axes[i].scatter(x, y, s=200, c=history[step_index], 
                                   cmap="plasma", vmin=vmin, vmax=vmax)
                axes[i].set_title(f"Etapa {step_index}")
                axes[i].set_aspect("equal")
                axes[i].axis("off")
                
                # Colorbar individual
                plt.colorbar(sc, ax=axes[i], orientation="vertical", 
                            fraction=0.06, pad=0.02)
            
            # Desliga eixos vazios
            for i in range(steps_in_this_graph, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f"Evolução da População - Gráfico {graph_num + 1} (Etapas {start_step} a {end_step-1})")
            plt.tight_layout()
            plt.show()

# Uso corrigido
plot_evolution_steps(history, n=12, m=3)