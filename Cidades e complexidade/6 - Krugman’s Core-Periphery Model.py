import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parâmetros da simulação
# ----------------------------
N = 24               # número de células
steps = 20          # número de etapas (ajuste se quiser mais)
K1, K2 = 1.0, 0.7    # escalas
alpha, beta = 1.4, 0.25  # decaimentos

# Distribuição inicial
P = np.ones(N) + 0.2 * np.random.rand(N)

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
    
    # Normalização para manter soma constante
    P = P * (len(P) / P.sum())
    
    history.append(P.copy())

# ----------------------------
# Visualização: círculo em cada etapa
# ----------------------------
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
x, y = np.cos(theta), np.sin(theta)

fig, axes = plt.subplots(4, 5, figsize=(12,9))  # grade 3x4 para 12 passos
axes = axes.flatten()

for t, ax in enumerate(axes):
    sc = ax.scatter(x, y, s=250, c=history[t], cmap="plasma")
    ax.set_title(f"Etapa {t}")
    ax.set_aspect("equal")
    ax.axis("off")

fig.colorbar(sc, ax=axes, orientation="vertical", fraction=0.02, pad=0.05, label="População")
plt.suptitle("Evolução da distribuição da população no círculo")
plt.tight_layout()
plt.show()
