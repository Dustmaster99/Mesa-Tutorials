# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 21:18:25 2025

@author: eosjo
"""

import mesa
import math
import matplotlib.pyplot as plt
import numpy as np
import random

import numpy as np
import matplotlib.pyplot as plt

def plot_population_grid(model, t):
    """
    Plota a população atual do modelo em um grid n x n 
    com escala de cinza (preto = mínimo, branco = máximo).
    Cada célula mostra o valor P do agente (ou soma de P se houver mais de um agente).
    
    Args:
        model: objeto Mesa model
        t: tempo atual (step do modelo)
    """
    n = model.n
    grid_data = np.zeros((n, n))

    # Preenche os valores da grade
    for (contents, (x, y)) in model.grid.coord_iter():
        # soma a população dos agentes naquela célula
        total_P = sum([agent.P for agent in contents])
        grid_data[y, x] = total_P   # y = linha, x = coluna

    # Plota a imagem
    plt.figure(figsize=(6,6))
    plt.imshow(grid_data, cmap="gray", origin="lower", vmin=0, vmax=np.max(grid_data))
    plt.colorbar(label="População")

    # Coloca o valor de P dentro de cada célula
    #for y in range(n):
    #    for x in range(n):
    #        plt.text(x, y, f"{grid_data[y, x]/1000:.1f}", ha='center', va='center', color='red')


    # Título com o tempo atual
    plt.title(f"Distribuição da população no grid - t = {t}")
    plt.show()


import matplotlib.pyplot as plt

def Simulate(model, steps):
    """
    Simula o modelo e plota o valor máximo de P entre todos os agentes ao longo do tempo.
    
    Args:
        model: objeto Mesa model
        steps: número de passos de simulação
    """
    max_P_over_time = []

    for t in range(steps):
        plot_population_grid(model,model.t)
        model.step()  # avança o modelo um passo
        max_P = max(agent.P for agent in model.agents)
        max_P_over_time.append(max_P)

    # Plota o gráfico
    plt.figure(figsize=(8,5))
    plt.plot(range(steps), max_P_over_time, marker='o')
    plt.xlabel("Tempo")
    plt.ylabel("Valor máximo de P")
    plt.title("Evolução do valor máximo de P ao longo do tempo")
    plt.grid(True)
    plt.show()
    
    return max_P_over_time  # opcional, se quiser usar os dados depois

class CellAgent(mesa.Agent):
    def __init__(self, model, C, P0, lam):
        super().__init__(model)
        self.C = C          # capacidade de suporte
        self.P0 = P0        # população inicial
        self.lam = lam      # taxa de crescimento
        self.t = 0          # tempo
        self.P = P0         # população atual

    def step(self):
        # Atualiza população segundo equação logística
        self.t += 1
        self.P = self.P + self.lam * self.P * (1 - self.P / self.C)


class LogisticModel(mesa.Model):
    def __init__(self, n=10, C=5000, lam=3.4,seed=None):
        super().__init__(seed=seed)
        x_pos = []
        y_pos = []
        self.t = 0
        self.n = n
        self.grid = mesa.space.MultiGrid(n, n, torus=False)
        # create a vector with random initial population between 1 and 20
        init_pop = [random.randint( 1, 10) for _ in range(n*n)]
        # Create n agents
        agents = CellAgent.create_agents(model=self, n=n*n, C=C,lam=lam, P0 =  init_pop)
        # Cria um agente para cada célula
        for x in range(self.n):
            for y in range(self.n):
                x_pos.append(x)
                y_pos.append(y)
        for a, i, j in zip(agents, x_pos, y_pos):
            # Add 1 agent to each grid cell
            self.grid.place_agent(a, (i, j))

    def step(self):
        self.t += 1
        self.agents.do("step")
        
        
model = LogisticModel()
Simulate(model, 20)
    
    
            
            
