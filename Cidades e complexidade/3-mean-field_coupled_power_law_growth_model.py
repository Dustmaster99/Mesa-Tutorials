# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 22:57:42 2025

@author: eosjo
"""

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

def plot_potential_grid(model, t):
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
    plt.imshow(grid_data, cmap="gray", origin="lower", vmin=0, vmax=6)
    plt.colorbar(label="Potencia")


    # Título com o tempo atual
    plt.title(f"Distribuição do potencial no grid - t = {t}")
    plt.show()


def count_developed_agents(model):
    """
    Conta quantos agentes do modelo têm a propriedade isDeveloped = True
    """
    return sum(1 for agent in model.agents if agent.isDeveloped)



def count_reach_developed_agents(model):
    """
    Conta quantos agentes do modelo têm a propriedade isDeveloped = True
    """
    return sum(1 for agent in model.agents if agent.reachDevelopment)

def Simulate(model, steps):
    """
    Simula o modelo e plota o valor máximo de P entre todos os agentes ao longo do tempo.
    
    Args:
        model: objeto Mesa model
        steps: número de passos de simulação
    """
    max_P_over_time = []
    developed_over_time = []
    reach_developed_over_time = []
    for t in range(steps):
        plot_potential_grid(model,model.t)
        model.step()  # avança o modelo um passo
        max_P = max(agent.P for agent in model.agents)
        max_P_over_time.append(max_P)
        developed_over_time.append(count_developed_agents(model))
        reach_developed_over_time.append(count_reach_developed_agents(model))
   
    # Plot do valor máximo de P entre as células ao longo do tempo
    # Plota o gráfico
    plt.figure(figsize=(8,5))
    plt.plot(range(steps), max_P_over_time, marker='o')
    plt.xlabel("Tempo")
    plt.ylabel("Valor máximo de P")
    plt.title("Evolução do valor máximo de P ao longo do tempo")
    plt.grid(True)
    plt.show()
    
    # Contagem dos agentes P que estão acima do limiar de desenvolvimento ao longo do tempo
    plt.figure(figsize=(8,4))
    plt.plot(range(steps), developed_over_time, marker="o")
    plt.xlabel("Tempo (t)")
    plt.ylabel("Células desenvolvidos")
    plt.title("Evolução da quantidade de células desenvolvidos ")
    plt.grid(True)
    plt.show()
    
    
    # Contagem dos agentes P que atingiram desenvolvimento ao longo do tempo em algum momento
    plt.figure(figsize=(8,4))
    plt.plot(range(steps), reach_developed_over_time, marker="o")
    plt.xlabel("Tempo (t)")
    plt.ylabel("Células que atingiram desenvolvimento")
    plt.title("Evolução da quantidade de células que atingiram desenvolvimento em algum momento ")
    plt.grid(True)
    plt.show()


class CellAgent(mesa.Agent):
    def __init__(self, model, P0):
        """
        Creates a new agent following Mean_field_coupled_Power_law_with_noise

        Args:
            model: The model instance the agent belongs to.
            P0: Initial population/value at time t=0.
            t : Initial agent time
        """
        super().__init__(model)
        self.P = P0  # Current potential 
        self.t = 0   # Internal time step counter
        self.isDeveloped = False
        self.reachDevelopment = False
        self.developmentThreshold = 4.5
        
    def step(self):
        self.t += 1
        p_sum = 0
        epsilon = random.choice([-1, 1]) # Ruído aleátorio que pode assumir valores de 1 e -1.
        neighbors = self.model.grid.get_neighbors(
                    self.pos,
                    moore=True,       # True = vizinhança de Moore; False = de Von Neumann
                    include_center=True  # não inclui a própria célula
                    )
        
        if len(neighbors) > 0:
            for neighbor in neighbors:
                p_sum +=  neighbor.P
        self.P = (p_sum/len(neighbors)) + epsilon
        
        if self.P > self.developmentThreshold:
            self.isDeveloped = True
        if self.P <= self.developmentThreshold:
            self.isDeveloped = False 
        if self.P > self.developmentThreshold:
            self.reachDevelopment = True
            
        
        
class Mean_field_coupled_Power_law_with_noise_model(mesa.Model):
    def __init__(self, n=100,seed=None):
        super().__init__(seed=seed)
        x_pos = []
        y_pos = []
        self.t = 0
        self.n = n
        self.grid = mesa.space.MultiGrid(n, n, torus=False)
        # create a vector with random initial potencial
        init_pot = [random.randint(-1,1) for _ in range(n*n)]
        # Create n agents
        agents = CellAgent.create_agents(model=self, n=n*n, P0 = init_pot)
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
        
        
model = Mean_field_coupled_Power_law_with_noise_model()
Simulate(model, 100)
    
    
            
            
