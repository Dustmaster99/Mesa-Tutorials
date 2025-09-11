# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 15:32:32 2025

@author: eosjo
"""

from enum import Enum
import mesa
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import numpy as np

def plot_opinion_grid(model, t):
    """
    Plota o grid de opiniões:
    - Branco = sem preferência
    - Azul   = cerveja
    - Vermelho = vinho
    """
    # cria matriz do grid
    grid_data = np.zeros((model.grid.width, model.grid.height, 3))  # RGB

    for cell_contents, (x, y) in model.grid.coord_iter():
        if len(cell_contents) > 0:
            agent = cell_contents[0]
            if agent.preference == BEVERAGE_PREFERENCE.NO_PREFERENCE:
                grid_data[x, y] = [1, 1, 1]      # branco
            elif agent.preference == BEVERAGE_PREFERENCE.BEER:
                grid_data[x, y] = [0, 0, 1]      # azul
            elif agent.preference == BEVERAGE_PREFERENCE.WINE:
                grid_data[x, y] = [1, 0, 0]      # vermelho


    plt.figure(figsize=(5,5))
    plt.imshow(grid_data, interpolation="nearest")
    plt.title(f"Passo {t}")
    plt.axis("off")
    plt.show()


def Simulate(model, steps):
    """
    Simula o modelo e plota:
    1. Grid de opiniões a cada passo
    2. Evolução percentual de agentes que mudaram de opinião em relação ao estado inicial
    """
    # Guardar estado inicial das preferências
    initial_preferences = [agent.preference for agent in model.agents]

    # Lista para armazenar evolução percentual
    changed_percentage = []

    for t in range(steps):
        # Plota o grid colorido
        plot_opinion_grid(model, t)

        # Passo do modelo
        model.step()

        # Contar agentes que mudaram de opinião em relação ao inicial
        changes = sum(agent.preference != initial for agent, initial in zip(model.agents, initial_preferences))
        percent = (changes / len(model.agents)) * 100
        changed_percentage.append(percent)

    # Plot da evolução percentual
    plt.figure(figsize=(8,5))
    plt.plot(range(steps), changed_percentage, marker='o')
    plt.xlabel("Tempo")
    plt.ylabel("Percentual de agentes que mudaram de opinião (%)")
    plt.title("Evolução da mudança de opinião ao longo do tempo")
    plt.grid(True)
    plt.show()



class BEVERAGE_PREFERENCE(Enum):
        NO_PREFERENCE = 0    
        WINE= 1
        BEER = 2


class CellAgent(mesa.Agent):
    def __init__(self, model, Opinion: BEVERAGE_PREFERENCE):
        """
        Creates a new agent following the change of opinio model

        Args:
            model: The model instance the agent belongs to.
            preference: Initial preference for the agent
            t : Initial agent time
        """
        super().__init__(model)
        self.t = 0
        self.preference = Opinion
        
    def step(self):
        self.t += 1
        sum_0 = 0
        sum_1 = 0
        sum_2 = 0
        
        neighbors = self.model.grid.get_neighbors(
                    self.pos,
                    moore=True,       # True = vizinhança de Moore; False = de Von Neumann
                    include_center= False  # não inclui a própria célula
                    )
        
        if len(neighbors) > 0:
            for neighbor in neighbors:
                if neighbor.preference == BEVERAGE_PREFERENCE.WINE:
                    sum_1 += 1
                elif neighbor.preference == BEVERAGE_PREFERENCE.BEER:
                    sum_2 += 1
                elif neighbor.preference == BEVERAGE_PREFERENCE.NO_PREFERENCE:
                    sum_0 += 1
           
            if self.preference == BEVERAGE_PREFERENCE.WINE:
                if sum_2> sum_1:
                    self.preference = BEVERAGE_PREFERENCE.BEER
            
            elif self.preference == BEVERAGE_PREFERENCE.BEER:
                if sum_1> sum_2:
                    self.preference = BEVERAGE_PREFERENCE.WINE
                    
            else:
                if sum_1> sum_2:
                    self.preference = BEVERAGE_PREFERENCE.WINE
                if sum_2> sum_1:
                    self.preference = BEVERAGE_PREFERENCE.BEER

                    
        
class Change_of_opinion_model(mesa.Model):
    def __init__(self, n=100,seed=None):
        super().__init__(seed=seed)
        x_pos = []
        y_pos = []
        self.t = 0
        self.n = n
        self.grid = mesa.space.MultiGrid(n, n, torus=False)
        # create a vector with random Preference distribution at start
        
        # quantidade de NO_PREFERENCE (95%)
        n_no_preference = int(n * n * 0.95)
        n_other = (n*n) - n_no_preference
        # criar lista inicial
        init_preferences = [BEVERAGE_PREFERENCE.NO_PREFERENCE] * n_no_preference
        # distribuir os outros 5% aleatoriamente entre WINE e BEER
        for _ in range(n_other):
            init_preferences.append(random.choice([BEVERAGE_PREFERENCE.WINE, BEVERAGE_PREFERENCE.BEER]))
        
        random.shuffle(init_preferences)
        #init_preferences = [random.choice(list(BEVERAGE_PREFERENCE)) for _ in range(n*n)]
        
        # Create n agents
        agents = CellAgent.create_agents(model=self, n=n*n, Opinion = init_preferences)
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
        self.agents.shuffle_do("step")
        
        
model = Change_of_opinion_model()
Simulate(model, 10)