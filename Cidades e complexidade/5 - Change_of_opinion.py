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
    - Branco = NO_PREFERENCE
    - Azul = BEER
    - Vermelho = WINE
    - Marrom = COFFEE
    """
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
            elif agent.preference == BEVERAGE_PREFERENCE.COFFEE:
                grid_data[x, y] = [0.6, 0.3, 0]  # marrom

    plt.figure(figsize=(5,5))
    plt.imshow(grid_data, interpolation="nearest")
    plt.title(f"Passo {t}")
    plt.axis("off")

def Simulate(model, steps):
    """
    Simula o modelo, plota o grid a cada passo, 
    e no final plota a evolução percentual de cada opinião.
    """
    # Armazenar estado inicial (para comparação se quiser)
    initial_preferences = [agent.preference for agent in model.agents]

    # Listas para evolução percentual
    no_pref_perc = []
    wine_perc = []
    beer_perc = []
    coffee_perc = []

    for t in range(steps):
        # Plot do grid
        plot_opinion_grid(model, t)

        # Contar agentes por opinião ANTES de atualizar o modelo
        counts = {BEVERAGE_PREFERENCE.NO_PREFERENCE: 0,
                  BEVERAGE_PREFERENCE.WINE: 0,
                  BEVERAGE_PREFERENCE.BEER: 0,
                  BEVERAGE_PREFERENCE.COFFEE: 0}

        for agent in model.agents:
            counts[agent.preference] += 1

        total_agents = len(model.agents)
        no_pref_perc.append(100 * counts[BEVERAGE_PREFERENCE.NO_PREFERENCE] / total_agents)
        wine_perc.append(100 * counts[BEVERAGE_PREFERENCE.WINE] / total_agents)
        beer_perc.append(100 * counts[BEVERAGE_PREFERENCE.BEER] / total_agents)
        coffee_perc.append(100 * counts[BEVERAGE_PREFERENCE.COFFEE] / total_agents)

        # Agora atualiza modelo
        model.step()
        if model.t == 10:
            model.insert_coffee_lovers(0.3)
            

    # Plot da evolução percentual
    plt.figure(figsize=(8,5))
    plt.plot(range(steps), no_pref_perc, label="No Preference", color="gray")
    plt.plot(range(steps), wine_perc, label="Wine", color="red")
    plt.plot(range(steps), beer_perc, label="Beer", color="blue")
    plt.plot(range(steps), coffee_perc, label="coffee", color="brown")
    plt.xlabel("Tempo")
    plt.ylabel("Porcentagem de agentes (%)")
    plt.title("Evolução percentual das opiniões ao longo do tempo")
    plt.legend()
    plt.grid(True)
    plt.show()




class BEVERAGE_PREFERENCE(Enum):
        NO_PREFERENCE = 0    
        WINE= 1
        BEER = 2
        COFFEE = 3 


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
        self.epsilon1 = 1  #Chance  that your opinion changes at a given time
        self.epsilon2 = 0  # Chance that you opionio sudenly changes from one opinio to another
        self.epsilon3 = 0.5 
        
    def step(self):
        self.t += 1
        sum_0 = 0
        sum_1 = 0
        sum_2 = 0
        sum_3 = 0
    
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False
        )
    
        for neighbor in neighbors:
            if neighbor.preference == BEVERAGE_PREFERENCE.WINE:
                sum_1 += 1
            elif neighbor.preference == BEVERAGE_PREFERENCE.BEER:
                sum_2 += 1
            elif neighbor.preference == BEVERAGE_PREFERENCE.NO_PREFERENCE:
                sum_0 += 1
            elif neighbor.preference == BEVERAGE_PREFERENCE.COFFEE:
                sum_3 += 1
    
        # Influência dos vizinhos
        if random.random() < self.epsilon1:
            if self.preference == BEVERAGE_PREFERENCE.WINE:
                if sum_2> sum_1 and sum_2 > sum_3:
                    self.preference = BEVERAGE_PREFERENCE.BEER
                elif sum_3> sum_1 and sum_3 > sum_2:
                    self.preference = BEVERAGE_PREFERENCE.COFFEE
            
            elif self.preference == BEVERAGE_PREFERENCE.BEER:
                if sum_1 > sum_2 and sum_1> sum_3:
                    self.preference = BEVERAGE_PREFERENCE.WINE
                if sum_3 > sum_1 and sum_3> sum_2:
                    self.preference = BEVERAGE_PREFERENCE.COFFEE
           
            else: 
                if sum_1> sum_2 and sum_1> sum_3 : 
                    self.preference = BEVERAGE_PREFERENCE.WINE 
                if sum_2> sum_1 and sum_2> sum_3: 
                    self.preference = BEVERAGE_PREFERENCE.BEER 
                if sum_3> sum_1 and sum_3> sum_2: 
                    self.preference = BEVERAGE_PREFERENCE.COFFEE 
                
        # Mudança aleatória
        if random.random() < self.epsilon2:
            if self.preference == BEVERAGE_PREFERENCE.WINE:
                self.preference = BEVERAGE_PREFERENCE.BEER
           
            elif self.preference == BEVERAGE_PREFERENCE.BEER:
                self.preference = BEVERAGE_PREFERENCE.WINE
                
            else :
                if random.random() <= self.epsilon3:
                   self.preference = BEVERAGE_PREFERENCE.WINE 
                if random.random() > self.epsilon3:    
                    self.preference = BEVERAGE_PREFERENCE.BEER

        
class Change_of_opinion_model(mesa.Model):
    def __init__(self, n=70,seed=None):
        super().__init__(seed=seed)
        x_pos = []
        y_pos = []
        self.t = 0
        self.n = n
        self.grid = mesa.space.MultiGrid(n, n, torus=False)
        # create a vector with random Preference distribution at start
        
        # quantidade de NO_PREFERENCE (95%)
        n_no_preference = int(n * n * 0.95)
        n_wine_preference = int(n * n * 0.025)
        n_beer_preference = (n*n) - (n_no_preference + n_wine_preference)
        # criar lista inicial
        init_preferences_no_preferece = [BEVERAGE_PREFERENCE.NO_PREFERENCE] * n_no_preference
        init_preferences_wine = [BEVERAGE_PREFERENCE.WINE] * n_wine_preference
        init_preferences_beer = [BEVERAGE_PREFERENCE.BEER] * n_beer_preference
        
        init_preferences = (
            init_preferences_no_preferece + 
            init_preferences_wine + 
            init_preferences_beer
            )
        random.shuffle(init_preferences)
        # distribuir os outros 5% aleatoriamente entre WINE e BEER

        
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
            
        self.my_agents = agents
    
    def insert_coffee_lovers(self, percentage):
        n_coffee_agents = int((self.n*self.n)*percentage)
        random_agents = random.sample(self.my_agents, n_coffee_agents)
        for a in random_agents:
            a.preference = BEVERAGE_PREFERENCE.COFFEE
                
    def step(self):
        self.t += 1
        self.agents.shuffle_do("step")

            
        
        
        
model = Change_of_opinion_model()
Simulate(model, 20)