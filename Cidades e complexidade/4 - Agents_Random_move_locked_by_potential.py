# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 09:54:04 2025

@author: eosjo
"""


import mesa
import math
import matplotlib.pyplot as plt
import numpy as np
import random

import numpy as np
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def Simulate(model, steps, delay=0.2):
    """
    Simula o modelo por 'steps' passos.
    Plota o grid em escala de cinza a cada passo,
    e retorna o gráfico final do número de células desenvolvidas.
    
    Args:
        model : modelo Mesa
        steps : número de passos a simular
        delay : tempo em segundos entre cada plot do grid (default 0.2s)
    
    Returns:
        grid_data : grid final com contagem de IndividualAgents
        developed_counts : lista com número de células desenvolvidas a cada passo
    """
    n = model.n
    developed_counts = []  # Acumula contagem de células desenvolvidas

    plt.figure(figsize=(6, 6))
    
    for t in range(steps):
        model.step()  # Atualiza o modelo

        # Conta células desenvolvidas
        count = sum(cell.reachDevelopment for cell in model.cell_agents)
        developed_counts.append(count)

        # Cria matriz com contagem de IndividualAgents por célula
        grid_data = np.zeros((n, n))
        for x in range(n):
            for y in range(n):
                cellmates = model.grid.get_cell_list_contents([(x, y)])
                grid_data[y, x] = sum(isinstance(a, IndividualAgent) for a in cellmates)

        # Plota o grid em escala de cinza
        plt.imshow(grid_data, cmap='Greys', origin='lower', vmin=0, vmax=grid_data.max())
        plt.title(f"Passo {t+1}")
        plt.colorbar(label='Número de agentes')
        plt.pause(delay)
        plt.clf()  # Limpa o plot para o próximo passo

    plt.show()  # Mostra o último estado do grid

    # --- Gráfico final do desenvolvimento ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, steps+1), developed_counts, marker='o')
    plt.xlabel("Tempo (passos)")
    plt.ylabel("Número de células desenvolvidas")
    plt.title("Evolução das células desenvolvidas ao longo do tempo")
    plt.grid(True)
    plt.show()







class CellAgent(mesa.Agent):
    def __init__(self, model):
        """
        Creates a new agent following for cell
        Args:
            model: The model instance the agent belongs to.
            P0: Initial population
            t : Initial agent time
        """
        super().__init__(model)
        self.P = 0  # Current population
        self.t = 0   # Internal time step counter
        self.isDeveloped = False
        self.reachDevelopment = False
        self.developmentThreshold = 4
        
        
        
    def update_population(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        countAgents = sum(isinstance(a, IndividualAgent) for a in cellmates)
        self.P = countAgents
        
        
    def define_if_development(self):
        
        """ it will count the number of agents currently in the cell to calculate it population and try to 
        stabilish if itself reach development or not """
        self.t += 1
        p_sum = 0
        neighbors = self.model.grid.get_neighbors(
                    self.pos,
                    moore=True,       # True = vizinhança de Moore; False = de Von Neumann
                    include_center=True  # não inclui a própria célula
                    )
        for neighbor in neighbors:
            if isinstance(neighbor, CellAgent):
                p_sum += neighbor.P

        if p_sum > self.developmentThreshold:
            self.isDeveloped = True
        if p_sum <= self.developmentThreshold:
            self.isDeveloped = False 
        if p_sum > self.developmentThreshold:
            self.reachDevelopment = True
                   
class IndividualAgent(mesa.Agent):
    """An agent with starting position, that moves thought the grdi"""
    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)
        
    def move(self):
        # Pega todos os agentes na mesma célula
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        cell_agents = [a for a in cellmates if isinstance(a, CellAgent)]
        for cell in cell_agents:
            # supondo que o Cell tenha um atributo 'value'
            if cell.reachDevelopment == False:
                
                possible_steps = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=True
                )
                
                new_position = random.choice(possible_steps)
                self.model.grid.move_agent(self, new_position)
                
class Agents_random_movement_with_development_threshold(mesa.Model):
    def __init__(self, n=131, M =2500 ,seed=None):
        super().__init__(seed=seed)
        x_pos = []
        y_pos = []
        self.t = 0
        self.n = n
        self.M = M
        self.grid = mesa.space.MultiGrid(n, n, torus=False)
        # Create n agents
        self.cell_agents = CellAgent.create_agents(model=self, n=n*n)
        # Cria um agente para cada célula
        for x in range(self.n):
            for y in range(self.n):
                x_pos.append(x)
                y_pos.append(y)
        for a, i, j in zip(self.cell_agents, x_pos, y_pos):
            # Add 1 agent to each grid cell
            self.grid.place_agent(a, (i, j))
                        
        self.individual_agents = IndividualAgent.create_agents(model=self, n=M)
        for agent in self.individual_agents:
            x = random.randrange(self.n)
            y = random.randrange(self.n)
            self.grid.place_agent(agent, (x, y))


    def step(self):
        self.t += 1
        self.cell_agents.do("update_population")
        self.cell_agents.do("define_if_development")
        self.individual_agents.do("move")


model = Agents_random_movement_with_development_threshold()
Simulate(model, 200)


            