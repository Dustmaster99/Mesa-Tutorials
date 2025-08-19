# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 18:02:34 2025

@author: eosjo
"""

import mesa

# Data visualization tools.
import seaborn as sns

# Has multi-dimensional arrays and matrices. Has a large collection of
# mathematical functions to operate on these arrays.
import numpy as np

# Data manipulation and analysis.
import pandas as pd

import random

#%%       
class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        # Create the agent's attribute and set the initial values.
        self.wealth = random.randint(1, 5)  # valor inteiro entre 1 e 5
        

    def say_wealth(self):
        # The agent's step will go here.
        # For demonstration purposes we will print the agent's unique_id
        print(f"Hi, I am an agent, my wealth is  {str(self.wealth)}.")
        
    def exchange(self):
        # Verify agent has some wealth
        if self.wealth > 0:
            other_agent = self.random.choice(self.model.agents)
            if other_agent is not None:
                other_agent.wealth += 1
                self.wealth -= 1      
                
    def move(self):
    # 1. Pega vizinhos (incluindo a célula atual, já que o agente pode ficar parado)
        possible_steps = self.model.grid.get_neighborhood(
        self.pos,
        moore=True,
        include_center=True
        )

        best_position = self.pos
        best_avg_wealth = self.wealth  # inicializa como a riqueza do próprio agente

    # 2. Avalia cada célula vizinha
        for pos in possible_steps:
        # pega todos os agentes naquela célula
            agents_in_cell = self.model.grid.get_cell_list_contents([pos])

            if agents_in_cell:  # só calcula se houver agentes
                total_wealth = sum(a.wealth for a in agents_in_cell)
                avg_wealth = total_wealth / len(agents_in_cell)

            # 3. Se a média for maior que a riqueza do agente, é candidata
                if avg_wealth > best_avg_wealth:
                    best_avg_wealth = avg_wealth
                    best_position = pos

    # 4. Move para a melhor posição (se mudou)
        if best_position != self.pos:
            self.model.grid.move_agent(self, best_position)
        
    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        
        # Ensure agent is not giving money to itself
        cellmates.pop(
            cellmates.index(self)
        )
        if len(cellmates) > 0:
            other = self.random.choice(cellmates)
            if self.wealth > 0 :
                other.wealth += 1
                self.wealth -= 1


class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, n, width, height, seed=None):
        
        super().__init__(seed=seed)
        self.num_agents = n
        
        self.grid = mesa.space.MultiGrid(width, height, True)
        # Create n agents
        agents = MoneyAgent.create_agents(model=self, n=n)
        # Create x and y positions for agents
        x = self.rng.integers(0, self.grid.width, size=(n,))
        y = self.rng.integers(0, self.grid.height, size=(n,))
        for a, i, j in zip(agents, x, y):
            # Add the agent to a random grid cell
            self.grid.place_agent(a, (i, j))

    def step(self):
        """Advance the model by one step."""

        # This function psuedo-randomly reorders the list of agent objects and
        # then iterates through calling the function passed in as the parameter
        self.agents.shuffle_do("move")
        self.agents.shuffle_do("give_money")
        
        