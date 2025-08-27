# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 22:34:43 2025

@author: eosjo
"""

import mesa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# --- Agente 3D ---
class FluidCell(mesa.Agent):
    def __init__(self, pos, model,init_temp):
        # Pass the parameters to the parent class.
        super().__init__(model)
        # Create the agent's attribute and set the initial values.
        self.temperature = init_temp
        self.next_temperature = 0
        self.vx = 0 # velocidade horizontal
        self.vy = 0 # velocidade vertical
    
    def process_diffusion(self):
        
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        # Difusão de temperatura
        temp_sum = sum([n.temperature for n in neighbors])
        self.next_temperature = self.temperature + 0.1 * (temp_sum - len(neighbors) * self.temperature)
    
    def process_buoyancy(self):
       
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        
        # Empuxo vertical (buoyancy) baseado na diferença de temperatura
        avg_temp = np.mean([n.temperature for n in neighbors] + [self.temperature])
        self.vy = 0.1 * (self.temperature - avg_temp)
        
        # Movimento horizontal aleatório suave, proporcional a diferença local de temperatura
        temp_dx = (neighbors[0].temperature - neighbors[1].temperature) if len(neighbors) >= 2 else 0
        self.vx = 0.02 * temp_dx + np.random.uniform(-0.01, 0.01)        
        
    def move_cells(self):
        
        self.temperature = self.next_temperature
        # Atualização da posição
        x, y = self.pos
        new_x = min(max(int(round(x + self.vx)), 0), self.model.grid.width - 1)
        new_y = min(max(int(round(y + self.vy)), 0), self.model.grid.height - 1)
        self.model.grid.move_agent(self, (new_x, new_y))   
        
        
# --- Modelo ---
class BenardModel(mesa.Model):
    def __init__(self, width=10, height=10, seed=None):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.grid = mesa.space.MultiGrid(width, height, torus = True)

        # Inicialização do grid com temperatura crescente de cima para baixo
        for x in range(width):
            for y in range(height):
                temp = 1.0 - y / height  # topo frio, base quente
                cell = FluidCell((x,y), self, temp)
                self.grid.place_agent(cell, (x, y))

    def step(self):
        self.agents.do("process_diffusion")
        self.agents.do("process_buoyancy")
        self.agents.do("move_cells")