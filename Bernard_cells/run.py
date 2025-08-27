# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 22:34:45 2025

@author: eosjo
"""
from Bernard_cells import BenardModel
import matplotlib.pyplot as plt
import numpy as np


def plot_grid(model):
    width = model.width
    height = model.height

    grid_array = np.zeros((height, width))
    vx_array = np.zeros((height, width))
    vy_array = np.zeros((height, width))

    for agent in model.agents:
        x, y = agent.pos
        grid_array[y, x] = agent.temperature
        vx_array[y, x] = agent.vx
        vy_array[y, x] = agent.vy

    plt.imshow(grid_array, cmap='coolwarm')
    
    # Grade quadriculada mais visível
    plt.grid(color='k', linestyle='-', linewidth=0.7)
    plt.xticks(np.arange(-0.5, width, 1))
    plt.yticks(np.arange(-0.5, height, 1))
    
    # Setas de velocidade
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    plt.quiver(X, Y, vx_array, vy_array, color='black', pivot='mid', scale=0.5, width=0.004)

    # Escrever temperatura na célula
    for agent in model.agents:
        x, y = agent.pos
        plt.text(x, y, f'{agent.temperature:.1f}', color='white',
                 ha='center', va='center', fontsize=8, fontweight='bold')

    plt.colorbar(label='Temperatura')
    plt.title('Células de Bénard 2D')
    
    # Ajustes para melhor visualização
    plt.axis('equal')              # células quadradas
    plt.xlim(-0.5, width-0.5)
    plt.ylim(-0.5, height-0.5)
    
    plt.pause(0.01)

def run_model(model, steps=100):
    plt.ion()  # modo interativo ON
    fig = plt.figure(figsize=(6,6))
    for _ in range(steps):
        model.step()       # atualiza o modelo
        plt.clf()          # limpa o plot
        plot_grid(model)   # plota o estado atual

    plt.ioff()
    plt.show()
    
#Run the model
model = BenardModel()
run_model(model)