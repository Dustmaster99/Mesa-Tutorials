# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 18:17:58 2025

@author: eosjo
"""
import seaborn as sns  # Required import
import matplotlib.pyplot as plt  # Often used with Seaborn
from money_model import MoneyModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% 
def plot_wealth_per_id(df, y_max=10):
    """
    Gera gráficos de barras de wealth vs step para cada unique_id.
    Limita o eixo y a y_max.
    
    Parâmetros:
    df : pandas.DataFrame
        DataFrame com colunas 'unique_id', 'wealth', 'step'.
    y_max : float
        Valor máximo do eixo y (default=10).
    """
    unique_ids = df['unique_id'].unique()
    
    for uid in unique_ids:
        subset = df[df['unique_id'] == uid].sort_values('step')
        steps = np.array(subset['step'], dtype=float)
        wealth = np.array(subset['wealth'], dtype=float)
        
        plt.figure(figsize=(8, 5))
        plt.bar(steps, wealth, color='skyblue', width=0.4)
        
        plt.title(f'Wealth vs Step para unique_id {uid}')
        plt.xlabel('Step')
        plt.ylabel('Wealth')
        plt.ylim(0, y_max)  # Limita o eixo y
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
#%% 
# Suponha que você já tenha um DataFrame df (vazio ou com dados prévios)
run_data = pd.DataFrame(columns=["unique_id", "wealth", "step"])  # DataFrame vazio inicial

# Run the model
model = MoneyModel(10)
for step in range(50):
    agent_wealth = []
    model.step()
    agent_wealth = [
    {"unique_id": a.unique_id, "wealth": a.wealth, "step": step}
    for a in model.agents  # Itera sobre todos os agentes
    ]
    run_data = pd.concat([run_data, pd.DataFrame(agent_wealth)], ignore_index=True)
    # Store the results

plot_wealth_per_id(run_data)
