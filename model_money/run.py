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
from matplotlib.ticker import MaxNLocator
#%% 
def plot_wealth_per_id(df, y_max=10):
    """
    Gera gráficos de barras de wealth vs step para cada unique_id,
    exibindo 4 gráficos por figura (2x2).
    
    Parâmetros:
    df : pandas.DataFrame
        DataFrame com colunas 'unique_id', 'wealth', 'step'.
    y_max : float
        Valor máximo do eixo y (default=10).
    """
    unique_ids = df['unique_id'].unique()
    
    # Percorre os IDs de 4 em 4
    for i in range(0, len(unique_ids), 4):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()  # transforma matriz 2x2 em vetor
        
        for j, uid in enumerate(unique_ids[i:i+4]):
            subset = df[df['unique_id'] == uid].sort_values('step')
            steps = np.array(subset['step'], dtype=float)
            wealth = np.array(subset['wealth'], dtype=float)
            
            axs[j].bar(steps, wealth, color='skyblue', width=0.4)
            axs[j].set_title(f'Wealth vs Step - ID {uid}')
            axs[j].set_xlabel('Step')
            axs[j].set_ylabel('Wealth')
            axs[j].set_ylim(0, y_max)
            axs[j].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Remove eixos vazios se não tiver 4 plots completos
        for k in range(j+1, 4):
            fig.delaxes(axs[k])
        
        fig.tight_layout()
        plt.show()
        
   #%%     
def plot_wealth_distribution(df, step_values, n_per_page=4, y_max=None):
    """
    Plota histogramas para múltiplos steps, organizados em páginas com n_per_page gráficos.

    Parâmetros:
    df : pandas.DataFrame
        DataFrame com colunas 'unique_id', 'wealth', 'step'.
    step_values : list
        Lista de steps que serão plotados.
    n_per_page : int
        Número de gráficos por página (subplots por figure).
    y_max : float ou None
        Limite superior do eixo y (opcional).
    """
    total_steps = len(step_values)
    
    for start in range(0, total_steps, n_per_page):
        # Seleciona o bloco de steps desta "página"
        block_steps = step_values[start:start + n_per_page]
        
        # Cria figure com quantidade adequada de subplots
        fig, axes = plt.subplots(1, len(block_steps), figsize=(6 * len(block_steps), 5))
        
        # Garantir que axes seja iterável mesmo quando só houver 1 subplot
        if len(block_steps) == 1:
            axes = [axes]
        
        for ax, step in zip(axes, block_steps):
            subset = df[df['step'] == step]
            if subset.empty:
                ax.set_title(f"Step {step} (sem dados)")
                ax.axis("off")
                continue
            
            wealth_values = np.array(subset['wealth'], dtype=float)
            
            ax.hist(
                wealth_values,
                bins=range(int(wealth_values.min()), int(wealth_values.max()) + 2),
                color='skyblue',
                edgecolor='black',
                align='left'
            )
            ax.set_title(f"Distribuição da riqueza - Step {step}")
            ax.set_xlabel("Wealth")
            ax.set_ylabel("Quantidade de Agentes")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            if y_max is not None:
                ax.set_ylim(0, y_max)
            
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
#%%         

def plot_agents_by_step_heatmap(df, step_values, cmap="RdYlGn_r"):
    """
    Plota heatmap da distribuição espacial dos agentes, mostrando em cada célula
    a soma da riqueza e a quantidade de agentes que compõe a soma (soma/contagem).
    A intensidade da cor representa a média da riqueza por agente na célula.
    """
    for step in step_values:
        subset = df[df["step"] == step]
        if subset.empty:
            print(f"[Aviso] Nenhum dado encontrado para step {step}.")
            continue
        
        # Agrupar por posição: soma da riqueza e contagem de agentes
        grouped = subset.groupby(['y','x']).agg({'wealth':['sum','count']})
        wealth_sum = grouped['wealth']['sum'].unstack(fill_value=0).astype(float)
        wealth_count = grouped['wealth']['count'].unstack(fill_value=0).astype(int)

        # Média da riqueza por agente (para colormap)
        with np.errstate(divide='ignore', invalid='ignore'):  # evita divisão por zero
            wealth_avg = np.where(wealth_count > 0, wealth_sum / wealth_count, 0)

        plt.figure(figsize=(6,6))
        im = plt.imshow(wealth_avg, origin='lower', cmap=cmap, aspect='equal')
        cbar = plt.colorbar(im, label="Riqueza média por agente")
        cbar.locator = MaxNLocator(integer=True, prune='lower')
        cbar.update_ticks()

        # Escrever soma/contagem em cada célula
        for (i, j), val in np.ndenumerate(wealth_sum.values):
            count = wealth_count.values[i, j]
            plt.text(j, i, f"{int(val)}/{count}", ha='center', va='center', color='black', fontsize=10)

        plt.title(f"Distribuição Espacial dos Agentes - Step {step}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(False)
        plt.show()
#%% 
N_steps = 100
lista_steps = []
N_agents = 1000
grid_width = 10
grid_height = 10

#%% 

# Suponha que você já tenha um DataFrame df (vazio ou com dados prévios)
run_data = pd.DataFrame(columns=["unique_id", "wealth", "step"])  # DataFrame vazio inicial

# Run the model
model = MoneyModel(N_agents,grid_width,grid_height)
for step in range(N_steps):
    agent_wealth = []
    model.step()
    agent_wealth = [
    {
        "unique_id": a.unique_id,
        "wealth": a.wealth,
        "x": a.pos[0],
        "y": a.pos[1],
        "step": step
    }
    for a in model.agents
    ]
    
    run_data = pd.concat([run_data, pd.DataFrame(agent_wealth)], ignore_index=True)
    # Store the results


for step_value in range(N_steps):
    lista_steps.append(step_value)
#plot_wealth_per_id(run_data)
plot_wealth_distribution(run_data, lista_steps, y_max=1000 )
plot_agents_by_step_heatmap(run_data,lista_steps)
