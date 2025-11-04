# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 16:33:50 2025

@author: eosjo
"""

import mesa
import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd

class FightOrFlightAgent(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.fear = self.random.uniform(0, 0.2)
        self.anger = self.random.uniform(0, 0.2)
        self.state = "neutral"

    def step(self):
        
        # --- Percepção do ambiente ---
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )

        # Médias locais
        mean_fear = np.mean([agent.fear for agent in neighbors]) if neighbors else 0
        mean_anger = np.mean([agent.anger for agent in neighbors]) if neighbors else 0

        # --- Atualização emocional ---
        threat = self.model.threat_intensity
        self.fear += self.model.alpha_fear * (mean_fear + threat) - self.model.decay * self.fear
        fighting_neighbors = sum(1 for a in neighbors if a.state == "fight") / len(neighbors) if neighbors else 0
        self.anger += self.model.alpha_anger * (mean_anger + fighting_neighbors * threat) - self.model.decay * self.anger

        # Clamping
        self.fear = min(max(self.fear, 0), 1)
        self.anger = min(max(self.anger, 0), 1)

        # --- Decisão comportamental ---
        if self.fear > self.anger:
            self.state = "flight"
        elif self.anger > self.fear:
            self.state = "fight"
        else:
            self.state = "neutral"



class FightOrFlightModel(mesa.Model):
    def __init__(self, width=20, height=20, density=0.9,
                 alpha_fear=0.25, alpha_anger=0.2,
                 threat_intensity=0.3, decay=0.05, seed = None):
        super().__init__(seed=seed)
        self.num_agents = int(width * height * density)
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.alpha_fear = alpha_fear
        self.alpha_anger = alpha_anger
        self.threat_intensity = threat_intensity
        self.decay = decay
      
        # --- Criação e posicionamento dos agentes ---
        agents = FightOrFlightAgent.create_agents(model=self, n=self.num_agents)
        
        # Gera todas as posições possíveis do grid
        all_positions = [(x, y) for x in range(self.grid.width) for y in range(self.grid.height)]
        
        # Sorteia sem repetição
        chosen_positions = self.random.sample(all_positions, self.num_agents)
        
        # Posiciona cada agente em uma posição única
        for agent, pos in zip(agents, chosen_positions):
            self.grid.place_agent(agent, pos)


        # Coletor de dados
        self.datacollector = DataCollector(
            model_reporters={
                "FractionFight": lambda m: self.fraction_state("fight"),
                "FractionFlight": lambda m: self.fraction_state("flight"),
                "FractionNeutral": lambda m: self.fraction_state("neutral"),
            }
        )
        

    def fraction_state(self, state_name):
        total = len(self.agents)
        count = sum(1 for a in self.agents if a.state == state_name)
        return count / total if total > 0 else 0

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")
        
        
# # --- Intervalos de parâmetros ---
# param_range = np.arange(0, 1.1, 0.2)
# fixed_range = np.arange(0, 1.1, 0.25)

# results = []

# # --- Loop principal ---
# for threat_intensity in fixed_range:
#     for decay in fixed_range:
#         for density in fixed_range:
#             for alpha_fear in param_range:
#                 for alpha_anger in param_range:

#                     model = FightOrFlightModel(
#                         width=20, height=20,
#                         density=density,
#                         alpha_fear=alpha_fear,
#                         alpha_anger=alpha_anger,
#                         threat_intensity=threat_intensity,
#                         decay=decay
#                     )

#                     for _ in range(25):
#                         model.step()

#                     data = model.datacollector.get_model_vars_dataframe().iloc[-1]

#                     results.append({
#                         "alpha_fear": alpha_fear,
#                         "alpha_anger": alpha_anger,
#                         "decay": decay,
#                         "threat_intensity": threat_intensity,
#                         "density": density,
#                         "FractionFight": data["FractionFight"],
#                         "FractionFlight": data["FractionFlight"],
#                         "FractionNeutral": data["FractionNeutral"]
#                     })

# df_results = pd.DataFrame(results)
# print("Total de simulações:", len(df_results))

# # --- Função auxiliar para plotar mapas 2D ---
# def plot_heatmaps(df, threat_intensity, decay, density):
#     subset = df[
#         (df["threat_intensity"] == threat_intensity) &
#         (df["decay"] == decay) &
#         (df["density"] == density)
#     ]

#     fig, axes = plt.subplots(1, 3, figsize=(16, 5))

#     for ax, state, cmap, title in zip(
#         axes,
#         ["FractionFight", "FractionFlight", "FractionNeutral"],
#         ["Reds", "Oranges", "Greens"],
#         ["Lutar (Fight)", "Correr (Flight)", "Neutro (Neutral)"]
#     ):
#         pivot = subset.pivot(index="alpha_fear", columns="alpha_anger", values=state)
#         im = ax.imshow(pivot, origin="lower", cmap=cmap, extent=[0, 1, 0, 1], aspect="auto", vmin=0, vmax=1)
#         ax.set_title(f"{title}", fontsize=12)
#         ax.set_xlabel("α_anger")
#         ax.set_ylabel("α_fear")
#         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fração final")

#     fig.suptitle(
#         f"threat_intensity={threat_intensity:.1f}, decay={decay:.1f}, density={density:.1f}",
#         fontsize=14, fontweight="bold"
#     )
#     plt.tight_layout()
#     plt.show()


# # --- Plot automático para cada combinação de parâmetros fixos ---
# for threat_intensity in fixed_range:
#     for decay in fixed_range:
#         for density in fixed_range:
#             plot_heatmaps(df_results, threat_intensity, decay, density)


# ---------- Execução do Modelo ----------
model = FightOrFlightModel(width=20, height=20, density=1,
                           alpha_fear=0.4, alpha_anger=0.0,
                           threat_intensity=0.0, decay=0.5)


for step in range(100):
    model.step()


# ---------- Plot final das proporções ----------
data = model.datacollector.get_model_vars_dataframe()
data.plot(title="Dinâmica Lutar vs Correr (Mesa)")
plt.show()