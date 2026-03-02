import mesa
import numpy as np
from mesa.space import NetworkGrid
import networkx as nx
import random
from mesa.datacollection import DataCollector
import itertools
import matplotlib.pyplot as plt      
from mesa.datacollection import DataCollector      
import pandas as pd
import os

       
def get_initial_state(agent_type: str) -> int:
    states = {
        "A": 0,
        "B": 1,
        "C": 2
    }
    try:
        return states[agent_type.upper()]
    except KeyError:
        raise ValueError("Value Must be A, B or C")
    
def get_initial_adoption_threshold(agent_profile: str) -> float:
    states = {
        "immutable": 0,
        "conservative": 0.3,
        "influenceable": 0.8
    }
    
    try:
        return states[agent_profile.lower()]
    except KeyError:
        raise ValueError("Value Must be immutable, conservative or influenceable")
        

class Agent(mesa.Agent):
    def __init__(self, model, pos, ID, agent_type, agent_profile):
        super().__init__(model)

        self.pos = pos      # <- importante
        self.ID = ID
        self.agent_type = agent_type
        self.agent_profile = agent_profile
        # 👇 COLOQUE O DEBUG AQUI
        print("DEBUG agent_profile:", agent_profile, type(agent_profile))

        self.paradigm_state = get_initial_state(self.agent_type)
        self.adoption_threshold = get_initial_adoption_threshold(agent_profile)
        



class ComunityDifusionModel(mesa.Model):

    def __init__(
        self,
        nA: int,
        nB: int,
        nC: int,
        dA: float,
        dB: float,
        dC: float,
        rho_AB: float,
        rho_AC: float,
        rho_BC: float,
        delta_AB: float,
        delta_AC: float,
        delta_BC: float,
        profile_dist_A: dict,
        profile_dist_B: dict,
        profile_dist_C: dict,
        LogMetrics: bool = True,
        seed: int | None = None
    ):

        super().__init__(seed=seed)

     
    
        # ===============================
        # Salvar parâmetros
        # ===============================
        self.nA = nA
        self.nB = nB
        self.nC = nC
        self.dA = dA
        self.dB = dB
        self.dC = dC
        self.rho_AB = rho_AB
        self.rho_AC = rho_AC
        self.rho_BC = rho_BC
        self.delta_AB = delta_AB
        self.delta_AC = delta_AC
        self.delta_BC = delta_BC
        self.profile_dist_A = profile_dist_A
        self.profile_dist_B = profile_dist_B
        self.profile_dist_C = profile_dist_C
        self.LogMetrics = LogMetrics
        self.seed = seed
        
        # Inicializar tempo
        self.time = 0

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.G = self._generate_graph()
        self.grid = NetworkGrid(self.G)

        self._create_agents()
        self.setup_datacollector()
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # =========================================================
    # Função para criar o grafo
    # =========================================================
    def _generate_graph(self):

        GA = nx.erdos_renyi_graph(self.nA, self.dA, seed=self.seed)
        GB = nx.erdos_renyi_graph(self.nB, self.dB, seed=self.seed)
        GC = nx.erdos_renyi_graph(self.nC, self.dC, seed=self.seed)

        G = nx.Graph()

        def add_community(G_sub, prefix):
            for i, node in enumerate(G_sub.nodes()):
                node_name = f"{prefix}_{i}"
                G.add_node(node_name, community=prefix)

            for u, v in G_sub.edges():
                G.add_edge(f"{prefix}_{u}", f"{prefix}_{v}")

        add_community(GA, "A")
        add_community(GB, "B")
        add_community(GC, "C")

        nodes_A = sorted([n for n in G.nodes if n.startswith("A_")])
        nodes_B = sorted([n for n in G.nodes if n.startswith("B_")])
        nodes_C = sorted([n for n in G.nodes if n.startswith("C_")])

        def connect_interfaces(nodes_X, nodes_Y, rho, delta):

            size_X = int(rho * len(nodes_X))
            size_Y = int(rho * len(nodes_Y))

            interface_X = random.sample(nodes_X, size_X)
            interface_Y = random.sample(nodes_Y, size_Y)

            for x in interface_X:
                for y in interface_Y:
                    if random.random() < delta:
                        G.add_edge(x, y)

        connect_interfaces(nodes_A, nodes_B, self.rho_AB, self.delta_AB)
        connect_interfaces(nodes_A, nodes_C, self.rho_AC, self.delta_AC)
        connect_interfaces(nodes_B, nodes_C, self.rho_BC, self.delta_BC)

        return G
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # =========================================================
    # Função para gerar perfis multinomiais
    # =========================================================
    def _generate_profile(self, n, dist_dict):

        probs = [
            dist_dict["immutable"],
            dist_dict["conservative"],
            dist_dict["influenceable"]
        ]

        counts = np.random.multinomial(n, probs)

        profiles = (
            ["immutable"] * counts[0] +
            ["conservative"] * counts[1] +
            ["influenceable"] * counts[2]
        )

        random.shuffle(profiles)
        return profiles
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # -----------------------------------------
    # Criar agentes
    # -----------------------------------------
    def _create_agents(self):

        nodes_A = sorted([n for n in self.G.nodes if n.startswith("A_")])
        nodes_B = sorted([n for n in self.G.nodes if n.startswith("B_")])
        nodes_C = sorted([n for n in self.G.nodes if n.startswith("C_")])

        profiles_A = self._generate_profile(self.nA, self.profile_dist_A)
        profiles_B = self._generate_profile(self.nB, self.profile_dist_B)
        profiles_C = self._generate_profile(self.nC, self.profile_dist_C)
        
        def create_group(nodes, profiles, agent_type):

            for node, profile in zip(nodes, profiles):

                agent = Agent(
                    model=self,
                    pos=node,
                    ID=node,
                    agent_type=agent_type,
                    agent_profile=profile
                )

                # posiciona no grid
                self.grid.place_agent(agent, node)

        create_group(nodes_A, profiles_A, "A")
        create_group(nodes_B, profiles_B, "B")
        create_group(nodes_C, profiles_C, "C")
        
        
        
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def setup_datacollector(self):
        """
        Configura o DataCollector do modelo para registrar todas as propriedades
        dos agentes + o tempo t da simulação.
        """
    
        self.datacollector = DataCollector(
            model_reporters={
                "Time": lambda m: m.time
            },
            agent_reporters={
                "ID": lambda a: a.ID,
                "community": lambda a: a.agent_type,
                "profile": lambda a: a.agent_profile,
                "paradigm_state": lambda a: a.paradigm_state,
                "adoption_threshold": lambda a: a.adoption_threshold,
                "node": lambda a: a.pos
            }
        )
    
        # Coletar estado inicial
        self.datacollector.collect(self)
        
    def plot_network(self, seed=42, node_size=80, color_by="agent_type"):
        """
        Plota a rede do modelo.
    
        Parâmetros:
        -----------
        seed : int
            Semente para layout spring.
        node_size : int
            Tamanho dos nós.
        color_by : str
            "community"  → usa atributo do grafo
            "agent_type" → usa tipo do agente
            "paradigm"   → usa estado atual do agente
        """
    
        G = self.G
    
        # -----------------------------------
        # Layout (fixo se já existir)
        # -----------------------------------
        if not hasattr(self, "_pos"):
            self._pos = nx.spring_layout(G, seed=seed)
    
        pos = self._pos
    
        # -----------------------------------
        # Mapeamento de cores
        # -----------------------------------
        mapa_cores = {
            "A": "tab:blue",
            "B": "tab:orange",
            "C": "tab:green"
        }
    
        cores_nos = []
    
        for node in G.nodes():
    
            if color_by == "community":
                valor = G.nodes[node].get("community", None)
    
            elif color_by == "agent_type":
                agentes = self.grid.get_cell_list_contents([node])
                valor = agentes[0].agent_type if agentes else None
    
            elif color_by == "paradigm":
                agentes = self.grid.get_cell_list_contents([node])
                valor = agentes[0].paradigm_state if agentes else None
    
            else:
                valor = None
    
            cores_nos.append(mapa_cores.get(valor, "gray"))
    
        # -----------------------------------
        # Plot
        # -----------------------------------
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=cores_nos,
            node_size=node_size
        )
    
        plt.title("Rede com Comunidades A, B e C")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
    
    def export_data_to_excel(self, simulation_name: str = "default"):
        """
        Exporta os dados coletados para uma planilha Excel.
        
        Cria primeiro uma pasta 'simulacao_<simulation_name>' e salva o arquivo dentro.
        
        Args:
            simulation_name (str): nome para identificar a simulação
        """
        # 1️⃣ Criar pasta relativa
        folder_name = f"simulacao_{simulation_name}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        # 2️⃣ Pegar dados do DataCollector
        if not hasattr(self, "datacollector"):
            raise ValueError("O modelo não possui um DataCollector associado.")
        
        agent_data = self.datacollector.get_agent_vars_dataframe()
        
        # 3️⃣ Resetar índice para manipulação
        agent_data = agent_data.reset_index()  # cria colunas 'Step' e 'AgentID'
        
        # 4️⃣ Opcional: ordenar por comunidade e tempo
        if "community" in agent_data.columns:
            agent_data = agent_data.sort_values(by=["community", "Step"])
        
        # 5️⃣ Definir nome do arquivo
        file_path = os.path.join(folder_name, f"agents_log_{simulation_name}.xlsx")
        
        # 6️⃣ Salvar em Excel
        agent_data.to_excel(file_path, index=False)
        
        print(f"[INFO] Planilha exportada para: {file_path}")
#----------------------------------------------------------------------------------------------
    def step(self):
        # Coleta os dados do modelo e dos agentes
        self.datacollector.collect(self)  
        # Atualiza o tempo do modelo
        self.time += 1



   
        
        


