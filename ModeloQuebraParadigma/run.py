from model import ComunityDifusionModel
from model import Agent

model = ComunityDifusionModel(
    
    # =========================
    # Tamanho das comunidades
    # =========================
    nA=150,
    nB=120,
    nC=100,

    # =========================
    # Densidade interna
    # =========================
    dA=0.08,
    dB=0.06,
    dC=0.10,

    # =========================
    # Fração de interface
    # =========================
    rho_AB=0.15,
    rho_AC=0.10,
    rho_BC=0.12,

    # =========================
    # Densidade das conexões intercomunitárias
    # =========================
    delta_AB=0.25,
    delta_AC=0.15,
    delta_BC=0.20,

    # =========================
    # Distribuição de perfis
    # =========================
    profile_dist_A={
        "immutable": 0.20,
        "conservative": 0.50,
        "influenceable": 0.30
    },

    profile_dist_B={
        "immutable": 0.10,
        "conservative": 0.60,
        "influenceable": 0.30
    },

    profile_dist_C={
        "immutable": 0.05,
        "conservative": 0.35,
        "influenceable": 0.60
    },

    # =========================
    # Controle experimental
    # =========================
    LogMetrics=True,
    seed=42
)


model.plot_network(color_by="agent_type")

while model.time <= 5:
    print(model.time)
    model.step()

model.export_data_to_excel("run_1")