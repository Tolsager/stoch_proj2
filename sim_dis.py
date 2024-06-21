from src.SIR import CTMC_SIR, GenericSIR
# import tqdm
import sys

def disease_disappearing(model: GenericSIR, n_sims: int):
    n_disappeared = 0
    # for i in tqdm.trange(n_sims):
    for i in range(n_sims):
        model.simulate()
        if model.states["I"][-1] == 0 and model.states["S"][-1] != 0:
            n_disappeared += 1
        model.reset()
    return n_disappeared / n_sims

if __name__ == "__main__":
    ### likelihood of disease disappearing
    N = 50_000
    I0 = 50
    n_sims = 100
    t_max = 365*2

    states = {
        "S": [N-I0],
        "I": [I0],
        "R": [0],
        "T": [0]
    }
    disease_parameters = {
        "ebola": (0.2, 0.1),
        "covid": (0.17, 0.082),
        "flu": (1.37383, 0.98622),
        "swine_flu": (7/15, 1/3),
        "hypo": (1/10, 1/60)
    }

    idx_to_disease = {
        1: "ebola",
        2: "covid",
        3: "flu",
        4: "swine_flu",
        5: "hypo"
    }

    dis_idx = int(sys.argv[1])

    disease = idx_to_disease[dis_idx]

    parameters = disease_parameters[disease]
    model = CTMC_SIR(parameters, states, N, t_max,)
    p_disappeared = disease_disappearing(model, n_sims)
    print(f"{disease} disappearing prob: {p_disappeared}")

    # for disease, parameters in disease_parameters.items():
    #     model = CTMC_SIR(parameters, states, N, t_max,)
    #     p_disappeared = disease_disappearing(model, n_sims)
    #     print(f"{disease} disappearing prob: {p_disappeared}")








