import numpy as np
import copy
from src.SIR import CTMC_SIR_Vax
from src.SIR_parameters import covid_vax, corona_rates, N, I0, t_max, HOSPITAL_CAPACITY

from scipy.stats import ttest_ind


def plot(model, slice_start, figname=None):
    import matplotlib.pyplot as plt
    S = model.states["S"][slice_start:]
    Si = model.states["Si"][slice_start:]
    Hi = model.states["Hi"][slice_start:]
    I = model.states["I"][slice_start:]
    R = model.states["R"][slice_start:]
    V1 = model.states["V1"][slice_start:]
    V2 = model.states["V2"][slice_start:]
    T = model.states["T"][slice_start:]
    Dc = model.states["Dc"][slice_start:]
    Dn = model.states["Dn"][slice_start:]


    plt.plot(T, I, label="Infected")
    plt.plot(T, S, label="Susceptible")
    plt.plot(T, R, label="Recovered")
    plt.plot(T, Dc, label="Death from disease")
    plt.plot(T, Dn, label="Death from natural causes")
    plt.plot(T, Si, label="Severely infected")
    plt.plot(T, V1, label="Vaccination 1")
    plt.plot(T, V2, label="Vaccination 2")
    plt.plot(T, Hi, label="Hospitalized")
    plt.legend()
    plt.grid()
    plt.xlabel("Days")
    plt.tight_layout()
    if figname is not None:
        plt.savefig(f"{figname}_all.pdf")
        plt.clf()
    else:
        plt.show()

    plt.plot(T, np.array(Si) + np.array(Hi), label="Severely infected")
    plt.plot(T, Hi, label="Hospitalized")
    plt.hlines(HOSPITAL_CAPACITY, T[0], T[-1], colors="r", linestyles="dashed", label="Hospital capacity")
    plt.legend()
    plt.grid()
    plt.xlabel("Days")
    plt.tight_layout()
    if figname is not None:
        plt.savefig(f"{figname}_hospital.pdf")
        plt.clf()
    else:
        plt.show()


if __name__ == '__main__':
    from tqdm import trange

    states_vax = {"S": [N - I0],
                  "I": [I0],
                  "Si": [0],
                  "Hi": [0],
                  "R": [0],
                  "V1": [0],
                  "V2": [0],
                  "T": [0],
                  "Dn": [0],
                  "Dc": [0],
                  }

    params1 = covid_vax
    params2 = copy.deepcopy(covid_vax)
    params2[0] = corona_rates["Infection2"]
    params3 = copy.deepcopy(covid_vax)
    params3[0] = corona_rates["Infection3"]

    N_MODELS = 3

    model_none = CTMC_SIR_Vax(params1, copy.deepcopy(states_vax), t_max, blocked=True, vaccination_rollout=365, vaccination_threshold=0.5)
    # model_no_vax = CTMC_SIR_Vax(params2, states_vax1, t_max, blocked=True, vaccination_rollout=t_max + 1)
    model_mild = CTMC_SIR_Vax(params2, copy.deepcopy(states_vax), t_max, blocked=True, vaccination_rollout=365,
                              vaccination_threshold=0.5)
    # model_mild1 = CTMC_SIR_Vax(params2, copy.deepcopy(states_vax), t_max, blocked=True, vaccination_rollout=365,
    #                            vaccination_threshold=0.2)
    # model_mild2 = CTMC_SIR_Vax(params2, copy.deepcopy(states_vax), t_max, blocked=True, vaccination_rollout=365,
    #                            vaccination_threshold=0.8)
    model_severe = CTMC_SIR_Vax(params3, copy.deepcopy(states_vax), t_max, blocked=True, vaccination_rollout=365, vaccination_threshold=0.5)

    models = [model_none, model_mild, model_severe]

    model_none.name = "no social distancing"
    model_mild.name = "mild social distancing"
    model_severe.name = "severe social distancing"
    # model_mild2.name = "0.8"

    death_data = [[] for _ in range(N_MODELS)]

    n_simulations = 15
    for k in trange(n_simulations):
        for i, model in enumerate(models):
            model.simulate()
            # disease_deaths = np.array(model.states["Dc"])
            # time = np.array(model.states["T"])
            # t = np.min(np.where(time >= 365))
            # pre_deaths = disease_deaths[t - 1]
            # model.states["Dc"] = disease_deaths - pre_deaths

            death_data[i].append(model.states["Dc"][-1])

            if k == 0:
                plot(model, slice_start=0, figname=model.name)

            model.reset()

    # t-test
    for i in range(N_MODELS):
        for j in range(i + 1, N_MODELS):
            statistic, p_value = ttest_ind(death_data[i], death_data[j], equal_var=False)
            print(f"p-value for {models[i].name} vs {models[j].name}: {p_value}")
