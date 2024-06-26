import numpy as np
from src.SIR import CTMC_SIR_Vax

natural_causes_rates = {"Birth": 57_500 / (5_900_000 * 365), "Death": 1/(81 * 365)} 

# Vaccination rates
vaccination_rates = {"Vaccination": 100, "Vaccination1 I": 5/10, "Vaccination2 I": 1/10}


# Corona rates 1
corona_rates = {"Infection": 0.17,  # Infection without any precautions (from the link above)
                "Infection2": 0.13, # Infection with small precautions
                "Infection3": 0.10, # Infection with drastic precautions
                "Recovery": 0.082,  # From the link above
                "Death General": 0.02,      # General statistic when not distinguishing between severe and non-severe cases
                "Death": 0.001,  # Death when having mild corona 
                "Death Severe": 0.5, # Death when having severe corona without hospitalization
                "Death Treatment": 0.1, # Death when having severe corona with hospitalization
                "Reinfection": 1/60, # Reinfection rates could not be found, but we say that on average you can get reinfected every 2 months 
                "Severe": 130/3300, # Severe cases is calculated from 130_000 hospitalizations, and 3_300_000 cases. 
                "Severe Recovery": 3, # Times longer recovery when having servere corona
                "Severe Recovery Treatment": 2, # Times longer recovery when having servere corona and hospitalized
                }

covid_vax = (corona_rates["Infection"], # Infection rate
                corona_rates["Recovery"], # Recovery rate
                corona_rates["Death"], # Death rate frome disease 
                corona_rates["Death Severe"], # Death rate frome disease when being severly ill
                corona_rates["Death Treatment"], # Death rate frome disease when being severly ill and hospitalized
                corona_rates["Severe"], # Severe rate
                corona_rates["Recovery"]/corona_rates["Severe Recovery"], # Severe recovery rate
                corona_rates["Recovery"]/corona_rates["Severe Recovery Treatment"], # Severe recovery rate when hospitalized
                corona_rates["Reinfection"], # Reinfection rate
                vaccination_rates["Vaccination"], # Vaccination rate
                0, # This is a filler because we have a bonus event 
                vaccination_rates["Vaccination1 I"], # Infection rate for vaccinated round 1
                vaccination_rates["Vaccination2 I"], # Infection rate for vaccinated round 2
                natural_causes_rates["Birth"], # Birth rate 
                natural_causes_rates["Death"], # Death rate from natural causes
    )


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

    plt.plot(T, I, label = "Infected")
    plt.plot(T, S, label = "Susceptible")
    plt.plot(T, R, label = "Recovered")
    plt.plot(T, Dc, label = "Death from disease")
    plt.plot(T, Dn, label = "Death from natural causes")
    plt.plot(T, Si, label = "Severely infected")
    plt.plot(T, V1, label = "Vaccination 1")
    plt.plot(T, V2, label = "Vaccination 2")
    plt.plot(T, Hi, label = "Hospitalized")
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
    N = 50_000
    I0 = 100
    t_max = 365 * 3
    HOSPITAL_CAPACITY = 0.0025*N
    states_vax ={"S": [N - I0],
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
    
    states_vax ={"S": [N - I0],
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
    params2 = list(covid_vax)
    params2[0] = corona_rates["Infection2"]
    params2 = tuple(params2)
    params3 = list(covid_vax)
    params3[0] = corona_rates["Infection3"]
    params3 = tuple(params3)


    import copy
    states_vax1 = copy.deepcopy(states_vax)
    states_vax2 = copy.deepcopy(states_vax)
    states_vax3 = copy.deepcopy(states_vax)
    states_vax4 = copy.deepcopy(states_vax)

    # model_none = CTMC_SIR_Vax(params1, states_vax, t_max, blocked=True, vaccination_rollout=365)
    model_no_vax = CTMC_SIR_Vax(params2, states_vax1, t_max, blocked=True, vaccination_rollout=t_max+1)
    model_mild = CTMC_SIR_Vax(params2, states_vax2, t_max, blocked=True, vaccination_rollout=365, vaccination_threshold=0.5)
    model_mild1 = CTMC_SIR_Vax(params2, states_vax3, t_max, blocked=True, vaccination_rollout=365, vaccination_threshold=0.2)
    model_mild2 = CTMC_SIR_Vax(params2, states_vax4, t_max, blocked=True, vaccination_rollout=365, vaccination_threshold=0.8)
    # model_severe = CTMC_SIR_Vax(params3, states_vax, t_max, blocked=True, vaccination_rollout=365)
    
    death_data = [[], [], [], []]

    n_simulations = 15
    for k in trange(n_simulations):
        for i, model in enumerate([model_no_vax, model_mild, model_mild1, model_mild2]):
            model.simulate()
            disease_deaths = np.array(model.states["Dc"])
            time = np.array(model.states["T"])
            t = np.min(np.where(time >= 365))
            pre_deaths = disease_deaths[t-1]
            model.states["Dc"] = disease_deaths - pre_deaths


            death_data[i].append(model.states["Dc"][-1])
            
            if k == 0:
                fignames = ["no_vax", "05", "02", "08"]
                plot(model, slice_start=t, figname=fignames[i])

            model.reset()


    # t-test
    from scipy.stats import ttest_ind
    t_statistic, p_value = ttest_ind(death_data[0], death_data[1], equal_var=False)
    print(f"p-value for no vax vs 0.5: {p_value}")
    t_statistic, p_value = ttest_ind(death_data[0], death_data[2], equal_var=False)
    print(f"p-value for no vax vs 0.2: {p_value}")
    t_statistic, p_value = ttest_ind(death_data[0], death_data[3], equal_var=False)
    print(f"p-value for no vax vs 0.8: {p_value}")
    t_statistic, p_value = ttest_ind(death_data[1], death_data[2], equal_var=False)
    print(f"p-value for 0.5 vs 0.2: {p_value}")
    t_statistic, p_value = ttest_ind(death_data[1], death_data[3], equal_var=False)
    print(f"p-value for 0.5 vs 0.8: {p_value}")
    t_statistic, p_value = ttest_ind(death_data[2], death_data[3], equal_var=False)
    print(f"p-value for 0.2 vs 0.8: {p_value}")
