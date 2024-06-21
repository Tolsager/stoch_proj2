import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from enum import Enum

# TODO Event abstraction (if we want to)
class Event(Enum):
    INFECTION = 0
    RECOVERY = 1
    DEATH = 2
    BIRTH = 3


class GenericSIR(ABC):
    def __init__(self, parameters: tuple, states, t_max: float):
        self.parameters = parameters
        self.t_max = t_max
        self.states = states
        self.event_counts = np.zeros(len(self.parameters))

    def simulate(self):
        while self.states["T"][-1] < self.t_max and self.states["I"][-1] > 0:
            rates = self.rates()
            rate = rates.sum()
            time = np.random.exponential(1/rate)

            self.states["T"].append(self.states["T"][-1] + time)

            event = np.random.choice(range(len(rates)), p=rates/rate)

            self.event_counts[event] += 1

            self.event_callback(event)

    def update_states(self, increase: str, decrease: str | None):
        self.states[increase].append(self.states[increase][-1]+1)
        if decrease is not None:
            self.states[decrease].append(self.states[decrease][-1]-1)
        for key in self.states.keys():
            if key not in [increase, decrease, "T"]:
                self.states[key].append(self.states[key][-1])


    @abstractmethod
    def rates(self):
        raise NotImplementedError("TODO: Missing 1 line")

    @abstractmethod
    def event_callback(self, event: Event):
        raise NotImplementedError("TODO: Missing 1 line")


# Simple SIR model 
class CTMC_SIR(GenericSIR):
    def rates(self):
        S, I = self.states["S"][-1], self.states["I"][-1]
        beta, gamma = self.parameters # Infection rate, recovery rate

        # Exponential rates for : infection, recovery
        return np.array([beta*S*I/self.N, gamma*I])
    
    def event_callback(self, event):
        if event == 0:
            self.update_states(increase="I", decrease="S")
        else:
            self.update_states(increase="R", decrease="I")


# SIR with death by disease - total_recovered + total_dead = total_infected 
# and death*total_infected = total_dead  
class CTMC_SIR_Death(GenericSIR):
    def rates(self):
        S, I = self.states["S"][-1], self.states["I"][-1]
        beta, gamma, death = self.parameters
        # Calculate death_rate such that total_dead = death*total_infected
        death_rate = death*gamma/(1-death)
        return np.array([beta*S*I/self.N, gamma*I, death_rate*I])
    
    def event_callback(self, event):
        if event == 0:
            self.update_states(increase="I", decrease="S")
        elif event == 1:
            self.update_states(increase="R", decrease="I")
        elif event == 2:
            self.update_states(increase="D", decrease="I")
        
# SIR model with birth and death by disease and natural death
class CTMC_SIR_BirthDeath(GenericSIR):
    def rates(self):
        S, I, R = self.states["S"][-1], self.states["I"][-1], self.states["R"][-1]
        beta, gamma, death, birth_rate, natural_death = self.parameters
        n_living = S + I + R
        death_rate = death*gamma/(1-death) # Calculate death_rate such that total_dead = death*total_infected
        return np.array([beta*S*I/self.N, gamma*I, death_rate*I, birth_rate*n_living, natural_death*n_living])
    
    def event_callback(self, event):
        if event == 0: # S -> I
            self.update_states(increase="I", decrease="S")
        elif event == 1: # I -> R
            self.update_states(increase="R", decrease="I")
        elif event == 2: # I -> Dc
            self.update_states(increase="Dc", decrease="I")
        elif event == 3: # S -> S + 1
            self.update_states(increase="S", decrease=None)
        elif event == 4: # Someone alive -> Dn
            living = np.array([self.states["S"][-1], self.states["I"][-1],  self.states["R"][-1]])
            group = np.random.choice(["S", "I", "R"], p=living/np.sum(living))
            self.update_states(increase="Dn", decrease=group)

# SIR with birth, death by disease, reinfection and natural death 
class CTMC_SIR_BirthDeathReinfection(GenericSIR):
    def rates(self):
        S, I, R = self.states["S"][-1], self.states["I"][-1], self.states["R"][-1]
        infection, recovery, disease_death, reinfection, natural_birth, natural_death = self.parameters
        n_living = S + I + R
        disease_death = disease_death*recovery/(1-disease_death) # Calculate disease death such that total_dead = death*total_infected
        # Exponential rates for : infection,     recovery,   disease_death,   reinfection,   natural_birth,          natural_death
        return np.array([infection*S*I/n_living, recovery*I, disease_death*I, reinfection*R, natural_birth*n_living, natural_death*n_living])
    
    def event_callback(self, event):
        if event == 0: # S -> I
            self.update_states(increase="I", decrease="S")
        elif event == 1: # I -> R
            self.update_states(increase="R", decrease="I")
        elif event == 2: # I -> Dc
            self.update_states(increase="Dc", decrease="I")
        elif event == 3: # R -> S
            self.update_states(increase="S", decrease="R")
        elif event == 4: # S -> S + 1
            self.update_states(increase="S", decrease=None)
        elif event == 5: # Someone alive -> Dn
            living = np.array([self.states["S"][-1], self.states["I"][-1],  self.states["R"][-1]])
            group = np.random.choice(["S", "I", "R"], p=living/np.sum(living))
            self.update_states(increase="Dn", decrease=group)

# SIR model with birth, death by disease, reinfection, natural death and severe cases of disease 
class CTMC_SIR_Severe(GenericSIR):
    def rates(self):
        S, I, Si, R = self.states["S"][-1], self.states["I"][-1], self.states["Si"][-1], self.states["R"][-1]
        infection, recovery, disease_death, s_infection, s_recovery, s_disease_death, reinfection, natural_birth, natural_death = self.parameters
        n_living = S + I + Si + R
        # In this model severly sick people are not admitted to hospital and can still transmit the disease
        disease_death = disease_death*recovery/(1-disease_death) # Calculate disease death such that total_dead = death*total_infected
        severly_inf = s_infection*recovery/(1-s_infection)
        severly_death = s_disease_death*s_recovery/(1-s_disease_death)
        # Exponential rates for : 
        return np.array([infection*S*(I+Si)/n_living,   # infection
                         recovery*I,                    # recovery
                         disease_death*I,               # disease death
                         reinfection*R,                 # reinfection
                         I*severly_inf,                 # severely infected
                         Si*s_recovery,                 # severely recov
                         Si*severly_death,              # severely death
                         natural_birth*n_living,        # natural birth
                         natural_death*n_living])      # natural death
    
    def event_callback(self, event):
        if event == 0: # S -> I
            self.update_states(increase="I", decrease="S")
        elif event == 1: # I -> R
            self.update_states(increase="R", decrease="I")
        elif event == 2: # I -> Dc
            self.update_states(increase="Dc", decrease="I")
        elif event == 3: # R -> S
            self.update_states(increase="S", decrease="R")
        elif event == 4: # I -> Si
            self.update_states(increase="Si", decrease="I")
        elif event == 5: # Si -> R
            self.update_states(increase="R", decrease="Si")
        elif event == 6: # Si -> Dc
            self.update_states(increase="Dc", decrease="Si")
        elif event == 7: # S -> S + 1
            self.update_states(increase="S", decrease=None)
        elif event == 8: # Someone alive -> Dn
            living = np.array([self.states["S"][-1], self.states["I"][-1],  self.states["Si"][-1], self.states["R"][-1]])
            group = np.random.choice(["S", "I", "Si", "R"], p=living/np.sum(living))
            self.update_states(increase="Dn", decrease=group)

class CTMC_SIR_Vax(GenericSIR):
    def __init__(self, parameters: tuple, states, t_max: float, vaccination_rollout: int = None, blocked: bool = False):
        self.use_blocked = blocked
        self.vaccination_rollout = vaccination_rollout if vaccination_rollout is not None else t_max
        super().__init__(parameters, states, t_max)

    def rates(self):
        S, I, Si, Hi, V1, V2, R = self.states["S"][-1], self.states["I"][-1], self.states["Si"][-1], self.states["Hi"][-1], self.states["V1"][-1], self.states["V2"][-1], self.states["R"][-1]
        infection, recovery, disease_death, s_disease_death, h_disease_death, s_infection, s_recovery, h_recovery, reinfection, vaccination, _,  v_infection1, v_infection2, natural_birth, natural_death = self.parameters
        n_living = S + I + Si + Hi + V1 + V2 + R
        
        # Hospitialized will not transmit the disease (hopefully), hence only I and Si transmit
        disease_death = disease_death*recovery/(1-disease_death) # Calculate disease death such that total_dead = death*total_infected
        severly_inf = s_infection*recovery/(1-s_infection)
        severly_death = s_disease_death*s_recovery/(1-s_disease_death)
        hospitalized_death = h_disease_death*h_recovery/(1-h_disease_death)
        
        # Vaccinations
        v2_rate = vaccination if V1 + V2 > 0.5*n_living else 0
        v1_rate = vaccination if S > 0 and self.states["T"][-1] > self.vaccination_rollout  else 0 
        if v2_rate != 0 and v1_rate != 0:
            v2_rate = 0.5 * vaccination
            v1_rate = 0.5 * vaccination
        
        # Exponential rates for 
        return np.array([infection*S*I/n_living,                # infection
                         recovery*I,                            # recovery  
                         disease_death*I,                       # disease_death
                         reinfection*R,                         # reinfection
                         I*severly_inf,                         # severely_inf
                         Si*s_recovery,                         # severely_recov
                         Si*severly_death,                      # severely_death
                         Hi*h_recovery,                         # hospitalized_recov
                         Hi*hospitalized_death,                 # hospitalized_death
                         v1_rate,                               # vaccination round 1
                         v2_rate,                               # vaccination round 2                         
                         v_infection1*infection*V1*I/n_living,  # vaxed infected
                         v_infection2*infection*V2*I/n_living,  # vaxed infected
                         natural_birth*n_living,                # natural_birth
                         natural_death*n_living,])              # natural_death

    
    def event_callback(self, event):
        if event == 0: # S -> I
            self.update_states(increase="I", decrease="S")
        elif event == 1: # I -> R
            self.update_states(increase="R", decrease="I")
        elif event == 2: # I -> Dc
            self.update_states(increase="Dc", decrease="I")
        elif event == 3: # R -> S
            self.update_states(increase="S", decrease="R")
        elif event == 4: # I -> Hi if space otherwise I -> Si
            # Check if there is room in the hospital, otherwise the patient dies
            if self.use_blocked and self.states["Hi"][-1] > HOSPITAL_CAPACITY:
                 self.update_states(increase="Si", decrease="I")
            else: 
                self.update_states(increase="Hi", decrease="I")
        elif event == 5: # Si -> R
            self.update_states(increase="R", decrease="Si")
        elif event == 6: # Si -> Dc
            self.update_states(increase="Dc", decrease="Si")
        elif event == 7: # Hi -> R 
            self.update_states(increase="R", decrease="Hi")
        elif event == 8: # Hi -> Dc
            self.update_states(increase="Dc", decrease="Hi")
        elif event == 9: # S -> V1
            self.update_states(increase="V1", decrease="S")
        elif event == 10: # V1 -> V2
            self.update_states(increase="V2", decrease="V1")
        elif event == 11: # V1 -> I
            self.update_states(increase="I", decrease="V1")
        elif event == 12: # V2 -> I
            self.update_states(increase="I", decrease="V2")
        elif event == 13: # S -> S + 1
            self.update_states(increase="S", decrease=None)
        elif event == 14: # Someone alive -> Dn
            living = np.array([self.states["S"][-1], self.states["I"][-1],  self.states["Si"][-1], self.states["V1"][-1], self.states["V2"][-1], self.states["R"][-1]])
            group = np.random.choice(["S", "I", "Si", "V1", "V2", "R"], p=living/np.sum(living))
            self.update_states(increase="Dn", decrease=group)


# Rates for specific diseases (per day)  

# Natural causes rates
# In 2023 57500 where born in denmark (with population 5.9 million) 
# And the expected life span is 81 years 
natural_causes_rates = {"Birth": 57500 / (5900000 * 365), "Death": 1/(81 * 365)} 

# Vaccination rates
vaccination_rates = {"Vaccination": 100, "Vaccination1 I": 5/10, "Vaccination2 I": 1/10}

# Corona data
# Corona rates 1 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8642156/
corona_rates = {"Infection": 0.17,  # Infection without any precautions (from the link above)
                "Infection2": 0.13, # Infection with small precautions
                "Infection3": 0.10, # Infection with drastic precautions
                "Recovery": 0.082,  # From the link above
                "Death General": 0.02,      # General statistic when not distinguishing between severe and non-severe cases
                "Death": 0.001,  # Death when having mild corona 
                "Death Severe": 0.5, # Death when having severe corona without hospitalization
                "Death Treatment": 0.1, # Death when having severe corona with hospitalization
                "Reinfection": 1/60, # Reinfection rates could not be found, but we say that on average you can get reinfected every 2 months 
                "Severe": 130/3300, # Server cases is calculated from 130_000 hospitalizations, and 3_300_000 cases. 
                "Severe Recovery": 3, # Times longer recovery when having servere corona
                "Severe Recovery Treatment": 2, # Times longer recovery when having servere corona and hospitalized
                }

# Corona rates 2 - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265815
# corona_rates = {"Infection": 0.5,  # Infection without any precautions (from the link above)
#                 "Infection2": 0.25, # Infection with small precautions
#                 "Infection3": 0.10, # Infection with drastic precautions
#                 "Recovery": 0.13,  # From the link above
#                 "Death General": 0.02,      # General statistic when not distinguishing between severe and non-severe cases
#                 "Death": 0.002,  # Death when having mild corona 
#                 "Death Severe": 0.5, # Death when having severe corona without hospitalization
#                 "Death Treatment": 0.1, # Death when having severe corona with hospitalization
#                 "Reinfection": 1/60, 
#                 "Severe": 130/3300, 
#                 "Severe Recovery": 3, # Times longer recovery when having servere corona
#                 "Severe Recovery Treatment": 2, # Times longer recovery when having servere corona and hospitalized
#                 }

if __name__ == "__main__":
    N = 50_000
    t_max = 1000
    I0 = 100
    HOSPITAL_CAPACITY = 0.0025*N
    
    # For birth and death
    covid_birth_death =  (corona_rates["Infection"], # Infection rate
                      corona_rates["Recovery"], # Recovery rate
                      corona_rates["Death General"], # Death rate frome disease 
                      natural_causes_rates["Birth"], # Birth rate 
                      natural_causes_rates["Death"], # Death rate from natural causes
    )

    # For birth and death with reinfection
    extended_covid = (corona_rates["Infection"], # Infection rate
                      corona_rates["Recovery"], # Recovery rate
                      corona_rates["Death General"], # Death rate frome disease 
                      corona_rates["Reinfection"], # Reinfection rate
                      natural_causes_rates["Birth"], # Birth rate 
                      natural_causes_rates["Death"], # Death rate from natural causes
    )
    
    # For birth and death with reinfection and severe cases of disease
    covid_severe_disease = (corona_rates["Infection"], # Infection rate
                      corona_rates["Recovery"], # Recovery rate
                      corona_rates["Death"], # Death rate frome disease 
                      corona_rates["Severe"], # Severe rate
                      corona_rates["Recovery"]/corona_rates["Severe Recovery"], # Severe recovery rate
                      corona_rates["Death Severe"], # Severe death rate
                      corona_rates["Reinfection"], # Reinfection rate
                      natural_causes_rates["Birth"], # Birth rate 
                      natural_causes_rates["Death"], # Death rate from natural causes
    )
    
    # For birth and death with reinfection, severe cases of disease and vaccination
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

    states_birth_death = {  "S": [N - I0],
                            "I": [I0],
                            "R": [0],
                            "T": [0],
                            "Dc": [0],
                            "Dn" : [0]
                                    }
    
    states_severe ={"S": [N - I0],
                    "I": [I0],
                    "Si": [0],
                    "R": [0],
                    "T": [0],
                    "Dc": [0],
                    "Dn": [0]}
    
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
    
    # Corona with birth, death 
    # SIR = CTMC_SIR_BirthDeath(covid_birth_death, states_birth_death, t_max)

    # Corona with birth, death and reinfection
    # SIR = CTMC_SIR_BirthDeathReinfection(extended_covid, states_birth_death, t_max)

    # Corona with birth, death, reinfection and severe cases
    # SIR = CTMC_SIR_Severe(covid_severe_disease, states_severe, t_max)
    
    # Corona with birth, death, reinfection, severe cases and vaccination 
    SIR = CTMC_SIR_Vax(covid_vax, states_vax, t_max, blocked=True)

    SIR.simulate()
    S = SIR.states["S"]
    Si = SIR.states["Si"]
    Hi = SIR.states["Hi"]
    I = SIR.states["I"]
    R = SIR.states["R"]
    V1 = SIR.states["V1"]
    V2 = SIR.states["V2"]
    T = SIR.states["T"]
    Dc = SIR.states["Dc"]
    Dn = SIR.states["Dn"]

    print(SIR.event_counts)
    print(S[-1] + I[-1] + R[-1] + Dn[-1] + Dc[-1] + Si[-1] + V2[-1] + V1[-1])
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
    plt.show()

    plt.plot(T, np.array(Si) + np.array(Hi), label="Severely infected") 
    plt.plot(T, Hi, label="Hospitalized")
    plt.hlines(HOSPITAL_CAPACITY, 0, t_max, colors="r", linestyles="dashed", label="Hospital capacity")
    plt.legend()
    plt.grid()
    plt.xlabel("Days")
    plt.show()



            



        
    


    