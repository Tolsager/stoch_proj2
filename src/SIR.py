import numpy as np
import matplotlib.pyplot as plt
import copy

from abc import ABC, abstractmethod

from SIR_parameters import covid_vax, covid_severe_disease, covid_birth_death, extended_covid, N, I0, HOSPITAL_CAPACITY, t_max


class GenericSIR(ABC):
    def __init__(self, parameters: tuple, states, t_max: float):
        self.parameters = parameters
        self.t_max = t_max
        self.states = states
        self.init_states = copy.deepcopy(states)
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
    def event_callback(self, event: int):
        raise NotImplementedError("TODO: Missing 1 line")
    
    def reset(self):
        self.states = copy.deepcopy(self.init_states)


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
    def __init__(self, parameters: tuple, states, t_max: float, vaccination_rollout: int = None, blocked: bool = False, vaccination_threshold: float = 0.2):
        self.use_blocked = blocked
        self.vaccination_rollout = vaccination_rollout if vaccination_rollout is not None else t_max
        self.second_vaccination_threshold = vaccination_threshold
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
        v2_rate = vaccination if V1 + V2 > self.second_vaccination_threshold*n_living else 0
        v1_rate = vaccination if S > 0 and self.states["T"][-1] > self.vaccination_rollout else 0 
        if v2_rate != 0 and v1_rate != 0:
            v2_rate = 0.5 * vaccination
            v1_rate = 0.5 * vaccination
        
        # Exponential rates for 
        return np.array([infection*S*(I+Si)/n_living,                # infection
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
                         v_infection1*infection*V1*(I+Si)/n_living,  # vaxed infected
                         v_infection2*infection*V2*(I+Si)/n_living,  # vaxed infected
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
            # If hospital is full, then just add to severely infected
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




if __name__ == "__main__":
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
    SIR = CTMC_SIR_Vax(covid_vax, states_vax, t_max, blocked=True, vaccination_rollout=365, vaccination_threshold=0.5)

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
    plt.tight_layout()
    plt.savefig("no_vax_all_full.pdf")
    plt.show()

    plt.plot(T, np.array(Si) + np.array(Hi), label="Severely infected") 
    plt.plot(T, Hi, label="Hospitalized")
    plt.hlines(HOSPITAL_CAPACITY, 0, T[-1], colors="r", linestyles="dashed", label="Hospital capacity")
    plt.legend()
    plt.grid()
    plt.xlabel("Days")
    plt.tight_layout()
    plt.savefig("no_vax_hospital_full.pdf")
    plt.show()



            



        
    


    