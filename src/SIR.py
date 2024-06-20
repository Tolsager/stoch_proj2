import numpy as np
import matplotlib.pyplot as plt
import copy

from abc import ABC, abstractmethod
from enum import Enum

# TODO Event abstraction (if we want to)
class Event(Enum):
    INFECTION = 0
    RECOVERY = 1
    DEATH = 2
    BIRTH = 3


class GenericSIR(ABC):
    def __init__(self, parameters: tuple, states, N: int, t_max: float):
        self.parameters = parameters
        self.N = N
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
    def event_callback(self, event: Event):
        raise NotImplementedError("TODO: Missing 1 line")
    
    def reset(self):
        self.states = copy.deepcopy(self.init_states)


class CTMC_SIR(GenericSIR):
    def rates(self):
        S, I = self.states["S"][-1], self.states["I"][-1]
        beta, gamma = self.parameters

        return np.array([beta*S*I/self.N, gamma*I])
    
    def event_callback(self, event):
        if event == 0:
            self.update_states(increase="I", decrease="S")
        else:
            self.update_states(increase="R", decrease="I")


class CTMC_SIR_Death(GenericSIR):
    def rates(self):
        S, I = self.states["S"][-1], self.states["I"][-1]
        beta, gamma, mu = self.parameters

        return np.array([beta*S*I/self.N, gamma*I, mu*I])
    
    def event_callback(self, event):
        if event == 0:
            self.update_states(increase="I", decrease="S")
        elif event == 1:
            self.update_states(increase="R", decrease="I")
        elif event == 2:
            self.update_states(increase="D", decrease="I")
        

class CTMC_SIR_BirthDeath(GenericSIR):
    def rates(self):
        S, I, R = self.states["S"][-1], self.states["I"][-1], self.states["R"][-1]
        beta, gamma, mu, nu = self.parameters

        return np.array([beta*S*I/self.N, gamma*I, mu*I, nu*(S+I+R)])
    
    def event_callback(self, event):
        if event == 0: # S -> I
            self.update_states(increase="I", decrease="S")
        elif event == 1: # I -> R
            self.update_states(increase="R", decrease="I")
        elif event == 2: # I -> D
            self.update_states(increase="D", decrease="I")
        elif event == 3: # S -> S + 1
            self.update_states(increase="S", decrease=None)


class CTMC_SIR_BirthDeathReinfection(GenericSIR):
    def rates(self):
        S, I, R = self.states["S"][-1], self.states["I"][-1], self.states["R"][-1]
        beta, gamma, mu, nu, alpha = self.parameters

        return np.array([beta*S*I/self.N, gamma*I, mu*I, nu*(S+I+R), alpha*R])
    
    def event_callback(self, event):
        if event == 0: # S -> I
            self.update_states(increase="I", decrease="S")
        elif event == 1: # I -> R
            self.update_states(increase="R", decrease="I")
        elif event == 2: # I -> D
            self.update_states(increase="D", decrease="I")
        elif event == 3: # S -> S + 1
            self.update_states(increase="S", decrease=None)
        elif event == 4: # R -> S
            self.update_states(increase="S", decrease="R")


if __name__ == "__main__":
    ebola = (0.2, 0.1)
    covid = (0.17, 0.082)
    flu = (1.37383, 0.98622)
    swine_flu = (7/15, 1/3)
    custom_disease = (0.45, 1/30)
    N = 50_000
    t_max = 1000
    I0 = 100

    extended_covid = (0.17, 0.082*7/10, 0.0082, 0.0003, 0.05)

    states = {"S": [N - I0],
              "I": [I0],
              "R": [0],
              "T": [0],
              "D": [0]
                       }
    # SIR = CTMC_SIR(*flu, N, I0, t_max)
    # SIR = CTMC_SIR_Death(extended_covid, states, N, t_max)
    # SIR = CTMC_SIR_BirthDeath(extended_covid, states, N, t_max)
    SIR = CTMC_SIR_BirthDeathReinfection(extended_covid, states, N, t_max)

    SIR.simulate()
    S = SIR.states["S"]
    I = SIR.states["I"]
    R = SIR.states["R"]
    T = SIR.states["T"]
    D = SIR.states["D"]

    print(SIR.event_counts)
    print(S[-1] + I[-1] + R[-1] + D[-1])
    plt.plot(T, I)
    plt.plot(T, S)
    plt.plot(T, R)
    plt.plot(T, D)
    plt.legend(["I", "S", "R", "Dead:)"])
    plt.xlabel("Days")
    plt.show()




            



        
    


    