import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class GenericSIR(ABC):
    def __init__(self, parameters: tuple, states, N: int, t_max: float):
        self.parameters = parameters
        self.N = N
        self.t_max = t_max
        self.states = states

    def simulate(self):
        while self.states["T"][-1] < self.t_max and self.states["I"][-1] > 0:
            rates = self.rates()
            rate = rates.sum()
            time = np.random.exponential(1/rate)

            self.states["T"].append(self.states["T"][-1] + time)

            event = np.random.choice(range(len(rates)), p=rates/rate)

            self.event_callback(event)


    @abstractmethod
    def rates(self):
        raise NotImplementedError("TODO: Missing 1 line")

    @abstractmethod
    def event_callback(self, event):
        raise NotImplementedError("TODO: Missing 1 line")


class CTMC_SIR3(GenericSIR):
    def rates(self):
        S, I = self.states["S"][-1], self.states["I"][-1]
        beta, gamma = self.parameters

        return np.array([beta*S*I/self.N, gamma*I])
    
    def event_callback(self, event):
        if event == 0:
            self.states["S"].append(self.states["S"][-1]-1)
            self.states["I"].append(self.states["I"][-1]+1)
            self.states["R"].append(self.states["R"][-1])
        else:
            self.states["S"].append(self.states["S"][-1])
            self.states["I"].append(self.states["I"][-1]-1)
            self.states["R"].append(self.states["R"][-1]+1)


class CTMC_SIR_Death(GenericSIR):
    def rates(self):
        S, I = self.states["S"][-1], self.states["I"][-1]
        beta, gamma, mu = self.parameters

        return np.array([beta*S*I/self.N, gamma*I, mu*I])
    
    def event_callback(self, event):
        if event == 0:
            self.states["S"].append(self.states["S"][-1]-1)
            self.states["I"].append(self.states["I"][-1]+1)
            self.states["R"].append(self.states["R"][-1])
            self.states["D"].append(self.states["D"][-1])
        elif event == 1:
            self.states["S"].append(self.states["S"][-1])
            self.states["I"].append(self.states["I"][-1]-1)
            self.states["R"].append(self.states["R"][-1]+1)
            self.states["D"].append(self.states["D"][-1])
        elif event == 2:
            self.states["I"].append(self.states["I"][-1]-1)
            self.states["D"].append(self.states["D"][-1]+1)
            self.states["S"].append(self.states["S"][-1])
            self.states["R"].append(self.states["R"][-1])
        



class CTMC_SIR:
    def __init__(self, beta: float, gamma: float, N: int, I0: int, t_max: float):
        self.beta = beta
        self.gamma = gamma
        self.N = N
        self.t_max = t_max
        self.I0 = I0
        self.S = [N-I0]
        self.I = [I0]
        self.R = [0]
        self.T = [0]
    
    def simulate(self):
        while self.T[-1] < self.t_max and self.I[-1] > 0:
            a = self.beta * self.S[-1] * self.I[-1] / self.N
            b = self.gamma * self.I[-1]

            p1 = a / (a+b)
            u1 = np.random.rand()
            u2 = np.random.rand()

            if u1 < p1:
                self.S.append(self.S[-1]-1)
                self.I.append(self.I[-1]+1)
                self.R.append(self.R[-1])
            else:
                self.I.append(self.I[-1]-1)
                self.R.append(self.R[-1]+1)
                self.S.append(self.S[-1])
            
            self.T.append(self.T[-1]- np.log(u2) / (a + b))
        return self.S, self.I, self.R, self.T

if __name__ == "__main__":
    ebola = (0.2, 0.1)
    covid = (0.17, 0.082)
    flu = (1.37383, 0.98622)
    swine_flu = (7/15, 1/3)
    custom_disease = (0.45, 1/30)
    N = 50_000
    t_max = 1000
    I0 = 100

    extended_covid = (0.17, 0.082*9/10, 0.0082)

    states = {"S": [N - I0],
              "I": [I0],
              "R": [0],
              "T": [0],
              "D": [0]
                       }
    # SIR = CTMC_SIR(*flu, N, I0, t_max)
    SIR = CTMC_SIR_Death(extended_covid, states, N, t_max)

    SIR.simulate()
    S = SIR.states["S"]
    I = SIR.states["I"]
    R = SIR.states["R"]
    T = SIR.states["T"]
    D = SIR.states["D"]

    print(S[-1], I[-1], R[-1])
    plt.plot(T, I)
    plt.plot(T, S)
    plt.plot(T, R)
    plt.plot(T, D)
    plt.legend(["I", "S", "R", "Dead:)"])
    plt.xlabel("Days")
    plt.show()




            



        
    


    