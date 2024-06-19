import numpy as np
import matplotlib.pyplot as plt

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
    beta = 0.1
    gamma = 0.05
    N = 10_000
    t_max = 2000
    I0 = 100

    SIR = CTMC_SIR(beta, gamma, N, I0, t_max)
    S, I, R, T = SIR.simulate()
    plt.plot(T, I)
    plt.plot(T, S)
    plt.plot(T, R)
    plt.show()




            



        
    


    
