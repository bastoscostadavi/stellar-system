import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import EDO

class stellar_system:
   

    def __init__(self, state, time = 0, history = np.array([[],[]])):
        self.state = state
        self.history = history        

    def change(self,t,n=1000):
        self.history = EDO(t,n,self.state)
        self.state = np.transpose(self.history)[-1]

    def trajectory(self):
        N = int(len(self.state)/6)
        for k in range(N):
            x = self.history[6*k]
            y = self.history[6*k+1]
            plt.plot(x,y)    
        plt.show()
        
