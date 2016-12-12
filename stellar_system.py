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
        xs = self.history[0]
        ys = self.history[1]
        xe = self.history[6]
        ye = self.history[7]
        xl = self.history[12]
        yl = self.history[13]
        plt.plot(xs,ys,color = 'black')    
        plt.plot(xe,ye, color = 'red')
        plt.plot(xl,yl, color = 'blue')
        plt.show()
        
