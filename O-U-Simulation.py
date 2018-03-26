# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import stochastic
import pyprocess


class OU:  
    def __init__(self, mu, sigma=0.3, theta=1.2, T=30, N=30, x0=20): 
        self.T = T
        self.N = N
        self.theta = theta
        self.x0 = x0
        self.mu = self.x0*mu  
        self.sigma = sigma  
        self.dt = T/N  
        self.x_prev = self.x0*np.ones(mu.shape)
  
  
    def OU_simulation(self):
        result = []
        result.append(self.x_prev)
        for i in range(1, self.T):
            x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) *np.random.normal(size=self.mu.shape)  
            self.x_prev = x  
            result.append(x)
        plt.figure('OU data')  
        x = np.linspace(1,self.T,self.T)
        plt.plot(x, result)
        plt.show()
        
    def stochastic_test(self):
        
        return
    
    def pyprocess_test(self):
        return
    
  
  
if __name__=="__main__":  
    test = OU(mu=np.ones(3),x0=20) 
    test.OU_simulation()