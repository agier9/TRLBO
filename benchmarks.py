import numpy as np

class Levy:#multimodal
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -10.0 * np.ones(dim)
        self.ub = 10.0 * np.ones(dim)
        self.optimizers = 1.0*np.ones(dim)
        self.optimal_value = 0.0#a global minimal value
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 1 + (x - 1.0) / 4.0
        val = np.sin(np.pi * w[0]) ** 2 + np.sum((w[1:self.dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dim - 1] + 1) ** 2)) +             (w[self.dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[self.dim - 1])**2)
        return val

class Griewank:#unimodal
    def __init__(self,dim=10):
        self.dim = dim
        self.lb = -600.0 * np.ones(dim)
        self.ub = 600.0 * np.ones(dim)
        self.optimizers = np.zeros(dim)
        self.optimal_value = 0.0#minimum
    
    def __call__(self,x):
        p1 = np.sum(x**2/4000.0)
        factor = np.array([i+1 for i in range(0,self.dim)])
        p2 = -np.prod(np.cos(x/np.sqrt(factor)))
        return p1 + p2 + 1.0

class Ackley:#multimodal
    def __init__(self,dim=10):
        self.dim = dim
        self.lb = -32.768*np.ones(dim)
        self.ub = 32.768*np.ones(dim)
        self.optimizers = np.zeros(dim)
        self.optimal_value = 0.0#minimum
    
    def __call__(self,x):
        a,b,c = 20.0,0.2,2*np.pi
        p1 = -a *np.exp(-b / np.sqrt(self.dim) * np.linalg.norm(x))
        p2 = -(np.exp(np.mean(np.cos(c*x))))
        return p1 + p2 + a + np.e

class Rosenbrock:#unimodal
    def __init__(self,dim=10):
        self.dim = dim
        self.lb = -32.768*np.ones(dim)
        self.ub = 32.768*np.ones(dim)
        self.optimizers = np.ones(dim)
        self.optimal_value = 0.0#minimum
    
    def __call__(self,x):
        p1 = x[1:]
        p2 = x[:-1]**2
        p3 = (x[:-1]-1)**2
        return np.sum(100.0*(p1-p2)**2+p3)