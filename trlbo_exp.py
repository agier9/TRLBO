import sys,os
path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(path)
from trlbo.trlbo import TRLBO
from benchmarks import Levy,Griewank,Ackley
import numpy as np
import torch
import json,os,datetime

dim = [50]
iters = [10000]
funcs = [
    {'func':Levy,'name':'levy','iteration':{'dim':dim,'iter':iters}},
    {'func':Griewank,'name':'griewank','iteration':{'dim':dim,'iter':iters}},
    {'func':Ackley,'name':'ackley','iteration':{'dim':dim,'iter':iters}},
    ]

def worker(repeat,device):
    for func in funcs:
        iteration = func['iteration']['iter']
        if 'dim' in func['iteration']:
            f = [func['func'](dim) for dim in func['iteration']['dim']]
            paths = ['data//trlbo//'+func['name']+'//'+str(dim) for dim in func['iteration']['dim']]
        else:
            f = [func['func']()]
            paths = ['data//trlbo//'+func['name']+'//']
        for path in paths:
            if not os.path.exists(path):
                    os.makedirs(path)
        #counting result files
        for f_,path,max_iter in zip(f,paths,iteration):
            repeat_ = repeat - len(os.listdir(path))
            for _ in range(repeat_):
                turbo1 = TRLBO(
                    f=f_,  # Handle to objective function
                    lb=f_.lb,  # Numpy array specifying lower bounds
                    ub=f_.ub,  # Numpy array specifying upper bounds
                    n_init=20,  # Number of initial bounds from an Latin hypercube design
                    max_evals = max_iter,  # Maximum number of evaluations
                    batch_size= 10,  # How large batch size TuRBO uses
                    verbose=False,  # Print information from each batch
                    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                    min_cuda=1024,  # Run on the CPU for small datasets
                    device=device,  # "cpu" or "cuda"
                    dtype="float64",  # float64 or float32
                )

                turbo1.optimize()

                fX = turbo1.fX  # Observed values
                data = np.minimum.accumulate(fX).tolist()
                id=datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
                with open(path+'//'+id,'w') as f:
                    json.dump(data,f)
        print(func['name'],' done')
        sys.stdout.flush()
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    repeat = 30
    worker(repeat,device)