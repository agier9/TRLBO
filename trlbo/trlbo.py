import math,random
import sys,multiprocessing
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from scipy.optimize import minimize,Bounds
from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, to_unit_cube

class UpperConfidenceBound():
    def __init__(self,model,beta=0.1,maximum = True):
        self.model = model
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.beta = beta
        self.maximum = maximum
    
    def __call__(self,X):
        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        f_predict = self.model(X)
        mean = f_predict.mean
        variance = f_predict.variance
        mean_min = torch.min(mean)
        mean_max = torch.max(mean)
        mean = (mean-mean_min)/(mean_max-mean_min) if not mean_min==mean_max else mean/mean.shape[0]
        
        var_min = torch.min(variance)
        var_max = torch.max(variance)
        variance = (variance-var_min)/(var_max-var_min) if not var_min==var_max else variance/variance.shape[0]

        self.beta = self.beta.to(mean)
        delta = (self.beta.expand_as(mean) * variance).sqrt()
        return mean + delta if self.maximum else -mean + delta

class TRLBO:
    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        num_restarts=5,
        turn_acqf_lenth = 0.08,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
    ):

        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > n_init and max_evals > batch_size
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"

        # Save function information
        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        # Settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.num_restarts = num_restarts
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.ones(self.dim) if self.use_ard else np.zeros((0, 1))

        # Tolerances and counters
        self.n_cand = min(100 * self.dim, 5000)
        self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
        self.succtol = 3
        self.n_evals = 0

        # Trust region sizes
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.8
        self.turn_acqf_lenth = turn_acqf_lenth
        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        # Initialize parameters
        self._restart()

    def _restart(self):
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

    def _adjust_length(self, fX_next):
        if np.min(fX_next) < np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            self.length /= 2.0
            self.failcount = 0

    def _create_candidates(self, X, fX, length, n_training_steps, hypers,turn=False):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values.
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), self.dtype
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )

            # Save state dict
            hypers = gp.state_dict()

        # Create the trust region boundaries
        x_center = X[fX.argmin().item(), :][None, :]
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        #save the lengthscale
        self.lengthscales = weights
        # Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert #do not use bound to limit the acqusition function's optimal val

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]

        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), self.dtype
        else:
            device, dtype = self.device, self.dtype
        #device, dtype = torch.device("cpu"), torch.float64
        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)
        
        ucb = UpperConfidenceBound(gp,1.0,self.length*self.dim)
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = ucb(X_cand_torch).t().cpu().detach().numpy()

        del X_torch
        del y_torch
        del X_cand_torch
        del gp
        torch.cuda.empty_cache()
        # De-standardize the sampled values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers

    def _select_candidates(self, X_cand, y_cand):
        X_next = np.ones((self.batch_size, self.dim))
        for i in range(self.batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = np.argmin(y_cand)
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_cand[indbest] = np.inf
        return X_next

    def optimize(self):
        """Run the full optimization process."""
        while self.n_evals < self.max_evals:
            if len(self._fX) > 0 and self.verbose:
                n_evals, fbest = self.n_evals, self._fX.min()
                print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
                sys.stdout.flush()

            # Initialize parameters
            self._restart()

            # Generate and evalute initial design points
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            fX_init = np.array([[self.f(x)] for x in X_init])

            # Update budget and set as initial data for this TR
            self.n_evals += self.n_init
            self._X = deepcopy(X_init)
            self._fX = deepcopy(fX_init)

            # Append data to the global history
            self.X = np.vstack((self.X, deepcopy(X_init)))
            self.fX = np.vstack((self.fX, deepcopy(fX_init)))

            if self.verbose:
                fbest = self._fX.min()
                print(f"Starting from fbest = {fbest:.4}")
                sys.stdout.flush()

            while self.n_evals < self.max_evals and self.length >= self.length_min:
                # Warp inputs
                
                X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)

                # Standardize values
                fX = deepcopy(self._fX).ravel()
                #here we find the points souround in the minimum point with ard self.length
                minimum_x = X[np.argmin(self._fX)]
                minimum_vals = np.array([minimum_x for _ in range(fX.shape[0])])
                dis_to_center = np.linalg.norm(X-minimum_vals,axis=1)
                max_lengthscale = np.max(self.lengthscales)
                valid_points_indices = np.argwhere(dis_to_center-self.length*max_lengthscale*np.sqrt(X.shape[1])/2<=0)
                X = X[valid_points_indices]
                X = X.reshape((X.shape[0],self._X.shape[1]))
                fX = fX[valid_points_indices].ravel()

                # Create th next batch
                X_cand, y_cand, _ = self._create_candidates(
                X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
                )
                X_next = self._select_candidates(X_cand, y_cand)
             
                # Undo the warping
                X_next = from_unit_cube(np.array(X_next), self.lb, self.ub)

                # Evaluate batch
                fX_next = np.array([[self.f(x)] for x in X_next])

                # Update trust region
                self._adjust_length(fX_next)

                # Update budget and append data
                self.n_evals += self.batch_size
                self._X = np.vstack((self._X, X_next))
                self._fX = np.vstack((self._fX, fX_next))

                if self.verbose and fX_next.min() < self.fX.min():
                    n_evals, fbest = self.n_evals, fX_next.min()
                    print(f"{n_evals}) New best: {fbest:.4}")
                    sys.stdout.flush()

                # Append data to the global history
                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.vstack((self.fX, deepcopy(fX_next)))
