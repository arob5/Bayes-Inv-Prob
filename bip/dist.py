import copy
import numpy as np
import scipy.stats as sp
from abc import ABC, abstractmethod
from param import ParamInfo, ParamGroup, ParamGroupValues, NoConstraint

class Dist(ABC): 
    """
    Abstract base class for probability distributions.

    Attributes:
        rv_info (ParamGroup): Metadata for the random variable's parameters.
        _dist_params (ParamGroupValues): Internal storage of parameter values.

    Methods:
        sample(n): Draw n samples from the distribution.
        density(x): Evaluate the probability density (or mass) function at x.
        log_density(x): Evaluate the log of the density (or mass) at x.
    """

    def __init__(self, param_group: ParamGroup, param_values: dict):
        self.rv_info = param_group
        self._dist_params = ParamGroupValues(param_group, param_values)

    @property
    def dist_params(self) -> ParamGroupValues:
        """
        Read-only access to distribution parameters.
        """
        return self._dist_params

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        pass

    @abstractmethod
    def density(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def log_density(self, x: np.ndarray) -> np.ndarray:
        pass 
    
class Norm(Dist): 
    """
    Multivariate Gaussian Distribution (x ~ N(m, C)) for d-dimensional x.

    Parameters:
        - m: Mean vector (of shape (d,)) with no additional constraints.
        - C: Covariance matrix (of shape (d, d)) that must be positive semidefinite (psd).

    The class uses the param package:
        * Metadata for 'm' and 'C' are stored in a ParamGroup (rv_info).
        * The actual parameter values are stored in dist_param as a ParamGroupValues object.
   
    Methods:
        - sample(n): Generates n samples from the distribution.
        - density(x): Evaluates the probability density at x.
        """
    
    def __init__(self, m, C):
        m = np.array(m)
        C = np.array(C)
        d = m.shape[0]  # Determine the dimension from the mean vector.

        # Create parameter's metadata.
        param_group = ParamGroup({
            "m": ParamInfo(value_type="float", shape=(d,), constraint=NoConstraint),
            "C": ParamInfo(value_type="float", shape=(d, d), constraint="psd")
        })
        # Create the initial parameter values (these will be validated by ParamGroupValues).
        param_values = {"m": m, "C": C}

        # Initialize the Distribution's attributes.
        super().__init__(param_group, param_values)
        
        # Create the underlying SciPy multivariate normal distribution instance.
        m_val = self._dist_params.values["m"]
        C_val = self._dist_params.values["C"] # Use the value property defined in ParamGroupValues
        self._rv = sp.multivariate_normal(mean=m_val, cov=C_val)
        
    def sample(self, n: int) -> np.ndarray:
        """
        Generate n samples from the Normal distribution.
        
        Returns:
            np.ndarray: An array of shape (n, d), where d is the dimension of the mean vector.
        """
        samples = self._rv.rvs(size=n)
        # Ensure a 2D array is returned even when n == 1.
        if samples.ndim == 1:
            samples = samples[np.newaxis, :]
        return samples
    
    def density(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the probability density function (pdf) at x.
        
        Parameters:
            x (np.ndarray): A point in the d-dimensional space.
        
        Returns:
            np.ndarray: The density (pdf) evaluated at x.
        """
        return self._rv.pdf(x)

    def log_density(self, x: np.ndarray) -> np.ndarray:
        return self._rv.logpdf(x)

class Dirichlet(Dist):
    """
    Dirichlet distribution with concentration parameters alpha.
    """
    def __init__(self, alpha):
        alpha = np.array(alpha)
        k = alpha.shape[0]
        param_group = ParamGroup({
            "alpha": ParamInfo(value_type="float", shape=(k,), constraint=NoConstraint)
        })
        param_values = {"alpha": alpha}
        super().__init__(param_group, param_values)
        a_val = self._dist_params.values["alpha"]
        self._rv = sp.dirichlet(a_val)

    def sample(self, n: int) -> np.ndarray:
        samples = self._rv.rvs(size=n)
        if samples.ndim == 1:
            samples = samples[np.newaxis, :]
        return samples

    def density(self, x: np.ndarray) -> np.ndarray:
        return self._rv.pdf(x)

    def log_density(self, x: np.ndarray) -> np.ndarray:
        return self._rv.logpdf(x)

class Poisson(Dist):
    """
    Poisson distribution with rate parameter mu.
    """
    def __init__(self, mu):
        mu = float(mu)
        param_group = ParamGroup({
            "mu": ParamInfo(value_type="float", shape=(), constraint=NoConstraint)
        })
        param_values = {"mu": mu}
        super().__init__(param_group, param_values)
        mu_val = self._dist_params.values["mu"]
        self._rv = sp.poisson(mu=mu_val)

    def sample(self, n: int) -> np.ndarray:
        return np.atleast_1d(self._rv.rvs(size=n))

    def density(self, x: np.ndarray) -> np.ndarray:
        return self._rv.pmf(x)

    def log_density(self, x: np.ndarray) -> np.ndarray:
        return self._rv.logpmf(x)

class Multinomial(Dist):
    """
    Multinomial distribution with n trials and category probabilities p.
    """
    def __init__(self, n_trials, p):
        n_trials = int(n_trials)
        p = np.array(p)
        k = p.shape[0]
        param_group = ParamGroup({
            "n": ParamInfo(value_type="int", shape=(), constraint=NoConstraint),
            "p": ParamInfo(value_type="float", shape=(k,), constraint=NoConstraint)
        })
        param_values = {"n": n_trials, "p": p}
        super().__init__(param_group, param_values)
        n_val = self._dist_params.values["n"]
        p_val = self._dist_params.values["p"]
        self._rv = sp.multinomial(n=n_val, p=p_val)

    def sample(self, num_samples: int) -> np.ndarray:
        return self._rv.rvs(size=num_samples)

    def density(self, x: np.ndarray) -> np.ndarray:
        return self._rv.pmf(x)

    def log_density(self, x: np.ndarray) -> np.ndarray:
        return self._rv.logpmf(x)
