import numpy as np

class SimulationData:
    def __init__(self):
        self.censoring_range = 2

    def true_survival_function(self, t, x):
        """
        Ground truth survival function S(t|x).
        x should be a numpy array of 0s or 1s.
        t should be a numpy array of times.
        """
        # S(t|x=0) = exp(-2t)
        s0 = np.exp(-2 * t)
        # S(t|x=1) = exp(-2 * (t**2))
        s1 = np.exp(-2 * (t**2))
        
        return s0 * (x == 0) + s1 * (x == 1)

    def true_cumulative_hazard(self, t, x):
        """
        True cumulative hazard Lambda(t|x) = -ln(S(t|x)).
        """
        # Lambda(t|x=0) = 2t
        L0 = 2.0 * t
        # Lambda(t|x=1) = 2t^2
        L1 = 2.0 * (t**2)
        
        return L0 * (x == 0) + L1 * (x == 1)
    
    def true_hazard_function(self, t, x):
        """
        Ground truth hazard function h(t|x) = -d/dt log S(t|x).
        x should be a numpy array of 0s or 1s.
        t should be a numpy array of times.
        """
        # h(t|x=0) = 2 (constant)
        h0 = 2.0 * np.ones_like(t)
        # h(t|x=1) = 4t
        h1 = 4.0 * t
        
        return h0 * (x == 0) + h1 * (x == 1)

    def generate_batch(self, batch_size=1024, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        
        # 1. Generate x ~ Bernoulli(0.5)
        x = rng.binomial(1, 0.5, size=(batch_size, 1)).astype(np.float32)
        
        # 2. Generate true event times (inverse CDF method)
        # u ~ Uniform(0, 1)
        u = rng.uniform(0, 1, size=(batch_size, 1))
        
        # If x=0: S(t) = exp(-2t) -> t = -ln(u)/2
        t0 = -np.log(u) / 2.0
        
        # If x=1: S(t) = exp(-2t^2) -> t = sqrt(-ln(u)/2)
        t1 = np.sqrt(-np.log(u) / 2.0)
        
        t_true = t0 * (1-x) + t1 * x
        
        # 3. Generate Censoring times ~ Uniform(0, 2)
        c = rng.uniform(0, self.censoring_range, size=(batch_size, 1))
        
        # 4. Observed data
        t_obs = np.minimum(t_true, c).astype(np.float32)
        event = (t_true <= c).astype(np.float32) # delta
        
        import torch
        return torch.tensor(x), torch.tensor(t_obs), torch.tensor(event)


class ModerateSimulationData:
    def __init__(self):
        self.censoring_range = 3 # Slightly longer follow-up
        
    def true_hazard_function(self, t, x):
        """
        x=0: Smooth Oscillation (Easier than Stiff, but non-monotonic)
             h(t) = 1.2 + 0.8 * sin(3t)
        x=1: Bathtub Curve (High risk -> Low -> High)
             h(t) = 2.0 * (t - 1.2)^2 + 0.5
        """
        # Group 0: Sine Wave
        # Mean 1.4, Amplitude 0.8
        h0 = 1.4 + 0.8 * np.sin(4.3 * t)
        
        # Group 1: Parabola (Flatter)
        # Minimum at t=0.6. Coeff 3.0, Min 1.0
        h1 = 3.0 * (t - 0.6)**2 + 1.0
        
        return h0 * (1 - x) + h1 * x

    def true_cumulative_hazard(self, t, x):
        """
        Analytical Integrals
        """
        # Int(1.4 + 0.8sin(4.3t)) = 1.4t - (0.8/4.3)cos(4.3t)
        c0 = 0.8 / 4.3
        Lambda0 = 1.4 * t - c0 * np.cos(4.3 * t) + c0
        
        # Int(3.0(t-0.6)^2 + 1.0) = (3.0/3)(t-0.6)^3 + 1.0t
        # Constant C: -(3.0/3) * (-0.6)^3 = 1.0 * 0.216 = 0.216
        c1 = 0.216
        Lambda1 = (3.0/3.0) * ((t - 0.6)**3) + 1.0 * t + c1
        
        return Lambda0 * (1 - x) + Lambda1 * x

    def true_survival_function(self, t, x):
        return np.exp(-self.true_cumulative_hazard(t, x))

    def inverse_transform_sampling(self, u, x):
        """
        Newton-Raphson to solve Lambda(t) = -ln(u)
        """
        target = -np.log(u)
        t_est = target / 1.0 # Rough linear guess
        
        for _ in range(15):
            Lambda = self.true_cumulative_hazard(t_est, x)
            h = self.true_hazard_function(t_est, x)
            diff = Lambda - target
            # Update
            t_est = t_est - diff / (h + 1e-6)
            t_est = np.maximum(t_est, 0.0)
            
        return t_est

    def generate_batch(self, batch_size=1024, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

        x = rng.binomial(1, 0.5, size=(batch_size, 1)).astype(np.float32)
        u = rng.uniform(1e-5, 0.9999, size=(batch_size, 1))
        t_true = self.inverse_transform_sampling(u, x).astype(np.float32)
        
        c = rng.uniform(0, self.censoring_range, size=(batch_size, 1)).astype(np.float32)
        
        t_obs = np.minimum(t_true, c)
        event = (t_true <= c).astype(np.float32)
        
        import torch
        return torch.tensor(x), torch.tensor(t_obs), torch.tensor(event)
