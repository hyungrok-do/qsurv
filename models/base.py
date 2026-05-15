import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import copy
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

eps = 1e-8

class SurvivalModel(nn.Module, ABC):
    def __init__(self, network, optimizer_class=optim.Adam, scheduler_class=None, 
                 optimizer_params=None, scheduler_params=None, random_seed=42,
                 batch_size=128, epochs=100, lr=1e-3,
                 time_norm_mode='none'):
        super().__init__()
        self.network = network
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.scheduler_params = scheduler_params if scheduler_params is not None else {}
        self.random_seed = random_seed
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_epochs = epochs
        self.lr = lr
        self.time_norm_mode = time_norm_mode
        
        self.best_val_loss = float('inf')

    def _to_tensor(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(np.array(data)).float()
        else:
            data = data.float()
        return data.to(next(self.parameters()).device)

    @abstractmethod
    def calculate_loss(self, x, t, e):
        pass

    @abstractmethod
    def predict_survival_probability(self, x, t):
        pass
    
    def _update_time_normalization_stats(self, t):
        """Update mu/sigma buffers on the network (or its time_embed) from training times."""
        target_mod = None
        candidates = []
        if hasattr(self, 'network'):
            candidates.append(self.network)
            if hasattr(self.network, 'resnet'):
                candidates.append(self.network)

        for mod in candidates:
            if hasattr(mod, 'mu') and hasattr(mod, 'sigma'):
                target_mod = mod
                break
        if target_mod is None:
            for mod in candidates:
                if hasattr(mod, 'time_embed') and hasattr(mod.time_embed, 'mu'):
                    target_mod = mod.time_embed
                    break
        if target_mod is None:
            return

        if isinstance(t, np.ndarray):
            t_tensor = torch.from_numpy(t).float()
        elif isinstance(t, torch.Tensor):
            t_tensor = t.float()
        else:
            return

        mu = torch.tensor(0.0)
        sigma = torch.tensor(1.0)
        if self.time_norm_mode == 'log_std':
            u = torch.log1p(t_tensor)
            mu, sigma = u.mean(), u.std(unbiased=False)
        elif self.time_norm_mode == 'identity_std':
            mu, sigma = t_tensor.mean(), t_tensor.std(unbiased=False)
        elif self.time_norm_mode == 'quantile':
            # sigma = t_99; mu = 0 maps t to t/t_99. MPS lacks torch.quantile, so fall back to CPU.
            t_99 = torch.quantile(t_tensor.cpu(), 0.99).to(t_tensor.device) if t_tensor.device.type == 'mps' else torch.quantile(t_tensor, 0.99)
            sigma = t_99
        elif self.time_norm_mode == 'min_max':
            # mu=0, sigma=t_max keeps normalized times non-negative so v(0)=0 has no extrapolation.
            sigma = t_tensor.max()

        if sigma < 1e-6:
            sigma = torch.tensor(1.0)

        if hasattr(target_mod, 'mu'):
            target_mod.mu.fill_(mu.to(target_mod.mu.device))
        if hasattr(target_mod, 'sigma'):
            target_mod.sigma.fill_(sigma.to(target_mod.sigma.device))

    def fit(self, x, t, e, val_x=None, val_t=None, val_e=None):
        if x is None or t is None or e is None:
            raise ValueError("x, t, and e must be provided.")

        self._update_time_normalization_stats(t)

        n = x.shape[0]
        idxs = np.arange(n)
        e_cpu = e.cpu().numpy() if isinstance(e, torch.Tensor) else e

        if val_x is None:
            try:
                train_idxs, val_idxs = train_test_split(idxs, test_size=0.2, stratify=e_cpu, random_state=self.random_seed)
            except ValueError:
                # Stratified split can fail when an event class has too few samples.
                train_idxs, val_idxs = train_test_split(idxs, test_size=0.2, random_state=self.random_seed)
            train_x, val_x = x[train_idxs], x[val_idxs]
            train_t, val_t = t[train_idxs], t[val_idxs]
            train_e, val_e = e[train_idxs], e[val_idxs]
            val_size = len(val_idxs)
        else:
            train_x, train_t, train_e = x, t, e
            val_size = val_x.shape[0]

        train_dataset = TensorDataset(self._to_tensor(train_x), self._to_tensor(train_t), self._to_tensor(train_e))
        if val_size > 0:
            val_dataset = TensorDataset(self._to_tensor(val_x), self._to_tensor(val_t), self._to_tensor(val_e))

        g = torch.Generator()
        g.manual_seed(self.random_seed)

        # num_workers=0 is required: data is already on GPU, and worker processes spawned
        # inside joblib.Parallel cause CUDA initialization crashes.
        nw = 0
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  drop_last=False, generator=g, num_workers=nw)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=nw) if val_size > 0 else None

        opt_params = self.optimizer_params.copy()
        opt_params['lr'] = self.lr
        optimizer = self.optimizer_class(self.parameters(), **opt_params)
        scheduler = self.scheduler_class(optimizer, **self.scheduler_params) if self.scheduler_class else None

        self.best_val_loss = float('inf')
        best_model_state = None
        device = next(self.parameters()).device

        for epoch in range(self.epochs):
            self.train()
            train_loss = 0.0
            actual_train_size = 0
            for batch in train_loader:
                batch_x, batch_t, batch_e = batch
                batch_x = batch_x.to(device)
                batch_t = batch_t.to(device)
                batch_e = batch_e.to(device)
                optimizer.zero_grad()
                loss = self.calculate_loss(batch_x, batch_t, batch_e)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
                actual_train_size += batch_x.size(0)
            train_loss /= max(actual_train_size, 1)

            self.eval()
            val_loss = 0.0
            actual_val_size = 0
            if val_loader is not None:
                with torch.no_grad():
                    for batch in val_loader:
                        batch_x, batch_t, batch_e = batch
                        batch_x = batch_x.to(device)
                        batch_t = batch_t.to(device)
                        batch_e = batch_e.to(device)
                        loss = self.calculate_loss(batch_x, batch_t, batch_e)
                        val_loss += loss.item() * batch_x.size(0)
                        actual_val_size += batch_x.size(0)
                val_loss /= max(actual_val_size, 1)
            else:
                val_loss = train_loss

            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}

        if best_model_state is not None:
            self.load_state_dict(best_model_state)
        return self.best_val_loss
