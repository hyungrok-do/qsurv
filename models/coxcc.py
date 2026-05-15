import torch
import torch.nn as nn
import numpy as np
import copy
from models.base import SurvivalModel

from torch.utils.data import DataLoader, TensorDataset
from models._coxdataset import CoxCCDataset

class CoxCC(SurvivalModel):
    def __init__(self, network, n_controls=1, optimizer_class=torch.optim.AdamW, scheduler_class=None, 
                 optimizer_params=None, scheduler_params=None, random_seed=42,
                 batch_size=128, epochs=100, lr=1e-3):
        super().__init__(network, optimizer_class, scheduler_class, optimizer_params, scheduler_params, random_seed,
                         batch_size, epochs, lr)
        self.network = network
        self.n_controls = n_controls
        self.baseline_hazards_ = None
        self.baseline_cumulative_hazards_ = None
        self.train_x = None
        self.train_t = None
        self.train_e = None

    def calculate_loss(self, x_case, x_controls):
        """Pycox case-control loss: softplus(g_control - g_case).mean()."""
        log_risk_case = self.network(x_case)                                     # (B, 1)
        B, n_ctrl = x_controls.shape[:2]
        rest_dims = x_controls.shape[2:]
        log_risk_controls = self.network(x_controls.reshape(-1, *rest_dims)).reshape(B, n_ctrl)
        return torch.nn.functional.softplus(log_risk_controls - log_risk_case).mean()

    def fit(self, x, t, e, val_x=None, val_t=None, val_e=None):
        rng = np.random.RandomState(self.random_seed)
        n = x.shape[0]

        if val_x is None:
            val_size = int(n * 0.2)
            idxs = rng.permutation(n)
            train_idxs = idxs[:n - val_size]
            val_idxs = idxs[n - val_size:]
            train_x, val_x = x[train_idxs], x[val_idxs]
            train_t, val_t = t[train_idxs], t[val_idxs]
            train_e, val_e = e[train_idxs], e[val_idxs]
        else:
            train_x, train_t, train_e = x, t, e
        
        train_dataset = CoxCCDataset(train_x, train_t, train_e, n_controls=self.n_controls, random_seed=self.random_seed)
        val_dataset = CoxCCDataset(val_x, val_t, val_e, n_controls=self.n_controls, random_seed=self.random_seed)
        
        g = torch.Generator()
        g.manual_seed(self.random_seed)

        device = next(self.parameters()).device
        pin_memory = device.type == 'cuda'
        nw = 0

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True, 
            generator=g,
            num_workers=nw,
            pin_memory=pin_memory,
            persistent_workers=(nw > 0)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=nw,
            pin_memory=pin_memory,
            persistent_workers=(nw > 0)
        )
            
        # Optimizer
        opt_params = self.optimizer_params.copy()
        opt_params['lr'] = self.lr
        optimizer = self.optimizer_class(self.parameters(), **opt_params)
        scheduler = None
        if self.scheduler_class:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_params)
            
        self.best_val_loss = float('inf')
        best_model_state = None
        device = next(self.parameters()).device
        
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0.0

            for x_case, x_controls in train_loader:
                x_case = x_case.float().to(device).contiguous()
                x_controls = x_controls.float().to(device).contiguous()
                
                optimizer.zero_grad()
                loss = self.calculate_loss(x_case, x_controls)
                loss.backward()
                    
                optimizer.step()
                train_loss += loss.item() * x_case.size(0)
            
            if len(train_dataset) > 0:
                train_loss /= len(train_dataset)
            
            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_case, x_controls in val_loader:
                    x_case = x_case.float().to(device).contiguous()
                    x_controls = x_controls.float().to(device).contiguous()
                    loss = self.calculate_loss(x_case, x_controls)
                    val_loss += loss.item() * x_case.size(0)
            
            if len(val_dataset) > 0:
                val_loss /= len(val_dataset)
            else:
                val_loss = train_loss
            
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            
        self.compute_baseline_hazards(train_x, train_t, train_e)
        
        return self.best_val_loss

    def compute_baseline_hazards(self, x, t, e):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=device)
                t = torch.tensor(t, dtype=torch.float32, device=device)
                e = torch.tensor(e, dtype=torch.float32, device=device)

            t = t.view(-1)
            e = e.view(-1)
            
            loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)
            risk_scores_list = []
            for batch in loader:
                batch_x = batch[0].to(device)
                batch_scores = self.network(batch_x).view(-1)
                risk_scores_list.append(batch_scores)
            risk_scores = torch.cat(risk_scores_list, dim=0)
            
            t = t.cpu()
            e = e.cpu()
            risk_scores = risk_scores.cpu()
            
            sorted_idx = torch.argsort(t)
            t_sorted = t[sorted_idx]
            e_sorted = e[sorted_idx]
            risk_scores_sorted = risk_scores[sorted_idx]
            
            unique_times = torch.unique(t_sorted, sorted=True)
            unique_indices = torch.searchsorted(t_sorted, unique_times)
            
            exp_risk = torch.exp(risk_scores_sorted.clamp(max=30))
            risk_set_sums = torch.cumsum(exp_risk.flip(0), 0).flip(0)
            
            risk_set_sums_at_unique = risk_set_sums[unique_indices]
            
            dataset_event_cumsum = torch.cat([torch.tensor([0.0]), torch.cumsum(e_sorted, 0)])
            
            next_indices = torch.cat([unique_indices[1:], torch.tensor([len(t_sorted)])])
            event_counts = dataset_event_cumsum[next_indices] - dataset_event_cumsum[unique_indices]
            
            hazard = event_counts / (risk_set_sums_at_unique + 1e-8)
            hazard[event_counts == 0] = 0.0
            
            cum_hazard = torch.cumsum(hazard, 0)
            
            self.baseline_hazards_ = (unique_times.numpy(), hazard.numpy())
            self.baseline_cumulative_hazards_ = (unique_times.numpy(), cum_hazard.numpy())

    def predict_survival_probability(self, x, t):
        if self.baseline_cumulative_hazards_ is None:
             raise RuntimeError("Model must be fitted before prediction. Baseline hazards not found.")
            
        self.eval()
        x = self._to_tensor(x)
        t = self._to_tensor(t)
        
        with torch.no_grad():
            if t.dim() == 0:
                t = t.unsqueeze(0)
            t = t.view(-1)

            unique_times = self.baseline_cumulative_hazards_[0] # numpy
            cum_hazards = self.baseline_cumulative_hazards_[1] # numpy
            
            device = x.device
            unique_times_t = torch.tensor(unique_times, device=device, dtype=torch.float32)
            cum_hazards_t = torch.tensor(cum_hazards, device=device, dtype=torch.float32)

            indices = (torch.bucketize(t, unique_times_t, right=True) - 1).clamp(0, len(cum_hazards) - 1)
            base_cum_haz = cum_hazards_t[indices]
            base_cum_haz[t < unique_times_t[0]] = 0.0

            loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)
            all_survival = []
            for batch in loader:
                batch_x = batch[0].to(device)
                exp_risk = torch.exp(self.network(batch_x).view(-1).clamp(max=30))
                cum_haz = exp_risk.unsqueeze(1) * base_cum_haz.unsqueeze(0)            # H(t|x) = H0(t) exp(risk)
                all_survival.append(torch.exp(-cum_haz.clamp(max=30)).cpu())
            return torch.cat(all_survival, dim=0)

    def predict_hazard(self, x, t):
        if self.baseline_hazards_ is None:
            raise RuntimeError("Model must be fitted before prediction. Baseline hazards not found.")
            
        self.eval()
        x = self._to_tensor(x)
        t = self._to_tensor(t)
        
        with torch.no_grad():
            unique_times = self.baseline_hazards_[0]
            baseline_hazards = self.baseline_hazards_[1]
            
            device = x.device
            
            unique_times_t = torch.tensor(unique_times, device=device, dtype=torch.float32)
            baseline_hazards_t = torch.tensor(baseline_hazards, device=device, dtype=torch.float32)
            
            if t.dim() == 0: t = t.unsqueeze(0)
            t = t.view(-1)

            indices = (torch.bucketize(t, unique_times_t, right=True) - 1).clamp(0, len(baseline_hazards_t) - 1)
            widths = torch.diff(unique_times_t)
            widths = torch.cat([widths, widths[-1].unsqueeze(0)]) if len(widths) > 0 else torch.tensor([1.0], device=device)
            base_haz_rate = baseline_hazards_t / widths.clamp(min=1e-6)
            h0_rate = base_haz_rate[indices]
            h0_rate[t < unique_times_t[0]] = 0.0

            loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)
            all_hazard = []
            for batch in loader:
                batch_x = batch[0].to(device)
                exp_risk = torch.exp(self.network(batch_x).view(-1).clamp(max=30))
                all_hazard.append((exp_risk.unsqueeze(1) * h0_rate.unsqueeze(0)).cpu())
            return torch.cat(all_hazard, dim=0)

    def predict_risk(self, x, t=None):
        self.eval()
        x = self._to_tensor(x)
        if t is not None:
            t = self._to_tensor(t)

        with torch.no_grad():
            loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)
            all_risk = []
            for batch in loader:
                batch_x = batch[0].to(x.device)
                exp_risk = torch.exp(self.network(batch_x).view(-1).clamp(max=30))
                if t is not None:
                    exp_risk = exp_risk.unsqueeze(1).expand(-1, t.numel())
                all_risk.append(exp_risk.cpu())
            return torch.cat(all_risk, dim=0)
