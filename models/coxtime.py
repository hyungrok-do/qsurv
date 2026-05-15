import torch
import torch.nn as nn
import numpy as np
import copy
from models.base import SurvivalModel

from torch.utils.data import DataLoader, TensorDataset
from models._coxdataset import CoxTimeDataset

class CoxTime(SurvivalModel):
    def __init__(self, network, n_controls=1, optimizer_class=torch.optim.AdamW, scheduler_class=None, 
                 optimizer_params=None, scheduler_params=None, random_seed=42,
                 batch_size=128, epochs=100, lr=1e-3,
                 time_norm_mode='none'):
        super().__init__(network, optimizer_class, scheduler_class, optimizer_params, scheduler_params, random_seed,
                         batch_size, epochs, lr, time_norm_mode)
        self.network = network
        self.n_controls = n_controls
        self.baseline_hazards_ = None
        self.baseline_cumulative_hazards_ = None
        self.train_x = None
        self.train_t = None
        self.train_e = None

    def calculate_loss(self, x_case, x_controls, t_case):
        """Pycox case-control loss: softplus(g_control - g_case).mean()."""
        log_risk_case = self.network(x_case, t_case)                            # (B, 1)
        B, n_ctrl = x_controls.shape[:2]
        rest_dims = x_controls.shape[2:]
        x_controls_flat = x_controls.reshape(-1, *rest_dims)
        t_case_expanded = t_case.unsqueeze(1).expand(t_case.size(0), n_ctrl, *t_case.size()[1:]).reshape(-1, *t_case.size()[1:])
        log_risk_controls = self.network(x_controls_flat, t_case_expanded).reshape(B, n_ctrl)
        return torch.nn.functional.softplus(log_risk_controls - log_risk_case).mean()

    def fit(self, x, t, e, val_x=None, val_t=None, val_e=None):
        rng = np.random.RandomState(self.random_seed)
        n = x.shape[0]
        self._update_time_normalization_stats(t)

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
        
        train_dataset = CoxTimeDataset(train_x, train_t, train_e, n_controls=self.n_controls, random_seed=self.random_seed)
        val_dataset = CoxTimeDataset(val_x, val_t, val_e, n_controls=self.n_controls, random_seed=self.random_seed)
        
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
            
        opt_params = self.optimizer_params.copy()
        opt_params['lr'] = self.lr
        optimizer = self.optimizer_class(self.parameters(), **opt_params)
        scheduler = None
        if self.scheduler_class:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_params)
            
        self.best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(self.epochs):
            self.train()
            train_loss = 0.0
            
            for x_case, x_controls, t_case in train_loader:
                param_device = next(self.network.parameters()).device
                x_case = x_case.float().to(param_device)
                x_controls = x_controls.float().to(param_device)
                t_case = t_case.float().to(param_device)
                
                optimizer.zero_grad()
                loss = self.calculate_loss(x_case, x_controls, t_case)
                loss.backward()
                    
                optimizer.step()
                train_loss += loss.item() * x_case.size(0)
            
            if len(train_dataset) > 0:
                train_loss /= len(train_dataset)
            
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_case, x_controls, t_case in val_loader:
                    device = next(self.parameters()).device
                    x_case = x_case.float().to(device)
                    x_controls = x_controls.float().to(device)
                    t_case = t_case.float().to(device)
                    loss = self.calculate_loss(x_case, x_controls, t_case)
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
            else:
                 x = x.to(device)
                 t = t.to(device)
                 e = e.to(device)
                 
            t = t.view(-1)
            e = e.view(-1)
            
            sorted_idx = torch.argsort(t)
            x_sorted = x[sorted_idx]
            t_sorted = t[sorted_idx]
            e_sorted = e[sorted_idx]
            
            event_times = t_sorted[e_sorted == 1]
            unique_event_times = torch.unique(event_times, sorted=True)
            
            if len(unique_event_times) == 0:
                self.baseline_hazards_ = (np.array([]), np.array([]))
                self.baseline_cumulative_hazards_ = (np.array([]), np.array([]))
                return

            n_samples = x_sorted.shape[0]
            n_times = unique_event_times.shape[0]

            mask_events = (t_sorted.unsqueeze(1) == unique_event_times.unsqueeze(0)) & (e_sorted.unsqueeze(1) == 1)
            event_counts = mask_events.sum(dim=0).float()

            # Chunk over time to keep N*T memory bounded (e.g. 5000x5000 = 25M cells).
            time_batch_size = 128
            denom_list = []
            for i in range(0, n_times, time_batch_size):
                t_batch = unique_event_times[i:i + time_batch_size]
                current_batch_size = t_batch.shape[0]
                t_rep = t_batch.unsqueeze(0).expand(n_samples, -1).reshape(-1, 1)

                if i == 0 and hasattr(self.network, 'precompute') and hasattr(self.network, 'forward_nodes_from_cache'):
                    h_all, y_static_all, v_all = self.network.precompute(x_sorted)

                if hasattr(self.network, 'precompute') and hasattr(self.network, 'forward_nodes_from_cache'):
                    risk_scores_batch = self.network.forward_nodes_from_cache(h_all, y_static_all, v_all, t_rep, K=current_batch_size).view(n_samples, current_batch_size)
                elif hasattr(self.network, 'forward_split'):
                    risk_scores_batch = self.network.forward_split(x_sorted, t_rep).view(n_samples, current_batch_size)
                else:
                    x_rep = x_sorted.unsqueeze(1).expand(x_sorted.size(0), current_batch_size, *x_sorted.size()[1:]).reshape(-1, *x_sorted.size()[1:])
                    risk_scores_batch = self.network(x_rep, t_rep).view(n_samples, current_batch_size)

                exp_risk_batch = torch.exp(risk_scores_batch.clamp(max=30))
                mask_risk_batch = t_sorted.unsqueeze(1) >= t_batch.unsqueeze(0)
                denom_list.append((exp_risk_batch * mask_risk_batch).sum(dim=0))

            denom = torch.cat(denom_list)
            hazard = event_counts / (denom + 1e-8)
            cum_hazard = torch.cumsum(hazard, dim=0)
            
            self.baseline_hazards_ = (unique_event_times.cpu().numpy(), hazard.cpu().numpy())
            self.baseline_cumulative_hazards_ = (unique_event_times.cpu().numpy(), cum_hazard.cpu().numpy())

    def predict_survival_probability(self, x, t):
        if self.baseline_cumulative_hazards_ is None:
            raise RuntimeError("Model must be fitted before prediction. Baseline hazards not found.")
            
        self.eval()
        
        # Ensure correct device
        model_device = next(self.parameters()).device
        x = self._to_tensor(x)
        t_tensor = self._to_tensor(t).to(model_device)
        
        with torch.no_grad():
            unique_times = self.baseline_cumulative_hazards_[0]
            device = model_device 
            unique_times_tensor = torch.tensor(unique_times, dtype=torch.float32, device=device)
            baseline_hazards_tensor = torch.tensor(self.baseline_hazards_[1], dtype=torch.float32, device=device)
            
            if t_tensor.dim() == 0:
                t_tensor = t_tensor.unsqueeze(0)
            t_tensor = t_tensor.view(-1)
            
            max_t = t_tensor.max()
            mask = unique_times_tensor <= max_t
            rel_times = unique_times_tensor[mask]
            rel_hazards = baseline_hazards_tensor[mask]
            
            n_rel = len(rel_times)
            
            loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)
            all_survival = []
            
            for batch in loader:
                batch_x = batch[0].to(device)
                n_batch = batch_x.shape[0]

                if n_rel > 0:
                    t_rep = rel_times.unsqueeze(0).expand(n_batch, -1).reshape(-1, 1)
                    
                    if hasattr(self.network, 'precompute') and hasattr(self.network, 'forward_nodes_from_cache'):
                        h_cache, y_static, v_cache = self.network.precompute(batch_x)
                        risk_scores = self.network.forward_nodes_from_cache(h_cache, y_static, v_cache, t_rep, K=n_rel).view(n_batch, n_rel)
                    elif hasattr(self.network, 'forward_split'):
                        risk_scores = self.network.forward_split(batch_x, t_rep).view(n_batch, n_rel)
                    else:
                        x_rep = batch_x.unsqueeze(1).expand(batch_x.size(0), n_rel, *batch_x.size()[1:]).reshape(-1, *batch_x.size()[1:])
                        risk_scores = self.network(x_rep, t_rep).view(n_batch, n_rel)
                        
                    exp_risk = torch.exp(risk_scores.clamp(max=30))
                    H_at_times = torch.cumsum(exp_risk * rel_hazards.unsqueeze(0), dim=1)

                    indices = (torch.bucketize(t_tensor, rel_times, right=True) - 1).clamp(0, n_rel - 1)
                    H_selected = H_at_times[:, indices]
                    H_selected[:, t_tensor < rel_times[0]] = 0.0
                else:
                    H_selected = torch.zeros(n_batch, len(t_tensor)).to(device)

                all_survival.append(torch.exp(-H_selected.clamp(max=30)).cpu())

            return torch.cat(all_survival, dim=0)

    def predict_hazard(self, x, t):
        if self.baseline_hazards_ is None:
            raise RuntimeError("Model must be fitted before prediction. Baseline hazards not found.")

        self.eval()
        model_device = next(self.parameters()).device
        x = self._to_tensor(x)
        t_tensor = self._to_tensor(t).to(model_device)

        with torch.no_grad():
            unique_times = self.baseline_hazards_[0]
            device = model_device
            unique_times_tensor = torch.tensor(unique_times, dtype=torch.float32, device=device)
            baseline_hazards_tensor = torch.tensor(self.baseline_hazards_[1], dtype=torch.float32, device=device)

            if t_tensor.dim() == 0: t_tensor = t_tensor.unsqueeze(0)
            t_tensor = t_tensor.view(-1)

            indices = (torch.bucketize(t_tensor, unique_times_tensor, right=True) - 1).clamp(0, len(unique_times) - 1)
            diffs = torch.diff(unique_times_tensor)
            diffs = torch.cat([diffs, diffs[-1].unsqueeze(0)]) if len(diffs) > 0 else torch.tensor([1.0], device=device)
            base_haz_rate = baseline_hazards_tensor / diffs.clamp(min=1e-6)
            h0_vals = base_haz_rate[indices]
            h0_vals[t_tensor < unique_times_tensor[0]] = 0.0
            n_times = t_tensor.shape[0]

            loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)
            all_hazard = []
            for batch in loader:
                batch_x = batch[0].to(device)
                n_batch = batch_x.shape[0]
                t_rep = t_tensor.unsqueeze(0).expand(n_batch, -1).reshape(-1, 1)

                if hasattr(self.network, 'precompute') and hasattr(self.network, 'forward_nodes_from_cache'):
                    h_cache, y_static, v_cache = self.network.precompute(batch_x)
                    risk_scores = self.network.forward_nodes_from_cache(h_cache, y_static, v_cache, t_rep, K=n_times).view(n_batch, n_times)
                elif hasattr(self.network, 'forward_split') and n_times > 1:
                    risk_scores = self.network.forward_split(batch_x, t_rep).view(n_batch, n_times)
                else:
                    x_rep = batch_x.unsqueeze(1).expand(batch_x.size(0), n_times, *batch_x.size()[1:]).reshape(-1, *batch_x.size()[1:])
                    risk_scores = self.network(x_rep, t_rep).view(n_batch, n_times)

                exp_risk = torch.exp(risk_scores.clamp(max=30))
                all_hazard.append((h0_vals.unsqueeze(0) * exp_risk).cpu())
            return torch.cat(all_hazard, dim=0)

    def predict_risk(self, x, t):
        """exp(g(x, t))."""
        self.eval()
        model_device = next(self.parameters()).device
        x = self._to_tensor(x).to(model_device)
        t = self._to_tensor(t).to(model_device)

        with torch.no_grad():
            if t.dim() == 0: t = t.unsqueeze(0)
            t = t.view(-1)
            n_samples = x.shape[0]
            n_times = t.shape[0]
            t_rep = t.unsqueeze(0).expand(n_samples, -1).reshape(-1, 1)

            if hasattr(self.network, 'forward_split'):
                log_risk = self.network.forward_split(x, t_rep).view(n_samples, n_times)
            else:
                expanded_shape = list(x.shape)
                expanded_shape.insert(1, n_times)
                x_rep = x.unsqueeze(1).expand(*expanded_shape).reshape(-1, *x.shape[1:])
                log_risk = self.network(x_rep, t_rep).view(n_samples, n_times)
            return torch.exp(log_risk.clamp(max=30))
