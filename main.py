import argparse
import json
import os
import random
import signal
import sys
import time

import numpy as np
import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Data Imports
from data.tabular import get_metabric, get_support2, get_flchain, get_gbsg, get_nwtco, get_framingham
from data.covid.dataset import get_covid_dataset
from data.c4kc_kits.dataset import get_c4kc_kits_dataset
from data.brats.dataset import get_brats_dataset

# Models (Tabular & Generic)
from models.soden import SODEN
from models.qsurv import QSurv
from models.mdn import MDN
from models.coxcc import CoxCC
from models.coxtime import CoxTime
from models.nnetsurv import NnetSurv
from models.deephit import DeepHit
from models.desurv import DeSurv

# Networks (Tabular)
from networks import NeuralHazard, MDNNet, CoxNet, CoxTimeNet, NnetSurvNet, DeepHitNet

# Networks (Image / ResNet)
from networks import (
    ResNetDeepHit, ResNetNnetSurv, ResNetCox, 
    ResNetMDN,
    TimeLoRASurvivalWrapper, BaseMLP, ResNetBackbone, TimeConcatenatedMLP,
    TimeFiLMSurvivalWrapper, TimeConcatSurvivalWrapper,
)

# Utils
from tools.eval import (
    get_tau_quantiles, ipcw_uno_concordance_index, cluster_survival_curves,
    integrated_brier_score as ibs_func, integrated_binomial_log_likelihood as ibll_func,
    d_calibration, brier_score_at_times, bll_at_times, concordance_at_times
)
from models._discretizer import LabTransDiscreteTime

TABULAR_DATASETS = ['metabric', 'support2', 'flchain', 'gbsg', 'nwtco', 'framingham']
IMAGE_DATASETS = ['covid', 'c4kc_kits', 'brats']
RUNNABLE_MODELS = ['soden', 'desurv', 'qsurv', 'qsurv_concat', 'qsurv_film',
                   'mdn', 'coxcc', 'coxtime', 'nnetsurv', 'deephit']
DEFAULT_MODELS = ['coxcc', 'coxtime', 'nnetsurv', 'mdn', 'desurv',
                  'qsurv', 'qsurv_concat', 'qsurv_film', 'deephit']


def set_global_seed(seed, deterministic=True):
    seed = int(seed) % (2**32 - 1)
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        pass

def get_dataset_type(dataset_name):
    if dataset_name in IMAGE_DATASETS:
        return 'image'
    return 'tabular'

_DATA_LOADERS = {
    'metabric':   get_metabric,
    'support2':   get_support2,
    'flchain':    get_flchain,
    'gbsg':       get_gbsg,
    'nwtco':      get_nwtco,
    'framingham': get_framingham,
    'covid':      get_covid_dataset,
    'c4kc_kits':  get_c4kc_kits_dataset,
    'brats':      lambda **kwargs: get_brats_dataset(modality='flair', **kwargs),
}


def get_data_loader(dataset_name):
    try:
        return _DATA_LOADERS[dataset_name]
    except KeyError:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def sample_hyperparameters(model_name, n_disc, random_state, epochs, dataset_type='tabular', dataset_name=None):
    opt_class = torch.optim.AdamW

    # t_hidden capacity for the time-conditioning MLP. QSurv, DeSurv, and CoxTime
    # use the larger 64 (these are the LoRA-conditioned methods that benefit from
    # extra time-MLP capacity). Other models stay at 32.
    t_hidden = 64 if model_name in ('qsurv', 'desurv', 'coxtime') else 32

    if dataset_type == 'image':
        lr_log = random_state.uniform(-5, -2)
        lr = 10**lr_log
        
        wd_log = random_state.uniform(-8, -3)
        weight_decay = 10**wd_log
        
        batch_size = int(random_state.choice([16, 32, 64]))
        
        dropout = float(random_state.choice([0.0, 0.1, 0.3, 0.5]))
        
        params = {
            'lr': lr,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'hidden_dims': None,
            'optimizer_class': opt_class,
            't_hidden': t_hidden,
            'dropout': dropout
        }
        
    else:
        n_layers = random_state.choice([2, 3, 4])
        hidden_size = int(random_state.choice([32, 64, 128, 256]))
        hidden_dims = [hidden_size] * n_layers
        
        lr_log = random_state.uniform(-4, -2)
        lr = 10**lr_log
        
        wd_log = random_state.uniform(-8, -3)
        weight_decay = 10**wd_log
        
        dropout = float(random_state.choice([0.0, 0.1, 0.3, 0.5]))
        batch_size = int(random_state.choice([64, 128, 256]))
        batch_norm = bool(random_state.choice([True, False]))
        
        params = {
            'hidden_dims': hidden_dims,
            'lr': lr,
            'weight_decay': weight_decay,
            'dropout': dropout,
            'batch_size': batch_size,
            'batch_norm': batch_norm,
            'optimizer_class': opt_class,
            't_hidden': t_hidden
        }
    
    if model_name in ['nnetsurv', 'deephit']:
        params['n_disc'] = 50
        params['cuts'] = params['n_disc'] + 1

        if model_name == 'deephit':
            params['alpha'] = 0.2
            params['sigma'] = 0.1

    if model_name == 'mdn':
        params['n_components'] = 5
        
    if model_name == 'desurv':
        params['n_nodes'] = 15

    if 'qsurv' in model_name:
        params['n_nodes'] = 15

    # QSurv, DeSurv, and CoxTime use the basic LoRA config (rank=32, alpha=32)
    # for an apples-to-apples comparison. t_hidden=64 is set above.
    if model_name in ('qsurv', 'desurv', 'coxtime'):
        params['rank'] = 32
        params['alpha'] = 32

    return params

def _model_kwargs(seed, epochs, lr, batch_size, optimizer_class, optimizer_params):
    """Common kwargs for all survival models."""
    return dict(
        random_seed=seed, epochs=epochs, lr=lr, batch_size=batch_size,
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params={'T_max': epochs, 'eta_min': 0.0},
        optimizer_params=optimizer_params, optimizer_class=optimizer_class
    )

def _time_net_image(backbone_name, input_channels, pretrained, conditioning, mu_t, sigma_t, rank, alpha, t_hidden, t_depth=3, output_activation='softplus', output_bias=True, dropout=0.0):
    """Create time-conditioned network for image models."""
    bb = ResNetBackbone(model_name=backbone_name, input_channels=input_channels, pretrained=pretrained)
    if conditioning == 'lora':
        return TimeLoRASurvivalWrapper(bb, feature_dim=bb.output_dim, output_dim=1, output_activation=output_activation, output_bias=output_bias,
                                       mu=mu_t, sigma=sigma_t, time_norm_mode='min_max', rank=rank, alpha=alpha,
                                       adapter_activation=torch.nn.Tanh, t_hidden=t_hidden, t_depth=t_depth, dropout=dropout)
    elif conditioning == 'film':
        return TimeFiLMSurvivalWrapper(bb, feature_dim=bb.output_dim, output_dim=1, output_activation=output_activation, output_bias=output_bias,
                                       mu=mu_t, sigma=sigma_t, time_norm_mode='min_max',
                                       t_hidden=t_hidden, t_depth=t_depth)
    else:  # concat
        return TimeConcatSurvivalWrapper(bb, feature_dim=bb.output_dim, output_dim=1, output_activation=output_activation, output_bias=output_bias,
                                         mu=mu_t, sigma=sigma_t, time_norm_mode='min_max',
                                         head_hidden_dims=[t_hidden]*max(1, t_depth-1))



def _time_net_tabular(input_dim, hidden_dims, dropout, batch_norm, conditioning, mu_t, sigma_t, rank, alpha, t_hidden, t_depth=3, output_activation='softplus', output_bias=True):
    """Create time-conditioned network for tabular models."""
    if conditioning == 'lora':
        bb = BaseMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=hidden_dims[-1], activation=torch.nn.Tanh, dropout=dropout, batch_norm=batch_norm)
        return TimeLoRASurvivalWrapper(bb, feature_dim=hidden_dims[-1], output_dim=1, output_activation=output_activation, output_bias=output_bias,
                                       mu=mu_t, sigma=sigma_t, time_norm_mode='min_max', rank=rank, alpha=alpha,
                                       adapter_activation=torch.nn.Tanh, t_hidden=t_hidden, t_depth=t_depth)
    elif conditioning == 'film':
        bb = BaseMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=hidden_dims[-1], activation=torch.nn.Tanh, dropout=dropout, batch_norm=batch_norm)
        return TimeFiLMSurvivalWrapper(bb, feature_dim=hidden_dims[-1], output_dim=1, output_activation=output_activation, output_bias=output_bias,
                                       mu=mu_t, sigma=sigma_t, time_norm_mode='min_max',
                                       t_hidden=t_hidden, t_depth=t_depth)
    else:  # concat
        return TimeConcatenatedMLP(input_dim=input_dim, hidden_dims=hidden_dims, activation=torch.nn.Tanh, dropout=dropout, batch_norm=batch_norm,
                                   mu=mu_t, sigma=sigma_t, time_norm_mode='min_max', output_dim=1, output_activation=output_activation, output_bias=output_bias)

def get_model(model_name, input_dim_or_channels, seed, epochs, params, discretizer=None, dataset_type='tabular', device='cpu', backbone='resnet18', pretrained=False):
    lr, batch_size, weight_decay = params['lr'], params['batch_size'], params['weight_decay']
    optimizer_class = params.get('optimizer_class', torch.optim.Adam)
    optimizer_params = {'weight_decay': weight_decay}
    t_hidden = params.get('t_hidden', 32)
    
    conditioning = params.get('conditioning', 'lora')
    rank = params.get('rank', params.get('lora_rank', 32))
    alpha = params.get('alpha', params.get('lora_alpha', rank))  # alpha should match rank by default
    t_depth = params.get('t_depth', 3)
    mu_t, sigma_t = params.get('mu_t', 0.0), params.get('sigma_t', 1.0)
    
    mkw = _model_kwargs(seed, epochs, lr, batch_size, optimizer_class, optimizer_params)
    
    if dataset_type == 'image':
        ic = input_dim_or_channels

        if model_name in ['soden', 'desurv', 'qsurv', 'qsurv_concat', 'qsurv_film']:
            # Softplus activation - network outputs positive hazard directly
            act = 'softplus'
            dropout = params.get('dropout', 0.0)
            net = _time_net_image(backbone, ic, pretrained, conditioning, mu_t, sigma_t, rank, alpha, t_hidden, t_depth, output_activation=act, dropout=dropout)
            if model_name == 'soden':
                return SODEN(net, **mkw, time_norm_mode='min_max', use_adjoint=False)
            if model_name == 'desurv':
                n_nodes = params.get('n_nodes', 15)
                return DeSurv(net, n_nodes=n_nodes, **mkw, time_norm_mode='min_max')
            n_nodes = params.get('n_nodes', 10)
            return QSurv(net, n_nodes=n_nodes, **mkw, time_norm_mode='min_max')
        
        if model_name == 'coxtime':
            dropout = params.get('dropout', 0.0)
            net = _time_net_image(backbone, ic, pretrained, conditioning, mu_t, sigma_t, rank, alpha, t_hidden, t_depth, output_activation='identity', output_bias=False, dropout=dropout)
            return CoxTime(net, n_controls=1, **mkw, time_norm_mode='min_max')
        
        if model_name == 'coxcc':
            return CoxCC(ResNetCox(model_name=backbone, input_channels=ic, pretrained=pretrained), n_controls=1, **mkw)
        if model_name == 'mdn':
            n_comps = params.get('n_components', 5)
            return MDN(ResNetMDN(model_name=backbone, input_channels=ic, n_components=n_comps, pretrained=pretrained), **mkw, time_norm_mode='min_max')
        
        if model_name in ['nnetsurv', 'deephit']:
            if discretizer is None: raise ValueError(f"Discretizer required for {model_name}")
            n_bins = len(discretizer.bin_edges_)
            if model_name == 'nnetsurv':
                return NnetSurv(ResNetNnetSurv(output_dim=n_bins, model_name=backbone, input_channels=ic, pretrained=pretrained), discretizer=discretizer, **mkw)
            return DeepHit(ResNetDeepHit(output_dim=n_bins, model_name=backbone, input_channels=ic, pretrained=pretrained), discretizer=discretizer, alpha=params.get('alpha', 0.1), sigma=params.get('sigma', 2.0), **mkw)
    
    else:  # Tabular
        hidden_dims, dropout = params['hidden_dims'], params['dropout']
        batch_norm = params.get('batch_norm', False)
        input_dim = input_dim_or_channels
        
        if model_name in ['soden', 'desurv', 'qsurv', 'qsurv_concat', 'qsurv_film']:
            # Softplus activation - network outputs positive hazard directly
            act = 'softplus'
            net = _time_net_tabular(input_dim, hidden_dims, dropout, batch_norm, conditioning, mu_t, sigma_t, rank, alpha, t_hidden, t_depth, output_activation=act)
            if model_name == 'soden':
                return SODEN(net, **mkw, time_norm_mode='min_max', use_adjoint=False)
            if model_name == 'desurv':
                n_nodes = params.get('n_nodes', 15)
                return DeSurv(net, n_nodes=n_nodes, **mkw, time_norm_mode='min_max')
            n_nodes = params.get('n_nodes', 10)
            return QSurv(net, n_nodes=n_nodes, **mkw, time_norm_mode='min_max')
        
        if model_name == 'coxtime':
            net = _time_net_tabular(input_dim, hidden_dims, dropout, batch_norm, conditioning, mu_t, sigma_t, rank, alpha, t_hidden, t_depth, output_activation='identity', output_bias=False)
            return CoxTime(net, n_controls=1, **mkw, time_norm_mode='min_max')
        
        if model_name == 'coxcc':
            return CoxCC(CoxNet(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, batch_norm=batch_norm, output_bias=False, output_activation='identity'), n_controls=1, **mkw)
        if model_name == 'mdn':
            n_comps = params.get('n_components', 5)
            return MDN(MDNNet(input_dim=input_dim, hidden_dims=hidden_dims, n_components=n_comps, dropout=dropout, batch_norm=batch_norm, mu=mu_t, sigma=sigma_t), **mkw, time_norm_mode='min_max')
        
        if model_name in ['nnetsurv', 'deephit']:
            if discretizer is None:
                raise ValueError(f"Discretizer required for {model_name}")
            # Always use discretizer's bin count for network output dim to guarantee consistency
            n_bins = len(discretizer.bin_edges_)
            if model_name == 'nnetsurv':
                return NnetSurv(NnetSurvNet(input_dim=input_dim, output_dim=n_bins, hidden_dims=hidden_dims, dropout=dropout, batch_norm=batch_norm), discretizer=discretizer, **mkw)
            return DeepHit(DeepHitNet(input_dim=input_dim, output_dim=n_bins, hidden_dims=hidden_dims, dropout=dropout, batch_norm=batch_norm), discretizer=discretizer, alpha=params.get('alpha', 0.1), sigma=params.get('sigma', 2.0), **mkw)
    
    raise ValueError(f"Unknown model: {model_name}")

def _as_np(arr):
    return arr.cpu().numpy().flatten() if isinstance(arr, torch.Tensor) else np.asarray(arr).flatten()


def _serialize_params(params):
    """JSON-safe serialization of an HPO params dict.

    Handles class objects (e.g., optimizer_class), numpy arrays/scalars, and bool (which
    must be checked before int since bool is a subclass of int).
    """
    out = {}
    if params is None:
        return out
    for k, v in params.items():
        if isinstance(v, type):
            out[k] = v.__name__
        elif isinstance(v, (bool, np.bool_)):
            out[k] = str(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _safe_call(fn, *args, fallback=np.nan, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return fallback


def compute_validation_ibs(model, val_x, val_t, val_e, train_t, train_e, discretizer, dataset_name=None):
    """Validation IBS at the train-fit censoring-KM quantile G(tau)=0.001."""
    model.eval()
    with torch.no_grad():
        val_t_np, val_e_np = _as_np(val_t), _as_np(val_e)
        train_t_np, train_e_np = _as_np(train_t), _as_np(train_e)

        # Censoring KM is fit on train to avoid leaking val censoring into the eval horizon.
        tau = float(get_tau_quantiles(train_t_np, train_e_np, (0.001,))[0])
        if tau <= 0:
            return float('inf')

        def surv_func(times_arr):
            try:
                probs = model.predict_survival_probability(val_x, times_arr)
                if isinstance(probs, torch.Tensor):
                    probs = probs.detach().cpu().numpy()
                return np.clip(probs, 0.0, 1.0)
            except Exception:
                return None

        try:
            ibs = ibs_func(train_t_np, train_e_np, val_t_np, val_e_np,
                           surv_func, tau, n_points=500, max_weight=10)
            if np.isnan(ibs) or np.isinf(ibs):
                return float('inf')
            return float(ibs)
        except Exception:
            return float('inf')


def compute_validation_cindex(model, val_x, val_t, val_e, train_t, train_e, dataset_name=None, model_name=None):
    """Validation C-index with IPCW capped via max_weight inside ipcw_uno_concordance_index."""
    model.eval()
    with torch.no_grad():
        val_t_np, val_e_np = _as_np(val_t), _as_np(val_e)
        train_t_np, train_e_np = _as_np(train_t), _as_np(train_e)

        tau = float(get_tau_quantiles(train_t_np, train_e_np, (0.001,))[0])
        if tau <= 0:
            return 0.0

        def surv_func(times_arr):
            probs = model.predict_survival_probability(val_x, times_arr)
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()
            return np.clip(probs, 0.0, 1.0)

        c_idx = ipcw_uno_concordance_index(
            train_t_np, train_e_np, val_t_np, val_e_np, surv_func, tau, max_weight=10,
        )
        if np.isnan(c_idx) or np.isinf(c_idx):
            return 0.0
        return float(c_idx)


def run_single_trial(trial_idx, seed, model_name, train_data, epochs, n_disc, device, eval_discretizer, args, dataset_type='tabular', dataset_name=None, mu_t=0.0, sigma_t=1.0):
    result = {
        'trial': trial_idx + 1,
        'params': None,
        'val_loss': float('inf'),
        'runtime': 0.0,
        'model_object': None, 'error': None
    }
    
    try:
        set_global_seed(seed * 100000 + trial_idx)
        random_state = np.random.RandomState(seed + trial_idx)
        params = sample_hyperparameters(model_name, n_disc, random_state, epochs,
                                        dataset_type=dataset_type, dataset_name=dataset_name)
        result['params'] = params
        params['mu_t'] = mu_t
        params['sigma_t'] = sigma_t

        # QSurv suffix (`_film`/`_concat`) selects the time-conditioning variant; bare 'qsurv' is LoRA.
        if 'qsurv' in model_name:
            params['conditioning'] = 'film' if 'film' in model_name else 'concat' if 'concat' in model_name else 'lora'
        elif model_name in ('coxtime', 'desurv', 'soden'):
            params['conditioning'] = 'lora'
        else:
            params['conditioning'] = 'concat'

        tr_x, tr_t, tr_e, vl_x, vl_t, vl_e = train_data
        input_dim_or_channels = tr_x.shape[1]

        # Discrete-time models use a per-trial discretizer sized by params['n_disc'];
        # all other models share eval_discretizer (only consulted for validation IBS).
        trial_discretizer = eval_discretizer
        if model_name in ('nnetsurv', 'deephit') and 'n_disc' in params:
            trial_discretizer = LabTransDiscreteTime(num_durations=params['n_disc'], scheme='quantile')
            tr_t_np = _as_np(tr_t)
            tr_e_np = _as_np(tr_e)
            trial_discretizer.fit(tr_t_np[tr_e_np.astype(bool)])

        model = get_model(model_name, input_dim_or_channels, seed, epochs, params, trial_discretizer,
                          dataset_type, device, backbone=args.backbone, pretrained=args.pretrained)
        model.to(device)

        start_time = time.time()
        model.fit(x=tr_x, t=tr_t, e=tr_e, val_x=vl_x, val_t=vl_t, val_e=vl_e)

        result['runtime'] = time.time() - start_time
        result['val_loss'] = model.best_val_loss
        val_cindex = compute_validation_cindex(model, vl_x, vl_t, vl_e, tr_t, tr_e,
                                               dataset_name=dataset_name, model_name=model_name)
        val_ibs = compute_validation_ibs(model, vl_x, vl_t, vl_e, tr_t, tr_e, eval_discretizer,
                                         dataset_name=dataset_name)
        result['val_cindex'] = val_cindex
        result['val_ibs'] = val_ibs

        # HPO selection: maximize val_cindex (primary); tie-break by minimizing val_ibs.
        # Within-trial early stopping still uses val_loss.
        result['hpo_metric'] = (val_cindex, -val_ibs) if np.isfinite(val_cindex) else (-float('inf'), -float('inf'))

        # Free GPU VRAM before joblib pickles the result back.
        model.to('cpu')
        result['model_object'] = model

    except Exception as e:
        import traceback
        traceback.print_exc()
        result['error'] = str(e)
        result['val_cindex'] = -float('inf')
        result['val_ibs'] = float('inf')
        result['hpo_metric'] = (-float('inf'), -float('inf'))

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', required=True, choices=TABULAR_DATASETS + IMAGE_DATASETS)
    parser.add_argument('--model', type=str, nargs='+', required=True, choices=RUNNABLE_MODELS + ['all'])
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone for image models (resnet18, resnet34, resnet50)')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone for image models')
    parser.add_argument('--seed', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=None, help='Override max epochs. Default 200.')
    parser.add_argument('--trials', type=int, default=None, help='Override max trials. Defaults to 30 for tabular, 20 for imaging.')
    parser.add_argument('--n_disc', type=int, default=50)
    parser.add_argument('--test_prop', type=float, default=0.2, help='Proportion of dataset for testing')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'], help='Force specific device')
    parser.add_argument('--force', action='store_true', help='Re-run even if output JSON exists (default: skip already-completed (dataset, model, seed) cells).')
    args = parser.parse_args()
    if args.seed:
        set_global_seed(args.seed[0])

    # Resilience: SIGUSR1 marks the process to exit cleanly after the current
    # (dataset, model, seed) finishes. Slurm sends SIGUSR1 N seconds before
    # walltime via `--signal=B:USR1@N`; combined with `--requeue`, the job
    # auto-restarts and the skip-if-exists guard below picks up where we left off.
    _should_stop = {'flag': False}
    def _on_sigusr1(signum, frame):
        _should_stop['flag'] = True
        print(f"[signal] SIGUSR1 received — will exit cleanly after current model.", flush=True)
    try:
        signal.signal(signal.SIGUSR1, _on_sigusr1)
    except (ValueError, OSError):
        # SIGUSR1 not available on this platform (e.g., Windows); skip silently
        pass
    
    parsed_epochs = args.epochs
    parsed_trials = args.trials

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    if 'all' in args.model:
        args.model = DEFAULT_MODELS

    # Image Transforms
    # Resizing to 224x224 for ResNet. Inputs stay in [0,1] after ToTensor;
    # no Normalize (matches the training of the saved _weights.pt files).
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for dataset_name in args.dataset:
        dtype = get_dataset_type(dataset_name)
        
        # Enforce max epochs and trials based on modality or user input
        args.epochs = parsed_epochs if parsed_epochs is not None else 200
        args.trials = parsed_trials if parsed_trials is not None else (30 if dtype == 'tabular' else 20)

        print(f"\nLoading data for {dataset_name} ({dtype})...")
        
        loader_fn = get_data_loader(dataset_name)
        
        current_train_transform = train_transform
        current_eval_transform = eval_transform

        for seed in args.seed:
            set_global_seed(seed)
            print(f"\nProcessing seed {seed}...")
            
            if dtype == 'image':
                # covid returns (train_subset, test_subset)
                train_total_ds, test_ds = loader_fn(seed=seed, test_prop=args.test_prop, train_transform=current_train_transform, eval_transform=current_eval_transform)
                # These are Subsets - access via .indices and .dataset
                train_indices = train_total_ds.indices
                orig = train_total_ds.dataset 
                train_times = np.array([orig.samples[i]['time'] for i in train_indices], dtype=float)
                train_events = np.array([orig.samples[i]['event'] for i in train_indices], dtype=float)
                test_indices = test_ds.indices
                test_orig = test_ds.dataset
                test_times = np.array([test_orig.samples[i]['time'] for i in test_indices], dtype=float)
                test_events = np.array([test_orig.samples[i]['event'] for i in test_indices], dtype=float)
                # EVAL DISCRETIZER: Fixed discretization for evaluation metrics (independent of model HPO)
                eval_discretizer = LabTransDiscreteTime(num_durations=args.n_disc, scheme='quantile')
                eval_discretizer.fit(train_times[train_events.astype(bool)])
                
                # Pre-format data for evaluation
                train_y = np.array([(e, t) for e, t in zip(train_events, train_times)], dtype=[('e', bool), ('t', float)])
                test_y = np.array([(e, t) for e, t in zip(test_events, test_times)], dtype=[('e', bool), ('t', float)])
                
                train_t_raw = train_times
                train_e_raw = train_events
                test_t_raw = test_times
                test_e_raw = test_events
                
                train_data_arg = train_total_ds # Pass DS
                test_data_arg = test_ds 
                
                # [standardized] Per-Seed Train/Val Split (Image)
                # Split train_total_ds into train (80%) and val (20%) STRATIFIED by event
                total_len = len(train_total_ds)
                
                # Stratified split
                from sklearn.model_selection import train_test_split
                orig = train_total_ds.dataset
                indices = train_total_ds.indices
                event_labels = np.array([orig.samples[i]['event'] for i in indices])
                
                all_indices = np.arange(total_len)
                # 0.25 of train_total = 20% of full → final 60/20/20 split
                try:
                    train_indices, val_indices = train_test_split(
                        all_indices, test_size=0.25, stratify=event_labels, random_state=seed
                    )
                except ValueError:
                    train_indices, val_indices = train_test_split(
                        all_indices, test_size=0.25, random_state=seed
                    )
                
                ds_train_static = Subset(train_total_ds, train_indices.tolist())
                ds_val_static = Subset(train_total_ds, val_indices.tolist())
                
                print("Extracting image features to memory to prevent I/O bottlenecks during HPO...")
                from torch.utils.data import DataLoader
                def extract_all(ds):
                    loader = DataLoader(ds, batch_size=32, shuffle=False)
                    xs, ts, es = [], [], []
                    for x, t, e in loader:
                        xs.append(x)
                        ts.append(t)
                        es.append(e)
                    return torch.cat(xs).float(), torch.cat(ts).float(), torch.cat(es).float()
                
                tr_x, tr_t, tr_e = extract_all(ds_train_static)
                vl_x, vl_t, vl_e = extract_all(ds_val_static)
                
                # Pass tuple of tensors
                train_data_arg = (tr_x, tr_t, tr_e, vl_x, vl_t, vl_e)
                
                # Calculate Normalization Stats on TRAIN SPLIT ONLY
                # Map Subset indices correctly to the original dataset
                orig_train_indices = [train_total_ds.indices[i] for i in ds_train_static.indices]
                orig_ds = train_total_ds.dataset
                train_times_static = np.array([orig_ds.samples[i]['time'] for i in orig_train_indices], dtype=float)
                
                t_min, t_max = train_times_static.min(), train_times_static.max()
                mu_t_static = 0.0  # Simple normalization: divide by t_max
                sigma_t_static = max(t_max, 1e-6)

            else:
                train_x, train_e, train_t, test_x, test_e, test_t = loader_fn(train_prop=0.8, seed=seed)
                test_t = np.clip(test_t, a_min=None, a_max=train_t.max())
                
                from sklearn.model_selection import train_test_split
                # 0.25 of train_total = 20% of full → final 60/20/20 split
                try:
                    tr_x_np, vl_x_np, tr_t_np, vl_t_np, tr_e_np, vl_e_np = train_test_split(
                        train_x, train_t, train_e, test_size=0.25, stratify=train_e, random_state=seed
                    )
                except ValueError:
                    tr_x_np, vl_x_np, tr_t_np, vl_t_np, tr_e_np, vl_e_np = train_test_split(
                        train_x, train_t, train_e, test_size=0.25, random_state=seed
                    )
                
                # Input features are already normalized by get_dataset
                
                tr_x_tensor = torch.tensor(tr_x_np).float().to(device)
                tr_t_tensor = torch.tensor(tr_t_np).float().unsqueeze(1).to(device)
                tr_e_tensor = torch.tensor(tr_e_np).float().unsqueeze(1).to(device)
                
                vl_x_tensor = torch.tensor(vl_x_np).float().to(device)
                vl_t_tensor = torch.tensor(vl_t_np).float().unsqueeze(1).to(device)
                vl_e_tensor = torch.tensor(vl_e_np).float().unsqueeze(1).to(device)
                
                # Calculate normalization stats on TRAIN SPLIT
                t_min = tr_t_tensor.min().item()
                t_max = tr_t_tensor.max().item()
                mu_t_static = 0.0  # Simple normalization: divide by t_max
                sigma_t_static = float(t_max) if t_max > 1e-6 else 1.0

                # EVAL DISCRETIZER: Fixed discretization for evaluation metrics (independent of model HPO)
                eval_discretizer = LabTransDiscreteTime(num_durations=args.n_disc, scheme='quantile')
                eval_discretizer.fit(tr_t_np)

                train_y = np.array([(e, t) for e, t in zip(train_e, train_t)], dtype=[('e', bool), ('t', float)])
                test_y = np.array([(e, t) for e, t in zip(test_e, test_t)], dtype=[('e', bool), ('t', float)])
                
                train_t_raw = train_t # Keep full for G(t) estimation or switch to tr_t_np?
                train_e_raw = train_e 
                test_t_raw = test_t
                test_e_raw = test_e
                
                train_data_arg = (tr_x_tensor, tr_t_tensor, tr_e_tensor, vl_x_tensor, vl_t_tensor, vl_e_tensor)

                
            # Loop Models
            for model_name in args.model:
                # All conditioning types (lora, film, concat) now supported for images

                # Idempotent resume: skip if final results JSON already exists.
                # Use --force to override (e.g., to re-run with new HPO settings).
                final_metrics_file = os.path.join('output', f"{dataset_name}+{model_name}+{seed:02d}_hpo.json")
                if os.path.exists(final_metrics_file) and not args.force:
                    print(f"\n[skip] {dataset_name}+{model_name}+{seed:02d}: output exists at {final_metrics_file} — pass --force to re-run.")
                    continue

                print(f"\nRunning HPO for {model_name} on {dataset_name} (seed={seed}, trials={args.trials})")

                # Execution
                start_time_total = time.time()
                # qsurv_film and soden HPO workers blow past slurm --mem with n_jobs>1 (joblib OOM)
                # Image jobs share a single --gres=gpu:1 across workers; n_jobs=2 → CUDA OOM at ~17h.
                n_jobs = 1 if (dtype == 'image' or model_name in ('qsurv_film', 'soden')) else 2
                results = Parallel(n_jobs=n_jobs, prefer="processes")(
                    delayed(run_single_trial)(
                        i, seed, model_name, train_data_arg, args.epochs, args.n_disc, device, eval_discretizer, args, dataset_type=dtype, dataset_name=dataset_name, mu_t=mu_t_static, sigma_t=sigma_t_static
                    ) for i in range(args.trials)
                )
                
                best_hpo_metric = (-float('inf'), -float('inf'))
                best_model = None
                best_params = None
                trials_log = []

                for r in results:
                    hpo_pair = r.get('hpo_metric', (-float('inf'), -float('inf')))
                    log_entry = {'trial': r['trial'], 'params': _serialize_params(r['params']),
                                 'val_loss': float(r['val_loss']) if not np.isinf(r['val_loss']) else None,
                                 'val_cindex': float(r.get('val_cindex', -float('inf'))),
                                 'val_ibs': float(r.get('val_ibs', float('inf'))),
                                 'hpo_metric_cindex': float(hpo_pair[0]) if hpo_pair[0] != -float('inf') else None,
                                 'hpo_metric_neg_ibs': float(hpo_pair[1]) if hpo_pair[1] != -float('inf') else None,
                                 'runtime': r['runtime'], 'error': r['error']}
                    trials_log.append(log_entry)
                    if r['error'] is not None:
                        print(f"    Trial {r['trial']} failed: {r['error']}")
                    if r['error'] is None and hpo_pair > best_hpo_metric:
                        best_hpo_metric = hpo_pair
                        best_model = r['model_object']
                        best_params = r['params']

                total_runtime = time.time() - start_time_total
                print(f"Total HPO Runtime: {total_runtime:.2f}s")
                print(f"Best (val_cindex, -val_ibs) = ({best_hpo_metric[0]:.4f}, {best_hpo_metric[1]:.4f})")
                print(f"Best Params: {best_params}")

                
                 # Save Logs
                log_dir = 'logs'
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"{dataset_name}+{model_name}+{seed:02d}_hpo.json")
                try:
                     with open(log_file, 'w') as f:
                          json.dump(trials_log, f, indent=4)
                except Exception as e:
                     print(f"Failed to save JSON log {log_file}: {e}")
                     
                # Evaluation
                if best_model is None:
                    print("No successful trials.")
                    continue

                # Persist weights immediately so a downstream eval crash can't lose them
                output_dir = 'output'
                os.makedirs(output_dir, exist_ok=True)
                weights_file = os.path.join(output_dir, f"{dataset_name}+{model_name}+{seed:02d}_weights.pt")
                try:
                    torch.save(best_model.state_dict(), weights_file)
                    print(f"Model weights saved to {weights_file}")
                except Exception as e:
                    print(f"Failed to save model weights to {weights_file}: {e}")

                print("Evaluating Best Model...")
                
                # Predict
                try:
                    if dtype == 'image':
                        # Extract test tensors for prediction compatibility
                        loader = DataLoader(test_data_arg, batch_size=32, shuffle=False)
                        xs = []
                        for x, t, e in loader:
                            xs.append(x)
                        test_x = torch.cat(xs).float().to(device)

                    # Recompute Cox-family baseline hazards on the full 80% train slice used by
                    # IPCW + tau. fit() left them on the 60% inner-train; without this re-fit
                    # the eval would use a different risk set than the IPCW horizon below.
                    if model_name in ('coxcc', 'coxtime') and hasattr(best_model, 'compute_baseline_hazards'):
                        if dtype == 'image':
                            # train_data_arg is a tuple (tr_x, tr_t, tr_e, vl_x, vl_t, vl_e)
                            # of pre-extracted feature tensors. Concatenate the 60% inner-train
                            # and 20% val slices in matched order so X/T/E remain row-aligned.
                            tr_x_full = torch.cat([train_data_arg[0], train_data_arg[3]], dim=0).float().to(device)
                            tr_t_full = torch.cat([train_data_arg[1], train_data_arg[4]], dim=0).float().to(device)
                            tr_e_full = torch.cat([train_data_arg[2], train_data_arg[5]], dim=0).float().to(device)
                        else:
                            tr_x_full = train_x if torch.is_tensor(train_x) else torch.from_numpy(np.asarray(train_x)).float()
                            tr_x_full = tr_x_full.float().to(device)
                            tr_t_full = torch.from_numpy(np.asarray(train_t_raw)).float().to(device)
                            tr_e_full = torch.from_numpy(np.asarray(train_e_raw)).float().to(device)
                        best_model.compute_baseline_hazards(tr_x_full, tr_t_full, tr_e_full)

                    # Create survival function wrapper for the new API
                    def surv_func(times_arr):
                        """Wrapper that returns survival probabilities at given times."""
                        with torch.no_grad():
                            probs = best_model.predict_survival_probability(test_x, times_arr)
                        if isinstance(probs, torch.Tensor):
                            probs = probs.detach().cpu().numpy()
                        return np.clip(probs, 0.0, 1.0)
                    
                    # Horizons: full (train-fit censoring-KM G(τ)=0.001) + raw observed-time quartiles.
                    # NOTE: censoring-KM is fit on TRAIN here to avoid leaking test censoring into the eval horizon.
                    full_quantile = 0.001
                    time_quantiles = (0.25, 0.5, 0.75)
                    full_tau = get_tau_quantiles(train_t_raw, train_e_raw, (full_quantile,))[0]
                    q_taus = np.quantile(np.asarray(test_t_raw).flatten(), time_quantiles)
                    taus = np.concatenate([[full_tau], q_taus])
                    tau_labels = ('full', 'Q1', 'Q2', 'Q3')

                    metrics = {}
                    metrics['taus'] = taus.tolist()
                    metrics['tau_labels'] = list(tau_labels)
                    
                    max_weight_eval = 10

                    # Per-tau metrics; event instances exceeding max_weight_eval IPCW are
                    # dropped inside ipcw_uno_concordance_index.
                    metrics['C_index'] = [
                        ipcw_uno_concordance_index(train_t_raw, train_e_raw, test_t_raw, test_e_raw,
                                                   surv_func, tau, max_weight=max_weight_eval)
                        for tau in taus
                    ]
                    metrics['IBS'] = [
                        _safe_call(ibs_func, train_t_raw, train_e_raw, test_t_raw, test_e_raw,
                                   surv_func, tau, n_points=500, max_weight=max_weight_eval)
                        for tau in taus
                    ]
                    metrics['IBLL'] = [
                        _safe_call(ibll_func, train_t_raw, train_e_raw, test_t_raw, test_e_raw,
                                   surv_func, tau, n_points=500, max_weight=max_weight_eval)
                        for tau in taus
                    ]
                    
                    # Calibration: keep D-calibration only.
                    d_cal = _safe_call(d_calibration, test_t_raw, test_e_raw, surv_func, num_bins=10,
                                       fallback=(np.nan, np.nan))
                    metrics['D_cal_stat'], metrics['D_cal_pval'] = float(d_cal[0]), float(d_cal[1])
                    
                    # Cluster survival curves using K-means (K=5, 50 equidistant points from 0 to 99% of test-max)
                    # Clusters are sorted by risk: 0=highest risk, 4=lowest risk
                    try:
                        cluster_result = cluster_survival_curves(surv_func, test_t_raw, n_clusters=5, n_points=50)
                        if cluster_result is not None:
                            metrics['clustering'] = {
                                'cluster_labels': cluster_result['cluster_labels'],
                                'cluster_centers': cluster_result['cluster_centers'],
                                'cluster_auc': cluster_result['cluster_auc'],
                                'times': cluster_result['times'],
                                'n_clusters': cluster_result['n_clusters'],
                                'n_points': cluster_result['n_points']
                            }
                            # Count samples per cluster (sorted by risk)
                            cluster_counts = [0] * cluster_result['n_clusters']
                            for label in cluster_result['cluster_labels']:
                                cluster_counts[label] += 1
                            metrics['clustering']['cluster_counts'] = cluster_counts
                            print(f"  Clustering (risk-sorted): {dict(enumerate(cluster_counts))}")
                    except Exception as e:
                        print(f"  Clustering failed: {e}")
                    
                    # Print summary
                    print(f"Results at horizons {tau_labels}:")
                    print(f"  Taus: {[f'{t:.2f}' for t in taus]}")
                    print(f"  C-index: {[f'{c:.4f}' if not np.isnan(c) else 'NaN' for c in c_indices]}")
                    print(f"  IBS: {[f'{v:.4f}' if not np.isnan(v) else 'NaN' for v in ibs_values]}")
                    print(f"  IBLL: {[f'{v:.4f}' if not np.isnan(v) else 'NaN' for v in ibll_values]}")
                    if 'D_cal_pval' in metrics and not np.isnan(metrics['D_cal_pval']):
                        print(f"  D-Cal Statistic: {metrics['D_cal_stat']:.4f}, p-value: {metrics['D_cal_pval']:.4f}")
                    
                    os.makedirs(output_dir, exist_ok=True)
                    res_dict = {
                        'dataset': dataset_name, 'model': model_name, 'seed': seed,
                        'runtime': total_runtime,
                        'best_params': _serialize_params(best_params),
                    }
                    res_dict.update(metrics)
                    
                    json_file = os.path.join(output_dir, f"{dataset_name}+{model_name}+{seed:02d}_hpo.json")
                    try:
                        with open(json_file, 'w') as f:
                            json.dump(res_dict, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x) if isinstance(x, type) else x)
                    except Exception as e:
                        print(f"Failed to save results JSON: {e}")
                    print(f"Results saved to {json_file}")

                except Exception as e:
                    import traceback
                    print(f"Evaluation failed: {e}")
                    traceback.print_exc()

                # Resilience: SIGUSR1 received → exit cleanly so Slurm requeues.
                # Skip-if-exists at the start of next run resumes from this point.
                if _should_stop['flag']:
                    print(f"[signal] Exiting cleanly after {dataset_name}+{model_name}+{seed:02d} — Slurm will requeue.", flush=True)
                    sys.exit(0)

if __name__ == '__main__':
    main()
