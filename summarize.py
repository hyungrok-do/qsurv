import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Data Imports
# Ensure we can import from current directory
sys.path.append(os.getcwd())
from data.tabular import get_metabric, get_support2, get_flchain, get_gbsg, get_nwtco, get_framingham
from data.covid.dataset import get_covid_dataset
from data.brats.dataset import get_brats_dataset

def get_data_loader(dataset_name):
    if dataset_name == 'metabric': return get_metabric
    elif dataset_name == 'support2': return get_support2
    elif dataset_name == 'flchain': return get_flchain
    elif dataset_name == 'gbsg': return get_gbsg
    elif dataset_name == 'nwtco': return get_nwtco
    elif dataset_name == 'framingham': return get_framingham
    elif dataset_name == 'covid': return get_covid_dataset
    elif dataset_name == 'brats': return get_brats_dataset
    else:
        return None

def get_dataset_max_time(dataset_name):
    loader = get_data_loader(dataset_name)
    if loader is None:
        return np.nan
    
    try:
        # Check if it's image based on name seems safest shortcut given get_dataset_type exists in main but not here
        # But we can just try/except blocks or check return type
        
        # Call loader
        # Covid dataset
        # covid has all defaults
        data = loader()
        
        if isinstance(data, tuple) and len(data) == 2:
             # Likely Image (train_ds, test_ds)
             # Check if Subset or Dataset
             train_ds = data[0]
             
             # Access underlying dataset
             if hasattr(train_ds, 'dataset'):
                 ds = train_ds.dataset
             else:
                 ds = train_ds
                 
             if hasattr(ds, 'samples'):
                 times = [s['time'] for s in ds.samples]
                 if times:
                     return max(times)
             
        elif isinstance(data, tuple) and len(data) >= 6:
            # Tabular: train_x, train_e, train_o, test_x, test_e, test_o
            train_t = data[2]
            test_t = data[5]
            # Ensure numpy
            if hasattr(train_t, 'numpy'): train_t = train_t.numpy()
            if hasattr(test_t, 'numpy'): test_t = test_t.numpy()
            
            return float(max(np.max(train_t), np.max(test_t)))
            
    except Exception as e:
        # print(f"Error calculating Tau for {dataset_name}: {e}")
        return np.nan
    return np.nan


def visualize_improvements(df_all):
    datasets = df_all['dataset'].unique()
    non_metric_cols = ['dataset', 'model', 'seed', 'runtime', 'best_params', 'Tau']
    metrics = [c for c in df_all.columns if c not in non_metric_cols]
    
    def metric_sort_key(m):
        parts = m.split('_')
        base = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""
        return (base, suffix)
        
    metrics.sort(key=metric_sort_key)
    
    if not metrics:
        return

    n_datasets = len(datasets)
    n_metrics = len(metrics)
    
    # Define friendly names globally for plotting
    friendly_names = {'coxcc': 'CoxCC', 'coxtime': 'CoxTime', 'deephit': 'DeepHit', 'nnetsurv': 'NnetSurv', 'mdn': 'MDN', 'soden': 'SODEN', 'desurv': 'DeSurv', 'qsurv_concat': 'Q-Surv (Concat)', 'qsurv': 'Q-Surv (LoRA)', 'qsurv_film': 'Q-Surv (FiLM)'}

    # Aesthetic Setup
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Sort models by preferred order
    unique_models = df_all['model'].unique().tolist()
    preferred_order = ['coxcc', 'coxtime', 'deephit', 'nnetsurv', 'mdn', 'soden', 'desurv', 'qsurv_concat', 'qsurv', 'qsurv_film']
    models_sorted = [m for m in preferred_order if m in unique_models] + [m for m in unique_models if m not in preferred_order]
    
    # Create mapped preferred order
    mapped_preferred_order = [friendly_names.get(m, m) for m in models_sorted]
    
    palette = sns.color_palette("husl", len(models_sorted))
    color_map = {friendly_names.get(m, m): c for m, c in zip(models_sorted, palette)}

    if n_metrics == 0:
        return
        
    fig, axes = plt.subplots(n_datasets, n_metrics, figsize=(4 * n_metrics + 2, 5 * n_datasets), squeeze=False)
    
    for i, dataset in enumerate(datasets):
        df_ds = df_all[df_all['dataset'] == dataset]
        
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            records = []
            
            for model in models_sorted:
                m_data = df_ds[df_ds['model'] == model].set_index('seed')
                m_vals = m_data[metric].dropna()
                
                for val in m_vals:
                    records.append({
                        'Model': friendly_names.get(model, model),
                        'Value': val
                    })
            
            if not records:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue
                
            plot_df = pd.DataFrame(records)
            
            # Filter order to only include models present in this plot
            plot_order = [m for m in mapped_preferred_order if m in plot_df['Model'].unique()]
            
            if plot_df.empty or not plot_order:
                 ax.text(0.5, 0.5, "No Data (Empty DF)", ha='center', va='center')
                 continue

            try:
                sns.boxplot(data=plot_df, x='Model', y='Value', hue='Model', ax=ax, palette=color_map, order=plot_order,
                            legend=False, showfliers=False, width=0.6, boxprops=dict(alpha=0.8), medianprops=dict(color='black', linewidth=1.5))
            except Exception as e:
                try:
                    sns.boxplot(data=plot_df, x='Model', y='Value', hue='Model', ax=ax, legend=False, showfliers=False)
                except Exception as e2:
                    ax.text(0.5, 0.5, f"Plot Failed: {e}", ha='center', va='center')
            
            # Titles and Labels
            metric_title = metric.upper()
            if i == 0:
                ax.set_title(f"{metric_title}", fontsize=14, fontweight='bold', pad=15)
            else:
                 ax.set_title(f"{metric_title}", fontsize=12)
                 
            if j == 0:
                ax.set_ylabel(f"{dataset.upper()}", fontsize=14, fontweight='bold', labelpad=10)
            else:
                ax.set_ylabel("")
                
            ax.set_xlabel("")
            ax.tick_params(axis='x', rotation=45)
            
            # Improve tick labels font
            for label in ax.get_xticklabels():
                label.set_fontsize(10)
                
    plt.tight_layout()
    # Add a global title
    fig.suptitle("Absolute Performance Metrics Comparison", fontsize=18, fontweight='bold', y=1.02)

    
    output_path = 'output/summary_boxplot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Boxplots saved to {output_path}")


def visualize_tail_behaviors(output_dir='output'):
    """
    Create line plots showing metrics across ALL time points (full grid) for each dataset.
    Highlights QSurv models with thicker lines and distinct colors.
    """
    import json
    
    # Collect all JSON files
    json_files = [f for f in os.listdir(output_dir) if f.endswith('_hpo.json')]
    
    if not json_files:
        print("No JSON files found for tail behavior visualization.")
        return
    
    # Parse all data - store full grids
    all_grids = {}  # {(dataset, model, metric): list of grid arrays}
    
    for jf in json_files:
        try:
            parts = jf.replace('_hpo.json', '').split('+')
            if len(parts) != 3:
                continue
            dataset, model, seed = parts
            
            with open(os.path.join(output_dir, jf)) as f:
                jd = json.load(f)
            
            if 'C_grid' not in jd or len(jd['C_grid']) == 0:
                continue
            
            metrics_grids = {
                'C': 'C_grid',
                'BS': 'BS_grid', 
                'BLL': 'BLL_grid',
            }
            
            for metric_name, grid_key in metrics_grids.items():
                if grid_key in jd:
                    key = (dataset, model, metric_name)
                    if key not in all_grids:
                        all_grids[key] = []
                    all_grids[key].append(np.array(jd[grid_key]))
        except Exception as e:
            pass
    
    if not all_grids:
        print("No valid data for tail behavior visualization.")
        return
    # Get unique datasets and metrics (exclude covid only)
    exclude_datasets = {'covid'}
    datasets = sorted([d for d in set(k[0] for k in all_grids.keys()) if d not in exclude_datasets])
    metrics = ['C', 'BS', 'BLL']
    metric_labels = {'C': 'C-index (↑)', 'BS': 'Brier Score (↓)', 'BLL': 'Log-Likelihood (↑)'}
    
    # Model order - QSurv models last to be drawn on top
    model_order = ['coxcc', 'coxtime', 'deephit', 'nnetsurv', 'mdn', 'desurv', 'qsurv_concat', 'qsurv']
    qsurv_models = {'qsurv_concat', 'qsurv'}
    
    # Colors: muted for baselines, blue tones for QSurv
    baseline_palette = sns.color_palette("Set2", 6)
    qsurv_colors = {'qsurv_concat': '#1E88E5', 'qsurv': '#0D47A1'}  # Light blue and Dark blue
    
    # Color map with friendly names
    color_map = {}
    for i, m in enumerate(model_order):
        name = friendly_names.get(m, m)
        if m in qsurv_models:
            color_map[name] = qsurv_colors[m]
        else:
            color_map[name] = baseline_palette[i % len(baseline_palette)]
    
    # Create figure
    plt.style.use('seaborn-v0_8-whitegrid')
    n_datasets = len(datasets)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(n_datasets, n_metrics, figsize=(4 * n_metrics, 2.5 * n_datasets), squeeze=False)
    
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            # Find all models for this dataset+metric
            models_in_ds = [m for m in model_order if (dataset, m, metric) in all_grids]
            
            # Plot baselines first (thinner, more transparent)
            for model in models_in_ds:
                if model in qsurv_models:
                    continue
                    
                grids = all_grids[(dataset, model, metric)]
                if not grids:
                    continue
                
                # Stack and compute mean/std across seeds
                # Handle different grid lengths by using only grids with matching length
                grid_lengths = [len(g) for g in grids]
                if not grid_lengths:
                    continue
                most_common_len = max(set(grid_lengths), key=grid_lengths.count)
                matching_grids = [g for g in grids if len(g) == most_common_len]
                if not matching_grids:
                    continue
                stacked = np.vstack([g.reshape(1, -1) for g in matching_grids])
                mean = np.nanmean(stacked, axis=0)
                std = np.nanstd(stacked, axis=0)
                
                x = np.arange(len(mean))
                fname = friendly_names.get(model, model)
                color = color_map.get(fname, 'gray')
                
                ax.plot(x, mean, label=fname, color=color, linewidth=1.5, alpha=0.8)
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
            
            # Plot QSurv models on top (thicker, more visible)
            for model in models_in_ds:
                if model not in qsurv_models:
                    continue
                    
                grids = all_grids[(dataset, model, metric)]
                if not grids:
                    continue
                
                # Handle different grid lengths
                grid_lengths = [len(g) for g in grids]
                if not grid_lengths:
                    continue
                most_common_len = max(set(grid_lengths), key=grid_lengths.count)
                matching_grids = [g for g in grids if len(g) == most_common_len]
                if not matching_grids:
                    continue
                stacked = np.vstack([g.reshape(1, -1) for g in matching_grids])
                mean = np.nanmean(stacked, axis=0)
                std = np.nanstd(stacked, axis=0)
                
                x = np.arange(len(mean))
                fname = friendly_names.get(model, model)
                color = color_map.get(fname, 'red')
                
                ax.plot(x, mean, label=fname, color=color, linewidth=1.5, alpha=0.8)
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
            
            # Formatting
            ax.set_xlabel('Time Quantile' if i == n_datasets - 1 else '', fontsize=9)
            
            if i == 0:
                ax.set_title(metric_labels[metric], fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(dataset.upper(), fontsize=11, fontweight='bold')
            
            # Legend on last column of first row
            if i == 0 and j == n_metrics - 1:
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    fig.suptitle("Tail Behavior Analysis: Full Grid Metrics (QSurv Highlighted)", fontsize=14, fontweight='bold', y=1.02)
    
    output_path = 'output/tail_behaviors.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Tail behavior plot saved to {output_path}")

def visualize_clustering():
    """Visualize clustering results from JSON files, one plot per seed."""
    import json
    
    output_dir = 'output'
    if not os.path.exists(output_dir):
        print("Output directory not found.")
        return
    
    # Collect clustering data
    clustering_data = {}  # {(dataset, model, seed): clustering_dict}
    
    for f in os.listdir(output_dir):
        if not f.endswith('_hpo.json'):
            continue
        try:
            parts = f.replace('_hpo.json', '').split('+')
            if len(parts) != 3:
                continue
            dataset, model, seed = parts
            
            with open(os.path.join(output_dir, f)) as fp:
                data = json.load(fp)
            
            if 'clustering' in data and data['clustering']:
                clustering_data[(dataset, model, int(seed))] = data['clustering']
        except Exception:
            pass
    
    if not clustering_data:
        print("No clustering data found.")
        return
    
    # Get unique datasets and models
    datasets = sorted(set(k[0] for k in clustering_data.keys()))
    models = sorted(set(k[1] for k in clustering_data.keys()))
    
    # Define friendly names globally for plotting
    friendly_names = {'coxcc': 'CoxCC', 'coxtime': 'CoxTime', 'deephit': 'DeepHit', 'nnetsurv': 'NnetSurv', 'mdn': 'MDN', 'soden': 'SODEN', 'desurv': 'DeSurv', 'qsurv_concat': 'Q-Surv (Concat)', 'qsurv': 'Q-Surv (LoRA)', 'qsurv_film': 'Q-Surv (FiLM)'}
    
    # Color palette for clusters
    cluster_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    
    for dataset in datasets:
        # Get all seeds for this dataset
        seeds_in_ds = sorted(set(k[2] for k in clustering_data.keys() if k[0] == dataset))
        
        for seed in seeds_in_ds:
            # Get models with data for this dataset+seed
            models_with_data = [m for m in models if (dataset, m, seed) in clustering_data]
            if not models_with_data:
                continue
            
            n_models = len(models_with_data)
            fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4), squeeze=False)
            fig.suptitle(f'{dataset.upper()} - Seed {seed}: Cluster Centers', fontsize=12, fontweight='bold')
            
            for idx, model in enumerate(models_with_data):
                ax = axes[0, idx]
                clust = clustering_data[(dataset, model, seed)]
                
                times = np.array(clust.get('times', []))
                centers = clust.get('cluster_centers', [])
                counts = clust.get('cluster_counts', [])
                
                if len(centers) == 0 or len(times) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(model)
                    continue
                
                # Plot each cluster's predicted survival curve
                for c_idx, (center, count) in enumerate(zip(centers, counts)):
                    color = cluster_colors[c_idx % len(cluster_colors)]
                    ax.plot(times, center, color=color, linewidth=2, 
                           label=f'Cluster {c_idx} (n={count})')
                
                ax.set_xlim(times[0], times[-1])
                ax.set_ylim(0, 1.05)
                ax.set_xlabel('Time')
                ax.set_ylabel('Survival Probability' if idx == 0 else '')
                fname = friendly_names.get(model, model)
                ax.set_title(fname, fontweight='bold')
                ax.legend(loc='lower left', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = f'output/clustering_{dataset}_seed{seed:02d}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved {output_path}")

def summarize_results():
    import json
    
    output_dir = 'output'
    if not os.path.exists(output_dir):
        print("Output directory not found.")
        return

    # Collect data for all horizons: full + Q1/Q2/Q3 of raw observed times
    data_full = []
    data_q1 = []
    data_q2 = []
    data_q3 = []
    
    active_models = {'coxcc', 'coxtime', 'deephit', 'nnetsurv', 'mdn',
                     'soden', 'desurv', 'qsurv_concat', 'qsurv', 'qsurv_film'}
    active_datasets = {'metabric', 'support2', 'flchain', 'gbsg', 'nwtco',
                       'framingham', 'covid', 'brats', 'c4kc_kits'}

    # Track which seeds we've processed
    processed = set()
    
    # Primary: Read from JSON files
    json_files = [f for f in os.listdir(output_dir) if f.endswith('_hpo.json')]
    
    for jf in json_files:
        try:
            parts = jf.replace('_hpo.json', '').split('+')
            if len(parts) != 3:
                continue
            dataset, model, seed = parts
            if dataset not in active_datasets or model not in active_models:
                continue
            key = (dataset, model, seed)
            
            with open(os.path.join(output_dir, jf)) as f:
                jd = json.load(f)
            
            c_index = jd.get('C_index', [])
            ibs = jd.get('IBS', [])
            ibll = jd.get('IBLL', [])

            # Eps-truncated metrics (legacy field; parallel arrays at eps in {0.0, 0.2, 0.4})
            ibs_eps = jd.get('IBS_eps', [])
            ibll_eps = jd.get('IBLL_eps', [])

            def _at(arr, i):
                return arr[i] if isinstance(arr, list) and len(arr) > i else None

            if isinstance(c_index, list) and len(c_index) >= 4:
                processed.add(key)

                # index 0 = full (censoring-KM G(τ)=0.1)
                data_full.append({
                    'dataset': dataset, 'model': model, 'seed': int(seed),
                    'C_index': c_index[0], 'IBS': _at(ibs, 0), 'IBLL': _at(ibll, 0),
                    # Eps-truncated columns (legacy; only present in older JSONs)
                    'IBS_eps0': _at(ibs_eps, 0), 'IBS_eps02': _at(ibs_eps, 1), 'IBS_eps04': _at(ibs_eps, 2),
                    'IBLL_eps0': _at(ibll_eps, 0), 'IBLL_eps02': _at(ibll_eps, 1), 'IBLL_eps04': _at(ibll_eps, 2),
                    'D_cal_stat': jd.get('D_cal_stat'), 'D_cal_pval': jd.get('D_cal_pval'),
                    'runtime': jd.get('runtime', np.nan)
                })

                # indices 1/2/3 = Q1/Q2/Q3 of raw observed-time quantiles
                for idx, bucket in [(1, data_q1), (2, data_q2), (3, data_q3)]:
                    bucket.append({
                        'dataset': dataset, 'model': model, 'seed': int(seed),
                        'C_index': _at(c_index, idx),
                        'IBS': _at(ibs, idx),
                        'IBLL': _at(ibll, idx),
                        'D_cal_stat': np.nan, 'D_cal_pval': np.nan,
                        'runtime': jd.get('runtime', np.nan)
                    })
        except Exception as e:
            pass  # Skip corrupted JSON

    if not data_q1:
        print("No results found.")
        return

    def print_combined_table(df_full, df_q1, df_q2, df_q3):
        """Print summary tables organized by dataset, showing all horizons together."""
        model_order = ['coxcc', 'coxtime', 'deephit', 'nnetsurv', 'mdn', 'soden', 'desurv', 'qsurv_concat', 'qsurv', 'qsurv_film']
        
        # Key metrics to show
        metrics = ['C_index', 'IBS', 'IBLL']
        metric_labels = {'C_index': 'C-idx ↑', 'IBS': 'IBS ↓', 'IBLL': 'IBLL ↑'}
        
        # Collect all datasets from all dataframes
        all_datasets = set()
        for df in [df_full, df_q1, df_q2, df_q3]:
            if df is not None and not df.empty:
                all_datasets.update(df['dataset'].unique())
        datasets = sorted(all_datasets)

        for dataset in datasets:
            print("\n" + "="*100)
            print(f"  {dataset.upper()}")
            print("="*100)

            horizons = [
                ('Full (censoring-KM G(τ)=0.1)', df_full),
                ('Q1 (raw observed-time 25%)', df_q1),
                ('Q2 / median (raw observed-time 50%)', df_q2),
                ('Q3 (raw observed-time 75%)', df_q3),
            ]
            
            for horizon_name, df in horizons:
                if df is None or df.empty:
                    continue
                df_ds = df[df['dataset'] == dataset]
                if df_ds.empty:
                    continue
                
                print(f"\n  {horizon_name}")
                print("-" * 96)
                
                # Header row
                header = f"  {'Model':<12}"
                for metric in metrics:
                    header += f"  {metric_labels[metric]:^16}"
                header += "   N"
                print(header)
                
                # Get models present in this dataset
                models_present = [m for m in model_order if m in df_ds['model'].unique()]
                
                for model in models_present:
                    df_model = df_ds[df_ds['model'] == model]
                    n_samples = len(df_model)
                    
                    row = f"  {model:<12}"
                    for metric in metrics:
                        values = pd.to_numeric(df_model[metric], errors='coerce')
                        mean = values.mean()
                        std = values.std()
                        
                        if pd.isna(mean):
                            row += f"  {'—':^16}"
                        else:
                            row += f"  {mean:.4f} ± {std:.4f} "
                    
                    row += f"  {n_samples}"
                    print(row)
    
    # Build DataFrames
    df_full = pd.DataFrame(data_full) if data_full else None
    df_q1 = pd.DataFrame(data_q1) if data_q1 else None
    df_q2 = pd.DataFrame(data_q2) if data_q2 else None
    df_q3 = pd.DataFrame(data_q3) if data_q3 else None

    # Print combined table
    print_combined_table(df_full, df_q1, df_q2, df_q3)
    
    # Save aggregated results to CSV
    def save_aggregated_csv(df, horizon_name, output_path):
        if df is None or df.empty:
            return
        metrics = ['C_index', 'IBS', 'IBLL']
        rows = []
        for dataset in df['dataset'].unique():
            df_ds = df[df['dataset'] == dataset]
            for model in df_ds['model'].unique():
                df_model = df_ds[df_ds['model'] == model]
                row = {'dataset': dataset, 'model': model, 'N': len(df_model)}
                for m in metrics:
                    vals = pd.to_numeric(df_model[m], errors='coerce')
                    row[f'{m}_mean'] = vals.mean()
                    row[f'{m}_std'] = vals.std()
                rows.append(row)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Saved {output_path}")
    
    # Create single unified CSV with all horizons
    def save_unified_csv(df_full, df_q1, df_q2, df_q3, output_path):
        metrics = ['C_index', 'IBS', 'IBLL',
                   'IBS_eps0', 'IBS_eps02', 'IBS_eps04',
                   'IBLL_eps0', 'IBLL_eps02', 'IBLL_eps04',
                   'runtime', 'D_cal_stat', 'D_cal_pval']
        horizons = [('full', df_full), ('Q1', df_q1), ('Q2', df_q2), ('Q3', df_q3)]
        
        # Collect all unique dataset-model pairs
        all_pairs = set()
        for _, df in horizons:
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    all_pairs.add((row['dataset'], row['model']))
        
        rows = []
        for dataset, model in sorted(all_pairs):
            row = {'dataset': dataset, 'model': model}
            
            for hz_name, df in horizons:
                if df is None or df.empty:
                    for m in metrics:
                        row[f'{m}_{hz_name}_mean'] = np.nan
                        row[f'{m}_{hz_name}_std'] = np.nan
                    row[f'N_{hz_name}'] = 0
                    continue
                    
                df_match = df[(df['dataset'] == dataset) & (df['model'] == model)]
                if df_match.empty:
                    for m in metrics:
                        row[f'{m}_{hz_name}_mean'] = np.nan
                        row[f'{m}_{hz_name}_std'] = np.nan
                    row[f'N_{hz_name}'] = 0
                else:
                    row[f'N_{hz_name}'] = len(df_match)
                    for m in metrics:
                        if m not in df_match.columns:
                            row[f'{m}_{hz_name}_mean'] = np.nan
                            row[f'{m}_{hz_name}_std'] = np.nan
                            continue
                        vals = pd.to_numeric(df_match[m], errors='coerce')
                        row[f'{m}_{hz_name}_mean'] = vals.mean()
                        row[f'{m}_{hz_name}_std'] = vals.std()
            rows.append(row)
        
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Saved {output_path}")
    
    save_unified_csv(df_full, df_q1, df_q2, df_q3, 'output/summary_all_horizons.csv')
    
    # Generate visualizations (using df_full which has the most complete data)
    # visualize_improvements(df_full if df_full is not None and not df_full.empty else df_20)
    
    # Visualize clustering results
    # visualize_clustering()

if __name__ == "__main__":
    summarize_results()
