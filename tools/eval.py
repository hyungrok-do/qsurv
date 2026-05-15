import numpy as np
from lifelines import KaplanMeierFitter


def _event_time_floor(test_t, test_e, tau, q=0.05, eps=1e-7):
    """5th-percentile observed event time on the test set, clipped to (eps, tau)."""
    test_t = np.asarray(test_t).flatten()
    test_e = np.asarray(test_e).flatten().astype(bool)
    if not test_e.any():
        return eps
    floor = float(np.percentile(test_t[test_e], q * 100))
    floor = max(floor, eps)
    if floor >= tau:
        return eps
    return floor


def get_tau_quantiles(times, events, quantiles=(1e-4, 0.2, 0.4)):
    """Tau values where the censoring KM G(tau) hits each quantile (G fit on inverted events)."""
    times = np.asarray(times).flatten()
    events = np.asarray(events).flatten()

    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=1 - events)

    taus = []
    for q in quantiles:
        try:
            surv_func = kmf.survival_function_at_times(kmf.survival_function_.index)
            mask = surv_func.values.flatten() <= q
            tau = surv_func.index[mask][0] if mask.any() else times.max()
            taus.append(float(tau))
        except Exception:
            taus.append(float(np.quantile(times, 1 - q)))

    return np.array(taus)


def ipcw_uno_concordance_index(train_t, train_e, test_t, test_e, surv_func, tau, max_weight=10.0):
    """Time-dependent Uno C-index (Faruk et al. 2025).

    Evaluates concordance S(T_i; Z_i) < S(T_i; Z_j) at each event time T_i (not at a fixed
    tau). G is fit on train data to avoid leakage. Event instances with implied IPCW weight
    1/G(T_i)^2 > max_weight are dropped to prevent a single event in a tiny-G region from
    dominating; pass max_weight=None to disable the cap.
    """
    from lifelines import KaplanMeierFitter
    kmf_c = KaplanMeierFitter()
    kmf_c.fit(train_t, event_observed=1 - train_e)

    def G(t_vals):
        pred = kmf_c.predict(t_vals)
        return pred.values if hasattr(pred, 'values') else np.array(pred)

    event_mask = (test_e == 1) & (test_t <= tau)
    event_indices = np.where(event_mask)[0]
    if len(event_indices) == 0:
        return np.nan

    T_i_vals = test_t[event_indices]
    G_Ti = G(T_i_vals)

    eps_G = 1e-8
    if max_weight is not None and max_weight > 0:
        eps_G = max(eps_G, 1.0 / np.sqrt(max_weight))
    valid_G = G_Ti > eps_G
    event_indices = event_indices[valid_G]
    T_i_vals = T_i_vals[valid_G]
    G_Ti = G_Ti[valid_G]
    if len(event_indices) == 0:
        return np.nan

    try:
        S_matrix = surv_func(T_i_vals)                       # (N_test, len(T_i_vals))
    except Exception:
        return np.nan
    if S_matrix is None:
        return np.nan

    numerator = 0.0
    denominator = 0.0
    for k, idx_i in enumerate(event_indices):
        T_i = T_i_vals[k]
        G_weight = 1.0 / (G_Ti[k] ** 2)
        S_i = S_matrix[idx_i, k]
        mask_j = test_t > T_i
        if not np.any(mask_j):
            continue
        S_j = S_matrix[mask_j, k]
        conc = np.sum(S_i < S_j) + 0.5 * np.sum(S_i == S_j)
        numerator += conc * G_weight
        denominator += np.sum(mask_j) * G_weight

    if denominator == 0:
        return np.nan
    return float(numerator / denominator)

def _fit_censoring_km(train_t, train_e):
    kmf_c = KaplanMeierFitter()
    kmf_c.fit(train_t, event_observed=1 - train_e)

    def G(t_vals):
        pred = kmf_c.predict(t_vals)
        return pred.values if hasattr(pred, 'values') else np.array(pred)

    return G


def integrated_brier_score(train_t, train_e, test_t, test_e, surv_func, tau, n_points=10, max_weight=None):
    """IPCW Integrated Brier Score.

    G is fit on train data to avoid leakage. IPCW weights are capped by flooring G at
    1/max_weight (when set), so no subject is dropped and the survival term keeps its
    at-risk mass. Integration starts at the 5th-percentile event time so discrete-time
    models' trivial S=1 region below their first cut point does not skew the comparison.
    """
    G = _fit_censoring_km(train_t, train_e)
    t_min = _event_time_floor(test_t, test_e, tau)
    times = np.linspace(t_min, tau, n_points)

    S = surv_func(times)                                                         # (N_test, n_points)
    if S is None:
        return float('inf')
    if S.size == 0:
        return float('nan')
    S = S.T                                                                      # (M, N)

    min_G = 1.0 / max_weight if max_weight is not None else 1e-8
    G_T = np.maximum(G(test_t), min_G).reshape(1, -1)
    G_t = np.maximum(G(times), min_G).reshape(-1, 1)
    ind = (test_t.reshape(1, -1) <= times.reshape(-1, 1)).astype(float)
    labels = test_e.reshape(1, -1)

    brier = (S ** 2) * labels * ind / G_T + ((1 - S) ** 2) * (1 - ind) / G_t
    return float(brier.mean())


def integrated_binomial_log_likelihood(train_t, train_e, test_t, test_e, surv_func, tau, n_points=10, max_weight=None):
    """IPCW Integrated Binomial Log-Likelihood; same G / IPCW / integration grid as IBS."""
    G = _fit_censoring_km(train_t, train_e)
    t_min = _event_time_floor(test_t, test_e, tau)
    times = np.linspace(t_min, tau, n_points)

    S = surv_func(times)
    if S is None:
        return -float('inf')
    if S.size == 0:
        return float('nan')
    S = S.T

    min_G = 1.0 / max_weight if max_weight is not None else 1e-8
    G_T = np.maximum(G(test_t), min_G).reshape(1, -1)
    G_t = np.maximum(G(times), min_G).reshape(-1, 1)
    ind = (test_t.reshape(1, -1) <= times.reshape(-1, 1)).astype(float)
    labels = test_e.reshape(1, -1)

    bll = np.log(1 - S + 1e-10) * labels * ind / G_T + np.log(S + 1e-10) * (1 - ind) / G_t
    return float(bll.mean())


def cluster_survival_curves(surv_func, test_t, n_clusters=5, n_points=50):
    """K-means cluster survival curves on a 0..0.99*t_max grid; sort clusters by AUC ascending.

    Cluster 0 has the lowest AUC (highest risk); cluster K-1 the highest. Sorting by AUC
    keeps cluster ids semantically aligned across seeds.
    """
    from sklearn.cluster import KMeans

    times = np.linspace(1e-7, np.max(test_t) * 0.99, n_points)
    surv_curves = surv_func(times)
    if surv_curves is None:
        return None
    surv_curves = np.clip(np.nan_to_num(surv_curves, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    original_labels = kmeans.fit_predict(surv_curves)
    cluster_aucs = np.trapz(kmeans.cluster_centers_, times, axis=1)

    sorted_indices = np.argsort(cluster_aucs)
    label_mapping = {old: new for new, old in enumerate(sorted_indices)}
    sorted_labels = np.array([label_mapping[l] for l in original_labels])

    return {
        'cluster_labels': sorted_labels.tolist(),
        'cluster_centers': kmeans.cluster_centers_[sorted_indices].tolist(),
        'times': times.tolist(),
        'cluster_auc': cluster_aucs[sorted_indices].tolist(),
        'n_clusters': n_clusters,
        'n_points': n_points,
    }


def brier_score_at_times(train_t, train_e, test_t, test_e, surv_probs, times, max_weight=None):
    """IPCW Brier score per time: BS(t) = mean(W_i(t) * (I(T_i > t) - S(t|x_i))^2)."""
    G = _fit_censoring_km(train_t, train_e)
    epsilon = max(1e-7, 1.0 / max_weight) if max_weight is not None else 1e-7
    G_Ti = np.maximum(G(test_t), epsilon)

    bs_scores = []
    for k, t_eval in enumerate(times):
        S_hat = surv_probs[:, k]
        G_t = G(t_eval)
        G_t = max(G_t, epsilon) if np.ndim(G_t) == 0 else np.where(G_t == 0, epsilon, G_t)

        term1 = (test_t > t_eval) * ((1 - S_hat) ** 2) / G_t
        term2 = ((test_t <= t_eval) & (test_e == 1)) * (S_hat ** 2) / G_Ti
        bs_scores.append(np.mean(term1 + term2))
    return np.array(bs_scores)


def bll_at_times(train_t, train_e, test_t, test_e, surv_probs, times, max_weight=None):
    """IPCW Binomial Log-Likelihood per time."""
    G = _fit_censoring_km(train_t, train_e)
    epsilon = max(1e-7, 1.0 / max_weight) if max_weight is not None else 1e-7
    G_Ti = np.maximum(G(test_t), epsilon)

    bll_scores = []
    for k, t_eval in enumerate(times):
        S_hat = np.clip(surv_probs[:, k], epsilon, 1 - epsilon)
        G_t = G(t_eval)
        G_t = max(G_t, epsilon) if np.ndim(G_t) == 0 else np.where(G_t == 0, epsilon, G_t)

        term1 = (test_t > t_eval) * np.log(S_hat) / G_t
        term2 = ((test_t <= t_eval) & (test_e == 1)) * np.log(1 - S_hat) / G_Ti
        bll_scores.append(np.mean(term1 + term2))
    return np.array(bll_scores)


def concordance_at_times(train_t, train_e, test_t, test_e, surv_probs, times):
    """Time-dependent concordance per t with risk = 1 - S(t); kept for per-horizon reporting."""
    from sksurv.metrics import concordance_index_ipcw

    train_y = np.array([(bool(e), t) for e, t in zip(train_e, train_t)], dtype=[('e', bool), ('t', float)])
    test_y = np.array([(bool(e), t) for e, t in zip(test_e, test_t)], dtype=[('e', bool), ('t', float)])

    c_scores = []
    for k, t_eval in enumerate(times):
        risk_at_t = 1.0 - surv_probs[:, k]
        try:
            ct, _, _, _, _ = concordance_index_ipcw(train_y, test_y, risk_at_t, tau=t_eval)
            c_scores.append(ct)
        except Exception:
            c_scores.append(np.nan)
            
    return np.array(c_scores)


from scipy.stats import chi2


def d_calibration(test_t, test_e, surv_func, num_bins=10):
    """D-Calibration statistic and chi^2 p-value (Haider et al. 2020)."""
    test_t = np.asarray(test_t)
    test_e = np.asarray(test_e)
    n = len(test_t)

    unique_times, time_indices = np.unique(test_t, return_inverse=True)
    surv_matrix = surv_func(unique_times)
    predicted_probs = surv_matrix[np.arange(n), time_indices]

    bin_width = 1.0 / num_bins
    bin_edges = np.linspace(0, 1, num_bins + 1)
    observed_counts = np.zeros(num_bins)

    for prob, event in zip(predicted_probs, test_e):
        if event == 1:
            observed_counts[min(int(prob / bin_width), num_bins - 1)] += 1
        elif prob > 0:
            for j in range(num_bins):
                intersection = max(0, min(bin_edges[j + 1], prob) - bin_edges[j])
                observed_counts[j] += intersection / prob

    effective_n = np.sum(observed_counts)
    expected_counts = np.full(num_bins, effective_n / num_bins)
    statistic = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
    p_value = float(chi2.sf(statistic, df=num_bins - 1))
    return statistic, p_value
