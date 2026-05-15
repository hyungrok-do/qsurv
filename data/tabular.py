"""Tabular survival dataset loaders.

Each loader returns `(train_x, train_e, train_t, test_x, test_e, test_t)`. Continuous features
are mean-imputed (with a missingness indicator) and standard-scaled; categorical features are
one-hot encoded with the first level dropped. Times and events are returned raw.

Per-dataset configuration lives in `DATASETS`. To add a dataset, append an entry there and a
thin `get_<name>` wrapper at the bottom of the file.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Framingham — requires application access; preprocessed before reaching the
# generic pipeline (cohort filter, event/time mapping, smoking fix).
# ---------------------------------------------------------------------------

_FRAMINGHAM_TIME_FOR_EVENT = {
    'ANGINA':   'TIMEAP',  'HOSPMI':   'TIMEMI',  'ANYCHD':   'TIMECHD',
    'STROKE':   'TIMESTRK', 'CVD':     'TIMECVD', 'DEATH':    'TIMEDTH',
    'HYPERTEN': 'TIMEHYP',
}


def _load_framingham_raw(target_event='CVD'):
    """Load and clean Framingham CSV; return a DataFrame with `event` and `duration` columns.

    Steps: filter to first exam (PERIOD == 1), rename the chosen event/time columns to
    event/duration, drop non-positive durations, fix CIGPDAY for non-smokers, mode-impute
    the categorical predictors so OneHotEncoder never sees NaN.
    """
    target = target_event.upper()
    if target not in _FRAMINGHAM_TIME_FOR_EVENT:
        raise ValueError(f"target_event must be one of {list(_FRAMINGHAM_TIME_FOR_EVENT)}")

    try:
        df = pd.read_csv('data/files/framingham.csv')
    except FileNotFoundError:
        raise FileNotFoundError(
            "Please obtain the Framingham dataset and place it at 'data/files/framingham.csv'"
        )

    df = df[df['PERIOD'] == 1].copy()
    df = df.rename(columns={target: 'event', _FRAMINGHAM_TIME_FOR_EVENT[target]: 'duration'})
    df = df[df['duration'] > 0]
    df.loc[df['CURSMOKE'] == 0, 'CIGPDAY'] = 0  # non-smokers can't have positive cigarettes/day

    # Mode-impute categoricals (notably `educ`) so OneHotEncoder never sees NaN.
    for col in DATASETS['framingham']['categorical']:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-dataset configuration. Optional keys:
#   csv_name: source CSV stem under data/files/ (default = dataset name)
#   rename:   {raw_col: standard_col} mapping applied before the generic pipeline
#   min_duration: minimum duration to keep (filter applied in addition to >0)
#   load_raw: custom callable returning a DataFrame with event/duration columns
# ---------------------------------------------------------------------------

DATASETS = {
    'metabric': {
        'continuous':  ['x0', 'x1', 'x2', 'x3', 'x8'],
        'categorical': ['x4', 'x5', 'x6', 'x7'],
    },
    'support2': {
        'continuous':  ['age', 'edu', 'meanbp', 'hrt', 'resp', 'temp', 'wblc', 'alb',
                        'bili', 'crea', 'sod', 'ph', 'pafi', 'num.co', 'scoma', 'adlp', 'adls'],
        'categorical': ['sex', 'race', 'income', 'diabetes', 'dementia', 'ca'],
        'rename':       {'d.time': 'duration', 'death': 'event'},
        'min_duration': 3,  # most labs were measured on day 3
    },
    'flchain': {
        'continuous':  ['age', 'creatinine', 'kappa', 'lambda'],
        'categorical': ['mgus', 'sex'],
    },
    'gbsg': {
        'csv_name':    'gbsg2',
        'continuous':  ['age', 'estrec', 'pnodes', 'tsize'],
        'categorical': ['horTh', 'menostat', 'tgrade'],
    },
    'nwtco': {
        'continuous':  ['age'],
        'categorical': ['stage', 'in.subcohort'],
        'rename':       {'rel': 'event', 'edrel': 'duration'},
    },
    'framingham': {
        'continuous':  ['AGE', 'TOTCHOL', 'HDLC', 'LDLC', 'SYSBP', 'DIABP', 'BMI',
                        'GLUCOSE', 'CIGPDAY', 'HEARTRTE'],
        'categorical': ['SEX', 'CURSMOKE', 'DIABETES', 'educ', 'BPMEDS',
                        'PREVAP', 'PREVCHD', 'PREVMI', 'PREVSTRK', 'PREVHYP'],
        'load_raw':     _load_framingham_raw,
    },
}


def _build_preprocessor(continuous_cols, categorical_cols):
    return ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean', add_indicator=True)),
            ('scaler',  StandardScaler()),
        ]), continuous_cols),
        ('cat', Pipeline([
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
        ]), categorical_cols),
    ])


def get_dataset(name, train_prop=0.7, seed=0):
    """Load `name` from DATASETS and return train/test arrays after the standard pipeline."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; known: {sorted(DATASETS)}")
    cfg = DATASETS[name]

    if 'load_raw' in cfg:
        df = cfg['load_raw']()
    else:
        df = pd.read_csv(f"data/files/{cfg.get('csv_name', name)}.csv")
    if 'rename' in cfg:
        df = df.rename(columns=cfg['rename'])

    cont, cat = cfg['continuous'], cfg['categorical']
    df = df[cont + cat + ['duration', 'event']].copy()
    df = df[df['duration'] > 0]
    if 'min_duration' in cfg:
        df = df[df['duration'] > cfg['min_duration']]
    df = df.reset_index(drop=True)

    indices = np.arange(len(df))
    train_idxs, test_idxs = train_test_split(
        indices, train_size=train_prop, stratify=df['event'].values, random_state=seed,
    )
    train_df = df.iloc[train_idxs].reset_index(drop=True)
    test_df = df.iloc[test_idxs].reset_index(drop=True)

    pre = _build_preprocessor(cont, cat)
    pre.fit(train_df[cont + cat])

    train_x = pre.transform(train_df[cont + cat]).astype(np.float32)
    test_x = pre.transform(test_df[cont + cat]).astype(np.float32)
    train_e = train_df['event'].values.astype(np.float32)
    train_t = train_df['duration'].values.astype(np.float32)
    test_e = test_df['event'].values.astype(np.float32)
    test_t = test_df['duration'].values.astype(np.float32)

    return train_x, train_e, train_t, test_x, test_e, test_t


# ---------------------------------------------------------------------------
# Per-dataset wrappers (kept as named functions so callers can import them by name).
# ---------------------------------------------------------------------------

def get_metabric(train_prop=0.7, seed=0):    return get_dataset('metabric',   train_prop, seed)
def get_support2(train_prop=0.7, seed=0):    return get_dataset('support2',   train_prop, seed)
def get_flchain(train_prop=0.7, seed=0):     return get_dataset('flchain',    train_prop, seed)
def get_gbsg(train_prop=0.7, seed=0):        return get_dataset('gbsg',       train_prop, seed)
def get_nwtco(train_prop=0.7, seed=0):       return get_dataset('nwtco',      train_prop, seed)
def get_framingham(train_prop=0.7, seed=0):  return get_dataset('framingham', train_prop, seed)
