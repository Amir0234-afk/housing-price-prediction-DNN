import argparse, json, os, sys, random, yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def set_seeds(seed: int = 42):
    import os, random, numpy as _np
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    _np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except Exception:
        pass

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith('.csv'):
        return pd.read_csv(path)
    elif path.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def coerce_bool(s):
    if s.dtype == bool:
        return s.astype(int)
    return s.astype(str).str.strip().str.lower().map({
        '1':1,'0':0,'true':1,'false':0,'yes':1,'no':0,'y':1,'n':0
    }).fillna(pd.to_numeric(s, errors='coerce')).fillna(0).astype(float)

def split_and_scale(df, features, target, test_size=0.2, random_state=42):
    X = df[features].copy()
    for col in features:
        if 'bool' in col.lower():
            X[col] = coerce_bool(X[col])
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    if target in df.columns:
        y = pd.to_numeric(df[target], errors='coerce')
    else:
        y = None
    data = pd.concat([X, y], axis=1) if y is not None else X
    data = data.dropna()
    X = data[features].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if y is None:
        return Xs, scaler, None, None, None, None
    y = data[target].values
    return train_test_split(Xs, y, test_size=test_size, random_state=random_state), scaler
