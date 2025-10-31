import argparse, os, pandas as pd, numpy as np
from joblib import load
from .common import load_config, read_table, split_and_scale

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='CSV/Excel with feature columns only')
    p.add_argument('--model', required=True, help='Path to .keras model file')
    p.add_argument('--scaler', required=True, help='Path to scaler.pkl')
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--out', default='results/predictions.csv')
    args = p.parse_args()

    cfg = load_config(args.config)
    df = read_table(args.data)

    # Scale using the provided scaler
    X = df[cfg['feature_columns']].copy()
    from .common import coerce_bool
    for col in cfg['feature_columns']:
        if 'bool' in col.lower():
            X[col] = coerce_bool(X[col])
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.dropna()
    scaler = load(args.scaler)
    Xs = scaler.transform(X.values)

    from tensorflow import keras
    model = keras.models.load_model(args.model)
    preds = model.predict(Xs, verbose=0).flatten()

    out_df = df.loc[X.index].copy()
    out_df['predicted_price_Mil_per_m2'] = preds
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out_df)} rows.")

if __name__ == '__main__':
    main()
