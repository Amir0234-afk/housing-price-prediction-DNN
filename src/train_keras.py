import os, json, argparse, matplotlib.pyplot as plt
from joblib import dump
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .common import set_seeds, load_config, read_table, split_and_scale

def build_model(n_features: int):
    return keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='CSV or Excel file with target column')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--outdir', default='models')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    set_seeds(42)
    cfg = load_config(args.config)
    df = read_table(args.data)
    (X_train, X_test, y_train, y_test), scaler = split_and_scale(
        df, cfg['feature_columns'], cfg['target_column'],
        test_size=cfg.get('test_size',0.2), random_state=cfg.get('random_state',42)
    )

    os.makedirs(args.outdir, exist_ok=True)
    model = build_model(X_train.shape[1])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    cb = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)]
    hist = model.fit(X_train, y_train, validation_split=0.2, epochs=args.epochs, batch_size=args.batch, verbose=1, callbacks=cb)

    # Evaluate
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0).flatten()
    from sklearn.metrics import r2_score
    r2 = float(r2_score(y_test, y_pred))

    # Save artifacts
    model_path = os.path.join(args.outdir, 'keras_model.keras')
    scaler_path = os.path.join(args.outdir, 'scaler.pkl')
    hist_png = os.path.join(args.outdir, 'training_history.png')
    metrics_json = os.path.join(args.outdir, 'metrics.json')

    model.save(model_path)
    dump(scaler, scaler_path)

    plt.figure(figsize=(10,6))
    plt.plot(hist.history['loss'], label='train_loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.xlabel('epoch'); plt.ylabel('mse'); plt.legend(); plt.title('Keras training history')
    plt.savefig(hist_png, bbox_inches='tight')

    with open(metrics_json,'w') as f:
        json.dump({'loss_mse': float(loss), 'mae': float(mae), 'r2': r2}, f, indent=2)

    print(f"Saved: {model_path}\nSaved: {scaler_path}\nSaved: {hist_png}\nSaved: {metrics_json}")
    print(json.dumps({'loss_mse': float(loss), 'mae': float(mae), 'r2': r2}, indent=2))

if __name__ == '__main__':
    main()
