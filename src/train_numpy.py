import argparse, os, json, numpy as np, pandas as pd
from .common import set_seeds, load_config, read_table, split_and_scale

def relu(z): return np.maximum(0, z)
def relu_deriv(z): return (z > 0).astype(float)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--outdir', default='models')
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()

    set_seeds(42)
    cfg = load_config(args.config)
    (X_train, X_test, y_train, y_test), _ = split_and_scale(
        read_table(args.data), cfg['feature_columns'], cfg['target_column'],
        test_size=cfg.get('test_size',0.2), random_state=cfg.get('random_state',42)
    )
    y_train = y_train.reshape(-1,1); y_test = y_test.reshape(-1,1)

    n_in = X_train.shape[1]; n_h1=32; n_h2=32; n_out=1
    W1 = np.random.randn(n_in, n_h1)*0.01; b1=np.zeros((1,n_h1))
    W2 = np.random.randn(n_h1, n_h2)*0.01; b2=np.zeros((1,n_h2))
    W3 = np.random.randn(n_h2, n_out)*0.01; b3=np.zeros((1,n_out))

    for epoch in range(args.epochs):
        Z1 = X_train@W1 + b1; A1 = relu(Z1)
        Z2 = A1@W2 + b2; A2 = relu(Z2)
        Z3 = A2@W3 + b3; A3 = Z3

        m = X_train.shape[0]
        loss = np.mean((A3 - y_train)**2)

        dZ3 = (A3 - y_train)
        dW3 = (A2.T @ dZ3)/m; db3 = dZ3.sum(axis=0, keepdims=True)/m
        dA2 = dZ3 @ W3.T
        dZ2 = dA2 * relu_deriv(Z2)
        dW2 = (A1.T @ dZ2)/m; db2 = dZ2.sum(axis=0, keepdims=True)/m
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_deriv(Z1)
        dW1 = (X_train.T @ dZ1)/m; db1 = dZ1.sum(axis=0, keepdims=True)/m

        W1 -= args.lr*dW1; b1 -= args.lr*db1
        W2 -= args.lr*dW2; b2 -= args.lr*db2
        W3 -= args.lr*dW3; b3 -= args.lr*db3

        if epoch % 100 == 0:
            print(f"epoch {epoch} train_mse {loss:.4f}")

    # test
    A1 = relu(X_test@W1 + b1); A2 = relu(A1@W2 + b2); A3 = A2@W3 + b3
    test_mse = float(np.mean((A3 - y_test)**2))

    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, 'numpy_weights.npz'),
             W1=W1,b1=b1,W2=W2,b2=b2,W3=W3,b3=b3)
    with open(os.path.join(args.outdir, 'numpy_metrics.json'), 'w') as f:
        json.dump({'test_mse': test_mse}, f, indent=2)
    print(json.dumps({'test_mse': test_mse}, indent=2))

if __name__ == '__main__':
    main()
