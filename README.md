# Housing Price Regression — Keras and NumPy baselines

Public MIT repository. Users provide their own tabular dataset and obtain predictions.
No private data is included.

## Features
- Keras DNN regressor with saved `scaler.pkl`, `.keras` model, `metrics.json`, `training_history.png`.
- Pure NumPy 2-hidden-layer baseline.
- CSV or Excel input. Schema is fixed and documented (see `src/data_schema.md`).

## Quickstart
```bash
# 1) install
#windows
.venv\Scripts\Activate.ps1
#linux
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2) prepare data
# Create your file matching the schema. See examples/sample_input.csv
# If you have Excel: keep sheet with these headers or export to CSV.

# 3) train Keras
python -m src.train_keras --data path/to/your.xlsx --outdir models

# 4) predict with trained Keras model
python -m src.predict_keras --data path/to/new_rows.csv --model models/keras_model.keras --scaler models/scaler.pkl --out results/preds.csv

# 5) train NumPy baseline
python -m src.train_numpy --data path/to/your.xlsx --outdir models
```

## Reproducibility
- Deterministic seeds set for NumPy, Python, and TensorFlow.
- Requirements pinned. See `requirements.txt`.
- Metrics and config saved to `models/`.

## License
MIT. See `LICENSE`.
