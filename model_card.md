# Model Card — Housing Price Regression

## Intended Use
Educational baselines for tabular regression. Not for financial advice or production use.

## Inputs
See `src/data_schema.md`. Data must be numeric or coerceable to numeric; booleans become 0/1.

## Metrics
- Keras: reports MSE, MAE, R² on held-out test split.
- NumPy: reports MSE.

## Limitations
- Sensitive to outliers. Consider robust scaling and outlier filtering if needed.
- No bias or fairness analysis.

## Security and Privacy
- Do not commit private datasets. This repo ships no data.
