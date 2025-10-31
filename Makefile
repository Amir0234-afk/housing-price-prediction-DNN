.PHONY: setup train predict clean

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	python -m src.train_keras --data examples/sample_input.csv --outdir models

predict:
	python -m src.predict_keras --data examples/sample_input.csv --model models/keras_model.keras --scaler models/scaler.pkl --out results/preds.csv

clean:
	rm -rf models results __pycache__ .venv
