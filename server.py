from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Загружаем свежую модель
MODEL_PATH = r"C:\server\plugins\FineAC\datacollection\model.pkl"
model = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # SlothAC шлёт бинарные flatbuffers, в которых идут float32 по признакам
        raw = request.data
        features = np.frombuffer(raw, dtype=np.float32)

        # Преобразуем в таблицу, где столько же колонок, сколько признаков в модели
        # (SlothAC шлёт sequence * 8 значений; наша модель обучена построчно на 8 — берём среднее)
        n_features = model.n_features_in_
        if len(features) % n_features == 0:
            # усредняем "окно" тиков до одних значений
            features = features.reshape(-1, n_features).mean(axis=0)
        else:
            # или просто отрезаем/дополняем
            features = features[:n_features]

        X = features.reshape(1, -1)

        # Probability для "читера" (класс 1)
        prob = model.predict_proba(X)[0, 1]

        return jsonify({"probability": float(prob)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)