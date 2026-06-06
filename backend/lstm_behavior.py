"""
Actual LSTM behavior predictor for learning trends.

This module predicts:
- improving
- stable
- struggling

using a real LSTM network (TensorFlow/Keras) over recent score sequences.
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np


LABEL_TO_ID = {"struggling": 0, "stable": 1, "improving": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


class LSTMBehaviorModel:
    def __init__(self, model_path: str, window_size: int = 3):
        self.model_path = model_path
        self.window_size = window_size
        self._model = None
        self._tf_ready = False

        # Lazy TensorFlow import so backend still runs without TF.
        try:
            import tensorflow as tf  # type: ignore

            self.tf = tf
            self._tf_ready = True
            if os.path.exists(self.model_path):
                self._model = tf.keras.models.load_model(self.model_path)
        except Exception:
            self.tf = None
            self._tf_ready = False
            self._model = None

    @staticmethod
    def _heuristic_label(sequence: List[float]) -> str:
        # Used only to create weak-supervision labels for initial training.
        if len(sequence) < 2:
            return "stable"
        diffs = [sequence[i] - sequence[i - 1] for i in range(1, len(sequence))]
        avg_diff = float(sum(diffs) / len(diffs))
        if avg_diff >= 10:
            return "improving"
        if avg_diff <= -10:
            return "struggling"
        return "stable"

    def _build_model(self):
        tf = self.tf
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.window_size, 1)),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _make_dataset(self, histories: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        X = []
        y = []
        for h in histories:
            if len(h) < self.window_size:
                continue
            for i in range(0, len(h) - self.window_size + 1):
                window = h[i : i + self.window_size]
                label = self._heuristic_label(window)
                X.append(window)
                y.append(LABEL_TO_ID[label])

        if not X:
            return np.array([]), np.array([])
        X = np.array(X, dtype=np.float32).reshape(-1, self.window_size, 1)
        y = np.array(y, dtype=np.int32)
        return X, y

    def train_if_possible(self, histories: List[List[float]], min_samples: int = 20) -> bool:
        if not self._tf_ready:
            return False

        X, y = self._make_dataset(histories)
        if len(X) < min_samples:
            return False

        if self._model is None:
            self._model = self._build_model()

        self._model.fit(X, y, epochs=20, batch_size=8, verbose=0)
        self._model.save(self.model_path)
        return True

    def predict(self, score_history: List[float]) -> str:
        # If TF/model unavailable, caller should fallback.
        if not self._tf_ready or self._model is None:
            raise RuntimeError("LSTM model not available")
        if len(score_history) < self.window_size:
            raise ValueError("Not enough history for LSTM")

        window = np.array(score_history[-self.window_size :], dtype=np.float32).reshape(1, self.window_size, 1)
        probs = self._model.predict(window, verbose=0)[0]
        label_id = int(np.argmax(probs))
        return ID_TO_LABEL.get(label_id, "stable")

