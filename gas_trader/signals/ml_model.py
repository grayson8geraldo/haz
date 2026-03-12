"""ML-based signal — XGBoost / LSTM for price direction prediction."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from gas_trader.config import Settings

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class MLPrediction:
    direction: int  # 1=long, -1=short, 0=neutral
    probability: float  # 0 to 1
    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    model_name: str


class MLSignal:
    """Trains and uses ML models to predict NG price direction."""

    FEATURE_COLS = [
        "rsi_14", "atr_14", "bb_width", "vwap_distance",
        "volume_ratio", "day_of_week", "month", "hour",
    ]

    def __init__(self, config: dict, settings: Settings):
        self.config = config
        self.settings = settings
        self.model_type = config.get("model", "xgboost")
        self.min_confidence = config.get("min_confidence", 0.65)
        self.retrain_days = config.get("retrain_interval_days", 7)
        self._model = None
        self._last_train: Optional[dt.datetime] = None
        self._feature_cols = self.FEATURE_COLS

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature DataFrame from OHLCV with indicators."""
        features = df.copy()

        # Time features
        features["day_of_week"] = features.index.dayofweek
        features["month"] = features.index.month
        features["hour"] = features.index.hour

        # Target: next bar direction (1 if price goes up, 0 if down)
        features["target"] = (features["close"].shift(-1) > features["close"]).astype(int)

        # Keep only needed columns
        available = [c for c in self._feature_cols if c in features.columns]
        result = features[available + ["target"]].dropna()
        return result

    def train(self, df: pd.DataFrame) -> dict:
        """Train the ML model on historical data."""
        features = self.prepare_features(df)
        if len(features) < 100:
            logger.warning(f"Not enough data for training: {len(features)} rows")
            return {"status": "insufficient_data", "rows": len(features)}

        X = features[self._feature_cols_available(features)]
        y = features["target"]

        # Train/test split (80/20, time-ordered)
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        if self.model_type == "xgboost":
            return self._train_xgboost(X_train, y_train, X_test, y_test)
        else:
            logger.warning(f"Model type '{self.model_type}' not implemented, using XGBoost")
            return self._train_xgboost(X_train, y_train, X_test, y_test)

    def _train_xgboost(self, X_train, y_train, X_test, y_test) -> dict:
        try:
            import xgboost as xgb
            from sklearn.metrics import accuracy_score, classification_report

            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            self._model = model
            self._last_train = dt.datetime.now()

            # Save model
            model_path = MODELS_DIR / "xgb_ng_model.json"
            model.save_model(str(model_path))

            logger.info(f"XGBoost trained — accuracy: {accuracy:.3f}")
            return {
                "status": "trained",
                "accuracy": accuracy,
                "train_size": len(X_train),
                "test_size": len(X_test),
            }

        except ImportError:
            logger.warning("XGBoost not installed — using fallback signal")
            return {"status": "xgboost_not_installed"}

    def predict(self, df: pd.DataFrame) -> MLPrediction:
        """Generate ML prediction for current market state."""
        # Check if model needs retraining
        if self._model is None:
            self._try_load_model()

        if self._model is None:
            # No model available — train on provided data
            result = self.train(df)
            if result.get("status") != "trained":
                return MLPrediction(
                    direction=0, probability=0.5,
                    signal=0.0, confidence=0.0,
                    model_name=self.model_type,
                )

        features = self.prepare_features(df)
        if features.empty:
            return MLPrediction(
                direction=0, probability=0.5,
                signal=0.0, confidence=0.0,
                model_name=self.model_type,
            )

        X = features[self._feature_cols_available(features)].iloc[[-1]]

        try:
            proba = self._model.predict_proba(X)[0]
            prob_up = proba[1] if len(proba) > 1 else 0.5

            if prob_up >= self.min_confidence:
                direction = 1
                signal = (prob_up - 0.5) * 2  # Scale [0.5, 1] → [0, 1]
            elif (1 - prob_up) >= self.min_confidence:
                direction = -1
                signal = (prob_up - 0.5) * 2  # Scale [0, 0.5] → [-1, 0]
            else:
                direction = 0
                signal = (prob_up - 0.5) * 2

            confidence = abs(prob_up - 0.5) * 2

            return MLPrediction(
                direction=direction,
                probability=prob_up,
                signal=float(np.clip(signal, -1, 1)),
                confidence=confidence,
                model_name=self.model_type,
            )
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return MLPrediction(
                direction=0, probability=0.5,
                signal=0.0, confidence=0.0,
                model_name=self.model_type,
            )

    def _try_load_model(self) -> None:
        """Try to load a previously saved model."""
        model_path = MODELS_DIR / "xgb_ng_model.json"
        if model_path.exists():
            try:
                import xgboost as xgb

                model = xgb.XGBClassifier()
                model.load_model(str(model_path))
                self._model = model
                logger.info("Loaded saved XGBoost model")
            except Exception as e:
                logger.warning(f"Could not load saved model: {e}")

    def needs_retrain(self) -> bool:
        if self._last_train is None:
            return True
        elapsed = (dt.datetime.now() - self._last_train).days
        return elapsed >= self.retrain_days

    def _feature_cols_available(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self._feature_cols if c in df.columns]
