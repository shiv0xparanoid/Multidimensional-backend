# app/ml_predict.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from app.indicators import sma, rsi, atr


def prepare_features(df, lags=[1, 5, 10]):
    if df is None or df.empty or "close" not in df.columns:
        raise ValueError("Data is empty or missing 'close' column. Try refetching.")

    df_feat = df.copy()
    close = df_feat['close'].squeeze()

    # Indicators
    df_feat['sma10'] = sma(close, 10)
    df_feat['sma20'] = sma(close, 20)
    df_feat['sma50'] = sma(close, 50)
    df_feat['sma200'] = sma(close, 200)
    df_feat['rsi14'] = rsi(close, 14)
    df_feat['mom10'] = close.pct_change(10)
    df_feat['mom20'] = close.pct_change(20)
    df_feat['mom60'] = close.pct_change(60)
    df_feat['atr14'] = atr(df, 14)
    df_feat['vol_chg'] = df_feat['volume'].pct_change()

    # Lagged returns
    for lag in lags:
        df_feat[f'ret_lag{lag}'] = close.pct_change(lag)

    # Smarter target: only predict meaningful moves (>0.2%)
    df_feat['target'] = np.where(df_feat['close'].shift(-1) > df_feat['close'] * 1.002, 1, 0)

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
    return df_feat


def evaluate_confidence(probs, preds, y_test, threshold=0.7):
    mask = (probs.max(axis=1) > threshold)
    if mask.sum() == 0:
        return 0, 0
    acc = accuracy_score(y_test[mask], preds[mask])
    coverage = mask.mean()
    return acc, coverage


def train_predict_model(df, symbol="RELIANCE", save_path="results/feature_importance.png"):
    df_feat = prepare_features(df)

    X = df_feat.drop(['target'], axis=1)
    y = df_feat['target']

    # Train = 1999–2023, Test = 2024–2025
    train = X.index < "2024-01-01"
    test = X.index >= "2024-01-01"

    X_train, y_train = X.loc[train], y.loc[train]
    X_test, y_test = X.loc[test], y.loc[test]

    if len(X_train) < 1000 or len(X_test) < 50:
        raise ValueError("Not enough data to train/test ML model.")

    # --- Ensemble Models ---
    rf = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=8)
    xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, eval_metric="logloss", use_label_encoder=False)
    logreg = LogisticRegression(max_iter=1000)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    logreg.fit(X_train, y_train)

    # Predictions
    rf_probs = rf.predict_proba(X_test)
    xgb_probs = xgb.predict_proba(X_test)
    logreg_probs = logreg.predict_proba(X_test)

    # Average probabilities = ensemble
    probs = (rf_probs + xgb_probs + logreg_probs) / 3.0
    preds = (probs[:, 1] > 0.5).astype(int)

    # --- Evaluation ---
    acc = accuracy_score(y_test, preds)
    print(f"\n=== Ensemble ML Model Performance ({symbol}) ===")
    print("Train Samples:", len(X_train), "| Test Samples:", len(X_test))
    print("Overall Test Accuracy:", round(acc, 4))
    print("\nClassification Report (2024–2025 test period):\n", classification_report(y_test, preds))

    # Selective accuracy (confidence > 0.7)
    conf_acc, coverage = evaluate_confidence(probs, preds, y_test, threshold=0.7)
    print(f"\nSelective Accuracy (conf>0.7): {conf_acc:.2f} with coverage {coverage*100:.1f}%")

    # --- Next Day Prediction ---
    last_features = X_test.iloc[[-1]]
    last_probs = (rf.predict_proba(last_features) + xgb.predict_proba(last_features) + logreg.predict_proba(last_features)) / 3.0
    prob = last_probs[0]
    pred_signal = "BUY" if prob[1] > 0.55 else "SELL" if prob[1] < 0.45 else "HOLD"

    print("\nNext Day Prediction:", pred_signal)
    print("Probabilities: SELL={:.2f}, BUY={:.2f}".format(prob[0], prob[1]))

    # --- Feature Importance (from RF + XGB) ---
    rf_importances = rf.feature_importances_
    xgb_importances = xgb.feature_importances_
    avg_importance = (rf_importances + xgb_importances) / 2.0

    feat_names = [str(c) for c in X.columns]
    sorted_idx = np.argsort(avg_importance)

    plt.figure(figsize=(10, 6))
    topn = 15 if len(feat_names) > 15 else len(feat_names)
    plt.barh([feat_names[i] for i in sorted_idx][-topn:], avg_importance[sorted_idx][-topn:])
    plt.title(f"Feature Importance (Ensemble: {symbol})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\nFeature importance chart saved to {save_path}")

    return (rf, xgb, logreg), pred_signal, prob
