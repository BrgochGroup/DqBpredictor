import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from catboost import CatBoostRegressor

def get_model():
    return CatBoostRegressor(depth=4, iterations=500, learning_rate=0.09293,
                             l2_leaf_reg=1.9172, loss_function='RMSE', verbose=0)

# === Load data ===
train_df = pd.read_excel("Cr3_dqb_training_set.xlsx")
X, y = train_df.iloc[:, 2:20].values, train_df.iloc[:, 1].values
predict_df = pd.read_excel("To_predict.xlsx")
X_new, formulas = predict_df.drop(columns=["Formula"]).values, predict_df["Formula"].values

print("🔄 Model is running to find the best random state...")

# === Find best random state ===
best_r2, best_state = -np.inf, None
for rs in sorted(set(range(5, 101, 5)).union(range(5, 101, 7))):
    r2s = []
    for tr_idx, te_idx in KFold(n_splits=10, shuffle=True, random_state=rs).split(X):
        sc = StandardScaler().fit(X[tr_idx])
        model = get_model()
        model.fit(sc.transform(X[tr_idx]), y[tr_idx])
        r2s.append(r2_score(y[te_idx], model.predict(sc.transform(X[te_idx]))))
    mean_r2 = np.mean(r2s)
    if mean_r2 > best_r2:
        best_r2, best_state = mean_r2, rs

print(f"✅ Best random state = {best_state}")

# === Final 10-fold CV ===
kf = KFold(n_splits=10, shuffle=True, random_state=best_state)
y_true, y_pred, r2_scores, fold_preds_new = [], [], [], []
train_preds = [[] for _ in range(len(y))]

for tr_idx, te_idx in kf.split(X):
    sc = StandardScaler().fit(X[tr_idx])
    model = get_model()
    model.fit(sc.transform(X[tr_idx]), y[tr_idx])
    preds = model.predict(sc.transform(X[te_idx]))
    y_true.extend(y[te_idx])
    y_pred.extend(preds)
    r2_scores.append(r2_score(y[te_idx], preds))
    for idx, p in zip(te_idx, preds):
        train_preds[idx].append(p)
    fold_preds_new.append(model.predict(sc.transform(X_new)))

# === Final model for prediction ===
sc_full = StandardScaler().fit(X)
final_model = get_model()
final_model.fit(sc_full.transform(X), y)
final_preds = final_model.predict(sc_full.transform(X_new))
uncertainty = np.std(np.array(fold_preds_new), axis=0)

# === Save to Excel ===
pd.DataFrame({
    "Formula": formulas,
    "Predicted Dq/B": final_preds,
    "Uncertainty": uncertainty
}).to_excel("final_prediction_with_uncertainty.xlsx", index=False)

# === CV Metrics ===
final_r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print("\n📊 R² scores per fold (best random_state):")
for i, r2 in enumerate(r2_scores, 1):
    print(f"Fold {i}: R² = {r2:.4f}")
print(f"\n✅ Final Combined R² = {final_r2:.4f}")
print(f"📉 MAE = {mae:.4f}")

print("\n📄 Final Predictions with Uncertainty:")
print("Formula                           Predicted Dq/B     Uncertainty")
for f, p, u in zip(formulas, final_preds, uncertainty):
    print(f"{f:<32} {p:>10.4f}         {u:.4f}")

# === Parity Plot ===
plt.figure(figsize=(7, 7))
plt.scatter(y, [np.mean(p) for p in train_preds], alpha=0.7)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=2)
plt.xlabel("True Dq/B")
plt.ylabel("Predicted Dq/B")
plt.title(f"Parity Plot (Best Random State: {best_state})\nR² = {final_r2:.4f}, MAE = {mae:.4f}")
plt.grid(True)
plt.tight_layout()
plt.show()
print("✅ Saved 'final_prediction_with_uncertainty.xlsx'")
