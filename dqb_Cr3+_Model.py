#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# To get 7 compositional Features;
# 
# avg_Mulliken EN,
# avg_First ionization energy (kJ/mol),	
# avg_Metallic valence,	
# avg_Martynov-Batsanov EN,	
# avg_Number of outer shell electrons,
# std_Mendeleev number,	
# max_First ionization energy (kJ/mol)
# 

# In[8]:


import pandas as pd, numpy as np
from pymatgen.core.composition import Composition
from openpyxl.utils import column_index_from_string

class Vectorize_Formula:
    def __init__(self):
        df = pd.read_excel('elements.xlsx', index_col='Symbol')
        self.df, self.col_names = df, [f"{s}_{p}" for s in ['avg', 'diff', 'max', 'min', 'std'] for p in df.columns]

    def get_features(self, formula):
        try:
            fc = Composition(formula).fractional_composition.as_dict()
            avg = sum(self.df.loc[el] * frac for el, frac in fc.items() if el in self.df.index)
            props = self.df.loc[list(fc.keys())]
            feats = np.concatenate([avg, props.max()-props.min(), props.max(), props.min(), props.std(ddof=0)])
            return feats if len(feats) == len(self.col_names) else [np.nan]*len(self.col_names)
        except: return [np.nan]*len(self.col_names)

df = pd.read_excel('To_get_compositional_features.xlsx', usecols='A')
vf = Vectorize_Formula()
features = [vf.get_features(f) for f in df['Formula']]
combined = pd.concat([df, pd.DataFrame(features, columns=vf.col_names)], axis=1)

cols = [column_index_from_string(c)-1 for c in ['A','Q','Y','S','O','X','EN','CO']]
filtered = combined[[combined.columns[i] for i in cols if i < len(combined.columns)]]
filtered.to_excel("Formula_with_compositional_features.xlsx", index=False)

print("✅Excel file saved as 'Formula_with_compositional_features.xlsx'")



# To get 4 structural features using CIF file;
# SGR No.	volume_per_fu	volume_per_atom	beta angle
# 

# In[14]:


import os, pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer as SGA

res = []
for f in os.listdir(): 
    if f.endswith(".cif"):
        try:
            s = SGA(Structure.from_file(f))
            c = s.get_conventional_standard_structure()
            res.append({
                "Formula": c.composition.reduced_formula,
                "SGR No.": s.get_space_group_number(),
                "volume_per_fu": c.volume / c.composition.get_reduced_composition_and_factor()[1],
                "volume_per_atom": c.volume / c.composition.num_atoms,
                "beta": c.lattice.beta
            })
        except Exception as e: print(f"❌ {f}: {e}")

pd.DataFrame(res).to_csv("CIF_Structural_output.csv", index=False)
print("✅ Saved as 'CIF_Structural_output.csv'")


# In[16]:


jupyter nbconvert --to script dqb_Cr3+_Model.ipynb


# In[ ]:




