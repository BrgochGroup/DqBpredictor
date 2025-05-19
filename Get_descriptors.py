#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


