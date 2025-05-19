#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:




