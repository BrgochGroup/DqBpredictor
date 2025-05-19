# DqB_Cr³⁺_Model  
**Predict the crystal field splitting parameter (Dq/B) for Cr³⁺-substituted phosphor**

This package provides a machine learning model trained on experimentally reported Dq/B values for Cr³⁺-activated phosphors. It predicts the Dq/B ratio using only compositional descriptors and structural parameters via the command line or Jupyter environment.

---

## 📑 Table of Contents
- [Citations](#citations)  
- [Prerequisites](#prerequisites)  
- [Usage](#usage)  
  - [Define the prediction set](#define-the-prediction-set)  
  - [Run the prediction model](#run-the-prediction-model)  
- [Authors](#authors)  

---

## 📚 Citations  
To cite the Dq/B prediction model, please reference the following work (or your paper when published):

**Amit Kumar**, *et al.* “Title of Your Paper” (*in preparation or journal name*).  

---

## ⚙️ Prerequisites  

This package requires:

- `pymatgen`  
- `catboost`  
- `scikit-learn`  
- `pandas`  
- `numpy`  
- `matplotlib`  
- `openpyxl`  

Install them using:

```bash
pip install -r requirements.txt
