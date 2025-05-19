# DqB_Cr³⁺_Model  
**Predict the crystal field splitting parameter (Dq/B) of Cr³⁺-substituted phosphor**

This package offers a machine learning model trained on experimentally reported Dq/B values for Cr³⁺ phosphors, each possessing a singular crystallographically independent octahedral coordination environment. 

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
