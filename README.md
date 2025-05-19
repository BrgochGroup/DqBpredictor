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

**Amit Kumar**, *et al.* “Machine Learning-Assisted Discovery of Cr³⁺-based NIR Phosphors” (*Submitted*).  

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

## 🚀 Usage
### 📄 Define the prediction data set
Create a .xlsx file titled To_predict.xlsx, in which the compositions of interest are enumerated in the first column under the header "Formula," accompanied by 15 additional features.
 There is an example of the to_predict.xlsx file in the repository
### 📄 to get 7 composition features
 Create a .xlsx file title To_get_compositional_features.xlsx, in which the compositions of interest are enumerated in the first column under the header "Formula". 
Get_descriptors.py will automatically read elements.xlsx to generate 7 composition features. After running, you will get .xlsx file named Formula_with_compositional_features.xlsx. 
These includes fetures values of 
avg_Mulliken EN, avg_First ionization energy (kJ/mol), avg_Metallic valence, avg_Martynov-Batsanov EN, avg_Number of outer shell electrons, std_Mendeleev number, max_First ionization energy (kJ/mol)
### 📄 Predict dq/B of Cr3+
After preparing To_predict.xlsx, you can get the dq/B prediction by:

python Eg_model.py
dqb_Cr3+_Model.py will automatically read Cr3_dqb_training_set.xlsx, and To_predict.xlsx to generate a prediction. After running, you will get a .xlsx file named final_prediction_with_uncertainty.xlsx in the same directory.
