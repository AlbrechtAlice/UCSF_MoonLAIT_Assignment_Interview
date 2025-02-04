# UCSF MoonLAIT Yeng Lab 

---

## 📌 Assignment Interview - Research Data Analyst  

### **Task 1: HRV Analysis & Survival Prediction**
1. Calculate HRV metrics (at least **SDNN, VLF, LF, HF**) from the **ECG channel**.
2. Perform **survival analysis** using HRV metrics to predict **mortality**.
3. Report performance metrics, including at least the **C-index**.

### **Task 2: Automatic Sleep Staging**
1. Run automatic **sleep staging** using a pre-trained model (**no model training required**). Suggested packages:
    - [YASA](https://github.com/raphaelvallat/yasa)
    - [U-Sleep](https://github.com/perslev/U-Sleep-API-Python-Bindings)
    - [Luna](https://github.com/remnrem/luna-base?tab=readme-ov-file)
    - Others, if applicable.
2. Report **confusion matrix** between actual and predicted sleep stages, and **performance metrics** such as **Cohen’s kappa, F1-score**, etc.

For questions, please contact:  
📧 Haoqi Sun: hsun3@bidmc.harvard.edu  
📧 Yue Leng: yue.leng@ucsf.edu  

---

## 📂 Repository Structure

### **`task1/` - HRV Analysis & Survival Prediction**
- **`individual_analysis.ipynb`**  
  → Jupyter Notebook that processes **ECG signals** from an EDF file for a **single subject**.  
  → Includes multiple **visualizations** to validate HRV metric extraction.  

- **`hrv_analysis.py`**  
  → Python script that **processes ECG data**, extracts **HRV metrics**, and generates **covariates & outcomes** for all subjects.  
  → Produces the output file: **`hrv_covariates_outcome.csv`**  

- **`hrv_covariates_outcome.csv`**  
  → Output from `hrv_analysis.py`, used as input for survival analysis.  

- **`python_survival_analysis.ipynb`**  
  → Jupyter Notebook performing **survival analysis** on HRV-based covariates and **reporting model performance**.  

---

### **`task2/` - Sleep Staging Analysis**
- **`sleep_staging_analysis.py`**  
  → Python script that **extracts actual and predicted sleep stages** and computes **performance metrics**.  
  → Produces the output files: **`all_metrics.csv`** and **`avg_metrics.csv`**  

- **`all_metrics.csv`**  
  → Output file containing **detailed performance metrics** for each sleep stage.  

- **`avg_metrics.csv`**  
  → Output file with **averaged performance scores** across all sleep stages.  

- **`results_visualization.ipynb`**  
  → Jupyter Notebook for **visualizing and comparing sleep staging results**, including confusion matrices and performance metrics.  

---

## 📦 Dependencies
This project uses Python and requires the following libraries:  
- `numpy`  
- `pandas`  
- `lifelines`  
- `matplotlib`  
- Other dependencies are listed in **`requirements.txt`**.  

---

## 🔧 Installation  
To install the required dependencies, run:  
```bash
pip install -r requirements.txt
