

# UCSF MoonLAIT Yeng Lab 

---

## Assignment interview - Research Data Analyst 
Task 1
1. calculate HRV metrics (at least SDNN, VLF, LF, HF) from the ECG channel.
2. perform survival analysis using HRV metrics to predict mortality.
3. report performances, at least C-index.

Task 2
1. run automatic sleep staging. There is no model training required. Please only run their trained model. Here are some possible package:
    - YASA, https://github.com/raphaelvallat/yasa
    - U-Sleep, https://github.com/perslev/U-Sleep-API-Python-Bindings
    - Luna, https://github.com/remnrem/luna-base?tab=readme-ov-file
    - others
2. report confusion matrix between actual and predicted sleep stages, and performance metrics such as Cohen's kappa, F1-score, etc.

For questions, please contact:
Haoqi Sun, hsun3@bidmc.harvard.edu
Yue Leng, yue.leng@ucsf.edu

---

## This repository contains the code to answer the 2 Tasks

## 📂 Repository Structure
- `task1/` 
    - individual_analysis.ipyn: Jupyter Notebook outlines the process of extracting HRV metrics from the ECG channels of an EDF file for a **single subject**. It includes multiple displays and visualizations at each step to ensure accurate HRV metric extraction and thorough verification of the process.
    - hrv_analysis.py: Python file the process the ECG, extract hrv metrics and get covariates and outcome for all the subject. Runing this files allow to get "hrv_covariates_outcome.csv" file
    - hrv_covariates_outcomes.csv Output from analysis.py and input for python_survival_analysis.ipynb
    - python_survival_analysis.ipynb: Jupyter Notebook that used HRV metrics covariates and outcomes to perform the survival analyssis and repport perfromance.

- `task2/` 
    - sleep_staging_analysis.py: Python file that cget actual and predicted sleep stage and then deduce all_metrics.csv and all_averages.csv
    - all_metrics.csv: output of sleep staging analysis use ad input ffor reuslts vizuqlition
    - avg_metrics.csv: output of sleep staging analysis use ad input ffor reuslts vizuqlition
    - results_visualization.ipynb: Jupyter notebook to show the reuslts of the ocmpariaosn report confusion matrix between actual and predicted sleep stages, and performance metrics such as Cohen's kappa, F1-score, etc.

## 📦 Dependencies
This project uses Python and the following libraries:
- `numpy`
- `pandas`
- `lifelines`
- `matplotlib`
- [Other required packages]

## 🔧 Installation
You can install the required dependencies using:
```bash
pip install -r requirements.txt