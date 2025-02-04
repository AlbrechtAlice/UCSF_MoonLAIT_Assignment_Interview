import os
import pyedflib
import numpy as np
import pandas as pd
import neurokit2 as nk
from tqdm import tqdm
from scipy.signal import lombscargle

# Need to be modified by user
assignment_folder_path = "/Users/alicealbrecht/Desktop/DrLeng_assignment"


def extract_ecg_signal(edf_file):
    """
    Extracts the ECG signal from an EDF file.

    Parameters:
    - edf_file (str): Path to the EDF file.

    Returns:
    - ecg_signal (numpy.ndarray): Inverted raw ECG signal.
    - sample_rate (float): Sampling rate of the ECG signal.

    Raises:
    - ValueError: If no ECG signal is found in the EDF file.
    """
    with pyedflib.EdfReader(edf_file) as f:
        signal_labels = f.getSignalLabels()
        ecg_index = None
        for i, label in enumerate(signal_labels):
            if 'ECG' in label.upper():
                ecg_index = i
                break
        else:
            raise ValueError(f"No ECG signal found in {edf_file}. Cannot proceed.")
        
        ecg_signal_raw = f.readSignal(ecg_index)
        sample_rate = f.getSampleFrequency(ecg_index)

    # Invert ECG signal
    ecg_signal = -ecg_signal_raw
    return ecg_signal, sample_rate


# Define function to calculate Temporal HRV metrics
def compute_temporal_hrv_metrics(nn_intervals):
    """
    Computes temporal HRV metrics from NN intervals.

    Parameters:
    - nn_intervals (numpy.ndarray): Array of NN intervals (in seconds).

    Returns:
    - avnn (float): Average NN interval (AVNN) in milliseconds.
    - sdnn (float): Standard deviation of NN intervals (SDNN) in milliseconds.
    - rmssd (float): Root mean square of successive RR interval differences (RMSSD) in milliseconds.
    - pnn50 (float): Percentage of successive RR intervals that differ by more than 50 ms.
    """
    avnn = np.mean(nn_intervals) * 1000
    sdnn = np.std(nn_intervals) * 1000
    rmssd = np.sqrt(np.mean(np.diff(nn_intervals) ** 2)) * 1000
    nn50 = np.sum(np.abs(np.diff(nn_intervals)) > 0.05) 
    pnn50 = (nn50 / len(nn_intervals)) * 100

    return avnn, sdnn, rmssd, pnn50


def compute_spectral_hrv_metrics(nn_intervals):
    """
    Computes spectral HRV metrics using Lomb-Scargle periodogram.

    Parameters:
    - nn_intervals (numpy.ndarray): Array of NN intervals (in seconds).

    Returns:
    - ULF_norm (float): Normalized power in the ULF band (%).
    - VLF_norm (float): Normalized power in the VLF band (%).
    - LF_norm (float): Normalized power in the LF band (%).
    - HF_norm (float): Normalized power in the HF band (%).
    - VHF_norm (float): Normalized power in the VHF band (%).
    - LF_HF_ratio (float): Ratio of LF power to HF power.
    """
    # Compute cumulative time (in seconds)
    t_original = np.cumsum(nn_intervals) 

    # Define frequency range (0.003 to 0.5 Hz, for example)
    frequencies = np.linspace(0.003, 0.5, 2000)  
    angular_freqs = 2 * np.pi * frequencies  # Convert to angular frequency (rad/s)

    # Compute Lomb-Scargle Periodogram
    psd = lombscargle(t_original, nn_intervals - np.mean(nn_intervals), angular_freqs, normalize="power", floating_mean=True)

    # Define frequency bands for ULF, VLF, LF, HF, VHF
    ULF_band = (0.0033, 0.01)  
    VLF_band = (0.01, 0.04)  
    LF_band = (0.04, 0.15)    
    HF_band = (0.15, 0.4)     
    VHF_band = (0.4, 0.5)   

    # Integrate power using the trapezoidal rule for better accuracy
    TOTPWR = np.trapezoid(psd, frequencies)  # Integrating over the entire frequency range

    # Compute power in the frequency bands
    ULF = np.trapezoid(psd[(frequencies >= ULF_band[0]) & (frequencies <= ULF_band[1])], frequencies[(frequencies >= ULF_band[0]) & (frequencies <= ULF_band[1])])
    VLF = np.trapezoid(psd[(frequencies >= VLF_band[0]) & (frequencies <= VLF_band[1])], frequencies[(frequencies >= VLF_band[0]) & (frequencies <= VLF_band[1])])
    LF = np.trapezoid(psd[(frequencies >= LF_band[0]) & (frequencies <= LF_band[1])], frequencies[(frequencies >= LF_band[0]) & (frequencies <= LF_band[1])])
    HF = np.trapezoid(psd[(frequencies >= HF_band[0]) & (frequencies <= HF_band[1])], frequencies[(frequencies >= HF_band[0]) & (frequencies <= HF_band[1])])
    VHF = np.trapezoid(psd[(frequencies >= VHF_band[0]) & (frequencies <= VHF_band[1])], frequencies[(frequencies >= VHF_band[0]) & (frequencies <= VHF_band[1])])

    # Normalize power in each band relative to total power (as percentage)
    ULF_norm = ULF / TOTPWR * 100
    VLF_norm = VLF / TOTPWR * 100
    LF_norm = LF / TOTPWR * 100
    HF_norm = HF / TOTPWR * 100
    VHF_norm = VHF / TOTPWR * 100

    # Calculate LF/HF ratio (avoid division by zero)
    LF_HF_ratio = LF / HF if HF != 0 else np.nan

    # Return the calculated spectral metrics
    return ULF_norm, VLF_norm, LF_norm, HF_norm, VHF_norm, LF_HF_ratio


def compute_nonlinear_hrv_metrics(nn_intervals):
    """
    Computes non-linear HRV metrics.

    Parameters:
    - nn_intervals (numpy.ndarray): Array of NN intervals (in seconds).

    Returns:
    - SD1 (float): Standard deviation of the Poincaré plot (SD1) in milliseconds.
    - SD2 (float): Standard deviation of the Poincaré plot (SD2) in milliseconds.
    """
    SD1 = np.sqrt(0.5) * np.std(nn_intervals[1:] - nn_intervals[:-1]) * 1000
    SD2 = np.sqrt(0.5) * np.std(nn_intervals[1:] + nn_intervals[:-1]) * 1000

    return SD1, SD2


def process_edf_file(edf_file):
    """
    Processes a single EDF file to extract HRV metrics.

    Parameters:
    - edf_file (str): Path to the EDF file.

    Returns:
    - hrv_metrics (dict): Dictionary containing HRV metrics for the given EDF file.
    """
    ecg_signal, sample_rate = extract_ecg_signal(edf_file)

    # Clean signal
    filtered_ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=sample_rate)

    # Find R-peaks
    ecg_peaks = nk.ecg_findpeaks(filtered_ecg_signal, 
                             sampling_rate=sample_rate, 
                             method= "promac",
                             promac_methods=["neurokit",
                                             "slopesumfunction",
                                             "engzee2012",
                                             "kalidas2017",
                                             "nabian2018",
                                             "rodrigues2021",
                                             "kalidas2017"])
    peaks = ecg_peaks['ECG_R_Peaks']
    rr_intervals = (np.diff(peaks) / sample_rate) # in seconds (not in samples)
    nn_intervals = rr_intervals[rr_intervals <= 2.5] # remove abnormal intervals
    
    # Calculate HRV metrics
    avnn, sdnn, rmssd, pnn50 = compute_temporal_hrv_metrics(nn_intervals)
    ULF_norm, VLF_norm, LF_norm, HF_norm, VHF_norm, LF_HF_ratio = compute_spectral_hrv_metrics(nn_intervals)
    SD1, SD2 = compute_nonlinear_hrv_metrics(nn_intervals)

    # Store all metrics in a dictionary
    hrv_metrics = {
        'NSRRID': os.path.basename(edf_file).split('-')[1].split('.')[0],  # Extract ID from the filename
        'AVNN': round(avnn, 3),
        'SDNN': round(sdnn, 3),
        'RMSSD': round(rmssd, 3),
        'pNN50': round(pnn50, 3),
        'ULF_norm': round(ULF_norm, 3),
        'VLF_norm': round(VLF_norm, 3),
        'LF_norm': round(LF_norm, 3),
        'HF_norm': round(HF_norm, 3),
        'VHF_norm': round(VHF_norm, 3),
        'LF_HF_ratio': round(LF_HF_ratio, 3),
        'SD1': round(SD1, 3),
        'SD2': round(SD2, 3)
    }

    return hrv_metrics


def process_all_edf_files(dataset_folder):
    """
    Processes all EDF files in the dataset folder and extracts HRV metrics.

    Parameters:
    - dataset_folder (str): Path to the folder containing EDF files.

    Returns:
    - hrv_df (pandas.DataFrame): DataFrame containing HRV metrics for all EDF files.
    """
    hrv_results = []
    edf_files = [f for f in os.listdir(dataset_folder) if f.endswith('.edf')]

    # Use tqdm to display a progress bar
    for file_name in tqdm(edf_files, desc="Processing EDF files", unit="file"):
        edf_file_path = os.path.join(dataset_folder, file_name)
        hrv_metrics = process_edf_file(edf_file_path)
        hrv_results.append(hrv_metrics)

    # Convert the list of dictionaries into a DataFrame
    hrv_df = pd.DataFrame(hrv_results)
    return hrv_df


def load_outcomes(outcomes_file):
    """
    Loads outcomes data from a CSV file.

    Parameters:
    - outcomes_file (str): Path to the outcomes CSV file.

    Returns:
    - outcomes_df (pandas.DataFrame): DataFrame containing the outcomes data.
    """
    outcomes_df = pd.read_csv(outcomes_file)
    outcomes_df.rename(columns={'nsrrid': 'NSRRID', 'vital': 'EVENT', 'censdate': 'DURATION'}, inplace=True)
    return outcomes_df


def load_covariates(covariates_file):
    """
    Loads confounding variables from a CSV file and renames columns to standardized names.

    Parameters:
    - covariates_file (str): Path to the SHHS subsampled data CSV file.

    Returns:
    - covariate_df (pandas.DataFrame): DataFrame containing the selected covariates. 
    """
    # Read the SHHS file that contians mainy features per subject
    covariates_df = pd.read_csv(covariates_file)

    # Select some features as potential covariates
    covariates_df.rename(columns={'nsrrid': 'NSRRID', 
                                   'age_s1': 'AGE', 
                                   'gender': 'GENDER',
                                   'bmi_s1': 'BMI', 
                                   'chol': 'CHOLESTEROL',
                                   'htnderv_s1': 'HYPERTENSION'}, inplace=True)
    covariates_df = covariates_df[['NSRRID', 'AGE', 'GENDER', 'BMI', 'CHOLESTEROL', 'HYPERTENSION']]
    
    return covariates_df


def main():
    """
    Main function to load, process, and combine HRV, confounding, and outcome data.
    """
    # Set the path to your dataset folder
    dataset_folder = os.path.join(assignment_folder_path, "dataset")

    # Load outcomes data (event and duration for survival analysis)
    outcomes_df = load_outcomes(os.path.join(assignment_folder_path, "outcomes.csv"))
    
    # Load covariates data (such as age, gender, etc.)
    covariates_df = load_covariates(os.path.join(assignment_folder_path, "shhs1-dataset-0.21.0-subsampled.csv"))

    # Process the EDF files to extract HRV metrics
    hrv_df = process_all_edf_files(dataset_folder)

    # Concatenate the confounding data, HRV data, and outcomes data along columns (axis=1) based on the common 'NSRRID'
    final_df = pd.concat([covariates_df, hrv_df, outcomes_df], axis=1)

    # Remove any duplicated columns (if they exist)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # Save the combined dataframe to a CSV file in the parent folder
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hrv_covariates_outcomes.csv')
    final_df.to_csv(output_file, index=False)
    print("Results saved to:", output_file)


# Run the main function
if __name__ == "__main__":
    main()
