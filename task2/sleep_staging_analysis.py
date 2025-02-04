import os
import pandas as pd
import numpy as np
import mne
import yasa
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, accuracy_score, precision_score, recall_score


# Please set the path to your Assignment Folder
assignment_folder_path = "/Users/alicealbrecht/Desktop/DrLeng_assignment"

# --- Step 1: Extract Subject Information ---
def extract_subject_data(dataset_folder):
    """
    Extract subject information from a CSV file containing details like nsrrid, age, gender.
    
    Parameters:
    - dataset_folder (str): The folder path where the dataset is stored.
    
    Returns:
    - subjects_data: A list of dictionaries, each containing information: nsrrid, age, gender, XML and EDF file paths for each subject.
    """
    # Extract age and gender to add as meta data for YASA model
    subjects_data = pd.read_csv(os.path.join(dataset_folder, "shhs1-dataset-0.21.0-subsampled.csv"))
    subjects_data = subjects_data[['nsrrid', 'age_s1', 'gender']] 
    subjects_data['gender'] = subjects_data['gender'].replace({2: 0}) # To have 0:female and 1:male

    # Assign file paths for the XML and EDF files for each subject
    subjects_data['xml_file'] = subjects_data['nsrrid'].apply(lambda x: os.path.join(dataset_folder, f'dataset/shhs1-{x}-nsrr.xml'))
    subjects_data['edf_file'] = subjects_data['nsrrid'].apply(lambda x: os.path.join(dataset_folder, f'dataset/shhs1-{x}.edf'))

    # Convert the DataFrame to a list of dictionaries for easier processing later
    subjects_data = subjects_data.to_dict(orient='records')

    return subjects_data


# --- Step 2: Read Actual Stages from XML ---
def get_actual_stages(xml_file):
    """
    Parse the XML file to extract the actual sleep stages based on the ScoredEvent information.
    
    Parameters:
    - xml_file (str): Path to the XML file containing sleep stage annotations.
    
    Returns:
    - actual_stages (list): A list of actual sleep stages (as integers like 0 for wake), corresponding to 30-second epochs.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    actual_stages = []

    for scored_event in root.findall(".//ScoredEvent"):
        event_type = scored_event.find('EventType').text if scored_event.find('EventType') is not None else ''
        if event_type and 'Stages' in event_type:
            stage = scored_event.find('EventConcept').text if scored_event.find('EventConcept') is not None else ''
            duration = float(scored_event.find('Duration').text)
            num_epoch = int(duration / 30)  # Calculate the number of 30-second epochs

            # Replace stage names as integers
            # Note that Stage3 and Stage4 are replaced by 3 to adapt to ASMM guileines
            stage = stage.replace('Wake|0', '0').replace('Stage 1 sleep|1', '1').replace('Stage 2 sleep|2', '2').replace('Stage 3 sleep|3', '3').replace('Stage 4 sleep|4', '3').replace('REM sleep|5', '4')
            
            # Append the stage multiple times based on the number of epochs 
            actual_stages.extend([stage] * num_epoch)

    actual_stages = [int(x) for x in actual_stages]
    return actual_stages


# --- Step 3: Using YASA for Sleep Staging Prediction ---
def get_predicted_stages(edf_file_path, age, gender):
    """
    Predict sleep stages using YASA for a single subject. YASA uses EEG, EOG, and EMG channels.
    
    Parameters:
    - edf_file_path (str): Path to the EDF file containing EEG, EOG, and EMG data.
    - age (int): Age of the subject.
    - gender (int): Gender of the subject (0: female, 1: male).
    
    Returns:
    - hypno_pred(list): A list of predicted sleep stages (as integers), corresponding to 30-second epochs.
    """
    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose='Error')

    # Extract channel names for EEG, EOG, and EMG
    # The EEG selected is C4 so using opposite EOG (in this case left EOG)
    channel_labels = raw.ch_names
    eeg_name = next((ch for ch in channel_labels if "EEG" in ch.upper() and len(ch) <= 3), None)
    eog_name = next((ch for ch in channel_labels if "EOG" in ch.upper() and "L" in ch.upper()), None)
    emg_name = next((ch for ch in channel_labels if "EMG" in ch.upper()), None)

    if not eeg_name or not eog_name or not emg_name:
        raise ValueError(f"Missing required channels! EEG: {eeg_name}, EOG: {eog_name}, EMG: {emg_name}")

    # Predict sleep stages using YASA
    sls = yasa.SleepStaging(raw, eeg_name=eeg_name, eog_name=eog_name, emg_name=emg_name, metadata=dict(age=age, male=gender))
    hypno_pred = sls.predict()
    hypno_pred = yasa.hypno_str_to_int(hypno_pred)
    
    return hypno_pred


# --- Step 4: Calculate Metrics ---
def calculate_metrics(actual_stages, predicted_stages):
    """
    Calculate confusion matrix and performance metrics between actual and predicted stages.
    
    Parameters:
    - actual_stages (list): List of actual sleep stages (integers).
    - predicted_stages (list): List of predicted sleep stages (integers).
    
    Returns:
    - tuple: Contains confusion matrix and performance metrics (kappa, f1, accuracy, precision, recall, specificity).
    """
    cm = confusion_matrix(actual_stages, predicted_stages)
    kappa = cohen_kappa_score(actual_stages, predicted_stages)
    f1 = f1_score(actual_stages, predicted_stages, average='weighted', zero_division=0)
    accuracy = accuracy_score(actual_stages, predicted_stages)
    precision = precision_score(actual_stages, predicted_stages, average='macro', zero_division=0)
    recall = recall_score(actual_stages, predicted_stages, average='macro', zero_division=0)
    specificity = recall_score(actual_stages, predicted_stages, average='macro', pos_label=0) 
    
    return cm, kappa, f1, accuracy, precision, recall, specificity


# --- Step 5: Process Each Subject ---
def process_subject(xml_file, edf_file, age, gender):
    """
    Process one subject, calculate all relevant metrics, and return results.
    
    Parameters:
    - xml_file (str): Path to the XML file containing actual sleep stages.
    - edf_file (str): Path to the EDF file containing data for predicted stages.
    - age (int): Age of the subject.
    - gender (int): Gender of the subject.
    
    Returns:
    - tuple: Contains confusion matrix and performance metrics for the subject.
    """
    actual_stages = get_actual_stages(xml_file)
    predicted_stages = get_predicted_stages(edf_file, age, gender)
    cm, kappa, f1, accuracy, precision, recall, specificity = calculate_metrics(actual_stages, predicted_stages)
    
    return cm, kappa, f1, accuracy, precision, recall, specificity


# --- Step 6: Process All Subjects ---
def process_all_subjects(assignment_folder_path):
    """
    Main function to process all subjects in the dataset and calculate performance metrics.
    
    Parameters:
    - assignment_folder_path (str): Path to the assignment folder containing the dataset.
    
    Returns:
    - all_metrics (pd.DataFrame): All metrics for each subject 
    - avg_metrics (pd.DataFrame): Aggregated average metrics across all subjects.
    """
    # Step 1: Process subject information
    subject_data = extract_subject_data(assignment_folder_path)

    # Step 2: Process each subject, calculate metrics, and store results
    all_metrics = []
    for subject in subject_data:
        nsrrid = subject['nsrrid']
        xml_file = subject['xml_file']
        edf_file = subject['edf_file']
        age = subject['age_s1']
        gender = subject['gender']

        # Step 3: Get actual and predicted stages
        actual_stages = get_actual_stages(xml_file)
        predicted_stages = get_predicted_stages(edf_file, age, gender)
        
        # Ensure stages are numpy arrays
        actual_stages = np.array(actual_stages)
        predicted_stages = np.array(predicted_stages)
        
        # Step 4: Calculate metrics for each subject
        cm, kappa, f1, accuracy, precision, recall, specificity = calculate_metrics(actual_stages, predicted_stages)
        
        # Append the results for this subject
        all_metrics.append({
            'nsrrid': nsrrid, 
            'cm': cm, 
            'kappa': kappa, 
            'f1': f1, 
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall, 
            'specificity': specificity, 
        })

    # --- Step 7: Aggregate Results ---
    # Extract confusion matrices
    confusion_matrices = [metric['cm'] for metric in all_metrics]
    cm_array = np.array(confusion_matrices)

    # Calculate average and standard deviation of confusion matrices
    average_cm = np.mean(cm_array, axis=0).astype(int)
    std_cm = np.std(cm_array, axis=0).astype(int)

    # Calculate average and standard deviation of other metrics
    avg_metrics = {
        'avg_cm': average_cm,
        'avg_kappa': np.mean([metric['kappa'] for metric in all_metrics]),
        'avg_f1': np.mean([metric['f1'] for metric in all_metrics]),
        'avg_accuracy': np.mean([metric['accuracy'] for metric in all_metrics]),
        'avg_precision': np.mean([metric['precision'] for metric in all_metrics]),
        'avg_recall': np.mean([metric['recall'] for metric in all_metrics]),
        'avg_specificity': np.mean([metric['specificity'] for metric in all_metrics]),
        'std_cm': std_cm,
        'std_kappa': np.std([metric['kappa'] for metric in all_metrics]),
        'std_f1': np.std([metric['f1'] for metric in all_metrics]),
        'std_accuracy': np.std([metric['accuracy'] for metric in all_metrics]),
        'std_precision': np.std([metric['precision'] for metric in all_metrics]),
        'std_recall': np.std([metric['recall'] for metric in all_metrics]),
        'std_specificity': np.std([metric['specificity'] for metric in all_metrics]),
    }

    # --- Step 8: Save Results to CSV ---
    # Convert the results into dataframes and save them
    all_metrics_df = pd.DataFrame(all_metrics)
    avg_metrics_df = pd.DataFrame([avg_metrics])

    # Save the results as CSV files
    all_metrics_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all_metrics.csv'), index=False)
    avg_metrics_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'avg_metrics.csv'), index=False)

    return all_metrics, avg_metrics


# --- Step 9: Main Execution ---
if __name__ == '__main__':
    # Run the processing
    all_metrics, avg_metrics = process_all_subjects(assignment_folder_path)
    print("Results saved to CSV files.")