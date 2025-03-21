import pandas as pd
from tqdm import tqdm
import sys, os

# Add the project directory to the Python path
sys.path.append(os.path.abspath("../src/"))
from data_helper import load_config

# Load configuration
config = load_config('../config/mimic_file.yaml')

# Load data
censor_death_df = pd.read_csv(
    '/data/padmalab_external/special_project/multi_event_data/MIMIC_IV_censor_death_date.csv',
    index_col=1
)
ecg_record_df = pd.read_csv(config['project']['ecg_record_path'])
ecg_record_df['ecg_time'] = pd.to_datetime(ecg_record_df['ecg_time'])
censor_death_df['censor_death_date'] = pd.to_datetime(censor_death_df['censor_death_date'])

# Function to process disease events
def process_disease_event(disease_event_path, dx_name, output_path):
    """
    Process disease events and calculate time from ECG to censor/death date.
    
    Args:
        disease_event_path (str): Path to the disease event DataFrame pickle file.
        dx_name (str): Name of the disease (e.g., 'Atribe_fibeliation', 'Heart_failure').
        output_path (str): Path to save the updated DataFrame.
    """
    # Load disease event DataFrame
    diseases_event_df = pd.read_pickle(disease_event_path)
    
    # Filter rows where 'subject_id' is in 'censor_death_df'
    diseases_event_df = diseases_event_df[diseases_event_df['subject_id'].isin(censor_death_df.index)]
    
    # Initialize progress bar
    with tqdm(total=diseases_event_df.shape[0]) as pbar:
        for index, row in diseases_event_df.iterrows():
            if row[f'{dx_name} event'] == False:
                # Get ECG time for the study_id
                ecg_time = ecg_record_df[ecg_record_df['study_id'] == row['study_id']].iloc[0]['ecg_time']
                # Get censor/death date for the subject_id
                last_date = censor_death_df.loc[row['subject_id']]['censor_death_date']
                # Calculate time difference in days
                diseases_event_df.loc[index, f'{dx_name} time'] = (last_date - ecg_time).days
            pbar.update(1)
    
    # Save the updated DataFrame to a pickle file
    diseases_event_df.to_pickle(output_path)

# Process AF event
process_disease_event(
    disease_event_path='/data/padmalab_external/special_project/multi_event_data/MIMIC_ECG_Atribe_fibeliation.pickle',
    dx_name='Atribe_fibeliation',
    output_path='/data/padmalab_external/special_project/multi_event_data/MIMIC_ECG_updated_AF.pickle'
)

# Process HF event
process_disease_event(
    disease_event_path='/data/padmalab_external/special_project/multi_event_data/MIMIC_ECG_Heart_failure.pickle',
    dx_name='Heart_failure',
    output_path='/data/padmalab_external/special_project/multi_event_data/MIMIC_ECG_updated_HF.pickle'
)