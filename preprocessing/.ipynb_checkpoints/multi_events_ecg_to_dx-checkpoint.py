#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A reusable script to process ECG data and label based on diagnosis time.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys, os
import argparse
import logging

# Add the project's src folder to sys.path to import custom modules
sys.path.append(os.path.abspath("../src/"))
from data_helper import get_days, load_config

def load_data(config):
    """
    Load required data including ECG records, admissions, and ICD10 diagnoses based on the configuration.
    
    Parameters:
        config (dict): Configuration dictionary containing file path information.
        
    Returns:
        ecg_record_df (DataFrame): ECG records data.
        admissions (DataFrame): Admissions data.
        diagnoses_icd10_df (DataFrame): ICD10 diagnoses data.
    """
    ecg_record_df = pd.read_csv(config['project']['ecg_record_path'])
    admissions_path = os.path.join(config['project']['mimic_parent_math'], 'hosp', 'admissions.csv.gz')
    diagnoses_path = os.path.join(config['project']['mimic_parent_math'], 'hosp', 'diagnoses_icd10.csv')
    
    admissions = pd.read_csv(admissions_path, compression='gzip')
    diagnoses_icd10_df = pd.read_csv(diagnoses_path, index_col=0)
    
    return ecg_record_df, admissions, diagnoses_icd10_df

def process_data(config, ICD_10_code='I48', Dx_name='Atribe_fibeliation', seq_threshold=3):
    """
    Process ECG data and label it with diagnosis information. For each patient,
    the script matches ECG records with corresponding admissions based on ICD10 codes and a sequence threshold,
    then calculates the time difference.
    
    Parameters:
        config (dict): Configuration dictionary.
        ICD_10_code (str): ICD10 code to filter diagnoses (default: 'I48').
        Dx_name (str): Prefix for the new columns (default: 'Atribe_fibeliation').
        seq_threshold (int): Sequence threshold for primary diagnoses (default: 3).
        
    Returns:
        ecg_record_df (DataFrame): Processed ECG records data with new time and event columns.
    """
    ecg_record_df, admissions, diagnoses_icd10_df = load_data(config)
    
    # Initialize new columns
    ecg_record_df[f'{Dx_name}_time'] = np.nan
    ecg_record_df[f'{Dx_name}_event'] = False

    # Filter diagnoses matching the specified ICD10 code
    dx_df = diagnoses_icd10_df[diagnoses_icd10_df['icd_10_code'].str.contains(f'^{ICD_10_code}', na=False)]
    patients = ecg_record_df['subject_id'].unique()

    for subject_id in tqdm(patients, desc="Processing Patients"):
        pat_ecg_df = ecg_record_df[ecg_record_df['subject_id'] == subject_id]
        pat_dx_df = dx_df[dx_df['subject_id'] == subject_id]
        dx_hadm_id_list = pat_dx_df[pat_dx_df['seq_num'] <= seq_threshold]['hadm_id'].unique().tolist()
        pat_dx_admission = admissions[admissions['hadm_id'].isin(dx_hadm_id_list)]
        
        # Iterate through each ECG record for the patient
        for index, row in pat_ecg_df.iterrows():
            # Select admissions that occurred after the ECG record
            relevant_admissions = pat_dx_admission[row['ecg_time'] < pat_dx_admission['admittime']]
            time = None
            event = False
            if not relevant_admissions.empty:
                # Choose the earliest admission as the diagnosis event
                next_dx_event_date = relevant_admissions['admittime'].min()
                time = get_days(next_dx_event_date, row['ecg_time'])
                event = True
            ecg_record_df.loc[index, [f'{Dx_name}_time', f'{Dx_name}_event']] = [time, event]
            
    return ecg_record_df

def save_output(df, output_path, Dx_name):
    """
    Save the processed results as a pickle file.
    
    Parameters:
        df (DataFrame): Processed ECG data.
        output_path (str): Directory path for the output file.
        Dx_name (str): Prefix used to generate the output file name.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, f"MIMIC_ECG_{Dx_name}.pickle")
    df[['subject_id', 'study_id', f'{Dx_name}_time', f'{Dx_name}_event']].to_pickle(output_file)
    logging.info("Saved processed data to: %s", output_file)

def main():
    parser = argparse.ArgumentParser(
        description="Process ECG records and associate them with diagnosis events."
    )
    parser.add_argument('--config', type=str, default='../config/mimic_file.yaml',
                        help="Path to the YAML configuration file.")
    parser.add_argument('--output_path', type=str,
                        default='/data/padmalab_external/special_project/multi_event_data/',
                        help="Output directory for the processed data.")
    parser.add_argument('--ICD_10_code', type=str, default='I48',
                        help="ICD-10 code to filter diagnoses (default: I48).")
    parser.add_argument('--Dx_name', type=str, default='Atribe_fibeliation',
                        help="Diagnosis name prefix for the new columns (default: Atribe_fibeliation).")
    parser.add_argument('--seq_threshold', type=int, default=3,
                        help="Sequence threshold for primary diagnoses (default: 3).")
    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Load configuration file
    config = load_config(args.config)
    # Process the data
    processed_df = process_data(config, args.ICD_10_code, args.Dx_name, args.seq_threshold)
    # Save the output
    save_output(processed_df, args.output_path, args.Dx_name)

if __name__ == "__main__":
    main()
