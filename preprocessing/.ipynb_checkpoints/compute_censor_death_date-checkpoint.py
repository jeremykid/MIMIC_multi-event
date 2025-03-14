#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A reusable script to compute censor death dates based on patients' data.
It uses death records (from patients.csv), ED stays, hospital admissions,
and ECG measurements to determine a censoring date for each subject.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from tqdm import tqdm

# Add the project's src folder to sys.path to import custom modules
sys.path.append(os.path.abspath("../src/"))
from data_helper import load_config, get_days  # get_days is imported if needed elsewhere

def load_data(config, ecg_path):
    """
    Load patients, ED stays, admissions, and ECG measurements data.
    
    Parameters:
        config (dict): Configuration dictionary containing file path information.
        ecg_path (str): Path to the ECG measurements CSV file.
    
    Returns:
        patients_df (DataFrame): Patients data.
        death_df (DataFrame): DataFrame with subject_id and date of death.
        ed_df (DataFrame): ED stays data.
        admission_df (DataFrame): Hospital admissions data.
        ecg_measurements_df (DataFrame): ECG measurements data.
        latest_ed_times (DataFrame): Latest ED 'outtime' per subject.
        latest_admission_times (DataFrame): Latest admission 'dischtime' per subject.
        latest_ecg_times (DataFrame): Latest ECG 'ecg_time' per subject.
    """
    # Get the parent directory path for MIMIC data
    MIMIC_parent_path = config['project']['mimic_parent_math']
    
    # Load patients data and create death dataframe (subjects with a date of death)
    patients_df = pd.read_csv(os.path.join(MIMIC_parent_path, 'hosp/patients.csv.gz'), compression='gzip')
    death_df = patients_df[~patients_df['dod'].isna()][['subject_id', 'dod']]
    death_df['dod'] = pd.to_datetime(death_df['dod'])
    
    # Load ED stays data
    ed_df = pd.read_csv(os.path.join(MIMIC_parent_path, 'ed/edstays.csv.gz'), compression='gzip')
    ed_df['outtime'] = pd.to_datetime(ed_df['outtime'])
    ed_df = ed_df.sort_values(by=['subject_id', 'outtime'])
    latest_ed_times = ed_df.groupby('subject_id')['outtime'].max().reset_index()
    
    # Load hospital admissions data
    admission_df = pd.read_csv(os.path.join(MIMIC_parent_path, 'hosp/admissions.csv.gz'), compression='gzip')
    admission_df['dischtime'] = pd.to_datetime(admission_df['dischtime'])
    admission_df = admission_df.sort_values(by=['subject_id', 'dischtime'])
    latest_admission_times = admission_df.groupby('subject_id')['dischtime'].max().reset_index()
    
    # Load ECG measurements data
    ecg_measurements_df = pd.read_csv(ecg_path)
    ecg_measurements_df['ecg_time'] = pd.to_datetime(ecg_measurements_df['ecg_time'])
    ecg_measurements_df = ecg_measurements_df.sort_values(by=['subject_id', 'ecg_time'])
    latest_ecg_times = ecg_measurements_df.loc[ecg_measurements_df.groupby('subject_id')['ecg_time'].idxmax()]
    
    return (patients_df, death_df, ed_df, admission_df, ecg_measurements_df,
            latest_ed_times, latest_admission_times, latest_ecg_times)

def print_data_summary(patients_df, ed_df, admission_df, ecg_measurements_df):
    """
    Print a summary of the number of unique subjects in each dataset.
    
    Parameters:
        patients_df (DataFrame): Patients data.
        ed_df (DataFrame): ED stays data.
        admission_df (DataFrame): Hospital admissions data.
        ecg_measurements_df (DataFrame): ECG measurements data.
    """
    logging.info("# patients in ECG measurements: %d", ecg_measurements_df['subject_id'].nunique())
    logging.info("# patients in patients data: %d", patients_df['subject_id'].nunique())
    logging.info("# patients in admissions: %d", admission_df['subject_id'].nunique())
    logging.info("# patients in ED stays: %d", ed_df['subject_id'].nunique())

def compute_censor_death_dates(patients_df, death_df, latest_ecg_times, latest_ed_times, latest_admission_times):
    """
    For each subject, compute the censor death date.
    If the subject has a recorded death, use that date; otherwise, 
    use the latest available date from ECG measurements, ED stays, or admissions.
    
    Parameters:
        patients_df (DataFrame): Patients data.
        death_df (DataFrame): DataFrame with subject_id and date of death.
        latest_ecg_times (DataFrame): Latest ECG 'ecg_time' per subject.
        latest_ed_times (DataFrame): Latest ED 'outtime' per subject.
        latest_admission_times (DataFrame): Latest admission 'dischtime' per subject.
    
    Returns:
        death_censor_df (DataFrame): DataFrame containing subject_id, censor_death_date, and death_event.
    """
    censor_death_dict = {
        'subject_id': [],
        'censor_death_date': [],
        'death_event': []
    }
    
    subject_ids = patients_df['subject_id'].unique().tolist()
    
    for subject_id in tqdm(subject_ids, desc="Computing censor death dates"):
        censor_death_dict['subject_id'].append(subject_id)
        temp_death_df = death_df[death_df['subject_id'] == subject_id]
        event = not temp_death_df.empty  # True if death event exists
        censor_death_date = pd.NaT
        
        if event:
            censor_death_date = temp_death_df.iloc[0]['dod']
        else:
            # Check latest ECG time
            temp_latest_ecg = latest_ecg_times[latest_ecg_times['subject_id'] == subject_id]
            if not temp_latest_ecg.empty:
                dates = [d for d in [censor_death_date, temp_latest_ecg.iloc[0]['ecg_time']] if pd.notna(d)]
                censor_death_date = max(dates) if dates else pd.NaT

            # Check latest ED outtime
            temp_latest_ed = latest_ed_times[latest_ed_times['subject_id'] == subject_id]
            if not temp_latest_ed.empty:
                dates = [d for d in [censor_death_date, temp_latest_ed.iloc[0]['outtime']] if pd.notna(d)]
                censor_death_date = max(dates) if dates else pd.NaT

            # Check latest admission dischtime
            temp_latest_adm = latest_admission_times[latest_admission_times['subject_id'] == subject_id]
            if not temp_latest_adm.empty:
                dates = [d for d in [censor_death_date, temp_latest_adm.iloc[0]['dischtime']] if pd.notna(d)]
                censor_death_date = max(dates) if dates else pd.NaT
                
        censor_death_dict['death_event'].append(event)
        censor_death_dict['censor_death_date'].append(censor_death_date)
    
    death_censor_df = pd.DataFrame.from_dict(censor_death_dict)
    return death_censor_df

def save_output(df, output_file):
    """
    Save the resulting DataFrame as a CSV file.
    
    Parameters:
        df (DataFrame): DataFrame to save.
        output_file (str): Path to the output CSV file.
    """
    df.to_csv(output_file, index=False)
    logging.info("Saved censor death dates to: %s", output_file)

def main():
    parser = argparse.ArgumentParser(
        description="Compute censor death dates from patients, ED, admissions, and ECG data."
    )
    parser.add_argument('--config', type=str, default='../config/mimic_file.yaml',
                        help="Path to the YAML configuration file.")
    parser.add_argument('--ecg_path', type=str,
                        default='/data/padmalab_external/special_project/MIMIC-IV_ECG/physionet.org/files/mimic-iv-ecg/1.0/machine_measurements.csv',
                        help="Path to the ECG measurements CSV file.")
    parser.add_argument('--output_file', type=str,
                        default='/data/padmalab_external/special_project/multi_event_data/MIMIC_IV_censor_death_date.csv',
                        help="Path to save the output CSV file.")
    args = parser.parse_args()
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Load configuration and data
    config = load_config(args.config)
    (patients_df, death_df, ed_df, admission_df, ecg_measurements_df,
     latest_ed_times, latest_admission_times, latest_ecg_times) = load_data(config, args.ecg_path)
    
    # Print data summary
    print_data_summary(patients_df, ed_df, admission_df, ecg_measurements_df)
    
    # Compute censor death dates
    death_censor_df = compute_censor_death_dates(patients_df, death_df, latest_ecg_times,
                                                   latest_ed_times, latest_admission_times)
    
    # Save the results
    save_output(death_censor_df, args.output_file)

if __name__ == "__main__":
    main()
