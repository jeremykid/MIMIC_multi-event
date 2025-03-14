from datetime import datetime
import pandas as pd
import yaml

def get_days(date_str1, date_str2):
    fmt = '%Y-%m-%d %H:%M:%S'

    dt1 = datetime.strptime(date_str1, fmt)
    dt2 = datetime.strptime(date_str2, fmt)

    delta = dt1 - dt2
    return (delta.days)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_ecg_measurement_df():
    ecg_measurements_df = pd.read_csv('/data/padmalab_external/special_project/MIMIC-IV_ECG/physionet.org/files/mimic-iv-ecg/1.0/machine_measurements.csv')
    ecg_measurements_df[['study_id', 'rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end','t_end', 'p_axis', 'qrs_axis', 't_axis']]
    return ecg_measurements_df