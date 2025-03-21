import numpy as np
import pandas as pd
from tqdm import tqdm

import pandas as pd
from tqdm import tqdm, trange
import sys, os
sys.path.append(os.path.abspath("../src/"))
from data_helper import get_days, load_config

config = load_config('../config/mimic_file.yaml')

censor_death_df = pd.read_csv('/data/padmalab_external/special_project/multi_event_data/MIMIC_IV_censor_death_date.csv', index_col = 1)
censor_death_df['censor_death_date'] = pd.to_datetime(censor_death_df['censor_death_date'])
ecg_record_df = pd.read_csv(config['project']['ecg_record_path'])
ecg_record_df['ecg_time'] = pd.to_datetime(ecg_record_df['ecg_time'])

# Initialize 'death_event' and 'death_time' columns
ecg_record_df['death_event'] = False
ecg_record_df['death_time'] = np.nan

# Filter rows where 'subject_id' is in 'censor_death_df'
ecg_record_df = ecg_record_df[ecg_record_df['subject_id'].isin(censor_death_df.index)]

# Merge with 'censor_death_df' to get 'censor_death_date' and 'death_event'
merged_df = ecg_record_df.merge(
    censor_death_df[['censor_death_date', 'death_event']],
    how='left',
    left_on='subject_id',
    right_index=True
)

# Calculate 'death_time' as the difference in days between 'censor_death_date' and 'ecg_time'
merged_df['death_time'] = (merged_df['censor_death_date'] - merged_df['ecg_time']).dt.days

# Ensure 'death_time' is non-negative
merged_df['death_time'] = merged_df['death_time'].clip(lower=0)

# Update 'death_event' and 'death_time' in the original DataFrame
ecg_record_df['death_event'] = merged_df['death_event']
ecg_record_df['death_time'] = merged_df['death_time']

# Save the updated DataFrame to a pickle file
ecg_record_df[['subject_id', 'study_id', 'death_event', 'death_time']].to_pickle(
    '/data/padmalab_external/special_project/multi_event_data/MIMIC_ECG_updated_death.pickle'
)
