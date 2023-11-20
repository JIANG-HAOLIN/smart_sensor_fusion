import os

import pandas as pd
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

csv_file_paths = ['/home/jin4rng/Downloads/2023-04-19 14-56-56.csv',
                 '/home/jin4rng/Downloads/2023-04-27 13-31-30.csv']


for csv_file_path in csv_file_paths:
    idx_shift = 16 if '2023-04-27 13-31-30' in csv_file_path else 0
    with open(csv_file_path, 'r') as raw_data_file:
        labels = raw_data_file.readline().rstrip('\n').split(',')
        num_col = len(labels)
    for i in tqdm(range(16)):
        print(i, datetime.now().strftime("%y:%m:%d %H:%M:%S"))
        cols = [ele for ele in labels if ele.split('.')[-1].strip() == f'{i + 1}']
        df = pd.read_csv(csv_file_path, usecols=cols)  # Assumes the header is in the first row
        nan_df = df.apply(pd.to_numeric, errors='coerce')
        for col in cols:
            out_path = os.path.join('/home/jin4rng/Documents/code/smart_sensor_fusion/data/processed_drilling_data/separate_data',
                                    col.split('.')[-2].strip())
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            np.save(os.path.join(out_path, f'{i+1+idx_shift}.npy'), nan_df[col])


# a = np.load('/home/jin4rng/Documents/code/smart_sensor_fusion/data/processed_drilling_data/separate_data/Phi1/1.npy')
# print(a)