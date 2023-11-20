import os

import pandas as pd
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

csv_file_path = '/home/jin4rng/Downloads/2023-04-19 14-56-56.csv'


idx_shift = 16 if '2023-04-27 13-31-30' in csv_file_path else 0
with open(csv_file_path, 'r') as raw_data_file:
    labels = raw_data_file.readline().rstrip('\n').split(',')
    num_col = len(labels)
for i in range(16):
    print(i, datetime.now().strftime("%y:%m:%d %H:%M:%S"))
    cols = ['                '] + [ele for ele in labels if ele.split('.')[-1].strip() == f'{i + 1}']
    out_path = f'{i+1+idx_shift}'
    # Open the file in write mode
    with open(os.path.join('/home/jin4rng/Documents/code/smart_sensor_fusion/data/processed_drilling_data', out_path+'.txt'), 'w') as txt_file:
        # Write each element of the list to a new line
        for item in cols:
            txt_file.write(str(item) + '\n')
    df = pd.read_csv(csv_file_path, usecols=cols)  # Assumes the header is in the first row
    nan_df = df.apply(pd.to_numeric, errors='coerce')
    np.save(os.path.join('/home/jin4rng/Documents/code/smart_sensor_fusion/data/processed_drilling_data', out_path+'.npy'), nan_df)



# plt.rc('font', size=4)
# for signal_index in tqdm(range(36, 53)):
#     for i in range(1, 33):
#         processed_data_folder_path = '/home/jin4rng/Documents/code/smart_sensor_fusion/data/processed_drilling_data/'
#         npy_path = os.path.join(processed_data_folder_path, f'{i}.npy')
#         txt_path = os.path.join(processed_data_folder_path, f'{i}.txt')
#         with open(txt_path, 'r') as txt_file:
#             lb = txt_file.readlines()[signal_index].strip().split('.')[-2]
#             if i == 1:
#                 pre_lb = lb
#             elif i >= 1:
#                 if lb != pre_lb:
#                     raise RuntimeError(f'label matching error: trajectory {i} and {i-1} at {signal_index}: {lb} and {pre_lb}')
#         arr = np.load(npy_path).T
#         masked_arr = np.ma.masked_invalid(arr)
#         x = arr[0]
#         y = masked_arr[signal_index]
#         x = x[~np.isnan(y)]
#         y = y[~np.isnan(y)]
#         plt.plot(x, y, '-', label=f'trajectory{i}', linewidth=0.2)
#     plt.legend()
#     plt.title(f'{lb}_index{signal_index}')
#     plt.savefig(os.path.join('/home/jin4rng/Documents/code/smart_sensor_fusion/data/processed_drilling_data/visualization', f'{lb}_index{signal_index}.png'),
#                 dpi=300,
#                 bbox_inches='tight',)
#     plt.clf()
#     plt.close('all')








