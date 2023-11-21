import os
import logging
import sys
import pandas as pd
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def process_raw_csv_data(csv_root_path: str):
    """
    Args:
        csv_root_path: path to directory that contains the raw csv files
    """
    csv_file_paths = [os.path.join(csv_root_path, '2023-04-4-27 14-56-56.csv'),
                      os.path.join(csv_root_path, '2023-04-27 13-31-30.csv')]
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
                out_path = os.path.join(os.path.abspath(os.path.join(__file__, '..', '..')),
                                        'data/processed_drilling_data/separate_data/np_arr',
                                        col.split('.')[-2].strip())
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                np.save(os.path.join(out_path, f'{i + 1 + idx_shift}.npy'), nan_df[col])


def visualization():
    signal_dict = {'Phi1-6_125': ['         Phi1.1 ',
                                  '         Phi2.1 ',
                                  '         Phi3.1 ',
                                  '         Phi4.1 ',
                                  '         Phi5.1 ',
                                  '         Phi6.1 ', ],
                   'XYZ_125': ['            X.1 ',
                               '            Y.1 ',
                               '            Z.1 ', ],
                   'Phixyz_125': ['         Phix.1 ',
                                  '         Phiy.1 ',
                                  '         Phiz.1 ', ],
                   'Fxyz_125': ['           Fx.1 ',
                                '           Fy.1 ',
                                '           Fz.1 ', ],
                   'Mxyz_125': ['           Mx.1 ',
                                '           My.1 ',
                                '           Mz.1 ', ],
                   'I1-6_125': ['           I1.1 ',
                                '           I2.1 ',
                                '           I3.1 ',
                                '           I4.1 ',
                                '           I5.1 ',
                                '           I6.1 ', ],
                   'DIO_125': ['     DIO_8bit.1 ', ],
                   'F_T_125': ['         Fx_T.1 ',
                               '         Fy_T.1 ',
                               '         Fz_T.1 ', ],
                   'M_T_125': ['         Mx_T.1 ',
                               '         My_T.1 ',
                               '         Mz_T.1 ', ],
                   'ac_C_50k': ['        acx_C.1 ',
                                '        acy_C.1 ',
                                '        acz_C.1 ', ],
                   'ap_C_50k': ['        apx_C.1 ',
                                '        apy_C.1 ',
                                '        apz_C.1 ', ],
                   'F_C_1k': ['         Fx_C.1 ',
                              '         Fy_C.1 ',
                              '         Fz_C.1 ', ],
                   'I_C_1k': ['         Is_C.1 ',
                              '         Iz_C.1 ', ],
                   'MakingHole_C_1k:': [' MakingHole_C.1 ', ],
                   's_C_1k': ['         s1_C.1 ',
                              '         s2_C.1 ', ],
                   'StartStop_1k': ['  StartStop_C.1 ', ],
                   'Yd_corr_125': ['           Yd.1 ',
                                   '       Y_corr.1 ', ],
                   's_C_corr_1k': ['    s1_C_corr.1 ',
                                   '    s2_C_corr.1 ', ],
                   'MakingHole_125': ['   MakingHole.1 ', ], }

    separate_preprocessed_numpy_data_path = os.path.join(os.path.abspath(os.path.join(__file__, '..', '..')),
                                                         'data/processed_drilling_data/separate_data/np_arr')
    separate_visualization_path = os.path.join(os.path.abspath(os.path.join(__file__, '..', '..')),
                                               'data/processed_drilling_data/separate_data/visualization')

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(asctime)s] - [%(levelname)s] --- %(message)s')
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(os.path.join(separate_visualization_path, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] - [%(levelname)s] --- %(message)s'))
    logger.addHandler(file_handler)

    for signal_group_name, signal_group in tqdm(signal_dict.items()):
        logger.info(f'processing {signal_group_name}')
        grid_shape = (2 if len(signal_group) > 3 else 1, 2 if len(signal_group) < 3 else 3)
        fig1, axes1 = plt.subplots(*grid_shape, figsize=(20 * grid_shape[1], 10 * grid_shape[0]))
        fig2, axes2 = plt.subplots(*grid_shape, figsize=(20 * grid_shape[1], 10 * grid_shape[0]))
        for idx, signal in enumerate(signal_group):
            signal = signal.strip().split('.')[0]
            logger.info(f'processing {signal}')
            pos = np.unravel_index(idx, grid_shape) if len(signal_group) > 3 else idx
            trajectories_folder = os.path.join(separate_preprocessed_numpy_data_path, signal)
            for traj in os.listdir(trajectories_folder):
                traj_idx = traj.split('.')[-2]
                y = np.load(os.path.join(trajectories_folder, traj))
                x = np.arange(0, len(y)) * 2e-5
                x = x[~np.isnan(y)]
                y = y[~np.isnan(y)]
                if len(x) >= 1e6:
                    x = x[::8]
                    y = y[::8]
                if int(traj_idx) <= 16:
                    axes1[pos].plot(x, y, '-', label=f'trajectory{traj_idx}', linewidth=0.2)
                else:
                    axes2[pos].plot(x, y, '-', label=f'trajectory{traj_idx}', linewidth=0.2)
            axes1[pos].set_title(signal)
            axes2[pos].set_title(signal)
        plt.legend()
        fig1.suptitle(f'19_{signal_group_name}')
        fig1.savefig(os.path.join(os.path.join(separate_visualization_path, '4-19'), f'{signal_group_name}.png'),
                     dpi=300,
                     bbox_inches='tight', )
        fig2.suptitle(f'27_{signal_group_name}')
        fig2.savefig(os.path.join(os.path.join(separate_visualization_path, '4-27'), f'{signal_group_name}.png'),
                     dpi=300,
                     bbox_inches='tight', )
        plt.clf()
        plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("--csv_root_path", help="path to directory that contains the raw csv files", type=str)
    parser.add_argument("--if_plot_figure", help="whether do you need visualization", action='store_true')
    args = parser.parse_args()
    process_raw_csv_data(args.csv_root_path)
    visualization()
