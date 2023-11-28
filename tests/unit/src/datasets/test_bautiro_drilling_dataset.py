import unittest
from src.datasets.bautiro_drilling_dataset import get_loaders


class TestBautiroDrillingDataset(unittest.TestCase):

    def test_bautiro_drilling_dataset(self):
        from tqdm import tqdm
        import os
        data_folder_path = '/fs/scratch/rng_cr_bcai_dl_students/jin4rng/'
        data_folder_path = os.path.join(data_folder_path, 'data/processed_drilling_data/separate_data/np_arr')

        # if not os.path.exists(data_folder_path):
        #     os.makedirs(data_folder_path)
        # for i in list(range(1, 14)) + list(range(17, 30)):
        #     for j in os.listdir(data_folder_path):
        #         arr = np.arange(i * 10, i * 10 + 10)
        #         np.save(os.path.join(data_folder_path, j, str(i) + '.npy'), arr)
        #
        # project_path = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
        # data_folder_path = os.path.join(project_path, 'processed_drilling_data/separate_data/np_arr')
        # for i in list(range(1, 14)) + list(range(17, 30)):
        #     for j in os.listdir(data_folder_path):
        #         print(np.load(os.path.join(data_folder_path, j, str(i) + '.npy')))
        batch_size = 32
        windows_size = 3 * 50000  # according to 3s
        step_size = 50000  # according to 1s
        train_loader = get_loaders(data_folder_path, shuffle=True, window_size=windows_size, step_size=step_size,
                                   batch_size=batch_size)
        for idx, data in tqdm(enumerate(train_loader)):
            for key, tensor in data.items():
                print(f'{key}: {tensor.shape}')
                if 'ap' in key or 'ac' in key:
                    self.assertEqual(tensor.shape, (batch_size, windows_size//4))


if __name__ == '__main__':
    unittest.main()
