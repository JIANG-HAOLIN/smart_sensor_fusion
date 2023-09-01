# smart_sensor_fusion
Repository for Smart Sensor Fusion project

## To install:
* First, install requirements: `pip install -r requirements.txt`
* Install package locally: `pip install -e .`

## To prepare dataset for progress prediction
"""https://github.com/JunzheJosephZhu/see_hear_feel"""
* First, download the example dataset from their [Google Drive](https://drive.google.com/drive/folders/13S6YcKJIIRKctB0SjdiaKEv_mvJEM_pk)
* Second, unzip and rename the folder to `data`, and place it under the project folder `smart_sensor_fusion`
* Third, preprocess the data by running `python utils/h5py_convert.py`
* Fourth, to split the training/testing dataset, run `python utils/split_train_val.py`
* Brief explanation for the example dataset: Under data/test_recordings, each folder is an episode. timestamps.json contains the human demo actions and the pose history of the robot, while each subfolder contains a stream of sensory inputs.

## Run training
* Train the model for transpose task by running: `python train.py --config-name config_transpose`
* Train the model for progress_prediction task by running: `python train.py --config-name config_progress_prediction`

## Run inference
* Test the model for transpose task by running: `python scripts/demo_transpose_numbers.py inference.ckpt_path='/home/jin4rng/Documents/code/smart_sensor_fusion/results/transpose/logits/simple_transformer/time_09-01-14:05:34/checkpoints/09-01-14:05:36-jobid=0-epoch=0-step=390.ckpt' inference.test_seq=[0,1,1,1,1,6,7,8,9,8,0,1,2,3,4,5,6]`
* Test the model for progress_prediction task by running: `python scripts/demo_progress_prediction.py models.inference.ckpt_path='/home/jin4rng/Documents/code/smart_sensor_fusion/results/progress_prediction/see_hear_feel_insert_audio/vit/time_08-31-10:32:13/checkpoints/08-31-10:32:15-jobid=0-epoch=7-step=3248.ckpt'`

## Help
* Hydra command line flags and override
  * override hydra arguments example: `python train.py task_name='override_args'` (check [here](https://hydra.cc/docs/advanced/override_grammar/basic/))
  * manipulate hydra config using command line: `python train.py --config-name config_transpose` (check [here](https://hydra.cc/docs/advanced/hydra-command-line-flags/))
  * manipulate hydra config and override hydra args: `python train.py --config-name config_transpose task_name='override_args'`