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
* Test the model for transpose task by running: `python scripts/demo_transpose_numbers.py ckpt_path=absolute path of the pretrained model`
* Test the model for progress_prediction task by running: `python scripts/demo_progress_prediction.py`
