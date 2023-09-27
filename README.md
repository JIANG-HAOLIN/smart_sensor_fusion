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
* Train the model for transpose task: `python train.py --config-name config_transpose`
* Train the model for progress_prediction task with default configuration: `python train.py --config-name config_progress_prediction`
* Train the model for progress_prediction task with multiple configuration(sweeper): `python train.py -cn config_progress_prediction -m`
* Train the model for progress_prediction task using both vision and audio signal with multiple configuration(sweeper):
  * first fill the datasets.dataloader.data_folder variable with absolute path of the dataset folder in config_progress_prediction_vision_audio.yaml
  * `python train.py -cn config_progress_prediction_vision_audio -m`

## Run inference
* load the pretrained model based on the config and ckpt that stored in the results directory
* Test the model for transpose task by running: `python scripts/demo_transpose_numbers.py 'inference.ckpt_path="path to .ckpt file" inference.test_seq=[0,1,1,1,1,6,7,8,9,8,0,1,2,3,4,5,6]'`
* Test the model for progress_prediction task by running in terminal: `python scripts/demo_progress_prediction.py -cp 'path to the result folder that contain the config' 'models.inference.ckpt_path="name of ckpt file"'` 
  e.g. `python demo_progress_prediction.py -cp '../results/progress_prediction/see_hear_feel_insert_audio/vit_time_patch_128_51_standardpe/exp_dim_batchsize/256_64/09-05-18:27:24/.hydra/' 'models.inference.ckpt_path="09-05-18:27:24-jobid=0-epoch=7-step=1624.ckpt"'`
* Test the model for progress_prediction task using both vision and audio signal by running in terminal: `python scripts/demo_progress_prediction_vision_audio.py -cp 'path to the result folder that contain the config' 'models.inference.ckpt_path="name of ckpt file"'` 
  e.g. `python demo_progress_prediction_vision_audio.py -cp '../results/progress_prediction/vision_audio/earlycat_newemb/exp_dim_batchsize/256_32/09-13-14:11:41/.hydra' 'models.inference.ckpt_path="09-13-14:11:41-jobid=0-epoch=13-step=5684.ckpt"'`

## Help
* Hydra command line flags and override
  * if using run configuration of IDE to pass the arguments:
    * override hydra arguments example: `python train.py task_name='override_args'` (check [here](https://hydra.cc/docs/advanced/override_grammar/basic/))
    * manipulate hydra config using command line: `python train.py --config-name config_transpose` (check [here](https://hydra.cc/docs/advanced/hydra-command-line-flags/))
    * manipulate hydra config and override hydra args: `python train.py --config-name config_transpose task_name='override_args'`
  * if using terminal to run the command line, because of interpretation of quote of shell, so you have to quote twice: 
    * override hydra arguments example: `python train.py 'task_name="override_args"'` (check [here](https://hydra.cc/docs/advanced/override_grammar/basic/))
    * manipulate hydra config using command line: `python train.py --config-name config_transpose` (check [here](https://hydra.cc/docs/advanced/hydra-command-line-flags/))
    * manipulate hydra config and override hydra args: `python train.py --config-name config_transpose 'task_name="override_args"'`