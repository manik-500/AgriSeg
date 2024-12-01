from datetime import date
import json
from pathlib import Path

dtype = "dataset-nsr-1"
if dtype=="dataset-nsr-1":
    mean= 0.1495
    std= 0.0825
elif dtype=="dataset-nsr-2":
    mean= 0.1836
    std= 0.1195
elif dtype=="dataset-nsr-3":
    mean= 0.152
    std= 0.0876
elif dtype=="dataset-nsr-4":
    mean= 0.0356
    std= 0.0824
elif dtype=="dataset-nsr-5":
    mean= 0.0306
    std= 0.0747
elif dtype=="dataset-sar-1":
    mean= -11.6673
    std=  5.1528
elif dtype=="dataset-sar-2":
    mean= -10.9157
    std= 4.8009
elif dtype=="dataset-sar-3":
    mean= -10.066135
    std= 4.200516
elif dtype=="dataset-sar-4":
    mean= 0.75226486
    std= 2.6914093
elif dtype=="dataset-sar-5":
    mean= 1.6018738
    std= 2.857205
else:
    print("please enter correct dtype")


# Image Input/Output
# ----------------------------------------------------------------------------------------------
channel_type = "nsr"
in_channels = 3 
num_classes = 2
height = 9000 # for PHR-CB experiment patch size = height = width
width = 9000
rename = False # True if we nned to name the files name to be compatible with our pipeline's file format otherwise False

# Training
# ----------------------------------------------------------------------------------------------
model_name = "planet-2"
batch_size = 1
epochs = 3000
learning_rate = 3e-4
val_plot_epoch = 20
augment = False
transfer_lr = False
gpu = "0"

# Dataset
# --------------------------------mask--------------------------------------------------------------
weights = True # False if cfr or phr, True if cfr_cb or phr_cb
balance_weights = [2.2,7.8, 0] #weight for handling data imbalance issues
root_dir = Path("/agriseg")
dataset_dir = root_dir / f"data/{dtype}"
train_size = 0.8
train_dir = dataset_dir / "data/csv/train.csv"
valid_dir = dataset_dir / "data/csv/valid.csv"
test_dir = dataset_dir / "data/csv/test.csv"
eval_dir = dataset_dir / "data/csv/eval.csv"

# Patchify (phr & phr_cb experiment)
# ----------------------------------------------------------------------------------------------
patchify = True
patch_class_balance = True # whether to use class balance while doing patchify
patch_size = 2048 # height = width, anyone is suitable
stride = 1024
p_train_dir = dataset_dir / f"data/json/train_patch_phr_cb_{patch_size}_{stride}.json"
p_valid_dir = dataset_dir / f"data/json/valid_patch_phr_cb_{patch_size}_{stride}.json"
p_test_dir = dataset_dir / f"data/json/test_patch_phr_cb_{patch_size}_{stride}.json"
p_eval_dir = dataset_dir / f"data/json/eval_patch_phr_cb_{patch_size}_{stride}.json"

# Logger/Callbacks
# ----------------------------------------------------------------------------------------------
csv = True # required for csv logger
val_pred_plot = True
lr = True
tensorboard = True
early_stop = False
checkpoint = True
patience = 300 # required for early_stopping, if accuracy does not change for 500 epochs, model will stop automatically

# Evaluation
# ----------------------------------------------------------------------------------------------
load_model_name = 'planet-2_ex_2024-04-29_e_3000_p_2048_s_1024_nsr-1_ep_3000.hdf5'
load_model_dir = None #  If None, then by befault root_dir/model/model_name/load_model_name
evaluation = False # default evaluation value will not work
video_path = None    # If None, then by default root_dir/data/video_frame

# Prediction Plot
# ----------------------------------------------------------------------------------------------
index = -1 # by default -1 means random image else specific index image provide by user

#  Create config path
# ----------------------------------------------------------------------------------------------
if patchify:
    height = patch_size
    width = patch_size
    
# Experiment Setup
# ----------------------------------------------------------------------------------------------
# cfr, cfr-cb, phr, phr-cb, phr-cbw
experiment = f"{str(date.today())}_e_{epochs}_p_{patch_size}_s_{stride}_{dtype}_finetuned"

# Create Callbacks paths
# ----------------------------------------------------------------------------------------------
tensorboard_log_name = "{}_ex_{}_ep_{}".format(model_name, experiment, epochs)
tensorboard_log_dir = root_dir / "logs/tens_logger" / model_name

csv_log_name = "{}_ex_{}_ep_{}_dtype_{}.csv".format(model_name, experiment, epochs,dtype)
csv_log_dir = root_dir / "logs/csv_logger" / model_name   
csv_logger_path = root_dir / "logs/csv_logger"

checkpoint_name = "{}_ex_{}_ep_{}.hdf5".format(model_name, experiment, epochs)
checkpoint_dir = root_dir / "logs/model" / model_name

# Create save model directory
# ----------------------------------------------------------------------------------------------
if load_model_dir == None:
    load_model_dir = root_dir / "logs/model" / model_name
    
# Create Evaluation directory
# ----------------------------------------------------------------------------------------------
prediction_test_dir = root_dir / "logs/prediction" / model_name / "test" / experiment
prediction_eval_dir = root_dir / "logs/prediction" / model_name / "eval" / experiment
prediction_val_dir = root_dir / "logs/prediction" / model_name / "validation" / experiment

# Create Visualization directory
# ----------------------------------------------------------------------------------------------
visualization_dir = root_dir / "logs/visualization"
