import os
import time
import argparse
from loss import *
from config import *
from metrics import *
from tensorflow import keras
from dataset import get_test_dataloader
from tensorflow.keras.models import load_model
from utils import create_paths, test_eval_show_predictions, frame_to_video

# Parsing variable
# ----------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir")
parser.add_argument("--model_name")
parser.add_argument("--load_model_name")
parser.add_argument("--index", type=int)
parser.add_argument("--experiment")
parser.add_argument("--gpu")
parser.add_argument("--evaluation")
parser.add_argument("--video_path")
args = parser.parse_args()

root_dir = args.root_dir if args.root_dir is not None else root_dir
model_name = args.model_name if args.model_name is not None else model_name
load_model_name = args.load_model_name if args.load_model_name is not None else load_model_name
index = args.index if args.index is not None else index
experiment = args.experiment if args.experiment is not None else experiment
gpu = args.gpu if args.gpu is not None else gpu
video_path = args.video_path if args.video_path is not None else video_path


print(f"root_dir:{root_dir} model_name: {model_name}  load_model_name:{load_model_name} \ index:{index}\ experiment:{experiment}\ gpu:{gpu}\ evaluation:{evaluation}\video_path:{video_path}")

# Training Start Time
# ----------------------------------------------------------------------------------------------
t0 = time.time()


# Set up test configaration
# ----------------------------------------------------------------------------------------------
if evaluation:
    create_paths(eval=True)
    print("evaluation")
else:
    create_paths(test=True)
    print("test")


# setup gpu
# ----------------------------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# Load Model
# ----------------------------------------------------------------------------------------------
print("Loading model {} from {}".format(load_model_name, load_model_dir))
# with strategy.scope(): # if multiple GPU is required
model = load_model((load_model_dir / load_model_name), compile=False)



# Dataset
# ----------------------------------------------------------------------------------------------
test_dataset = get_test_dataloader()


# Prediction Plot
# ----------------------------------------------------------------------------------------------
print("--------------------------------------")
print("Saving test/evaluation predictions...")
print("--------------------------------------")
test_eval_show_predictions(test_dataset, model)
print("Call test_eval_show_predictions")
print("--------------------------------------")


# Test Score
# ----------------------------------------------------------------------------------------------
if not evaluation:
    metrics = list(get_metrics().values())
    adam = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam, loss=focal_loss(), metrics=metrics)
    model.evaluate(test_dataset)


# Frame to Video
# ----------------------------------------------------------------------------------------------
if video_path == True:
    fname = dataset_dir + "prediction.avi"
    frame_to_video(fname, fps=30)


# Training time Calculation (End)
# ----------------------------------------------------------------------------------------------
print("training time sec: {}".format((time.time() - t0)))

print("--------------------------------------")
print(f"saving prediction: {prediction_val_dir}")
print("--------------------------------------")