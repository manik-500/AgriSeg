import os
import time
import numpy as np
import argparse
from loss import *
from config import *
from metrics import get_metrics
import tensorflow_addons as tfa
from dataset import get_train_val_dataloader
from tensorflow.keras.models import load_model
from model import get_model, get_model_transfer_lr
from utils import SelectCallbacks, create_paths
import random
seed_num = random.randint(1,1000)
random.seed(seed_num)
tf.config.optimizer.set_jit("True")


# Parsing variable ctrl + /
# ----------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir")
parser.add_argument("--model_name")
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--experiment")
parser.add_argument("--gpu")
parser.add_argument("--transfer_lr")

args = parser.parse_args()

root_dir = args.root_dir if args.root_dir is not None else root_dir
model_name = args.model_name if args.model_name is not None else model_name
epochs = args.epochs if args.epochs is not None else epochs
batch_size = args.batch_size if args.batch_size is not None else batch_size
experiment = args.experiment if args.experiment is not None else experiment
gpu = args.gpu if args.gpu is not None else gpu
transfer_lr = args.transfer_lr if args.transfer_lr is not None else transfer_lr


print(f"root_dir:{root_dir} model_name: {model_name}  epochs:{epochs} \ batch_size:{batch_size}\ experiment:{experiment}\ gpu:{gpu}")

# Set up train configaration
# ----------------------------------------------------------------------------------------------
create_paths(test = False)


# setup gpu
# ----------------------------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Print Experimental Setup before Training
# ----------------------------------------------------------------------------------------------
print("\n--------------------------------------------------------")
print("Model = {}".format(model_name))
print("Epochs = {}".format(epochs))
print("Batch Size = {}".format(batch_size))
print("Preprocessed Data = {}".format(os.path.exists(train_dir)))
print("Class Weigth = {}".format(str(weights)))
print("Experiment = {}".format(str(experiment)))
print("--------------------------------------------------------\n")



# Dataset
# ----------------------------------------------------------------------------------------------
train_dataset, val_dataset = get_train_val_dataloader()
# from viz import display_all
# display_all(train_dataset,"test")
# kk
# print(len(train_dataset[0]))
# im,msk=train_dataset[0]
# for im,msk in train_dataset:
# print(im.shape,msk.shape)

# lst = [data[0] for data in train_dataset]
# for im in lst:
#     print((np.mean(im),np.std(im)))
#     print("...................")
# print("final")
# print(np.mean(lst), np.std(lst))
# kk

# Metrics
# ----------------------------------------------------------------------------------------------
metrics = list(get_metrics().values())  # [list] required for new model 
custom_obj = get_metrics() # [dictionary] required for transfer learning & fine tuning

# Optimizer
# ----------------------------------------------------------------------------------------------
learning_rate = 0.001
weight_decay = 0.0001
adam = tfa.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay)

# Loss Function
# ----------------------------------------------------------------------------------------------
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True) # required for new model
custom_obj['loss'] = focal_loss() # required for transfer learning/fine-tuning

# Compile
# ----------------------------------------------------------------------------------------------
# transfer learning
if (os.path.exists(load_model_dir / load_model_name)) and transfer_lr:
    print("\n---------------------------------------------------------------------")
    print("Transfer Learning from model checkpoint {}".format(load_model_name))
    print("---------------------------------------------------------------------\n")
    # load model and compile
    model = load_model((load_model_dir / load_model_name), custom_objects=custom_obj, compile=True)
    model = get_model_transfer_lr(model, num_classes)
    model.compile(optimizer=adam, loss=loss, metrics=metrics)

else:
    # fine-tuning
    if (os.path.exists(load_model_dir / load_model_name)):
        print("\n---------------------------------------------------------------")
        print("Fine Tuning from model checkpoint {}".format(load_model_name))
        print("---------------------------------------------------------------\n")
        # load model and compile
        model = load_model((load_model_dir / load_model_name), custom_objects=custom_obj, compile=True)

    # new model
    else:
        print("\n-------------------------------------------------------------------")
        print("Training only")
        print("-------------------------------------------------------------------\n")
        model = get_model()
        model.compile(optimizer=adam, loss=loss, metrics=metrics)

# Callbacks
# ----------------------------------------------------------------------------------------------
loggers = SelectCallbacks(val_dataset, model)
model.summary()
# Fit
# ----------------------------------------------------------------------------------------------
t0 = time.time()
history = model.fit(train_dataset,
                    verbose = 1,
                    epochs = epochs,
                    validation_data = val_dataset,
                    shuffle = False,
                    callbacks = loggers.get_callbacks(val_dataset, model),
                    )

print("\n----------------------------------------------------")
print("training time minute: {}".format((time.time()-t0)/60))
print("\n----------------------------------------------------")