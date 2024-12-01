import os
import json
import math
import random
import pathlib
import numpy as np
from loss import *
import pandas as pd
from config import *
import config
import earthpy.plot as ep
import rasterio
from tensorflow import keras
import earthpy.spatial as es
from datetime import datetime
import matplotlib.pyplot as plt
from progressbar import ProgressBar
import moviepy.video.io.ImageSequenceClip
from rasterio.plot import show

from dataset import read_img, transform_data





# Callbacks and Prediction during Training
# ----------------------------------------------------------------------------------------------
class SelectCallbacks(keras.callbacks.Callback):
    def __init__(self, val_dataset, model):
        """
        Summary:
            callback class for validation prediction and create the necessary callbacks objects
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model object
        Return:
            class object
        """
        super(keras.callbacks.Callback, self).__init__()

        self.val_dataset = val_dataset
        self.model = model
        self.config = config
        self.callbacks = []

    def lr_scheduler(self, epoch):
        """
        Summary:
            learning rate decrease according to the model performance
        Arguments:
            epoch (int): current epoch
        Return:
            learning rate
        """
        drop = 0.5
        epoch_drop = self.config.epochs / 8.
        lr = self.config.learning_rate * \
            math.pow(drop, math.floor((1 + epoch) / epoch_drop))
        return lr

    def on_epoch_end(self, epoch, logs={}):
        """
        Summary:
            call after every epoch to predict mask
        Arguments:
            epoch (int): current epoch
        Output:
            save predict mask
        """
        if (epoch % self.config.val_plot_epoch == 0):  # every after certain epochs the model will predict mask
            val_show_predictions(self.val_dataset, self.model)  # save image/images with their mask, pred_mask and accuracy
            
        print("...............................................................")
        print(self.config.checkpoint_name)
        print("...............................................................")

    def get_callbacks(self, val_dataset, model):
        """
        Summary:
            creating callbacks based on configuration
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model class object
        Return:
            list of callbacks
        """
        if self.config.csv:  # save all type of accuracy in a csv file for each epoch
            self.callbacks.append(keras.callbacks.CSVLogger(os.path.join(
                self.config.csv_log_dir, self.config.csv_log_name), separator=",", append=False))

        if self.config.checkpoint:  # save the best model
            self.callbacks.append(keras.callbacks.ModelCheckpoint(os.path.join(
                self.config.checkpoint_dir, self.config.checkpoint_name), save_best_only=True))

        if self.config.tensorboard:  # Enable visualizations for TensorBoard
            self.callbacks.append(keras.callbacks.TensorBoard(log_dir=os.path.join(
                self.config.tensorboard_log_dir, self.config.tensorboard_log_name)))

        if self.config.lr:  # adding learning rate scheduler
            self.callbacks.append(
                keras.callbacks.LearningRateScheduler(schedule=self.lr_scheduler))

        if self.config.early_stop:  # early stop the training if there is no change in loss
            self.callbacks.append(keras.callbacks.EarlyStopping(
                monitor='my_mean_iou', patience=self.config.patience))

        if self.config.val_pred_plot:  # plot validated image for each epoch
            self.callbacks.append(SelectCallbacks(val_dataset, model))

        return self.callbacks

# Sub-ploting and save
# ----------------------------------------------------------------------------------------------
def display(display_list, idx, directory, score, exp, evaluation,src):
    """
    Summary:
        save all images into single figure
    Arguments:
        display_list (dict): a python dictionary key is the title of the figure
        idx (int) : image index in dataset object
        directory (str) : path to save the plot figure
        score (float) : accuracy of the predicted mask
        exp (str): experiment name
    Return:
        save images figure into directory
    """
    plt.figure(figsize=(12, 8))  # set the figure size
    title = list(display_list.keys())  # get tittle

    # plot all the image in a subplot
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        
        #plot nasadem image channel
        if title[i] == "DEM":  
            ax = plt.gca()
            #problems
            hillshade = es.hillshade(display_list[title[i]], azimuth=180)
            ep.plot_bands(
                display_list[title[i]],
                cbar=False,
                cmap="terrain",
                title=title[i],
                ax=ax
            )
            ax.imshow(hillshade, cmap="Greys", alpha=0.5)
            
        #plot VV or VH image channel
        elif title[i] == "VV" or title[i] == "VH":
            plt.title(title[i])
            plt.imshow((display_list[title[i]]), cmap='gray')
            plt.axis('off')
            
        # plot prediction mask on input image    
        elif 'Prediction_over_mask' in title[i]:
            plt.title(title[i])
            masked = np.ma.masked_where(display_list[title[i]] == 0, display_list[title[i]])
            plt.imshow(display_list["image"], 'jet', interpolation='none')
            plt.imshow(masked, 'jet', interpolation='none', alpha=0.8)
            plt.axis('off')
        elif title[i] == "combined_channels":
            ax = plt.gca()
            plt.title(title[i])
            show(display_list[title[i]],transform=src.transform, ax=ax)
            ax.grid(False)  # Turn off gridlines along the borders
            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_yticks([])  # Remove y-axis ticks
            ax.set_xlabel('')  # Remove x-axis label
            ax.set_ylabel('')  # Remove y-axis label
        else:
            plt.title(title[i])
            plt.imshow((display_list[title[i]]), cmap="gray")
            plt.axis('off')
    
    # create file name to save
    if evaluation:
        prediction_name = "{}_id_{}.png".format(exp, idx)
    else:
        prediction_name = "{}_id_{}_miou_{:.4f}.png".format(exp, idx, score) 
    
    plt.savefig((directory / prediction_name),
                bbox_inches = 'tight',
                dpi = 800)
    plt.clf()
    plt.cla()
    plt.close()
    
# this function is only required when we need prediction over mask during evaluation
def display_label(img, img_path, directory):
    """
    Summary:
        save only predicted labels
    Arguments:
        img (np.array): predicted label
        img_path (str) : source image path
        directory (str): saving directory
    Return:
        save images figure into directory
    """
    
    img_path_split = os.path.split(img_path)
    #problems
    if 'umm_' in img_path_split[1]:
        img_name = img_path_split[1][ : 4] + 'road_' + img_path_split[1][4 : ]
    elif 'um_' in img_path_split[1]:
        img_name = img_path_split[1][ : 3] + 'lane_' + img_path_split[1][3 : ]
    else:
        img_name = img_path_split[1][ : 3] + 'road_' + img_path_split[1][3 : ]
    
    plt.imsave(directory / img_name, img)
    

# Plot Test Data Prediction
# ----------------------------------------------------------------------------------------------
def test_eval_show_predictions(dataset, model):
    """
    Summary:
        predict test dataset and evaluation dataset and save images in the respective folder. for evaluation, 
        there will no mask/label available. So, the display function plot the figure for test and eval dataset in different way.
    Arguments:
        dataset (object): MyDataset class object
        model (object): keras.Model class object
    Return:
        merged patch image
    """
    global eval_dir, p_eval_dir, test_dir, p_test_dir
    # predict patch images and merge together
    if evaluation:
        var_list = [eval_dir, p_eval_dir]
    else:
        var_list = [test_dir, p_test_dir]

    with var_list[1].open() as j:  # opening the json file
        patch_test_dir = json.loads(j.read())

    df = pd.DataFrame.from_dict(patch_test_dir)  # read as panadas dataframe
    test_dir = pd.read_csv(var_list[0])  # get the csv file
    total_score = 0.0
    print(test_dir)
    # loop to traverse full dataset
    for i in range(len(test_dir)):
        mask_s = transform_data(
            read_img(test_dir["masks"][i], label=True), num_classes)
        mask_size = np.shape(mask_s)
        # for same mask directory get the index
        #problems
        idx = df[df["masks"] == test_dir["masks"][i]].index

        # construct a single full image from prediction patch images
        pred_full_label = np.zeros((mask_size[0], mask_size[1]), dtype=int)
        #problems
        for j in idx:
            p_idx = patch_test_dir["patch_idx"][j]          # p_idx 
            feature, mask, _ = dataset.get_random_data(j)
            pred_mask = model.predict(feature)
            pred_mask = np.argmax(pred_mask, axis=3)
            pred_full_label[p_idx[0]:p_idx[1],
                            p_idx[2]:p_idx[3]] = pred_mask[0]   # [start hig: end index, ]

        # read original image and mask
        feature_img = read_img(test_dir["feature_ids"][i])          #, in_channels=in_channels
        mask = transform_data(
            read_img(test_dir["masks"][i], label=True), num_classes)
        
        # calculate keras MeanIOU score
        m = keras.metrics.MeanIoU(num_classes=num_classes)
        m.update_state(np.argmax([mask], axis=3), [pred_full_label])
        score = m.result().numpy()
        total_score += score



            

        display({"VV": feature_img[:,:,0],      # change in the key "image" will have to change in the display
                "VH": feature_img[:,:,1],
                "DEM": feature_img[:,:,2],
                "Mask": np.argmax([mask], axis=3)[0],
                "Prediction (miou_{:.4f})".format(score): pred_full_label},
                i, prediction_eval_dir, score, experiment, evaluation)


# Plot Val Data Prediction
# ----------------------------------------------------------------------------------------------
def pct_clip(array,pct=[2.5,97.5]):
    array_min, array_max = np.nanpercentile(array,pct[0]), np.nanpercentile(array,pct[1])
    clip = (array - array_min) / (array_max - array_min)
    clip[clip>1]=1
    clip[clip<0]=0
    return clip

# def false_colour_read(path):
#     img= np.zeros((3,512,512))
#     with rasterio.open(path) as src:
#         for i in range(3):
#             img[i,:,:]= pct_clip(src.read(i+1))   
#     return img, src

def false_colour_read(path):
    with rasterio.open(path) as src:
        global h,w
        h,w =src.shape
        img= np.zeros((3,h,w))
        # print(img.shape)
        for i in range(3):
            img[i,:,:]= pct_clip(src.read(i+1))
            
    return img, src


def val_show_predictions(dataset, model):
    """
    Summary:
        predict patch images and merge together during training
    Arguments:
        dataset (object): MyDataset class object
        model (object): keras.Model class object
    Return:
        merged patch image
    """

    with p_valid_dir.open() as j:  # opening the json file
        patch_valid_dir = json.loads(j.read())

    df = pd.DataFrame.from_dict(patch_valid_dir)  # read as panadas dataframe
    val_dir = pd.read_csv(valid_dir)  # get the csv file

    if index == -1:
        i = random.randint(0,(len(val_dir)-1))
    else:
        i = index

    mask_s = transform_data(
            read_img(val_dir["masks"][i], label=True), num_classes)
    mask_size = np.shape(mask_s)              
    
    #checking mask from both csv and json file and taking indices from the json file
    idx = df[df["masks"] == val_dir["masks"][i]].index

    # construct a single full image from prediction patch images
    pred_full_label = np.zeros((mask_size[0], mask_size[1]), dtype=int)
    for j in idx:
        p_idx = patch_valid_dir["patch_idx"][j]
        feature, mask, indexNum = dataset.get_random_data(j)
        pred_mask = model.predict(feature)
        pred_mask = np.argmax(pred_mask, axis = 3)
        pred_full_label[p_idx[0]:p_idx[1], p_idx[2]:p_idx[3]] = pred_mask[0]   

    # read original image and mask
    feature_img , src = false_colour_read(val_dir["feature_ids"][i])          #, in_channels=in_channels
    mask = transform_data(read_img(val_dir["masks"][i], label=True), num_classes)
        
    # calculate keras MeanIOU score
    m = keras.metrics.MeanIoU(num_classes=num_classes)
    m.update_state(np.argmax([mask], axis = 3), [pred_full_label])
    score = m.result().numpy()




    display({"combined_channels": feature_img,      # change in the key "image" will have to change in the display
        "Mask": np.argmax([mask], axis=3)[0],
        "Prediction (miou_{:.4f})".format(score): pred_full_label 
        }, i, prediction_val_dir, score, experiment, evaluation,src=src)


# Model Output Path
# ----------------------------------------------------------------------------------------------

def create_paths(test = False, eval = False):
    """
    Summary:
        creating paths for train and test if not exists
    Arguments:
        test (bool): boolean variable for test directory create
    Return:
        create directories
    """
    if test:
        pathlib.Path(prediction_test_dir).mkdir(parents=True, exist_ok=True)
    if eval:
        if video_path == True:
            pathlib.Path(dataset_dir / "video_frame").mkdir(parents=True, exist_ok=True)
        pathlib.Path(prediction_eval_dir).mkdir(parents=True, exist_ok=True)
    else: # training
        pathlib.Path(csv_log_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(prediction_val_dir).mkdir(parents=True, exist_ok=True)

def frame_to_video(fname, fps=30):
    """
    Summary:
        create video from frames
    Arguments:
        fname (str): name of the video
    Return:
        video
    """
    
    image_folder=prediction_eval_dir
    image_names = os.listdir(image_folder)
    image_names = sorted(image_names)
    image_files = []
    for i in image_names:
        image_files.append(image_folder / i)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(fname)



        
        


