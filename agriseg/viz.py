import os
import cv2
import json
import config
import pathlib
import math
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
from config import *
import earthpy.plot as ep
import earthpy.spatial as es
from dataset import read_img
from matplotlib import pyplot as plt
import subprocess
import pyperclip

import rasterio as rio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np

train_df = pd.read_csv(config.train_dir)
test_df =  pd.read_csv(config.test_dir)
valid_df = pd.read_csv(config.valid_dir)
p_train_json = config.p_train_dir
p_test_json = config.p_test_dir
p_valid_json = config.p_valid_dir









def pct_clip(array,pct=[2.5,97.5]):
    array_min, array_max = np.nanpercentile(array,pct[0]), np.nanpercentile(array,pct[1])
    clip = (array - array_min) / (array_max - array_min)
    clip[clip>1]=1
    clip[clip<0]=0
    return clip








def false_colour_read_bt(path):
    with rasterio.open(path) as src:
        global h,w
        h,w =src.shape
        img= np.zeros((3,h,w))
        # print(img.shape)
        for i in range(3):
            img[i,:,:]= pct_clip(src.read(i+1))
            
    return img, src


visualization_dir = root_dir / "data/dataset-nsr-patch"

def re(p):
    with rasterio.open(p) as src:
        pass
    return src

src = re("/agriseg/data/dataset-nsr-3-same/input/tile_2_0_05-NSR_Area1_18-19_Fin.tif")
def display_all(data, name):
    """
    Summary:
        save all images into single figure
    Arguments:
        data : data file holding images path
        directory (str) : path to save images
    Return:
        save images figure into directory
    """
    pathlib.Path((visualization_dir / "display")).mkdir(parents=True, exist_ok=True)
    pathlib.Path((visualization_dir / "display"/"train")).mkdir(parents=True, exist_ok=True)
    pathlib.Path((visualization_dir / "display"/"test")).mkdir(parents=True, exist_ok=True)
    pathlib.Path((visualization_dir / "display"/"valid")).mkdir(parents=True, exist_ok=True)


    for i in range(len(data)):
        image,mask = data[i]
        display_list = {"image": image, "label": mask}

        plt.figure(figsize=(12, 8))
        title = list(display_list.keys())

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            if title[i]=='image':
                ax = plt.gca()
                display_list[title[i]] = np.swapaxes(display_list[title[i]],3,1)
                show(display_list[title[i]],transform=src.transform, ax=ax)
            else:
                display_list[title[i]] = np.squeeze(display_list[title[i]],axis=0)
                display_list[title[i]] = np.argmax(display_list[title[i]],axis=2)
                print(display_list[title[i]].shape)
                plt.imshow((display_list[title[i]]), cmap="gray")
            plt.axis("off")

        prediction_name = "img_id_{}.png".format(np.random.randint(0, len(data)))  # create file name to save
        plt.savefig(
            os.path.join((visualization_dir / "display"/ name), prediction_name),
            bbox_inches="tight",
            dpi=800,
        )
        plt.clf()
        plt.cla()
        plt.close()





