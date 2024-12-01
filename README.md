# AgriSeg
**AgriSeg** is a pipeline for semantic segmentation of rice crops using SAR imagery. It preprocesses SAR data and applies state-of-the-art Semantic Segmentation models for precise rice field segmentation. Scalable and customizable, AgriSeg supports agricultural monitoring, enabling data-driven decisions for sustainable crop management.




# **Pipeline**

## **Introduction**

The pipeline offers a robust solution for handling image data of arbitrary sizes by employing a systematic approach of patching images into fixed-size patches. This not only enables efficient data augmentation but also facilitates training models on very high-resolution data that would otherwise exceed GPU memory limitations.

The process of training begins with the `train.py` script, where paths are created and essential utilities from `utils.py` are invoked. The data loading and preparation steps are handled in `dataset.py`, where the dataset is generated, split, and train, valid and test CSV files are saved. The image patching process, including saving co-ordinate of patches to train, valid, and json files is executed, followed by data augmentation and transformation through the `Augment` class. The dataset is then prepared using `MyDataset` class, which handles data fetching and transformation.

Metrics calculation, focal loss computation, and model initialization are handled in separate modules respectively `metrics.py`, `loss.py`, `model.py`, ensuring modularity and ease of management. Callback selection, including learning rate scheduling and validation visualization, is orchestrated through `SelectCallbacks` class.

For evaluation, two distinct scenario are provided in `test.py`. In the first scenario (evaluation = False), the evaluation dataset is prepared similarly to the training dataset, and predictions are made and displayed for analysis. In the second scenario (evaluation = True), additional steps for generating evaluation CSVs named eval_csv_gen is called

Throughout the pipeline, emphasis is placed on modularity, allowing for easy integration of new components or modifications. Understanding the intricacies of this pipeline is paramount for leveraging its capabilities effectively. For a deeper dive into its workings, referencing the FAPNET paper is recommended, providing insights into the architecture and rationale behind its design.

## **Dataset**

## Input Data Organization Guidelines

To ensure proper organization of input features, adhere to the following guidelines:

1. **Separate Files for Different Input Channels**: Each channel of input feature should be stored in separate files.

2. **Unique Filename for Each Channel**: Ensure that filenames are unique for all channels, except for the last portion of the filename.

3. **Example Naming Convention**:
   - For instance, if the name of the VV channel is `rice_01_vv.tif`, then the corresponding VH and DEM files should be named as follows:
     - VH: `rice_01_vh.tif`
     - DEM: `rice_01_nasadem.tif`
   - The mask filename should retain the first unique portion, such as `rice_01.tif` in the example provided.

4. **Key Consideration**:
   - The first part of the filename for all channels should be unique, while the last part should vary to denote different channels. But the mask name should be the first unique portion of the name.

## Input Data Organization Guidelines

To ensure proper organization of input features, adhere to the following guidelines:

1. **Separate Files for Different Input Channels**: Each channel of input feature should be stored in separate files.

2. **Unique Filename for Each Channel**: Ensure that filenames are unique for all channels, except for the last portion of the filename.

3. **Example Naming Convention**:
   - For instance, if the name of the VV channel is `rice_01_vv.tif`, then the corresponding VH and DEM files should be named as follows:
     - VH: `rice_01_vh.tif`
     - DEM: `rice_01_nasadem.tif`
   - The mask filename should retain the first unique portion, such as `rice_01.tif` in the example provided.

4. **Key Consideration**:
   - The first part of the filename for all channels should be unique, while the last part should vary to denote different channels. But the mask name should be the first unique portion of the name.


## **Model**

Our current pipeline supports semantic segmentation for both binary and multi-class tasks. To configure the pipeline for your desired number of classes, simply adjust the `num_classes` variable in the `config.py` file. No modifications are required in the `model.py` file. This straightforward configuration process allows for seamless adaptation to different classification requirements without the need for extensive code changes.

## **Setup**

First clone the github repo in your local or server machine by following:

```
git clone https://github.com/manik-500/AgriSeg.git
```

Change the working directory to project root directory. Use Conda/Pip to create a new environment and install dependency from `requirement.txt` file. The following command will install the packages according to the configuration file `requirement.txt`.

```
pip install -r requirements.txt
```

Keep the above mention dataset in the data folder that give you following structure.

```
   data
      file_01_chip0_vv.tif
      file_01_chip0_vh.tif
      file_01_chip0_nasadem.tif
      file_01_chip0.tif
```


## **Experiment**

* ### **Comprehensive Full Resolution (CFR)**:
This experiment utilizes the dataset as it is. The image size must follow $2^n$ format, such as $256 \times 256$, $512 \times 512$, etc. If we choose $300 \times 300$, which is not in $2^n$ format, this experiment will not work.




```
python train.py --root_dir YOUR_ROOT_DIR \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --experiment cfr 
```
##### Example:
```
python train.py --root_dir /agriseg/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --experiment cfr 
```

* ### **Comprehensive Full Resolution with Class Balance (CFR-CB)**:
We balance the dataset biasness towards non-water class in this experiment.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --experiment cfr_cb \
```
##### Example:

```
python train.py --root_dir /agriseg/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --experiment cfr_cb 
```

* ### **Patchify Half Resolution (PHR)**:
In this experiment we take all the patch images for each chip. Data preprocessing can handle any image shape and convert it to a specific patch size.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --experiment phr \
```
##### Example:
```
python train.py --root_dir /agriseg/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --experiment phr 
```

* ### **Patchify Half Resolution with Class Balance (PHR-CB)**:
In this experiment we take a threshold value (19%) of water class and remove the patch images for each chip that are less than threshold value.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --experiment phr_cb 
```
##### Example:
```
python train.py --root_dir /agriseg/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --experiment phr_cb 
```

* **Patchify Half Resolution with Class Balance Weight (PHR-CBW)**:

### Resolving Data Imbalance Issues

To address data imbalance problems, one can utilize the following method:

If encountering a scenario where there are only two classes, with the first class representing 60% of the dataset and the second class comprising 40%, the imbalance can be rectified by setting `weights= True` and specifying `balance_weights = [4,6]` in the config.py file.


However, in cases where there are three classes, and one of them is considered a boundary that should be disregarded, the weight of the corresponding class must be set to 0. For instance, in the command line, this would be denoted as `balance_weights = [4,6,0]` in the config.py file.


```
python train.py --root_dir YOUR_ROOT_DIR \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --experiment phr_cb 
```
##### Example:
```
python train.py --root_dir /agriseg/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --experiment phr_cb \
```

## Transfer Learning, Fine-tuning, and Training from Scratch

In this project, the behavior of the model training process is determined by certain variables in the `config.py` file. Here's how they influence the training approach:

### **Transfer Learning**

- If `transfer_lr` variable is set to `True` and a `load_model_name` is provided in `config.py`, the model will undergo transfer learning.

### **Fine Tuning**

- If `transfer_lr` is set to `False` but a `load_model_name` is provided in `config.py`, the model will undergo fine-tuning.

### **Training from Scratch**

- If `transfer_lr` is set to `False` and no `load_model_name` is provided in `config.py`, the model will be trained from scratch.



## Testing

* ### **CFR and CFR-CB Experiment**

Run following model for evaluating train model on test dataset.

```
python test.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --experiment cfr 
```
##### Example:
```
python test.py \
    --dataset_dir /agriseg/data/ \
    --model_name unet \
    --load_model_name my_model.hdf5 \
    --experiment cfr 
```

* ### **PHR and PHR-CB Experiment**

```
python test.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name my_model.hdf5 \
    --experiment phr 
```
##### Example:
```
python test.py \
    --dataset_dir /agriseg/data/ \
    --model_name unet \
    --load_model_name my_model.hdf5 \
    --experiment phr 
```


### **Evaluation from Image**

If you have the images without mask, we need to do data pre-preprocessing before passing to the model checkpoint. In that case, run the following command to evaluate the model without any mask.

1. You can check the prediction of test images inside the `logs > prediction > YOUR_MODELNAME > eval > experiment`.

```
python project/test.py \
    --dataset_dir YOUR_IMAGE_DIR/ \
    --model_name fapnet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --experiment road_seg \
    --gpu YOUR_GPU_NUMBER \
    --evaluation True \
```
##### Example:
```
python project/test.py \
    --dataset_dir /agriseg/data/ \
    --model_name fapnet \
    --load_model_name my_model.hdf5 \
    --experiment road_seg \
    --gpu 0 \
    --evaluation True \
```






<!-- ## **SAR Rename Function**

This pipeline requires the dataset name to be in a specific format in order to correctly generate CSV files for the training, validation, and testing datasets. The details of the specific file format name are described in the dataset section. Our current `rename_files` function, defined in the `sar_visualization.ipynb` notebook, identifies whether the file name starts with 'VV' or 'VH', or 'DEM' and adapts it to our specific file name format. If the given dataset name is formatted differently other then file name starts with 'VV' or 'VH', or 'DEM', then we need to modify the `rename_files` function to ensure compatibility with the provided datasets. -->

## Question:

### How can we rename SAR files to be compatible with our pipeline?

---

## Answer:

To ensure compatibility with our pipeline, we need to modify the `rename_files` function defined in the `visualization.ipynb` notebook. This function currently identifies file names starting with 'VV', 'VH', or 'DEM' and adjusts them to fit our required format for generating CSV files. However, if the dataset names are formatted differently but still start with 'VV', 'VH', or 'DEM', we must update the `rename_files` function accordingly.

To achieve this, we can enhance the `rename_files` function to recognize various dataset name formats and adapt them to our specified format. This may involve adding conditional statements within the function to handle different naming conventions.

By refining the `rename_files` function in this manner, we can ensure that SAR files are renamed appropriately to seamlessly integrate with our pipeline, facilitating the generation of CSV files for training, validation, and testing datasets.




<!-- ## **Plot During Validation**
During the training process, validation will occur at specific epochs, which can be configured within the `config.py` module. Within `config.py`, there exists a variable named `val_plot_epoch`. If its value is set to 3, validation will be conducted after every 3 epochs. The resulting validation plots will be saved in the following folder where `experiment` means experimental names that we define in the config.py module.

```
root/logs/prediction/model_name/validation/experiment
``` -->

### Question: How to plot during validation?

### Answer:

During validation, plots can be generated by configuring the `val_plot_epoch` variable within the `config.py` module. This variable determines the frequency of validation during training. For instance, setting `val_plot_epoch` to 3 means validation will be conducted after every 3 epochs. The resulting validation plots will be saved in the following folder structure:

```
root/logs/prediction/model_name/validation/experiment
```

In this structure, `root` refers to the root directory of the project, `model_name` is the name of the model being trained, and `experiment` denotes the experimental configurations defined within the `config.py` module.


<!-- ## **Plot During Test**
### Running Test.py for Evaluation

To perform predictions on the test dataset and save plots, execute the following command in the terminal:

```
python test.py --evaluation False
```

This command will generate predictions on the test dataset and store plots in the following directory:

```
root_dir/logs/prediction/model_name/test/experiment
``` -->


### Question: How to plot during test?

### Answer:

To plot during testing, follow these steps:

1. Execute the command in the terminal:
    ```
    python test.py --evaluation False
    ```

   This command generates predictions on the test dataset without evaluation.

2. After executing the command, the plots will be stored in the following directory:
    ```
    root_dir/logs/prediction/model_name/test/experiment
    ```

   Ensure to replace `root_dir`, `model_name`, and `experiment` with the appropriate directory names based on your project setup.

  
<!-- ## **Plot During Evaluation**

To initiate evaluation on the test.py script, execute the following command in the terminal:

```
test.py --evaluation True
```

Upon execution, predictions will be generated using the evaluation dataset, and the resulting plots will be saved in the designated folder structure:

```
root_dir/logs/prediction/model_name/eval/experiment
``` -->

### Question: How to plot during evaluation?

### Answer:

To plot during evaluation, you need to follow these steps:

1. **Execute Evaluation Script**: Initiate evaluation on the `test.py` script by running the following command in the terminal:
   
   ```
   test.py --evaluation True
   ```

2. **Generate Predictions**: Upon execution, predictions will be generated using the evaluation dataset.

3. **View Resulting Plots**: The resulting plots will be automatically saved in the designated folder structure:
   
   ```
   root_dir/logs/prediction/model_name/eval/experiment
   ```

   Here, replace `root_dir` with the root directory of your project, `model_name` with the name of your model, and `experiment` with the specific experiment or evaluation run.


<!-- ## **Input Channel Selection**
In our `config.py` file, there exists a dedicated variable named "channel_type". This variable is designed to accept a list of strings representing specific channel names. For instance: `channel_type = ["vv", "vh", "nasadem"]`. You can include any number of channel names within this list as strings. The will dynamically calculate the total number of channels and perform normalization on a per-channel basis according to the provided channel names.
One thing to note here is that there is a variable named  `label_norm` where mean and std on per channel basis is stored in dictionary. If we want to include new channle type in the `channel_type` list then we must include the `channel_type` along with there mean and std in the `label_norm`  -->


### Question: How to select different channel types?
### Answer
To select different channel types, follow these steps:

1. **Open `config.py`**: Navigate to the `config.py` file in your project directory.

2. **Locate `channel_type` variable**: Inside the `config.py` file, find the variable named "channel_type".

3. **Update channel names**: Modify the `channel_type` variable to include the desired channel names as strings within a list. For example:
   
    ```python
    channel_type = ["vv", "vh", "nasadem", "new_channel"]
    ```

    You can include any number of channel names within this list as strings.

4. **Update label normalization**: If you're adding a new channel type to the `channel_type` list, ensure to update the `label_norm` variable. The `label_norm` variable contains mean and standard deviation values for each channel type. Add the mean and standard deviation values for the new channel type in the `label_norm` dictionary. For example:

    ```python
    label_norm = {
        'vv': ["_vv.tif", -12.204613877277495, 4.0422273221629785],
        'vh': ["_vh.tif", -18.96970667660666, 4.367189151420688],
        'nasadem': ["_nasadem.tif", 812.64422736, 425.41858324129265],
        'new_channel': ["_new_channel.tif", new_channel_mean, new_channel_std]
        # Add mean and std for the new channel type
    }
    ```

    Ensure to replace `new_channel_mean` and `new_channel_std` with the actual mean and standard deviation values for the new channel.

5. **Save changes**: After updating the `config.py` file, save the changes.

By following these steps, you can select different channel types by modifying the `channel_type` variable and ensuring proper label normalization in the `label_norm` variable.


## **Visualization of the Dataset**

To visualize the dataset, we must execute the display_all function as defined in the visualization.ipynb notebook. This function necessitates the CSV file that corresponds to the dataset we intend to visualize, along with a name parameter to establish a folder where the figure's visualization will be stored. For instance, calling 
```
display_all(data=train_df, name="train") 
```
accomplishes this task.


## **Action tree**


## Training Script: `train.py`
If we run train.py then the the following functions will be executed in the mentioned flow.
If we run train.py then the the following functions will be executed in the mentioned flow.
### Functions

- `create_paths(test=False)` - [utils.py]
- `get_train_val_dataloader()`
  - `data_csv_gen()`
    - `data_split()`
    - `save_csv()`
  - `patch_images()`
    - `save_patch_idx()`
      - `class_percentage_check()`
     - `write_json()`
  - `Augment()` - Class
    - `call()`
      - `read_img()`
  - `MyDataset()` - Class
    - `__getitem__()`
      - `read_img()`
      - `transform_data()`
    - `get_random_data()`
      - `read_img()`
      - `transform_data()`
- `get_metrics()` - [metrics.py]
  - `MyMeanIOU()` - Class
- `focal_loss()` - [loss.py]
- `get_model_transfer_lr(model, num_classes)` - [model.py]
- `get_model()` - [model.py]
  - `models` - call all model functions
- `SelectCallbacks()` - Class - [utils.py]
  - `__init__()`
  - `lr_scheduler()`
  - `on_epoch_end()`
    - `val_show_predictions()`
      - `read_img()`
      - `transform_data()`
      - `display()`
  - `get_callbacks()`

## Test Script: `test.py`
### Evaluation = False

#### If we run test.py with evaluation = False, then the the following functions will be executed in the mentioned flow.


#### If we run test.py with evaluation = False, then the the following functions will be executed in the mentioned flow.


- `create_paths()`
- `get_test_dataloader()`
  - `data_csv_gen()`
    - `data_split()`
    - `save_csv()`
  - `patch_images()`
    - `save_patch_idx()`
      - `class_percentage_check()`
    - `write_json()`
  - `MyDataset()` - Class
    - `__getitem__()`
      - `read_img()`
      - `transform_data()`
    - `get_random_data()`
      - `read_img()`
      - `transform_data()`
- `test_eval_show_predictions()`*
  - `read_img()`
  - `transform_data()`
  - `display()`
- `get_metrics()`

### Evaluation = True
#### If we run test.py with evaluation = True, then the the following functions will be executed in the mentioned flow.
#### If we run test.py with evaluation = True, then the the following functions will be executed in the mentioned flow.

- `create_paths()`
- `get_test_dataloader()`
  - `eval_csv_gen()`*
    - `save_csv()`
  - `patch_images()`
    - `save_patch_idx()`
      - `class_percentage_check()`
    - `write_json()`
  - `MyDataset()` - Class
    - `__getitem__()`
      - `read_img()`
      - `transform_data()`
    - `get_random_data()`
      - `read_img()`
      - `transform_data()`
- `test_eval_show_predictions()`*
  - `read_img()`
  - `transform_data()`
  - `display_label()`*
  - `display()`
- `get_metrics()`





