# A deep learning approach to galaxy morphology and rotation analysis.
COMP4560 12pt Honours project (S1/S2 2023) - Anthony Siharath

## Table of Contents

- [Project Description](#project-description)
- [File and Directory Structure](#file-and-directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Project Description

This project revolves around leveraging the capabilities of neural networks to classify galaxies based on their morphology and kinematics. By automating this classification process, we aim to enhance our understanding of galaxy evolution, formation, and their intricate interplay.

## File and Directory Structure

This section describes the files and directories included in your project.
- `src/`:  Source code files that define the project's core functionality.
    - `ModelWrapper.py`: Used to train, validate, and test the models.
    - `batch.bat`: A batch file for executing training, validation, and testing processes.
    - `classification_results.csv`: Testing results with model names, test accuracy, F1 scores, AUC (Area Under the Curve), and test loss.
    - `config.py`: Universal configuration of models: default directories, batch_size, max_epochs, learning rate. 
    - `data/`: Contains the spreadsheets and images of our gold sample.
    - `lightning_logs/`: Contains the checkpoints of trained models.
    - `models/`: Pytorch Lightning classes for EfficientNetB1, ResNet50, and ViT-Base.
    - `notebooks/`: Jupyter Notebooks for eda, prediction, training stability, and shap analysis.
    - `results.txt`: Contains the paths of checkpoints of trained models.
    - `top_3/`: Contains the paths to the checkpoints of top 3 performing models.
    - `train.py`: Orchestrates the process of model training, including data loading, model training, and checkpoint management.
    - `utils/`: Contains custom dataset classes, image transformations, and shap functions.
- `requirements.txt`: Specifies the Python dependencies and libraries required for the project to run successfully.

## Installation
You will need to download and extract the zip files `data.zip` and `top_3.zip` to the `src` directory.
The link for these files can be found here: https://drive.google.com/drive/folders/1AmCg_IFDaiDPt9ecrwKMDbRBUm7YhvwE?usp=sharing

Afterwards, install project dependencies by running the following command in the root directory:

```bash
pip install -r requirements.txt
```

## Usage
To train-validate-test a batch of models, add a line for each model to the file `batch.bat` or `batch.sh` in the following format:
```bash
python train.py -name {model}
```

For Example:
```bash
python train.py -name "effnet_6"
python train.py -name "effnet_7"
python train.py -name "effnet_8"
```

Run the batch file by using one of the following commands, depending on your operating system:  
For Windows:
```bash
batch.bat
```
For Unix-based systems:
```bash
./batch.sh
```

## Configuration
This configuration file (`config.py`) contains various parameters and settings used in the project. Here's a brief description of the attributes:

- `CURRENT_DIR`: The path to the current directory where this configuration file is located.
- `DATASET_DIR`: The directory path to the dataset used in the project, which includes data files.
- `IMG_DIR`: The directory path to the images within the dataset.
- `LOG_DIR`: The directory path for storing logs generated during training.
- `BATCH_SIZE`: The batch size used for training machine learning models.
- `LEARNING_RATE`: The learning rate used in the optimization process during training.
- `MAX_EPOCHS`: The maximum number of epochs for model training.
- `SEED`: The seed value used to ensure reproducibility in data splitting and randomness.
- `TRAIN_RATIO`, `VAL_RATIO`, and `TEST_RATIO`: The ratios used for splitting the dataset into training, validation, and test sets.

## Training
To train a specific model, you can simply run the following commands:
```bash
python train.py -name "effnet_6"
```
A checkpoint and train-validation logs will be saved to the folder `lightning_logs\`.
The path to each checkpoint is also saved to the text file `results.txt`.

## Results
The results of each trained model can be found in the file `classifcation_results.csv`.
- **Model Name**: The name of the model used for classification.
- **Test Accuracy (Balanced)**: The accuracy of the model on the test dataset, considering class balance.
- **Test Accuracy**: The overall accuracy of the model on the test dataset.
- **Test F1-Score (Macro)**: The F1-score of the model on the test dataset, calculated with a macro average.
- **Test F1-Score (Micro)**: The F1-score of the model on the test dataset, calculated with a micro average.
- **Test F1-Score (Weighted)**: The weighted F1-score of the model on the test dataset.
- **Test AUC (Area Under the ROC Curve)**: The area under the Receiver Operating Characteristic (ROC) curve for the model on the test dataset.
- **Test Loss**: The loss value of the model on the test dataset.

We have included the functions to plot the predictions according to the $\lambda_{Re}-\log{(M_{*} / M_{\Theta})}$ and $\lambda_{Re}-\varepsilon_{e}$ distributions for a model in the file `notebooks\prediction.ipynb`. 
By default, the notebook generates predictions using the best-performing models from EfficientNetB1, ResNet50, and ViT-Base.  
If you wish to use custom checkpoints for these models, you will need to modify the following checkpoint paths within the notebook:
```bash
from models.effnet_v import LitEffNetV
model_class = LitEffNetV
from utils.CustomDatasetImgVal import CustomDatasetImgVal
variables = ["CATID","SR_FR","LMSTAR"]
dataset_headings = "SR_FR"
model_params = {"in_channels": 1,"conditional_dim":1}
checkpoint_path = # use your custom checkpoint here
from utils.Transformations import greyscale_downscale_random_crop_rotate_transform
transformations = greyscale_downscale_random_crop_rotate_transform
effnet_wrapper = ModelWrapper(model_class,
                                dataset_headings,
                                CustomDatasetImgVal,
                                variables,
                                transformations,
                                model_name="effnet2",
                                model_params=model_params,
                                checkpoint_path=checkpoint_path)

```

We have also included functions to plot the training-validation curves in the file `notebooks\training_stability.ipynb`.  
If you wish to use custom checkpoints for these models, you will again need to modify the following checkpoint path in the notebook:
```bash
train_arr = []
val_arr = []
for i in range(5):
    log_dir = # use your custom checkpoint here
    train_data, val_data = extract_train_val_acc_epochs(log_dir, 43)
    train_arr.append(train_data)
    val_arr.append(val_data)
title = # your custom title
plot_train_val_acc_loss_epochs(train_arr, val_arr, title=title)
```

## SHAP
We have included the code used to visualise the shap outputs of a saved model in the file `notebooks/shap`.  
To use custom checkpoints for these models, modify the checkpoint path in the notebook.
```bash
from models.effnet_v import LitEffNetV
model_class = LitEffNetV
from utils.CustomDatasetImgVal import CustomDatasetImgVal
variables = ["CATID","SR_FR","LMSTAR"]
dataset_headings = "SR_FR"
model_params = {"in_channels": 1,"conditional_dim":1}
checkpoint_path = # use your custom checkpoint here
from utils.Transformations import *
transformations = greyscale_downscale_random_crop_rotate_transform
effnet_wrapper = ModelWrapper(model_class,
                                dataset_headings,
                                CustomDatasetImgVal,
                                variables,
                                transformations,
                                model_name="effnet2",
                                model_params=model_params,
                                checkpoint_path=checkpoint_path)

```

## Contact
Please do not hesitate to reach out with any inquiries by contacting me via email at: anthony_siharath@hotmail.com