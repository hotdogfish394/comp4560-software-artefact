import argparse
import torch
import warnings
from config import *
from ModelWrapper import ModelWrapper

# Suppress warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*exists and is not empty.*")
warnings.filterwarnings("ignore", ".*Set a lower value for log_every_n_steps.*")

# ... other imports ...
from models.effnet import LitEffNet
from models.resnet import LitResNet
from models.vit import LitViT
from models.effnet_v import LitEffNetV
from models.resnet_v import LitResNetV
from models.vit_v import LitViTV

# datasets
from utils.CustomDatasetImg import CustomDatasetImg
from utils.CustomDatasetImgVal import CustomDatasetImgVal

# transformations
from utils.Transformations import *

def train_and_evaluate_model(model_name, model_class, dataset_class, transform_func=None, extra_variables=None, model_params=None):
    """
    Train and evaluate a machine learning model.

    Args:
        model_name (str): Name of the model or experiment.
        model_class (class): The class of the machine learning model.
        dataset_class (class): The class of the dataset to use.
        transform_func (function, optional): A function for data augmentation and preprocessing.
        extra_variables (list, optional): Extra variables to condition the network during training.
        model_params (dict, optional): Additional model parameters.

    Returns:
        None
    """
    torch.set_float32_matmul_precision('medium')

    model_headings = "SR_FR"
    dataset = dataset_class
    
    variables = ["CATID", "SR_FR"]
    if extra_variables:
        variables.extend(extra_variables)
    
    if transform_func:
        my_transform = transform_func
    else:
        my_transform = None
    
    model_wrapper = ModelWrapper(model_class, model_headings, dataset, variables=variables, transform=my_transform, model_name=model_name, model_params=model_params)
    model_wrapper.train_model()
    model_wrapper.load_checkpoint() # load best checkpoint
    test_result = model_wrapper.test_model()
    predictions = model_wrapper.predict_model()
    model_wrapper.write_results_classification(predictions, test_result)

    print("Done!")
    print(model_wrapper.last_checkpoint)

    # write checkpoint to 
    with open("results.txt", "a") as f:
        checkpoint = f"{model_name}: {model_wrapper.last_checkpoint}\n"
        checkpoint = checkpoint.replace("\\", "/")
        f.write(checkpoint)

def main():
    parser = argparse.ArgumentParser(description="Training script with an experiment name.")
    parser.add_argument("-name", type=str, required=True, help="Name of the experiment")
    args = parser.parse_args()

    model_configs = {
        # models to train

        # models to train
        # model 1: black image
        "effnet_1": (LitEffNet, CustomDatasetImg, greyscale_downscale_random_crop_rotate_transform, [], {"in_channels": 1}),
        "resnet_1": (LitResNet, CustomDatasetImg, greyscale_downscale_centrecrop_rotate_transform, [], {"in_channels": 1}), 
        "vit_1": (LitViT, CustomDatasetImg, greyscale_downscale_centrecrop_rotate_transform, [], {"in_channels": 1}),

        # same accuracy
        # model 2: black image + stellar mass
        "effnet_2": (LitEffNetV, CustomDatasetImgVal, greyscale_downscale_random_crop_rotate_transform, ["LMSTAR"], {"in_channels": 1,"conditional_dim":1}),
        "resnet_2": (LitResNetV, CustomDatasetImgVal, greyscale_downscale_centrecrop_rotate_transform, ["LMSTAR"], {"in_channels": 1,"conditional_dim":1}),
        "vit_2": (LitViTV, CustomDatasetImgVal, greyscale_downscale_centrecrop_rotate_transform, ["LMSTAR"], {"in_channels": 1,"conditional_dim":1}),
        
        # model 3: black images + re
        "effnet_3": (LitEffNetV, CustomDatasetImgVal, greyscale_downscale_random_crop_rotate_transform, ["RE"], {"in_channels": 1,"conditional_dim":1}),
        "resnet_3": (LitResNetV, CustomDatasetImgVal, greyscale_downscale_centrecrop_rotate_transform, ["RE"], {"in_channels": 1,"conditional_dim":1}),
        "vit_3": (LitViTV, CustomDatasetImgVal, greyscale_downscale_centrecrop_rotate_transform, ["RE"], {"in_channels": 1,"conditional_dim":1}),
        
        # model 4: black images + re + stellar mass
        "effnet_4": (LitEffNetV, CustomDatasetImgVal, greyscale_downscale_random_crop_rotate_transform, ["RE","LMSTAR"], {"in_channels": 1,"conditional_dim":2}),
        "resnet_4": (LitResNetV, CustomDatasetImgVal, greyscale_downscale_centrecrop_rotate_transform, ["RE","LMSTAR"], {"in_channels": 1,"conditional_dim":2}),
        "vit_4": (LitViTV, CustomDatasetImgVal, greyscale_downscale_centrecrop_rotate_transform, ["RE","LMSTAR"], {"in_channels": 1,"conditional_dim":2}),

        # model 5: black image + re + stellar mass + ellip (shouldn't improve black image)
        "effnet_5": (LitEffNetV, CustomDatasetImgVal, greyscale_downscale_random_crop_rotate_transform, ["RE","LMSTAR","ELLIP"], {"in_channels": 1,"conditional_dim":3}), 
        "resnet_5": (LitResNetV, CustomDatasetImgVal, greyscale_downscale_centrecrop_rotate_transform, ["RE","LMSTAR","ELLIP"], {"in_channels": 1,"conditional_dim":3}),
        "vit_5": (LitViTV, CustomDatasetImgVal, greyscale_downscale_centrecrop_rotate_transform, ["RE","LMSTAR","ELLIP"], {"in_channels": 1,"conditional_dim":3}),

        # model 6: colour images
        "effnet_6": (LitEffNet, CustomDatasetImg, downscale_random_crop_rotate_transform, [], {"in_channels": 3}),

        # model 7: colour images + re + stellar mass
        "effnet_7": (LitEffNetV, CustomDatasetImgVal, downscale_random_crop_rotate_transform, ["LMSTAR", "RE"], {"in_channels": 3,"conditional_dim":2}), # expected
        "resnet_7": (LitResNetV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LMSTAR", "RE"], {"in_channels": 3,"conditional_dim":2}), # expected
        "vit_7": (LitViTV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LMSTAR", "RE"], {"in_channels": 3,"conditional_dim":2}), # expected
        
        # note: can predict stellar mass based off re
        # model 8: colour images + re + stellar mass + lumonisty age (shouldn't improve to 7)
        "effnet_8": (LitEffNetV, CustomDatasetImgVal, downscale_random_crop_rotate_transform, ["LMSTAR", "RE", "LW_AGE_RE"], {"in_channels": 3,"conditional_dim":3}),
        "resnet_8": (LitResNetV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LMSTAR", "RE", "LW_AGE_RE"], {"in_channels": 3,"conditional_dim":3}),
        "vit_8": (LitViTV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LMSTAR", "RE", "LW_AGE_RE"], {"in_channels": 3,"conditional_dim":3}),

        # model 9: colour images + re + stellar mass + sigma_re (shouldn't improve from 7 you should be able to calculate those numbers)
        "effnet_9": (LitEffNetV, CustomDatasetImgVal, downscale_random_crop_rotate_transform, ["LMSTAR", "RE", "SIGMA_RE"], {"in_channels": 3,"conditional_dim":3}),
        "resnet_9": (LitResNetV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LMSTAR", "RE", "SIGMA_RE"], {"in_channels": 3,"conditional_dim":3}),
        "vit_9": (LitViTV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LMSTAR", "RE", "SIGMA_RE"], {"in_channels": 3,"conditional_dim":3}),
        # model 10: colour images + re + stellar mass + sigma_re + h4_re (h4 should tell you how the stars are rotating - minor change)
        "effnet_10": (LitEffNetV, CustomDatasetImgVal, downscale_random_crop_rotate_transform, ["LMSTAR", "RE", "SIGMA_RE", "H4_RE"], {"in_channels": 3,"conditional_dim":4}),
        "resnet_10": (LitResNetV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LMSTAR", "RE", "SIGMA_RE", "H4_RE"], {"in_channels": 3,"conditional_dim":4}),
        "vit_10": (LitViTV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LMSTAR", "RE", "SIGMA_RE", "H4_RE"], {"in_channels": 3,"conditional_dim":4}),
        # model 11: colour images + re + stellar mass + sigma_re + h4_re + lumonisty age (shouldn't improve from 10)
        "effnet_11": (LitEffNetV, CustomDatasetImgVal, downscale_random_crop_rotate_transform, ["LMSTAR", "RE", "SIGMA_RE", "H4_RE", "LW_AGE_RE"], {"in_channels": 3,"conditional_dim":5}),
        "resnet_11": (LitResNetV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LMSTAR", "RE", "SIGMA_RE", "H4_RE", "LW_AGE_RE"], {"in_channels": 3,"conditional_dim":5}),
        "vit_11": (LitViTV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LMSTAR", "RE", "SIGMA_RE", "H4_RE", "LW_AGE_RE"], {"in_channels": 3,"conditional_dim":5}),

        # gold standard
        # model 14: lambdar_re + ellip + images
        "effnet_14": (LitEffNetV, CustomDatasetImgVal, downscale_random_crop_rotate_transform, ["LMSTAR", "LAMBDAR_RE", "ELLIP"], {"in_channels": 3,"conditional_dim":3}),
        "resnet_14": (LitResNetV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LMSTAR", "LAMBDAR_RE", "ELLIP"], {"in_channels": 3,"conditional_dim":3}),
        "vit_14": (LitViTV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LMSTAR", "LAMBDAR_RE", "ELLIP"], {"in_channels": 3,"conditional_dim":3}),
        # model 15: lambdar_re + images
        "effnet_15": (LitEffNetV, CustomDatasetImgVal, downscale_random_crop_rotate_transform, ["LMSTAR", "LAMBDAR_RE"], {"in_channels": 3,"conditional_dim":2}),
        "resnet_15": (LitResNetV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LMSTAR", "LAMBDAR_RE"], {"in_channels": 3,"conditional_dim":2}),
        "vit_15": (LitViTV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LMSTAR", "LAMBDAR_RE"], {"in_channels": 3,"conditional_dim":2}),
        # model 16: ellip + images
        "effnet_16": (LitEffNetV, CustomDatasetImgVal, downscale_random_crop_rotate_transform, ["LAMBDAR_RE", "ELLIP"], {"in_channels": 3,"conditional_dim":2}),
        "resnet_16": (LitResNetV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LAMBDAR_RE", "ELLIP"], {"in_channels": 3,"conditional_dim":2}),
        "vit_16": (LitViTV, CustomDatasetImgVal, downscale_centrecrop_rotate_transform, ["LAMBDAR_RE", "ELLIP"], {"in_channels": 3,"conditional_dim":2}),
    }

    if args.name in model_configs:
        model_class, dataset_class, transform_func, extra_vars, model_params = model_configs[args.name]
        train_and_evaluate_model(args.name, model_class, dataset_class, transform_func, extra_vars, model_params)
    else:
        print("Invalid model name")

if __name__ == "__main__":
    main()
