import shap
import os
import torch
import numpy as np
import random

def shap_deep_explainer_img_value(background_data, test_data, model):
    """
    Calculate Shapley values for an image classification model using DeepExplainer.

    Args:
        background_data (tuple): Background data for the model's inputs.
        test_data (tuple): Test data for which Shapley values are calculated.
        model (torch.nn.Module): The PyTorch model to explain.

    Returns:
        shap_numpy (list): List of Shapley values for the input images, organised as a list of two NumPy arrays
                           representing the contributions to the input channels.
        test_numpy (numpy.ndarray): NumPy array of test data, organised as a NumPy array with dimensions (batch, height, width, channels).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    background_data[0] = background_data[0].to(device)
    background_data[1] = background_data[1].to(device)

    model = model.to(device)

    shap_explainer = shap.DeepExplainer(model, data=background_data)

    test_data[0] = test_data[0].to(device)
    test_data[1] = test_data[1].to(device)

    shap_values = shap_explainer.shap_values(test_data)

    shap_numpy = [shap_values[0][0].transpose(0,2,3,1),shap_values[1][0].transpose(0,2,3,1)]
    test_numpy = test_data[0].cpu().numpy().transpose(0,2,3,1)

    # shap.image_plot(shap_numpy, test_numpy,show=False)
    return shap_numpy, test_numpy

def shap_deep_explainer_img_value_conditional(background_data, test_data, model):
    """
    Calculate Shapley values for an image classification model using DeepExplainer with conditional input.

    Args:
        background_data (tuple): Background data for the model's inputs.
        test_data (tuple): Test data for which Shapley values are calculated.
        model (torch.nn.Module): The PyTorch model to explain.

    Returns:
        shap_values (tuple): Tuple of Shapley values for the model's inputs.
        test_numpy (numpy.ndarray): NumPy array of test data, organised as a NumPy array with dimensions (batch, height, width, channels).
        shap_explainer (shap.Explainer): The Shapley explainer object used for explanation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    background_data[0] = background_data[0].to(device)
    background_data[1] = background_data[1].to(device)

    model = model.to(device)

    shap_explainer = shap.DeepExplainer(model, data=background_data)

    test_data[0] = test_data[0].to(device)
    test_data[1] = test_data[1].to(device)

    shap_values = shap_explainer.shap_values(test_data)
    test_numpy = test_data[0].cpu().numpy().transpose(0,2,3,1)

    return shap_values, test_numpy, shap_explainer