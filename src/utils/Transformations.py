from torchvision.transforms import CenterCrop, ToPILImage, Grayscale, transforms, ToTensor
crop_size = 360  # Desired cropped size
max_rotation_angle = 360  # Maximum rotation angle in degrees
import torch

# greyscale

def greyscale_only(img):
    return transforms.Compose([
        ToPILImage(),
        Grayscale(num_output_channels=1),
        transforms.RandomRotation(max_rotation_angle),
        ToTensor()
    ]) (img)

def greyscale_rotate(img):
    return transforms.Compose([
        ToPILImage(),
        Grayscale(num_output_channels=1),
        transforms.RandomRotation(max_rotation_angle),
        ToTensor()
    ]) (img)

def greyscale_downscale_centrecrop_rotate_transform(img):
    crop_size = 224
    return transforms.Compose([
        transforms.ToPILImage(),
        Grayscale(num_output_channels=1),
        transforms.CenterCrop(crop_size),
        transforms.RandomRotation(max_rotation_angle),
        transforms.ToTensor()
    ]) (img)

def greyscale_downscale_random_crop_rotate_transform(img):
    crop_size = 224
    return transforms.Compose([
        transforms.ToPILImage(),
        Grayscale(num_output_channels=1),
        transforms.RandomCrop(crop_size),
        transforms.RandomRotation(max_rotation_angle),
        transforms.ToTensor()
    ]) (img)

# colour images
def downscale_vit(img):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor()
    ]) (img)

def rotate_transform(img):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(max_rotation_angle),
        transforms.ToTensor()
    ]) (img)

def downscale_centrecrop_rotate_transform(img):
    crop_size = 224
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(crop_size),
        transforms.RandomRotation(max_rotation_angle),
        transforms.ToTensor()
    ]) (img)

def downscale_random_crop_rotate_transform(img):
    crop_size = 224
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(crop_size),
        transforms.RandomRotation(max_rotation_angle),
        transforms.ToTensor()
    ]) (img)

import random
import math
max_scale = math.log(1.3)
max_translate = (0.0625, 0.0625)
def random_rotate_translate_x_y_zoom_flip(img):
    scale = math.exp(random.uniform(-max_scale, max_scale))
    flip = transforms.RandomHorizontalFlip() if random.random() < 0.5 else transforms.RandomVerticalFlip()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=(0,max_rotation_angle),translate=max_translate, scale=(scale, scale)),
        flip,
        transforms.ToTensor()    
    ]) (img)

    return transform

def random_rotate_translate_x_y_zoom_flip_vit(img):
    scale = math.exp(random.uniform(-max_scale, max_scale))
    flip = transforms.RandomHorizontalFlip() if random.random() < 0.5 else transforms.RandomVerticalFlip()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.RandomAffine(degrees=(0,max_rotation_angle),translate=max_translate, scale=(scale, scale)),
        flip,
        transforms.ToTensor()    
    ]) (img)

    return transform