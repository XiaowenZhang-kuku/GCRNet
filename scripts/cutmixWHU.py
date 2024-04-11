import os.path

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import transforms


inputA_folder = '/data/CDdata/WHU/train/A'
inputB_folder = '/data/CDdata/WHU/train/B'
target_folder = '/data/CDdata/WHU/train/label'
output_folder = '/data/CDdata/WHU-cutmix30/train'

percentage = 0.3

image_files = [f for f in os.listdir(inputA_folder) if f.endswith(".png")]

num_selected_images = int(len(image_files) * percentage)
selected_images = random.sample(image_files, num_selected_images)
for i in selected_images:
    imageA_path = os.path.join(inputA_folder, i)
    imageB_path = os.path.join(inputB_folder, i)
    target_path = os.path.join(target_folder, i)
    imageA_ori = Image.open(imageA_path)
    imageB_ori = Image.open(imageB_path)
    target_ori = Image.open(target_path)
    selected_image_file = random.choice(image_files)
    imageA_choice = Image.open(os.path.join(inputA_folder, selected_image_file))
    imageB_choice = Image.open(os.path.join(inputB_folder, selected_image_file))
    target_choice = Image.open(os.path.join(target_folder, selected_image_file))
    tensorA_ori = transforms.ToTensor()(imageA_ori)
    tensorB_ori = transforms.ToTensor()(imageB_ori)
    tensortarget_ori = transforms.ToTensor()(target_ori)
    tensorA_choice = transforms.ToTensor()(imageA_choice)
    tensorB_choice = transforms.ToTensor()(imageB_choice)
    tensortarget_choice = transforms.ToTensor()(target_choice)

    alpha = 1

    height, width = 256, 256
    lam = np.random.beta(alpha, alpha)
    cut_h = int(height * np.sqrt(1 - lam))
    cut_w = int(width * np.sqrt(1 - lam))
    cut_x = np.random.randint(0, width - cut_w)
    cut_y = np.random.randint(0, height - cut_h)

    mixed_image_A = tensorA_ori.clone()
    mixed_image_A[:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = tensorA_choice[:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
    mixed_image_A_pil = transforms.ToPILImage()(mixed_image_A)
    mixed_image_B = tensorB_ori.clone()
    mixed_image_B[:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = tensorB_choice[:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
    mixed_image_B_pil = transforms.ToPILImage()(mixed_image_B)
    mixed_image_target = tensortarget_ori.clone()
    mixed_image_target[:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = tensortarget_choice[:, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w]
    mixed_image_target_pil = transforms.ToPILImage()(mixed_image_target)
    image_saved_name = "cutmix_" + i
    save_path_A = os.path.join(output_folder, 'A', image_saved_name)
    save_path_B = os.path.join(output_folder, 'B', image_saved_name)
    save_path_target = os.path.join(output_folder, 'label', image_saved_name) 
    mixed_image_A_pil.save(save_path_A)
    mixed_image_B_pil.save(save_path_B)
    mixed_image_target_pil.save(save_path_target)

