import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import io

LABEL_TO_INT = {
    "nature": 0, "ai": 1
}

ATTACKS = [
    'crop', 'blur', 'noise', 'compress', 'combined'
]

class AttackedGenImage(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.temp_data = []

        for data_class in os.listdir(data_dir):
            type_path = os.path.join(data_dir, data_class)
            label = LABEL_TO_INT.get(data_class)
            for image in os.listdir(type_path):
                file_path = os.path.join(type_path, image)
                self.temp_data.append((file_path, label))

        random.shuffle(self.temp_data)
        images_per_attack_per_label = len(self.temp_data)/len(ATTACKS)/len(LABEL_TO_INT)
        attack_counter = {
            'crop_nature': 0,
            'blur_nature': 0,
            'noise_nature': 0,
            'compress_nature': 0,
            'combined_nature': 0,
            'crop_ai': 0,
            'blur_ai': 0,
            'noise_ai': 0,
            'compress_ai': 0,
            'combined_ai': 0,
        }
        self.data = []

        # iterate through each attack and modify
        for image, label in self.temp_data:

            done_attack = False
            while not done_attack:
                attack = random.choice(ATTACKS)
                label_name = [k for k, v in LABEL_TO_INT.items() if v == label][0]
                attack_key = f'{attack}_{label_name}'
                if attack_counter.get(attack_key) < images_per_attack_per_label:

                    # assign attack
                    attack_counter[attack_key] += 1

                    # append to datastream for attacking
                    self.data.append((image, label, attack))
                    done_attack = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label, attack = self.data[idx]
        tensor = attack_img(torch.load(file_path, map_location='cpu'), attack)
        return tensor, label, attack


def attack_img(img_tensor, attack_type):

    # apply attack depending on condition
    if attack_type == 'crop':
        attacked_image = crop_image(image_tensor=img_tensor)
    elif attack_type == 'blur':
        attacked_image = blur_image(image_tensor=img_tensor)
    elif attack_type == 'noise':
        attacked_image = noise_image(image_tensor=img_tensor)
    elif attack_type == 'compress':
        attacked_image = compress_image(image_tensor=img_tensor)
    else:  # combined
        attacked_image = combination_image(image_tensor=img_tensor)
    return attacked_image


def crop_image(image_tensor):
    _, H, W = image_tensor.shape
    crop_ratio = random.uniform(0.80, 0.95)  # keep 80-95% of image
    new_H, new_W = int(H * crop_ratio), int(W * crop_ratio)
    resized_tensor = TF.resized_crop(image_tensor,
                           top=random.randint(0, H - new_H),
                           left=random.randint(0, W - new_W),
                           height=new_H, width=new_W,
                           size=(H, W))
    return resized_tensor

def blur_image(image_tensor):
    kernel_size = random.choice([3, 5, 7, 9])
    sigma = random.choice([1, 2, 3, 4])
    blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    blurred_image = blur(image_tensor)
    return blurred_image

def noise_image(image_tensor):
    variance = random.uniform(5, 20)
    sigma = variance ** 0.5  # sigma = sqrt(variance)
    noise = torch.randn_like(image_tensor) * sigma / 255.0
    noisy_image = image_tensor + noise
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    return noisy_image

def compress_image(image_tensor):
    image_pil = TF.to_pil_image(image_tensor.cpu())
    buffer = io.BytesIO()
    quality = random.randint(25, 90)  # Quality Factor
    image_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed_pil = Image.open(buffer)
    compressed_image = TF.to_tensor(compressed_pil).to(image_tensor.device)
    return compressed_image

def combination_image(image_tensor):
    attacked_tensor = image_tensor
    if random.random() < 0.5:
        attacked_tensor = crop_image(attacked_tensor)
    if random.random() < 0.5:
        attacked_tensor = blur_image(attacked_tensor)
    if random.random() < 0.5:
        attacked_tensor = noise_image(attacked_tensor)
    if random.random() < 0.5:
        attacked_tensor = compress_image(attacked_tensor)

    return attacked_tensor