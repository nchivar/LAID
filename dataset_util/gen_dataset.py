import os
from torchvision.transforms import transforms, v2
from PIL import Image
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import random

preprocess = v2.Compose([
    v2.PILToTensor(),
    v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
    v2.ConvertImageDtype(torch.float32),
])



def convert_to_freq(img_path):
    # Read the image in color (BGR format)
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Image at path '{img_path}' could not be loaded.")

    # Split the image into its Blue, Green, and Red channels
    b_channel, g_channel, r_channel = cv2.split(img_bgr)

    # Function to compute the magnitude spectrum for a single channel
    def magnitude_spectrum(channel):
        f_transform = np.fft.fft2(channel)
        f_shifted = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shifted)
        magnitude = np.log1p(magnitude)
        normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(normalized_magnitude)

    # Compute the magnitude spectrum for each channel
    b_magnitude = magnitude_spectrum(b_channel)
    g_magnitude = magnitude_spectrum(g_channel)
    r_magnitude = magnitude_spectrum(r_channel)

    # Merge the magnitude spectra back into an RGB image
    magnitude_rgb = cv2.merge([b_magnitude, g_magnitude, r_magnitude])

    return magnitude_rgb


def convert_to_tensor(img_in):
    img = None
    if isinstance(img_in, str) and os.path.isfile(img_in):
        img = Image.open(img_in).convert('RGB')
    elif isinstance(img_in, np.ndarray):
        if img_in.ndim == 3 and img_in.shape[2] == 3:
            img = Image.fromarray(img_in).convert('RGB')
    else:
        raise TypeError("Input must be a valid file path or a 2D NumPy array.")

    return preprocess(img)

def generate_dataset(in_dir, out_dir, num_train=100000, num_val=12500):

    output_dirs  = {
        'img_dir_train': f'{out_dir}/img/train',
        'img_dir_val': f'{out_dir}/img/val',
        'img_dir_test': f'{out_dir}/img/test',
        'spec_dir_train': f'{out_dir}/spec/train',
        'spec_dir_val': f'{out_dir}/spec/val',
        'spec_dir_test': f'{out_dir}/spec/test'
    }

    for dir in output_dirs.values():
        os.makedirs(os.path.join(dir,'ai'), exist_ok=True)
        os.makedirs(os.path.join(dir,'nature'), exist_ok=True)

    # get total number of generative models in GenImage (should be 8)
    num_generators = len(os.listdir(in_dir))

    # iterate through each generation model (BigGan, ADM, ...)
    for generator in os.listdir(in_dir):

        # due to subfolder structure, move down one folder
        generator_folder = os.path.join(in_dir, generator)
        children = [d for d in os.listdir(generator_folder)]
        if len(children) == 1:
            generator_folder = os.path.join(generator_folder, children[0])
        else:
            print(f'Generator folder structure for {generator} is incorrect.')

        # iterate through each data split (train/val)
        for data_split in os.listdir(generator_folder):

            data_folder = os.path.join(generator_folder, data_split)

            # iterate through both data types (ai, nature)
            for data_type in os.listdir(data_folder):
                counter = 0
                type_folder = os.path.join(data_folder, data_type)
                os.makedirs(type_folder, exist_ok=True)  # make the output directory if it doesnt exist already
                all_images = os.listdir(type_folder)


                if data_split == 'train':
                    n = num_train // 2 // num_generators
                else:
                    n = num_val // num_generators

                valid_tensors = []
                used_images = set()
                image_blacklist = set()

                print(f'Sampling {n} images from {type_folder}', flush=True)

                while len(valid_tensors) < n:
                    sampled_image = random.choice(all_images)
                    image_path = os.path.join(type_folder, sampled_image)

                    # Skip if already used or blacklisted
                    if image_path in used_images or image_path in image_blacklist:
                        continue

                    try:
                        img_tensor = convert_to_tensor(image_path)
                        spec_tensor = convert_to_tensor(convert_to_freq(image_path))
                        valid_tensors.append((sampled_image, img_tensor, spec_tensor))
                        used_images.add(image_path)
                    except Exception as e:
                        print(f"Failed on image: {image_path} ({e}), adding to blacklist...", flush=True)
                        image_blacklist.add(image_path)

                for image, img_tensor, spec_tensor in valid_tensors:

                    base = os.path.splitext(image)[0]
                    new_file_path = base + '.pth'

                    if data_split == 'train':
                        img_tensor_path = os.path.join(output_dirs['img_dir_train'], data_type, new_file_path)
                        spec_tensor_path = os.path.join(output_dirs['spec_dir_train'], data_type, new_file_path)

                    else:  # split validation set into val/test splits
                        if counter % 2 == 0:
                            img_tensor_path = os.path.join(output_dirs['img_dir_val'], data_type, new_file_path)
                            spec_tensor_path = os.path.join(output_dirs['spec_dir_val'], data_type, new_file_path)
                        else:
                            img_tensor_path = os.path.join(output_dirs['img_dir_test'], data_type, new_file_path)
                            spec_tensor_path = os.path.join(output_dirs['spec_dir_test'], data_type, new_file_path)
                        counter += 1

                    # save tensors
                    print(f'Saving image tensor to {img_tensor_path}')
                    torch.save(img_tensor, img_tensor_path)
                    print(f'Saving spectral tensor to {spec_tensor_path}')
                    torch.save(spec_tensor, spec_tensor_path)



