import argparse
import os
from torchvision.transforms import transforms, v2
from PIL import Image
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# python dataset/gen_dataset.py --in_dir 'dataset/raw' --out_dir_img 'dataset/preprocessed/images' --out_dir_spec 'dataset/preprocessed/spec'

# -------------------- ARGUMENTS -------------------- #
parser = argparse.ArgumentParser()

parser.add_argument("--in_dir", required=True, type=str,
                    help= f"Image directory of the dataset downloaded from: "
                          f"https://www.kaggle.com/datasets/yangsangtai/tiny-genimage?select=imagenet_ai_0508_adm")

parser.add_argument("--out_dir_img", required=True, type=str,
                    help= f"Output directory of the image version of the dataset")

parser.add_argument("--out_dir_spec", required=True, type=str,
                    help= f"Output directory of the spectral version of the dataset")

args = parser.parse_args()

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

def generate_dataset():

    counter = 0

    output_dirs  = {
        'img_dir_train': f'{args.out_dir_img}/train',
        'img_dir_val': f'{args.out_dir_img}/val',
        'img_dir_test': f'{args.out_dir_img}/test',
        'spec_dir_train': f'{args.out_dir_spec}/train',
        'spec_dir_val': f'{args.out_dir_spec}/val',
        'spec_dir_test': f'{args.out_dir_spec}/test'
    }

    # make output folders
    for dir in output_dirs.values():
        os.makedirs(dir, exist_ok=True)

    for folder in os.listdir(args.in_dir):
        model_path = os.path.join(args.in_dir, folder)
        for split in os.listdir(model_path):

            # get all the images in the dataset split
            split_path = os.path.join(model_path, split)

            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                all_images = [img for img in os.listdir(cls_path)]

                for dir in output_dirs.values():
                    os.makedirs(f'{dir}/{cls}', exist_ok=True)

                for image in all_images:
                    image_path = os.path.join(cls_path, image)
                    img_tensor = convert_to_tensor(image_path)
                    spec_tensor = convert_to_tensor(convert_to_freq(image_path))
                    base = os.path.splitext(image)[0]
                    new_file_path = base + '.pth'

                    if split == 'train':
                        img_tensor_path = os.path.join(output_dirs['img_dir_train'], cls, new_file_path)
                        spec_tensor_path = os.path.join(output_dirs['spec_dir_train'], cls, new_file_path)

                    else:
                        if counter % 2 == 0:
                            img_tensor_path = os.path.join(output_dirs['img_dir_val'], cls, new_file_path)
                            spec_tensor_path = os.path.join(output_dirs['spec_dir_val'], cls, new_file_path)
                        else:
                            img_tensor_path = os.path.join(output_dirs['img_dir_test'], cls, new_file_path)
                            spec_tensor_path = os.path.join(output_dirs['spec_dir_test'], cls, new_file_path)
                        counter += 1

                    # save tensors
                    print(f'Saving image tensor to {img_tensor_path}')
                    torch.save(img_tensor, img_tensor_path)
                    print(f'Saving spectral tensor to {spec_tensor_path}')
                    torch.save(spec_tensor, spec_tensor_path)

if __name__ == '__main__':
    generate_dataset()