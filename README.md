<div align="center">

<div>
   <a href="https://github.com/nchivar/LAID"><img src="https://visitor-badge.laobi.icu/badge?page_id=nchivar/LAID"/></a>
   <a href="https://github.com/nchivar/LAID"><img src="https://img.shields.io/github/stars/Ekko-zn/nchivar/LAID"/></a>
   <a href="https://drive.google.com/drive/folders/1FY7boXxIyKh8XYJwFwR104XL8_C35Umc?usp=sharing"><img src="https://img.shields.io/badge/Database-Release-green"></a>
</div>


</div>

#  LAID: Lightweight AI-Generated Image Detection in Spatial and Frequency Domains

## News
:new: [2025-04-28] Official release of LAID repository

# Collected Methods
|method|paper|test code|train code|
|:--------:|:------:|:----:|:------:|
|ShuffleNet|ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices|:white_check_mark:|:white_check_mark:|
|MobileNetV3|Searching for MobileNetV3|:white_check_mark:|:white_check_mark:|
|MnasNet|MnasNet: Platform-Aware Neural Architecture Search for Mobile|:white_check_mark:|:white_check_mark:|
|SqueezeNet|SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size|:white_check_mark:|:white_check_mark:|
|MobileNetV2|MobileNetV2: Inverted Residuals and Linear Bottlenecks|:white_check_mark:|:white_check_mark:|
|RegNet|RegNet: Self-Regulated Network for Image Classification|:white_check_mark:|:white_check_mark:|
|Lađević et al.|Detection of AI-Generated Synthetic Images with a Lightweight CNN|:white_check_mark:|:white_check_mark:|
|SpottingDiffusion |SpottingDiffusion: using transfer learning to detect latent diffusion model-synthesized images|:white_check_mark:|:white_check_mark:|


## Setup
1. Download the [GenImage](https://github.com/GenImage-Dataset/GenImage) dataset (Note: The local disk location of the GenImage dataset is flexible. You can specify its path using the `--data_dir` command-line argument when running `train.py`. For detailed instructions, refer to the [Training](#Training) section.
2. (Optional) Download pretrained detection model [weights](https://drive.google.com/drive/folders/1FY7boXxIyKh8XYJwFwR104XL8_C35Umc?usp=sharing)

## Training
For training, simply run `train.py` which will automatically subsample your GenImage download and save the subsampled dataset based on the `saved_dataset ` flag. 

### Training Parameters
|Argument | Type | Default | Description|
|:------------:|:------:|:----:|:------:|
-e, --epochs | int | 100 | Number of training epochs.
-b, --batch_size | int | 64 | Batch size for training.
-lr, --learning_rate | float | 1e-4 | Learning rate for the optimizer.
-wd, --weight_decay | float | 0 | Weight decay (L2 regularization) for the optimizer.
-d, --data_dir | str | 'GenImage/' | Directory containing the GenImage dataset.
--train_image_count | int | 100000 | Number of images to subsample for training set.
--val_image_count | int | 12500 | Number of images to subsample to use for validation and test sets.
--saved_dataset | str | 'dataset' | Locatiion of subsampled GenImage dataset.
-m, --model | str | Required | Model to train on AIGI detection. **Choices: {"ShuffleNet", "MobileNetV3", "MNASNet", "SqueezeNet", "MobileNetV2", "RegNet", "Ladevic", "Mulki"}.**
-dm, --modality | str | Required | Modality of input data (raw RGB (img) or 2D FFT plot ("freq"). **Choices: {"img", "freq"}.**
-mc, --model_checkpoint | str | None | Path to a previous model checkpoint for continued training (leave as None if you want to train from scratch).
--output_dir | str | Required | Directory where all output model weights and training loss plots will be saved.
--output_model | str | Required | Name of model weight file to save.
--output_plot | str | Required | Name of training loss plot to save.
-c, --cuda | bool | False | Use CUDA supported GPU for training if available.

Sample usage:
```
python train.py -b 1024 -m "ShuffleNet" -dm "img" --train_image_count 100000 --val_image_count 16000 --output_dir "outputs" --output_model "chk.pth"  --output_plot "chk_plot.png" -c True

```

# Testing
For test, simply run `test.py`.

### Testing Parameters
|Argument | Type | Default | Description|
|:-----------:|:------:|:----:|:------:|
-m, --models_dir | str | Required | Model to train on AIGI detection. **Choices: {"ShuffleNet", "MobileNetV3", "MNASNet", "SqueezeNet", "MobileNetV2", "RegNet", "Ladevic", "Mulki"}.**
--test_data_img | str | Required | Location of subsampled spatial test set (default location: ```dataset/spec/test```)
--test_data_spec | str | Required | Location of subsampled spectral test set (default location: ```dataset/spec/test```)
--attack | bool | True | Test models on adversarial attacks.
-c, --cuda | bool | False | Use CUDA supported GPU for training if available.

Sample usage:
```
python test.py -c True --test_data_img 'dataset/img/test' --test_data_spec 'dataset/spec/test' -m 'outputs/'

```
