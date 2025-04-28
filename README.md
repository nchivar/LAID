<div align="center">

<div>
   <a href="https://github.com/nchivar/LAID"><img src="https://visitor-badge.laobi.icu/badge?page_id=nchivar/LAID"/></a>
   <a href="https://github.com/nchivar/LAID"><img src="https://img.shields.io/github/stars/Ekko-zn/nchivar/LAID"/></a>
   <a href="https://drive.google.com/drive/folders/1p4ewuAo7d5LbNJ4cKyh10Xl9Fg2yoFOw?usp=drive_link"><img src="https://img.shields.io/badge/Database-Release-green"></a>
</div>


</div>

#  LAID: Lightweight AI-Generated Image Detection in Spatial and Frequency Domains

# News
:new: [2025-04-28] Official release of LAID repository

# Collected Methods
|method|paper|test code|train code|
|:--------:|:------:|:----:|:------:|
|ShuffleNet|ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices|:white_check_mark:|:white_check_mark:|
|MobileNetV3|Searching for MobileNetV3|:white_check_mark:|:white_check_mark:|
|MnasNet|MnasNet: Platform-Aware Neural Architecture Search for Mobile|:white_check_mark:|:white_check_mark:|
|RegNet|RegNet: Self-Regulated Network for Image Classification|:white_check_mark:|:white_check_mark:|
|EfficientNetV2|EfficientNetV2: Smaller Models and Faster Training|:white_check_mark:|:white_check_mark:|
|Lađević et al.|Detection of AI-Generated Synthetic Images with a Lightweight CNN|:white_check_mark:|:white_check_mark:|
|SpottingDiffusion |SpottingDiffusion: using transfer learning to
detect latent diffusion model-synthesized images|:white_check_mark:|:white_check_mark:|


# Setup
1. Download the [GenImage](https://github.com/GenImage-Dataset/GenImage) dataset (Note: The local disk location of the GenImage dataset is flexible. You can specify its path using the `--data_dir` command-line argument when running `train.py`. For detailed instructions, refer to the [Training](#Training) section.
2. Download project dependencies
    ```
    pip install -r requirements.txt
    ```
3. (Optional) Download pretrained detection model [weights](https://drive.google.com/drive/folders/1FY7boXxIyKh8XYJwFwR104XL8_C35Umc?usp=sharing)

# Training
For training, you simply need to run `train.py` which will automatically subsample your GenImage download and save the subsampled dataset based on the `saved_dataset ` flag. 

Argument | Type | Default | Description
-e, --epochs | int | 100 | Number of training epochs.
-b, --batch_size | int | 64 | Batch size for training.
-lr, --learning_rate | float | 1e-4 | Learning rate for the optimizer.
-wd, --weight_decay | float | 0 | Weight decay (L2 regularization) for the optimizer.
-d, --data_dir | str | 'GenImage/' | Directory containing the GenImage dataset.
--train_image_count | int | 100000 | Number of images to subsample for training set.
--val_image_count | int | 12500 | Number of images to subsample to use for validation and test sets.
--saved_dataset | str | 'dataset' | Locatiion of subsampled GenImage dataset.
-m, --model | str | Required | Model to train on AIGI detection. Choices: {"ShuffleNet", "MobileNetV3", "MNASNet", "SqueezeNet", "MobileNetV2", "RegNet", "EfficientNet", "Ladevic", "Mulki"}.
-dm, --modality | str | Required | Modality of input data (raw RGB (img) or 2D FFT plot ("freq"). Choices: {"img", "freq"}.
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
```

```
:exclamation: You should set your dataroot and dataset name in `eval_config.py`


All pre-trained detection models and necessary pre-processing models are available in `./weights`

For example, if you want to evaluate the performance of CNNSpot under blurring.
```
python eval_all.py --model_path ./weights/CNNSpot.pth --detect_method CNNSpot  --noise_type blur --blur_sig 1.0 --no_resize --no_crop --batch_size 1
```

## Dataset
### Training Set
We adopt the training set in [CNNSpot](https://github.com/peterwang512/CNNDetection).

### Test Set and Checkpoints
The whole test set and checkpoints we used in our experiments can be downloaded from [BaiduNetdisk](https://pan.baidu.com/s/1dZz7suD-X5h54wCC9SyGBA?pwd=l30u) or [modelscope](https://modelscope.cn/datasets/aemilia/AIGCDetectionBenchmark/file/view/master?id=88429&status=2&fileName=AIGCDetectionBenchmark%252Ftest.zip)


## Acknowledgments
Our code is developed based on [CNNDetection](https://github.com/peterwang512/CNNDetection), [FreDect](https://github.com/RUB-SysSec/GANDCTAnalysis), [Fusing](https://github.com/littlejuyan/FusingGlobalandLocal), [Gram-Net](https://github.com/liuzhengzhe/Global_Texture_Enhancement_for_Fake_Face_Detection_in_the-Wild), [LGrad](https://github.com/chuangchuangtan/LGrad), [LNP](https://github.com/Tangsenghenshou/Detecting-Generated-Images-by-Real-Images), [DIRE](https://github.com/ZhendongWang6/DIRE), [UnivFD](https://github.com/Yuheng-Li/UniversalFakeDetect) . Thanks for their sharing codes and models.:heart:
