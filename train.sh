# ShuffleNet
python train.py -m "ShuffleNet" --train_data "dataset/preprocessed/images/train" --val_data "dataset/preprocessed/images/val" --output_model "outputs/img_shufflenet.pth" --output_plot "outputs/img_loss_shufflenet.png" -c True # imgs
python train.py -m "ShuffleNet" --train_data "dataset/preprocessed/spec/train" --val_data "dataset/preprocessed/spec/val" --output_model "outputs/spec_shufflenet.pth" --output_plot "outputs/spec_loss_shufflenet.png" -c True

# MobileNetV3
python train.py -m "MobileNetV3" --train_data "dataset/preprocessed/images/train" --val_data "dataset/preprocessed/images/val" --output_model "outputs/img_mobilenetv3.pth" --output_plot "outputs/img_loss_mobilenetv3.png" -c True # imgs
python train.py -m "MobileNetV3" --train_data "dataset/preprocessed/spec/train" --val_data "dataset/preprocessed/spec/val" --output_model "outputs/spec_mobilenetv3.pth" --output_plot "outputs/spec_loss_mobilenetv3.png" -c True # imgs

# MNASNet
python train.py -m "MNASNet" --train_data "dataset/preprocessed/images/train" --val_data "dataset/preprocessed/images/val" --output_model "outputs/img_mnasnet.pth" --output_plot "outputs/img_loss_mnasnet.png" -c True # imgs
python train.py -m "MNASNet" --train_data "dataset/preprocessed/spec/train" --val_data "dataset/preprocessed/spec/val" --output_model "outputs/spec_mnasnet.pth" --output_plot "outputs/spec_loss_mnasnet.png" -c True

# SqueezeNet
python train.py -m "SqueezeNet" --train_data "dataset/preprocessed/images/train" --val_data "dataset/preprocessed/images/val" --output_model "outputs/img_squeezenet.pth" --output_plot "outputs/img_loss_squeezenet.png" -c True # imgs
python train.py -m "SqueezeNet" --train_data "dataset/preprocessed/spec/train" --val_data "dataset/preprocessed/spec/val" --output_model "outputs/spec_squeezenet.pth" --output_plot "outputs/spec_loss_squeezenet.png" -c True

# MobileNetV2
python train.py -m "MobileNetV2" --train_data "dataset/preprocessed/images/train" --val_data "dataset/preprocessed/images/val" --output_model "outputs/img_mobilenetv2.pth" --output_plot "outputs/img_loss_mobilenetv2.png" -c True # imgs
python train.py -m "MobileNetV2" --train_data "dataset/preprocessed/spec/train" --val_data "dataset/preprocessed/spec/val" --output_model "outputs/img_mobilenetv2.pth" --output_plot "outputs/spec_loss_mobilenetv2.png" -c True

# RegNet
python train.py -m "RegNet" --train_data "dataset/preprocessed/images/train" --val_data "dataset/preprocessed/images/val" --output_model "outputs/img_regnet.pth" --output_plot "outputs/img_loss_regnet.png" -c True # imgs
python train.py -m "RegNet" --train_data "dataset/preprocessed/spec/train" --val_data "dataset/preprocessed/spec/val" --output_model "outputs/img_regnet.pth" --output_plot "outputs/spec_loss_regnet.png" -c True

# EfficientNet
python train.py -m "EfficientNet" --train_data "dataset/preprocessed/images/train" --val_data "dataset/preprocessed/images/val" --output_model "outputs/img_efficientnet.pth" --output_plot "outputs/img_loss_efficientnet.png" -c True # imgs
python train.py -m "EfficientNet" --train_data "dataset/preprocessed/spec/train" --val_data "dataset/preprocessed/spec/val" --output_model "outputs/img_efficientnet.pth" --output_plot "outputs/spec_loss_efficientnet.png" -c True

# Ladevic
python train.py -m "Ladevic" --train_data "dataset/preprocessed/images/train" --val_data "dataset/preprocessed/images/val" --output_model "outputs/img_ladevic.pth" --output_plot "outputs/img_loss_ladevic.png" -c True -b 128 # imgs
python train.py -m "Ladevic" --train_data "dataset/preprocessed/spec/train" --val_data "dataset/preprocessed/spec/val" --output_model "outputs/img_ladevic.pth" --output_plot "outputs/spec_loss_ladevic.png" -c True

# Mulki
python train.py -m "Mulki" --train_data "dataset/preprocessed/images/train" --val_data "dataset/preprocessed/images/val" --output_model "outputs/img_mulki.pth" --output_plot "outputs/img_loss_mulki.png" -c True # imgs
python train.py -m "Mulki" --train_data "dataset/preprocessed/spec/train" --val_data "dataset/preprocessed/spec/val" --output_model "outputs/img_mulki.pth" --output_plot "outputs/spec_loss_mulki.png" -c True
