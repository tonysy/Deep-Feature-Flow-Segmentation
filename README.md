# Deeplab V2

## 1. Setup environment
- If you use our dockerfile, you can run the code easily.
- If you want to set up your own env, please follow these steps:
    - We only support `python2.7` now
    - Install tk: `sudo apt-get -y install python-tk`
    - Install OpenCV 3.4.1
    - Install needed python packages with `pip install -r requirements.txt`
        - If you are in China Mainland, you can use these to speedup
        `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

## 2. Prepare Data and Pretrained Model
### Cityscapes Data
You need to download the cityscapes data from the official webpapge and unzip the data
Put the data into `data/cityscapes`, you can use soft link to set the data path as the following:
`ln -s Dataset_path ./data/cityscapes`
### Pretrained Model
Download pretrained resnet model from: [resnet101-pretrained](), and put the model into `mode/pretrained_model/`

## Train and Test
### Training Deeplab V2
`python ./experiments/deeplab/deeplab_train_test.py --cfg ./experiments/deeplab/cfgs/deeplab_resnet_v1_101_cityscapes_segmentation_base.yaml`
### Training Deeplab V2 Deformable
`python ./experiments/deeplab/deeplab_train_test.py --cfg ./experiments/deeplab/cfgs/deeplab_resnet_v1_101_cityscapes_segmentation_dcn.yaml`
### Training DFF Deeplab V2
`python ./experiments/deeplab_dff/deeplab_dff_train.py --cfg ./experiments/deeplab_dff/cfgs/deeplab_resnet_v1_101_cityscapes_segmentation_video.yaml`

## Performance 
## TODO List
## FAQ
- Program hang if your system opencv is 2.x and your opencv-python is 3.x
- 