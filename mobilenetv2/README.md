# Cough Detection using MobileNetV2

Due to the limited number of cough audio files available, `main.py` fine-tunes a pre-trained [MobileNetV2](https://arxiv.org/abs/1801.04381) using PyTorch. This implementation is loosely based on [this](https://ieeexplore.ieee.org/document/8904554) paper. The dataset used is available [here](https://drive.google.com/file/d/19XyGihClOE4Vn0dM0IqQI1wA8NfZ9SQI/view?usp=sharing). It consists of 135 cough files and 52 non-cough files from Google's [AudioSet](https://research.google.com/audioset/), 40 cough files and 1,960 non-cough files from the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset, and 256 cough files and 10,801 non-cough files from the [FSDKaggle2018](https://zenodo.org/record/2552860#.XwscUud7kaE) dataset.

