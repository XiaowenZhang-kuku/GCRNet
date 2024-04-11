# GCRNet

## Prerequisites

> opencv-python==4.1.1  
  pytorch==1.6.0  
  torchvision==0.7.0  
  pyyaml==5.1.2  
  scikit-image==0.15.0  
  scikit-learn==0.21.3  
  scipy==1.3.1  
  tqdm==4.35.0

Tested using Python 3.7.4 on Ubuntu 16.04.

## Get Started

In `src/constants.py`, change the dataset locations to your own.

### Data Preprocessing

In `scripts/` there are preprocessing scripts for several datasets。

Use 30% cutmix operator to get better performance.

### Model Training

To train the model, use

```bash
python train.py train --exp_config PATH_TO_CONFIG_FILE
```

### Model Evaluation

To evaluate a model on the test subset, use

```bash
python train.py eval --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT --save_on --subset test
```
You can download the model weights from: https://drive.google.com/drive/u/0/folders/1l1wz9jsjoKcXI1lioH92BV7TJ5tfMx0e