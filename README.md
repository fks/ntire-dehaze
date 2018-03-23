Global Depth Local Dehaze
============================

NTIRE 2018 Ground truth-based single-image dehazing challenge

## Contents
0. [Requirements and Installation](#Requirements and Installation)
0. [Training](#Training)
0. [Testing / Submission](#Testing_Submission)


## Requirements and Installation

- Python 2.7
- Tensorflow 1.4
- Keras 2.1.4
- Keras Contrib
```bash
  pip install git+https://www.github.com/keras-team/keras-contrib.git
```
- Python modules numpy, pandas, cv2 (opencv), skimage (scikit-image), h5py 
- Download the preprocessed [NYU Depth V2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). File size is around 32GB so download might take a while.
```bash
  	cd data
  	wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz 
  	tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz 
  	cd ..
```
- Download Training/Validation-Datasets OutdoorTrainHazy.zip and OutdoorValidationHazy.zip.
  Unzip them to data/outdoor/train/haze/
```bash
    cd data/outdoor/train/haze/
    unzip OutdoorTrainHazy.zip
    unzip OutdoorValidationHazy.zip
```
- Download Training/Validation-Datasets OutdoorTrainGT.zip and OutdoorValidationGT.zip.
  Unzip them to data/outdoor/train/nohaze/
```bash
    cd data/outdoor/train/nohaze/
    unzip OutdoorTrainGT.zip
    unzip OutdoorValidationGT.zip
```
- Download Training/Validation-Datasets IndoorTrainHazy.zip and IndoorValidationHazy.zip.
  Unzip them to data/indoor/train/haze/
```bash
    cd data/indoor/train/haze/
    unzip IndoorTrainHazy.zip
    unzip IndoorValidationHazy.zip
```
- Download Training/Validation-Datasets IndoorTrainGT.zip and IndoorValidationGT.zip.
  Unzip them to data/indoor/train/nohaze/
```bash
    cd data/indoor/train/nohaze/
    unzip IndoorTrainGT.zip
    unzip IndoorValidationGT.zip    
```
- Download Test-Dataset OutdoorTestHazy.zip.
  Unzip to data/outdoor/test/
```bash
    cd data/indoor/test
    unzip OutdoorTestHazy.zip
```
- Download Test-Dataset IndoorTestHazy.zip.
  Unzip to data/indoor/test/
```bash
    cd data/indoor/test
    unzip IndoorTestHazy.zip
```

## Training
If training from scratch, depth model needs to be trained first. Depth prediction is used during training of dehaze model.

#### Depth Model
Make sure to download and extract the NYU depth dataset to data/nyudepthv2. (see above)
``` cd src
    python train_depth.py
```
Checkpoints and logs will be stored under log/depth4_xxx
To use the newly trained weights you will need to change the path to the weights file in class Depth4Predictor.

#### Dehaze Model
``` cd src
    python train_dehaze.py
```
Checkpoints and logs will be stored under log/dehaze10_xxx
To use the newly trained weights you will need to modify the weights_file path in submission.py.

## Testing / Submission
Make sure to download and extract the test images to data/indoor/test and data/outdoor/test.
To use other weights than the pretrained ones you will need to replace the path to the weights-file at the top of submission.py

#### Indoor Track
```bash
   cd src
   python submission.py indoor
```
Results will be stored under submission/submit_indoor.
#### Outdoor Track
```bash
   cd src
   python submission.py outdoor
```
Results will be stored under submission/submit_outdoor.

