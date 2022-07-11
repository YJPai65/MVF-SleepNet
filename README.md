# MVF-SleepNet
## How to use

### 1. Prepare the dataset:

Download the ISRUC-Sleep-S3 dataset (https://sleeptight.isr.uc.pt/)

### 2. Preprocess the data:

python rawdata_preprocess.py

### 3. Configuration:

Write the config file at /config/config.config

### 4. Feature extraction:

4.1. Run python train_FeatureNet.py with -c and -g parameters. <br>
4.2. Run python train_STFT_FeatureNet.py with -c and -g parameters.

-c: The configuration file. <br>
-g: The number of the GPU to use. E.g.,0,1,2.

### 5. Train MVF-SleepNet:

Run python train_MVF_SleepNet.py with -c and -g parameters.

### 6. Evaluate MVF-SleepNet:

Run python evaluate_MVF_SleepNet.py with -c and -g parameters.
