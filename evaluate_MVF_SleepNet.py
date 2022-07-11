import os
import numpy as np
import argparse
import gc

import keras
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from model.MVF_SleepNet import build_MVFSleepNet
from model.DataGenerator import kFoldGenerator
from model.Utils import *
import random

# # 1. Get configuration

# ## 1.1. Read .config file

# command line parameters -c -g
parser = argparse.ArgumentParser()
parser.add_argument("-c", type=str, help="configuration file", required=True)
parser.add_argument("-g", type=str, help="GPU number to use, set '-1' to use CPU", required=True)
args = parser.parse_args()
Path, _, cfgTrain, cfgModel = ReadConfig(args.c)

# set GPU number or use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = args.g
if args.g != "-1":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    print("Use GPU #" + args.g)
else:
    print("Use CPU only")

# ## 1.2. Analytic parameters

# [train] parameters
channels = int(cfgTrain["channels"])
fold = int(cfgTrain["fold"])
context = int(cfgTrain["context"])
num_epochs = int(cfgTrain["epoch"])
batch_size = int(cfgTrain["batch_size"])
optimizer = cfgTrain["optimizer"]
learn_rate = float(cfgTrain["learn_rate"])

# [model] parameters
GLalpha = float(cfgModel["GLalpha"])
num_of_chev_filters = int(cfgModel["cheb_filters"])
num_of_time_filters = int(cfgModel["time_filters"])
time_conv_strides = int(cfgModel["time_conv_strides"])
time_conv_kernel = int(cfgModel["time_conv_kernel"])
num_block = int(cfgModel["num_block"])
cheb_k = int(cfgModel["cheb_k"])
dropout = float(cfgModel["dropout"])

# # 2. Read data and process data

# ## 2.1. Read data
# Each fold corresponds to one subject's data (ISRUC-S3 dataset)
ReadList = np.load(Path['data'], allow_pickle=True)
Fold_Num = ReadList['Fold_len']  # Num of samples of each fold
Fold_Data = ReadList['Fold_data']  # Data of each fold
Fold_Label = ReadList['Fold_label']  # Labels of each fold

# ## 2.2. Read adjacency information

print("Read data successfully")
Fold_Num_c = Fold_Num + 1 - context
print('Number of samples: ', np.sum(Fold_Num), '(with context:', np.sum(Fold_Num_c), ')')

# ## 2.3. Build kFoldGenerator or DominGenerator
DataGenerator = kFoldGenerator(Fold_Data, Fold_Label)

# # 3. Model training (cross validation)

# k-fold cross validation
all_scores = []
for i in range(fold):
    print(128 * '_')
    print('Fold #', i)

    # Instantiation optimizer
    opt = Instantiation_optim(optimizer, learn_rate)

    # get i th-fold FeatureNet feature and label
    Features = np.load(Path['Save'] + 'Feature_' + str(i) + '.npz', allow_pickle=True)
    val_feature = Features['val_feature']
    val_targets = Features['val_targets']

    ## Combine adjacent phases
    print('Feature', val_feature.shape)
    val_feature, val_targets = AddContext_SingleSub(val_feature, val_targets, context)
    sample_shape1 = (val_feature.shape[1:])
    print('Feature with context:', val_feature.shape)

    # get i th-fold STFTNet feature
    Features = np.load(Path['save'] + 'STFT_Feature_' + str(i) + '.npz', allow_pickle=True)
    val_STFT_adjacent = Features['val_feature']
    val_STFT_adjacent, _ = AddContext_SingleSub_STFT_GenFeature(val_STFT_adjacent, val_targets, 5)

    sample_shape2 = tuple(np.shape(list(val_STFT_adjacent[0])))

    # build MVF-SleepNet

    model = build_MVFSleepNet(cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                              time_conv_kernel, sample_shape1, sample_shape2, num_block,
                              dropout, opt, GLalpha
                              )
    # Evaluate
    # Load weights of best performance
    model.load_weights(Path['save'] + 'MVF_SleepNet_Best_' + str(i) + '.h5')

    val_mse, val_acc = model.evaluate([val_feature, val_STFT_adjacent], val_targets, verbose=0)
    print('Evaluate', val_acc)
    all_scores.append(val_acc)

    # Predict
    predicts = model.predict([val_feature, val_STFT_adjacent])

    AllPred_temp = np.argmax(predicts, axis=1)
    AllTrue_temp = np.argmax(val_targets, axis=1)
    if i == 0:
        AllPred = AllPred_temp
        AllTrue = AllTrue_temp
    else:
        AllPred = np.concatenate((AllPred, AllPred_temp))
        AllTrue = np.concatenate((AllTrue, AllTrue_temp))

    # Fold finish
    keras.backend.clear_session()
    del model, val_feature, val_targets, val_STFT_adjacent
    gc.collect()

# # 4. Final results

# print acc of each fold
print(128 * '=')
print("All folds' acc: ", all_scores)
print("Average acc of each fold: ", np.mean(all_scores))

# Print score to console
print(128 * '=')
PrintScore(AllTrue, AllPred)
# Print score to Result.txt file
PrintScore(AllTrue, AllPred, savePath=Path['save'])

# Print confusion matrix and save
ConfusionMatrix(AllTrue, AllPred, classes=['W', 'N1', 'N2', 'N3', 'REM'], savePath=Path['save'])

print('End of evaluating MVF_SleepNet.')
print(128 * '#')
