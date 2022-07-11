import os
import argparse
import shutil
import gc
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from model.FeatureNet import build_FeatureNet
from model.DataGenerator import kFoldGenerator
from model.Utils import *

print(128 * '#')
print('Start to train FeatureNet.')

# # 1. Get configuration

# ## 1.1. Read .config file

# command line parameters -c -g
parser = argparse.ArgumentParser()
parser.add_argument("-c", type=str, help="configuration file", required=True)
parser.add_argument("-g", type=str, help="GPU number to use, set '-1' to use CPU", required=True)
args = parser.parse_args()
Path, cfgFeature, _, _ = ReadConfig(args.c)

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

# [train] parameters ('_f' means FeatureNet)
channels = int(cfgFeature["channels"])
fold = int(cfgFeature["fold"])
num_epochs_f = int(cfgFeature["epoch_f"])
batch_size_f = int(cfgFeature["batch_size_f"])
optimizer_f = cfgFeature["optimizer_f"]
learn_rate_f = float(cfgFeature["learn_rate_f"])

# ## 1.3. Parameter check and enable

# Create save pathand copy .config to it
if not os.path.exists(Path['Save']):
    os.makedirs(Path['Save'])
shutil.copyfile(args.c, Path['Save'] + "last.config")

# # 2. Read data and process data

# ## 2.1. Read data
# Each fold corresponds to one subject's data (ISRUC-S3 dataset)
ReadList = np.load(Path['data'], allow_pickle=True)
Fold_Num = ReadList['Fold_len']  # Num of samples of each fold
Fold_Data = ReadList['Fold_data']  # Data of each fold
Fold_Label = ReadList['Fold_label']  # Labels of each fold

print("Read data successfully")
print('Number of samples: ', np.sum(Fold_Num))

# ## 2.2. Build kFoldGenerator
DataGenerator = kFoldGenerator(Fold_Data, Fold_Label)

# # 3. Model training (cross validation)

# k-fold cross validation
all_scores = []
for i in range(fold):
    print(128 * '_')
    print('Fold #', i)

    # Instantiation optimizer
    opt_f = Instantiation_optim(optimizer_f, learn_rate_f)  # optimizer of FeatureNet

    # get i th-fold data
    train_data, train_targets, val_data, val_targets = DataGenerator.getFold(i)

    ## build FeatureNet & train
    featureNet, featureNet_p = build_FeatureNet(opt_f, channels)  # '_p' model is without the softmax layer
    history_fea = featureNet.fit(
        x=train_data,
        y=train_targets,
        epochs=num_epochs_f,
        batch_size=batch_size_f,
        shuffle=True,
        validation_data=(val_data, val_targets),
        verbose=2,
        callbacks=[keras.callbacks.ModelCheckpoint(Path['Save'] + 'FeatureNet_Best_' + str(i) + '.h5',
                                                   monitor='val_acc',
                                                   verbose=2,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   period=1)])
    # Save training information
    if i == 0:
        fit_loss = np.array(history_fea.history['loss']) * Fold_Num[i]
        fit_acc = np.array(history_fea.history['acc']) * Fold_Num[i]
        fit_val_loss = np.array(history_fea.history['val_loss']) * Fold_Num[i]
        fit_val_acc = np.array(history_fea.history['val_acc']) * Fold_Num[i]
    else:
        fit_loss = fit_loss + np.array(history_fea.history['loss']) * Fold_Num[i]
        fit_acc = fit_acc + np.array(history_fea.history['acc']) * Fold_Num[i]
        fit_val_loss = fit_val_loss + np.array(history_fea.history['val_loss']) * Fold_Num[i]
        fit_val_acc = fit_val_acc + np.array(history_fea.history['val_acc']) * Fold_Num[i]

    # load the weights of best performance
    featureNet.load_weights(Path['Save'] + 'FeatureNet_Best_' + str(i) + '.h5')

    # get and save the learned feature
    train_feature = featureNet_p.predict(train_data)
    val_feature = featureNet_p.predict(val_data)
    print('Save feature of Fold #' + str(i) + ' to' + Path['Save'] + 'Feature_' + str(i) + '.npz')
    np.savez(Path['Save'] + 'Feature_' + str(i) + '.npz',
             train_feature=train_feature,
             val_feature=val_feature,
             train_targets=train_targets,
             val_targets=val_targets
             )

    saveFile = open(Path['Save'] + "Result_FeatureNet.txt", 'a+')
    print('Fold #' + str(i), file=saveFile)
    print(history_fea.history, file=saveFile)
    saveFile.close()

    # Fold finish
    keras.backend.clear_session()
    del featureNet, featureNet_p, train_data, train_targets, val_data, val_targets, train_feature, val_feature
    gc.collect()

print(128 * '_')

print('End of training FeatureNet.')
print(128 * '#')
