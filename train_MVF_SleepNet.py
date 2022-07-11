import os
import argparse
import gc
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from model.MVF_SleepNet import build_MVFSleepNet
from model.DataGenerator import kFoldGenerator
from model.Utils import *
from keras.utils import plot_model


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


print("Read data successfully")
Fold_Num_c = Fold_Num + 1 - context
print('Number of samples: ', np.sum(Fold_Num), '(with context:', np.sum(Fold_Num_c), ')')

# ## 2.2. Build kFoldGenerator
DataGenerator = kFoldGenerator(Fold_Data, Fold_Label)


# # 3. Model training (cross validation)

for i in range(fold):
    print(128 * '_')
    print('Fold #', i)

    # Instantiation optimizer
    opt = Instantiation_optim(optimizer, learn_rate)

    # get i th-fold FeatureNet feature and label
    Features = np.load(Path['Save'] + 'Feature_' + str(i) + '.npz', allow_pickle=True)
    train_feature = Features['train_feature']
    val_feature = Features['val_feature']
    train_targets = Features['train_targets']
    val_targets = Features['val_targets']

    # Combine adjacent phases
    train_feature, train_targets = AddContext_MultiSub(train_feature, train_targets,
                                                       np.delete(Fold_Num.copy(), i), context, i)
    val_feature, val_targets = AddContext_SingleSub(val_feature, val_targets, context)

    sample_shape1 = (val_feature.shape[1:])

    print("sample_shape1: ", sample_shape1)
    print('Feature with context:', train_feature.shape, val_feature.shape)

    # get i th-fold STFTNet feature
    Features = np.load(Path['Save'] + 'STFT_Feature_' + str(i) + '.npz', allow_pickle=True)
    train_STFT_adjacent = Features['train_feature']
    val_STFT_adjacent = Features['val_feature']
    train_STFT_adjacent, _ = AddContext_MultiSub_STFT_GenFeature(train_STFT_adjacent, train_targets,
                                                                 np.delete(Fold_Num.copy(), i), 5, i)
    val_STFT_adjacent, _ = AddContext_SingleSub_STFT_GenFeature(val_STFT_adjacent, val_targets, 5)

    sample_shape2 = tuple(np.shape(list(train_STFT_adjacent[0])))

    # build_MVFSleepNet

    model = build_MVFSleepNet(cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                              time_conv_kernel, sample_shape1, sample_shape2, num_block, dropout,
                               opt, GLalpha,
                              )

    # train
    history = model.fit(
        x=[train_feature, train_STFT_adjacent],
        y=train_targets,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=([val_feature, val_STFT_adjacent], val_targets),
        verbose=2,
        callbacks=[keras.callbacks.ModelCheckpoint(Path['Save'] + 'MVF_SleepNet_Best_' + str(i) + '.h5',
                                                   monitor='val_acc',
                                                   verbose=3,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   period=1)])

    # Save training information
    if i == 0:
        fit_loss = np.array(history.history['loss']) * Fold_Num_c[i]
        fit_acc = np.array(history.history['acc']) * Fold_Num_c[i]
        fit_val_loss = np.array(history.history['val_loss']) * Fold_Num_c[i]
        fit_val_acc = np.array(history.history['val_acc']) * Fold_Num_c[i]
    else:
        fit_loss = fit_loss + np.array(history.history['loss']) * Fold_Num_c[i]
        fit_acc = fit_acc + np.array(history.history['acc']) * Fold_Num_c[i]
        fit_val_loss = fit_val_loss + np.array(history.history['val_loss']) * Fold_Num_c[i]
        fit_val_acc = fit_val_acc + np.array(history.history['val_acc']) * Fold_Num_c[i]

    saveFile = open(Path['Save'] + "Result_MVF_SleepNet.txt", 'a+')
    print('Fold #' + str(i), file=saveFile)
    print(history.history, file=saveFile)
    saveFile.close()

    # Fold finish
    keras.backend.clear_session()
    del model, train_feature, train_targets, val_feature, val_targets, Features, train_STFT_adjacent, val_STFT_adjacent
    gc.collect()

# # 4. Final results

# Average training performance
fit_acc = fit_acc / np.sum(Fold_Num_c)
fit_loss = fit_loss / np.sum(Fold_Num_c)
fit_val_loss = fit_val_loss / np.sum(Fold_Num_c)
fit_val_acc = fit_val_acc / np.sum(Fold_Num_c)

# Draw ACC / loss curve and save
VariationCurve(fit_acc, fit_val_acc, 'Acc', Path['Save'], figsize=(9, 6))
VariationCurve(fit_loss, fit_val_loss, 'Loss', Path['Save'], figsize=(9, 6))

saveFile = open(Path['Save'] + "Result_MVF_SleepNet.txt", 'a+')
print(history.history, file=saveFile)
saveFile.close()

print(128 * '_')
print('End of training MVF_SleepNet.')
print(128 * '#')
