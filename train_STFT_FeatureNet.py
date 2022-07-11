import keras.backend.tensorflow_backend as KTF
from model.STFT_FeatureNet import build_STFTNet
from model.DataGenerator import kFoldGenerator
from model.Utils import *
import os
import argparse
import shutil
import gc
import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


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

# [train] parameters ('_f' means STFTNet_with_feature)
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

    scaler = StandardScaler()
    train_data, train_targets, val_data, val_targets = DataGenerator.getFold(i)

    val_data = np.array(val_data)

    val_targets = np.array(val_targets)

    train_data = np.array(process_batch_signal(train_data))
    val_data = np.array(process_batch_signal(val_data))

    ## build STFTNet & train
    Net_Shape = tuple(np.shape(list(val_data[0])))
    STFTNet_f, STFTNet_p = build_STFTNet(Net_Shape, opt_f)

    history_fea = STFTNet_p.fit(
        x=train_data,
        y=train_targets,
        epochs=num_epochs_f,
        batch_size=batch_size_f,
        shuffle=True,
        validation_data=(val_data, val_targets),
        verbose=2,
        callbacks=[keras.callbacks.ModelCheckpoint(Path['Save'] + 'STFTNet_Best_' + str(i) + '.h5',
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

    saveFile = open(Path['Save'] + "Result_STFT.txt", 'a+')
    print('Fold #' + str(i), file=saveFile)
    print(history_fea.history, file=saveFile)
    saveFile.close()

    # load the weights of best performance
    STFTNet_p.load_weights(Path['Save'] + 'STFTNet_Best_' + str(i) + '.h5')

    # get and save the learned feature
    train_feature = STFTNet_f.predict(train_data)
    val_feature = STFTNet_f.predict(val_data)
    print('Save feature of Fold #' + str(i) + ' to' + Path['save'] + 'STFT_Feature_' + str(i) + '.npz')
    np.savez(Path['save'] + 'STFT_Feature_' + str(i) + '.npz',
             train_feature=train_feature,
             val_feature=val_feature,
             train_targets=train_targets,
             val_targets=val_targets
             )

    # Fold finish
    keras.backend.clear_session()
    del STFTNet_f, STFTNet_p, train_data, train_targets, val_data, val_targets
gc.collect()

# # 4. Final results

# Average training performance
fit_acc = fit_acc / np.sum(Fold_Num)
fit_loss = fit_loss / np.sum(Fold_Num)
fit_val_loss = fit_val_loss / np.sum(Fold_Num)
fit_val_acc = fit_val_acc / np.sum(Fold_Num)

# Draw ACC / loss curve and save
VariationCurve(fit_acc, fit_val_acc, 'Acc', Path['Save'], figsize=(9, 6))
VariationCurve(fit_loss, fit_val_loss, 'Loss', Path['Save'], figsize=(9, 6))

saveFile = open(Path['Save'] + "Result_STFT.txt", 'a+')
print(history_fea.history, file=saveFile)
saveFile.close()
print(128 * '_')
print('End of training STFTNet.')
print(128 * '#')
