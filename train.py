import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import tensorflow as tf
from data import *
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#====================0=========1==========2===========3===========4==============
# ==================================超参数=======================================
model_name = model_name_list[0]
TRAIN_IMAGE_SIZE = 256
TEST_IMAGE_SIZE = 1024
epochs = 100
batch_size = 8
# =============================================================================

model = model = UNet_line_v8(input_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 4), Falg_summary=True, Falg_plot_model=False)

# =============================================================================
savePath = mkSaveDir(model_name)
checkpointPath= savePath + "/"+model_name+"-{epoch:03d}-{val_seg_out_accuracy:.4f}.hdf5"
checkpoint = ModelCheckpoint(checkpointPath, monitor='val_seg_out_accuracy', verbose=1,
                             save_best_only=False, save_weights_only=True, mode='auto', period=1)
EarlyStopping = EarlyStopping(monitor='val_seg_out_acc', patience=200, verbose=1)
tensorboard = TensorBoard(log_dir=savePath, histogram_freq=0)
callback_lists = [tensorboard, EarlyStopping, checkpoint]
# ===================================训练=========================================
train_image = np.load('data/train0315/train.npy')
train_GT    = np.load('data/train0315/train_label.npy')
train_line = np.load('data/train0315/train_line_label.npy')
valid_image = np.load('data/valid0315/valid.npy')
valid_GT = np.load('data/valid0315/valid_label.npy')
valid_line = np.load('data/valid0315/valid_line_label.npy')

History = model.fit(train_image, [train_GT,train_line], batch_size=batch_size, validation_data=(valid_image, [valid_GT,valid_line]),
    epochs=epochs, verbose=1, shuffle=True,  callbacks=callback_lists)
with open(savePath + '/log_256.txt','w') as f:
    f.write(str(History.history))