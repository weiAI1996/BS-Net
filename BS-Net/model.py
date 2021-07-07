from keras.layers import *
from keras import backend as keras
import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Add,BatchNormalization
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.optimizers import SGD, Adam
import tensorflow as tf

def recall(y_true, y_pred):
    tp_fn_num =tf.count_nonzero(y_true)
    tp = tf.cast(tf.where(tf.equal(y_true, 1), y_pred, y_true),dtype=tf.uint8)
    tp_num = tf.count_nonzero(tp)
    return tp_num/(tp_fn_num+1)
def positive_num(y_true, y_pred):
    one = tf.ones_like(y_true)
    zero = tf.zeros_like(y_true)
    y_pred = tf.where(y_pred <0.5, x=zero, y=one)
    return tf.count_nonzero(y_pred)
def rcf_loss(y_true, y_pred):
    lambd =1.2
    nozero_num = tf.cast(tf.count_nonzero(y_true), tf.int32)
    zero_num = tf.size(y_true, out_type=tf.int32) - nozero_num
    alpha = lambd * tf.cast(nozero_num / (zero_num + nozero_num), tf.float32)
    beta = tf.cast(zero_num / (zero_num + nozero_num), tf.float32)
    ones = tf.ones_like(y_true)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    rcfloss = K.mean(-beta * y_true * K.log(y_pred) - alpha * (tf.cast(ones, tf.float32) - y_true) * K.log(tf.cast(ones, tf.float32) - y_pred))
    return rcfloss
def focal_loss(alpha, gamma):
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        # y_true 是个一阶向量, 下式按照加号分为左右两部分
        # 注意到 y_true的取值只能是 0或者1 (假设二分类问题)，可以视为“掩码”
        # 加号左边的 y_true*alpha 表示将 y_true中等于1的槽位置为标量 alpha
        # 加号右边的 (ones-y_true)*(1-alpha) 则是将等于0的槽位置为 1-alpha
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
        # 类似上面，y_true仍然视为 0/1 掩码
        # 第1部分 `y_true*y_pred` 表示 将 y_true中为1的槽位置为 y_pred对应槽位的值
        # 第2部分 `(ones-y_true)*(ones-y_pred)` 表示 将 y_true中为0的槽位置为 (1-y_pred)对应槽位的值
        # 第3部分 K.epsilon() 避免后面 log(0) 溢出
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed
#========================================Dice loss==============================================
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_coef_loss(y_true, y_pred):
	return 1 - dice_coef(y_true, y_pred, smooth=1)

def Conv_block(inputlayer,kernelnum):
    conv = Conv2D(kernelnum, 1,  padding='same', kernel_initializer='he_normal')(inputlayer)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv1 = Conv2D(kernelnum, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv1 = BatchNormalization()(conv1)#
    if kernelnum>=512:
        conv1 = Dropout(0.4)(conv1)
    conv2 = Conv2D(kernelnum, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    if kernelnum>=512:
        conv2 = Dropout(0.4)(conv2)
    return conv2

    
def sl_merge_block_V8(line_layer, smerge, kernel_num):
    smerge = Conv2D(kernel_num, 1, padding='same', kernel_initializer='he_normal')(smerge)
    smerge = BatchNormalization()(smerge)
    smerge = Activation('relu')(smerge)
    line_layer = Conv2D(kernel_num, 1, activation='relu', padding='same', kernel_initializer='he_normal')(line_layer)
    line_layer = BatchNormalization()(line_layer)
    line_layer = Activation('relu')(line_layer)
    line_layer = concatenate([line_layer, smerge], axis=3)
    line_decode_conv = Conv_block(line_layer, kernel_num)
    seg_decode_conv = Conv_block(smerge, kernel_num)
    seg_decode_conv = Add()([seg_decode_conv, line_decode_conv])
    return seg_decode_conv, line_decode_conv
def res_block(inputlayer,kernelnum):
    conv = Conv2D(kernelnum, 1, padding='same', kernel_initializer='he_normal')(inputlayer)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv1 = Conv2D(kernelnum, 3,padding='same', kernel_initializer='he_normal')(conv)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    #conv1 = BatchNormalization()(conv1)#
    #if kernelnum>=512:
        #conv1 = Dropout(0.4)(conv1)
    conv2 = Conv2D(kernelnum, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    #if kernelnum>=512:
        #conv2 = Dropout(0.4)(conv2)
    out = add([conv, conv2])
    return out
def BSNet(input_size, Falg_summary=False, Falg_plot_model=False, pretrained_weights=None):
    # ======================================编码器=============================================
    input1 = Input(input_size)
    conv = Conv2D(64, 7, padding='same', kernel_initializer='he_normal')(input1)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    encode1 = res_block(conv, 64) 
    pool1 = MaxPooling2D(pool_size=(2, 2))(encode1)
    encode2 = res_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(encode2)
    encode3 = res_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(encode3)
    encode4 = res_block(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(encode4)
    encode5 = res_block(pool4, 512)
    # ======================================解码.4=============================================
    lup4 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(encode5))

    sup4 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(encode5))

    smerge4 = concatenate([encode4, sup4], axis=3)
    lup4 = concatenate([encode4, lup4], axis=3)
    seg_decode_conv4, line_decode_conv4 = sl_merge_block_V8(lup4,smerge4,512)
    # ======================================解码.3=============================================
    lup3 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(line_decode_conv4))

    sup3 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(seg_decode_conv4))

    smerge3 = concatenate([encode3, sup3], axis=3)
    lup3 = concatenate([encode3, lup3], axis=3)
    seg_decode_conv3, line_decode_conv3 = sl_merge_block_V8(lup3,smerge3,256)
    # ======================================解码.2=============================================
    lup2 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(line_decode_conv3))

    sup2 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(seg_decode_conv3))

    smerge2 = concatenate([encode2, sup2], axis=3)
    lup2 = concatenate([encode2, lup2], axis=3)
    seg_decode_conv2, line_decode_conv2 = sl_merge_block_V8(lup2,smerge2,128)
    # ======================================解码.1=============================================
    lup1 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(line_decode_conv2))

    sup1 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(seg_decode_conv2))

    smerge1 = concatenate([encode1, sup1], axis=3)
    lup1 = concatenate([encode1, lup1], axis=3)
    seg_decode_conv1, line_decode_conv1 = sl_merge_block_V8(lup1,smerge1,64)
    # ===============================================输出==========================================================
    seg_out_conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(seg_decode_conv1)

    line_out_conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(line_decode_conv1)

    line_out_conv = concatenate([seg_out_conv, line_out_conv], axis=3)
    line_out_conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(line_out_conv)
 
    seg_out_conv = Add()([seg_out_conv,line_out_conv])
    seg_out = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(seg_out_conv)
    seg_out = Conv2D(1, 1, activation='sigmoid', name='seg_out')(seg_out)
    line_out = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(line_out_conv)
    line_out = Conv2D(1, 1, activation='sigmoid', name='line_out')(line_out)
    # =========================================================================================================
    model = Model(input=input1, output=[seg_out, line_out])

    opt = Adam(lr=1e-4)
    model.compile(optimizer=opt,
                  loss={'seg_out': 'binary_crossentropy',
                        'line_out': focal_loss(0.9,2.)},
                  loss_weights={
                      'seg_out': 1.2,
                      'line_out': 1.,
                  },
                  metrics={'seg_out': ['accuracy'],
                           'line_out': positive_num})
    if Falg_summary:
        model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model