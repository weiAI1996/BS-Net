# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:42:04 2019

@author: Administrator
"""

import matplotlib.pyplot as plt
from keras import backend as K
import math
import os

def plotHistory(History, savePath):
    # 绘制训练 & 验证的准确率值
    plt.plot(History.history['acc'])
    plt.plot(History.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.savefig(savePath + '/Accuracy.jpg')
#    plt.show()
    
    
    # 绘制训练 & 验证的损失值
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.savefig(savePath + '/Loss.jpg')
#    plt.show()


def plotFeaturemap(input_arr, model, savePath):
    # savePath = savePath + '/Featuremap' # make dir for saving
    # os.makedirs(savePath)
    #第一个 model.layers[0],不修改,表示输入数据；
    #第二个model.layers[you wanted],修改为你需要输出的层数的编号
    layer_ = K.function([model.layers[0].input], [model.layers[1].output])
    #输入图像: input_arr
    featureMap = layer_([input_arr])[0]
    # 显示前32张feature map
    if featureMap.shape[3] < 31:
        featureMap_num = 2
    else:
        featureMap_num = 32
    
    for _ in range(featureMap_num):
                #（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
                show_img = featureMap[:, :, :, _]
                show_img.shape = [featureMap.shape[1], featureMap.shape[2]]
                # 展示前32张feature map
                plt.subplot(4, 8, _ + 1)
                plt.imshow(show_img, cmap='gray')
                plt.axis('off')
    plt.savefig(savePath + '/layer-' + str(_+1) + '.jpg')
#    plt.show()
