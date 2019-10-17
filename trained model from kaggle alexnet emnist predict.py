# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:37:43 2019

@author: ragrawa1
"""

# =============================================================================
# kaggle trained alexnet 52000 images
# train - 41600
# val - 5200
# test - 5200
# epochs - 8
# adams
# lr - 0.001
# =============================================================================

from keras.models import load_model
from sklearn import metrics
from keras.preprocessing import image
import numpy as np
import glob
import os
import time

model_load = load_model('C:/Users/ragrawa1/Desktop/cnn/kaggle/alexnet.h5')

def predict_test(model_load):

    path_src = r'C:/Users/ragrawa1/Desktop/cnn/kaggle/aws_image_split/test/*'
    # model_load = load_model('C:/Users/ragrawa1/Desktop/cnn/kaggle/alexnet.h5')
    img_width, img_height = 224, 224
    
    test_label = []
    test_label_predicted = []
    
    for j, dir_src1 in enumerate(glob.glob(path_src)):
        label = os.path.basename(dir_src1)
        print("label : ",label)
        for j, file_src in enumerate(glob.glob(dir_src1+"/*")):
            img = image.load_img(file_src, target_size=(img_width, img_height))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
    
            # images = np.vstack([x])
            label_p = model_load.predict_classes(x)
            
            test_label_predicted.append(chr(label_p[0]+65))
            test_label.append(label)
            
    cm = metrics.confusion_matrix(test_label, test_label_predicted)
    print('Confusion Matrix :')
    print(cm)
    print('Accuracy Score : ', end="")
    print(metrics.accuracy_score(test_label, test_label_predicted))
    print('Report :')
    print(metrics.classification_report(test_label, test_label_predicted))

predict_test(model_load)

def predict_img(model_load, path_src):

    # path_src = r'/kaggle/input/aws_image_split/test/*'
    img_width, img_height = 224, 224

    img = image.load_img(path_src, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # images = np.vstack([x])
    label_p = model_load.predict_classes(x)
    
    print(chr(label_p[0]+65))

predict_img(model_load,"C:/Users/ragrawa1/Desktop/cnn/kaggle/img_test.png")


predict_img(model_load,"C:/Users/ragrawa1/Desktop/cnn/kaggle/aws_image_split/test/A/train_41_00025.png")










