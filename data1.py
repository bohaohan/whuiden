# -*- coding: UTF-8 -*-
__author__ = 'bohaohan'
import os
# from passage.models import RNN
from PIL import Image
import numpy as np
# from detect import *
from pre_pro import *

cha = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a',
       'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
       'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
       'x', 'y', 'z']

def get_index(x):
    return cha.index(x)

def get_name(name):
    if '_' in name:
        return name[:name.index('_')]
    else:
        return name



def load_n():
    data = np.empty((57218, 1, 24, 24), dtype="float32")
    label = []
    t = 0
    print 'load data'
    classes = os.listdir("./result")
    for i in range(len(classes)):
        imgs = os.listdir('./result/' + classes[i])
        # if i > 0:
        for img in imgs:
            arr1 = np.array(Image.open('./result/' + classes[i] + "/" + img), dtype="float32")
            data[t][0] = arr1
            label.append(get_index(classes[i]))
            t += 1


    print 'get data success'
    data = np.array(data, dtype="float32")
    return data, label

if __name__ == '__main__':
    d, l = load_n()
    print d.shape

