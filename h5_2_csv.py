# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 08:31:49 2018

@author: wangq
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from PIL import Image
import h5py

f = open('./mpii_train.csv', 'w')
data = h5py.File('./train.h5')
vis = False

for n in range(len(data['imgname'].value)):
    scale = 1.25
    name = data['imgname'].value[n].astype(str)
    points = data['part'].value[n]
    visible = data['visible'].value[n].reshape([-1, 1])
    img = Image.open('./images/' + name)

    print(n, name)
    if vis:
        plt.imshow(img)
    w, h = img.size

    points[:, 0][points[:, 0] >= w] = w - 1
    points[:, 1][points[:, 1] >= h] = h - 1
    if vis:
        plt.scatter(points[:, 0], points[:, 1])
        for line in [[6, 7, 8, 9], [6, 2, 1, 0], [6, 3, 4, 5], [8, 12, 11, 10], [8, 13, 14, 15]]:
            plt.plot(points[line, 0],
                     points[line, 1],
                     c='b')

    points_box = []
    for i in range(16):
        if (points[i, 0] == 0) & (points[i, 1] == 0):
            continue
        points_box.append(points[i])
    points_box = np.array(points_box)
    xmin = np.max([points_box[:, 0].min(), 0])
    xmax = np.min([points_box[:, 0].max(), (w - 1)])
    ymin = np.max([points_box[:, 1].min(), 0])
    ymax = np.min([points_box[:, 1].max(), (h - 1)])
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    w_box = (xmax - xmin) / 2 * scale
    h_box = (ymax - ymin) / 2 * scale
    xmin = np.max([x_center - w_box, 0])
    xmax = np.min([x_center + w_box, (w - 1)])
    ymin = np.max([y_center - h_box, 0])
    ymax = np.min([y_center + h_box, (h - 1)])
    if vis:
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], c='r')
        plt.show()

    points = np.concatenate([points, visible], axis=-1).reshape([-1, ])
    out = name+','+' '.join(np.array([xmin, ymin, xmax, ymax], dtype=str))+','+' '.join(np.array(points, dtype=str))
    f.write(out+'\n')
f.close()

