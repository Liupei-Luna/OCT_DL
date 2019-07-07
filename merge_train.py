# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:58:15 2019

@author: 48946
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image 

#from tensorflow.examples.tutorials.mnist import input_data
import os



#root = "3D_Fingerprint_1-2/number10/first_Dry"
#tmp=[]
#for dirpath, dirnames, filenames in os.walk(root):
#    for filepath in filenames:
#        image = Image.open(root+'/'+filepath)   # image is a PIL image
##        image=image.resize((5120,5120))
#        array = np.array(image) 
#        resized = tf.image.resize_images(array, [5120,5120], method=0)
#        array=array.reshape(-1)        # array is a numpy array 
#        tmp.append(array)
#tmp = np.array(tmp)


inputs = tf.placeholder(tf.float32,(None,5120,5120,3),name='inputs')
targets = tf.placeholder(tf.float32,(None,5120,5120,3),name='targets')

##Encoder
conv1 = tf.layers.conv2d(inputs=inputs,filters=2, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 5120x5120x16
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
# Now 2560x2560x16
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=2, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 2560x2560x16
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
# Now 1280x1280x16
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=2, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 1280x1280x16
maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
# Now 640x640x16
conv4 = tf.layers.conv2d(inputs=maxpool3, filters=2, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 640x640x16
maxpool4 = tf.layers.max_pooling2d(conv4, pool_size=(2,2), strides=(2,2), padding='same')
# Now 320x320x16
conv5 = tf.layers.conv2d(inputs=maxpool4, filters=2, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 320x320x16
maxpool5 = tf.layers.max_pooling2d(conv5, pool_size=(2,2), strides=(2,2), padding='same')
# Now 160x160x16
conv6 = tf.layers.conv2d(inputs=maxpool5, filters=2, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 28x28x32
maxpool6 = tf.layers.max_pooling2d(conv6, pool_size=(2,2), strides=(2,2), padding='same')
# Now 160x160x16
conv7 = tf.layers.conv2d(inputs=maxpool6, filters=1, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 80x80x8
encoded = tf.layers.max_pooling2d(conv7, pool_size=(2,2), strides=(2,2), padding='same')
##40×40×8


## Decoder
upsample1 = tf.image.resize_images(encoded, size=(80,80))
# Now 80x80x8
conv8 = tf.layers.conv2d(inputs=upsample1, filters=1, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 80x80x8
upsample2 = tf.image.resize_images(conv8, size=(160,160))
# Now 160x160x8
conv9 = tf.layers.conv2d(inputs=upsample2, filters=1, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 160x160x8
upsample3 = tf.image.resize_images(conv9, size=(320,320))
# Now 320x320x8
conv10 = tf.layers.conv2d(inputs=upsample3, filters=1, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 320x320x8
upsample4 = tf.image.resize_images(conv10, size=(640,640))
# Now 640x640x8
conv11 = tf.layers.conv2d(inputs=upsample4, filters=1, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 640x640x8
upsample5 = tf.image.resize_images(conv11, size=(1280,1280))
# Now 1280x1280x8
conv12 = tf.layers.conv2d(inputs=upsample5, filters=1, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 1280x1280x8
upsample6 = tf.image.resize_images(conv12, size=(2560,2560))
# Now 2560x2560x8
conv13 = tf.layers.conv2d(inputs=upsample6, filters=1, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 2560x2560x8
upsample7 = tf.image.resize_images(conv13, size=(5120,5120))
# Now 5120x5120x8
conv14 = tf.layers.conv2d(inputs=upsample7, filters=2, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 5120x5120x16
logits = tf.layers.conv2d(inputs=conv14, filters=3, kernel_size=(3,3), padding='same', activation=None)
# Now 5120x5120x16
decoded = tf.nn.sigmoid(logits)

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)
#cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(0.001).minimize(loss)



image = Image.open("F:/fingerprint/3D_Fingerprint_1-2/number10/first_Dry/zuoda.bmp")
full=np.reshape(image,[-1,5120,5120,3])
sess = tf.Session()
epochs = 100
batch_size = 1


sess.run(tf.global_variables_initializer())
for e in range(epochs):
#    for ii in range(1):
        
    
    cost, _ = sess.run([loss, opt], feed_dict={inputs: full,
                                                         targets: full})
    print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {:.4f}".format(cost))
#
