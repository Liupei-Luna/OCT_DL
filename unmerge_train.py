# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:35:33 2019

@author: lenovo
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image 
import random
#from tensorflow.examples.tutorials.mnist import input_data
import os
do_train=0
yanzheng=0
#mnist = input_data.read_data_sets('data/', validation_size=0)
#img = mnist.train.images[2]
#plt.imshow(img.reshape((28, 28)), cmap='Greys_r')、
def get_batch(batchsize):
    figurelists=["youda","youwuming","youxiao","youshi","youzhong","zuoda","zuoshi","zuowuming","zuoxiao","zuozhong"]
    getpeople={1:"number3",2:"number4",3:"number5"}
    getcondition={1:"first_Dry",2:"second_Wet",3:"third_Dirty"}
    people=getpeople[random.randint(1,3)]
    condition=getcondition[random.randint(1,3)]
    figure=figurelists[random.randint(0,9)]
    if condition=="second_Wet" or condition=="third_Dirty":
        root2="D:/fiugerprint/3D_Fingerprint_date_3-5/"+people+"/"+"first_Dry"+"/"+figure
    root="D:/fiugerprint/3D_Fingerprint_date_3-5/"+people+"/"+condition+"/"+figure
    picturenum=random.randint(1,400-batchsize+1)
    try:
        tmp2=[]
        if condition=="second_Wet" or condition=="third_Dirty": 
            for i in range(picturenum,picturenum+batchsize):
                image=Image.open(root2+"/"+str(i)+".bmp")
                image=image.resize((256,256))
                array = np.array(image)
                tmp2.append(array)
            tmp2=np.array(tmp2)    
        tmp=[]
        for i in range(picturenum,picturenum+batchsize):
            image=Image.open(root+"/"+str(i)+".bmp")
            image=image.resize((256,256))
            array = np.array(image)
            tmp.append(array)
        tmp=np.array(tmp)
    except:
        return get_batch(batchsize)
    return tmp,tmp2
batch=get_batch(4)    
    
#root = "8/youda"
#tmp=[]
#for dirpath, dirnames, filenames in os.walk(root):
#    for filepath in filenames:
#        image = Image.open(root+'/'+filepath)   # image is a PIL image
#        image=image.resize((256,256))
#        array = np.array(image) 
##        resized = tf.image.resize_images(array, [256, 256], method=0)
#        array=array.reshape(-1)        # array is a numpy array 
#        tmp.append(array)
#tmp = np.array(tmp)
total_size=12000
#n_input  = 256*256*3
#n_output =  256*256*3


inputs_ = tf.placeholder(tf.float32, (None, 256, 256, 3), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, 256, 256, 3), name='targets')


### 编码器--压缩
conv1 = tf.layers.conv2d(inputs_, 32, (3,3), padding='same', activation=tf.nn.relu)
# 当前shape: 256x256x32
maxpool1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
# 当前shape: 128x128x32
conv2 = tf.layers.conv2d(maxpool1, 16, (3,3), padding='same', activation=tf.nn.relu)
# 当前shape: 128x128x16
maxpool2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
# 当前shape: 64x64x16
conv3 = tf.layers.conv2d(maxpool2, 8, (3,3), padding='same', activation=tf.nn.relu)
# 当前shape: 64x64x8
encoded = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
# 当前shape: 32x32x8


### 解码器--还原
upsample1 = tf.image.resize_nearest_neighbor(encoded, (32,32))
# 当前shape: 32x32x8
conv4 = tf.layers.conv2d(upsample1, 8, (3,3), padding='same', activation=tf.nn.relu)
# 当前shape: 32x32x8
upsample2 = tf.image.resize_nearest_neighbor(conv4, (64,64))
# 当前shape: 64x64x8
conv5 = tf.layers.conv2d(upsample2, 16, (3,3), padding='same', activation=tf.nn.relu)
# 当前shape: 64x64x16
upsample3 = tf.image.resize_nearest_neighbor(conv5, (128,128))
# 当前shape: 128x128x16
conv6 = tf.layers.conv2d(upsample3, 16, (3,3), padding='same', activation=tf.nn.relu)
# 当前shape: 128x128x16
upsample4 = tf.image.resize_nearest_neighbor(conv5, (256,256))
# 当前shape: 256x256x16
conv7 = tf.layers.conv2d(upsample4, 32, (3,3), padding='same', activation=tf.nn.relu)
# 当前shape: 256x256x32


logits = tf.layers.conv2d(conv7, 3, (3,3), padding='same', activation=None)
#当前shape: 28x28x1


decoded = tf.nn.sigmoid(logits, name='decoded')


#计算损失函数
cost = tf.reduce_mean(tf.pow(targets_ - logits, 2),name="cost")

#loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
#cost = tf.reduce_mean(loss)
#使用adam优化器优化损失函数
opt = tf.train.AdamOptimizer(0.001).minimize(cost)

saver = tf.train.Saver(max_to_keep=3)
if do_train==1:
    sess = tf.Session()
    epochs = 1000
    batch_size = 30
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(total_size//batch_size):
#            ranint=random.randint(0,total_size-batch_size-1)
#            batch_xs = tmp[ranint:ranint+batch_size]
#            imgs = batch_xs.reshape((-1, 256, 256, 3))
            imgs,targets=get_batch(batch_size)
            if targets==[]:
                batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: imgs,targets_: imgs})
            else:
                batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: imgs,targets_: targets})
            print("Epoch: {}/{}...".format(e+1, epochs)
            ,
                  "Training loss: {:.4f}".format(batch_cost))
        if e%10==0 or e==epochs-1:
            saver.save(sess, "D:/fiugerprint/model/cnn_mnist_basic.ckpt-" + str(e))
 

lists=["youda","youwuming","youxiao","youshi","youzhong","zuoda","zuoshi","zuowuming","zuoxiao","zuozhong"]
if do_train==0 and yanzheng==0:
    save_dir = 'D:/fiugerprint/model/cnn_mnist_basic.ckpt-999'
    for dirname in lists:
        
        root = "D:/fiugerprint/3D_Fingerprint_date_3-5/number5/third_Dirty/"+dirname
        tmp=[]
        for dirpath, dirnames, filenames in os.walk(root):
            for filepath in filenames:
                image = Image.open(root+'/'+filepath)   # image is a PIL image
                image=image.resize((256,256))
                array = np.array(image) 
        #        resized = tf.image.resize_images(array, [256, 256], method=0)
                array=array.reshape(-1)        # array is a numpy array 
                tmp.append(array)
        tmp = np.array(tmp)
        tmp=tmp.reshape((-1,256,256,3))
        tmp.shape
        tmpe=np.zeros((1,32,32,8))
        for i in range(0,len(tmp),10):
            sess = tf.Session()
#        tf.reset_default_graph()
            saver.restore(sess,save_dir) 
            array = sess.run(encoded, feed_dict={inputs_: tmp[i:i+10]})
            array.shape
            tmpe=np.concatenate([tmpe,array],axis=0) #2表示最后一个维度
            sess.close()
        tmpe=np.delete(tmpe,0,axis=0)
        tmpe.shape

        np.save("D:/fiugerprint/3D_Fingerprint_date_3-5/number5/third_Dirty/"+dirname+".npy",tmpe)       

#验证
def raliative(a,b):  #计算两张图片的距离
    if len(a)!=len(b) and len(a[0])!=len(b[0]) and len(a[0][0])!=len(b[0][0]):
        return "error"
    errors=0
    for i in range(len(a)):
        for j in range(len(a[0])):
            for k in range(len(a[0][0])):
                for l in range(len(a[0][0][0])):
                    errors+=(a[i][j][k][l]-b[i][j][k][l])**2
    return errors**0.5
def cos(A,B): #计算两张图片的相似度
    num = float(np.dot(A.T , B)) #若为行向量则 A * B.T
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    cos = num / denom #余弦值
    sim = 0.5 + 0.5 * cos #归一化
    dist = np.linalg.norm(A - B)
    sim = 1.0 / (1.0 + dist) #归一化
    return sim
    
if do_train==0 and yanzheng==1:
    a = np.load("D:/fiugerprint/3D_Fingerprint_date_3-5/number3/first_Dry/youda.npy")
    a=a.reshape(-1,1)
    a.shape
    for dirname in lists:
        b=np.load("D:/fiugerprint/3D_Fingerprint_date_3-5/number3/second_Wet/"+dirname+".npy")
        b.shape
        b=b.reshape(-1,1)
        print(cos(a,b))  


   