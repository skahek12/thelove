# -*- coding: utf-8 -*-
"""

@author: NAMO KIM
"""
import os
import tensorflow as tf
import numpy as np




def search(dirname):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        return filenames
        
def weight_variable(shape):
    initial= tf.truncated_normal(shape,stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def Max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

    
def _CNNModel(N):
    x= tf.placeholder(tf.float32,[None, N])
    X_img = tf.reshape(x, [-1, 96, 96, 1])
    #imgin shaoe(?,96,96,1)
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(X_img,W_conv1)+b_conv1)
    h_pool1 = Max_pool_2x2(h_conv1)
    #print(np.shape(h_conv1),np.shape(h_pool1))
    # conv(?, 96, 96, 32) 
    # pool(?, 48, 48, 32)
    W_conv2 = weight_variable([3,3,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2 = Max_pool_2x2(h_conv2)
    #print(np.shape(h_conv2),np.shape(h_pool2))
    # conv(?, 48, 48, 64) 
    # pool(?, 24, 24, 64)
    W_conv3 = weight_variable([3,3,64,128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
    h_pool3 = Max_pool_2x2(h_conv3)
    # conv(?, 24, 24, 128) 
    # pool(?, 12, 12, 128)
    W_conv4 = weight_variable([3,3,128,256])
    b_conv4 = bias_variable([256])
    h_conv4 = tf.nn.relu(conv2d(h_pool3,W_conv4)+b_conv4)
    h_pool4 = Max_pool_2x2(h_conv4)
    #print(np.shape(h_conv4),np.shape(h_pool4))
    # conv(?, 12, 12, 256) 
    # pool(?, 6, 6, 256)
    print("%s---> CNN Model was built"%h_pool4)
    return h_pool4, x
    
def _FlatModel(CNNModel):
    W_fc1 = weight_variable([6*6*256,2048])
    b_fc1 = bias_variable([2048])
    h_pool2_flat = tf.reshape(CNNModel, [-1,6*6*256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    print("%s---> Flat Model was built"%h_fc1)
    #print(np.shape(CNNModel),np.shape(h_fc1))
    # input(?, 4, 4, 64)
    # out(?, 256)
    return h_fc1

def _DropOut(FlatModel):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(FlatModel,keep_prob)
    print("%s---> DropOut Model was built"%h_fc1_drop)
    return h_fc1_drop, keep_prob 


def _SoftMax(DropOut):
    W_fc2 = weight_variable([2048, 2])
    b_fc2 = bias_variable([2])
    y_conv = tf.matmul(DropOut,W_fc2)+b_fc2
    print("%s---> SoftMax Model was built"%y_conv)
    #print(np.shape(DropOut),np.shape(y_conv))
    return y_conv
    
def _SetAccuracy(SoftMaxModel, Y):
    y_=tf.placeholder(tf.float32,[None,Y])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=SoftMaxModel))
    train_step =  tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    print("%s---> Train Model was built"%train_step)
    correct_prediction = tf.equal(tf.argmax(SoftMaxModel,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("%s---> Accuracy Model was built"%accuracy)
    return train_step, accuracy, y_, correct_prediction,cross_entropy
    
def Nextbatch(data, label, batchsize):
    idx = np.random.randint(160, size=batchsize)
    return data[idx,:],label[idx,:]