# -*- coding: utf-8 -*-
"""

@author: NAMO KIM
"""

import urllib.request
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image as im
import numpy as np
import utills as ut
import tensorflow as tf
import time
import socket

#TB_SUMMARY_DIR='./tb/diabetes_DEEP'

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_address = ('192.168.0.21', 12345)

sess = tf.InteractiveSession()
train_epoch = 1000
# define parameter
class_num = 2
data_length = []
dir_image = []
data = []
label = []

inputimage = []

data_shape = [96, 96]
current_pwd = os.getcwd()

for i in range(class_num):
    dir_image.append(ut.search(current_pwd + '\\' + str(i+1)))
    data_length.append(len(dir_image[i]))
    data.append(np.zeros([data_length[i], data_shape[1], data_shape[0]]))
    label.append(np.zeros([data_length[i], class_num]))
    label[i][:,i] = 1 




# load data
for q in range(class_num):
    for i in range(data_length[q]):
        data[q][i,:,:] = np.mean(im.open(current_pwd + '\\' + str(q+1) + '\\' + dir_image[q][i]), -1)


        
rawdata = np.concatenate((data[0], data[1]),axis=0)
del data

raw_label = np.concatenate((label[0], label[1]),axis=0)
del label

total_data_poin = rawdata.shape[0]
permutation = np.random.permutation(total_data_poin)
rawdata = rawdata[permutation,:,:]
raw_label = raw_label[permutation,:]

rawdata = np.reshape(rawdata,[rawdata.shape[0], data_shape[0] * data_shape[1]])

##################################################################
train_size=int(len(rawdata)*0.8)
test_size=len(rawdata)-train_size


sess = tf.InteractiveSession()       

TrainX = np.array(rawdata[0:train_size])
TrainY = np.array(raw_label[0:train_size])


#idx = np.random.randint(test_size, size=test_size)
testX = np.array(rawdata[train_size:len(rawdata)])
testY = np.array(raw_label[train_size:len(raw_label)])

#inputX = np.array(raw_label[train_size:len(raw_label)])
#inputY = np.array(raw_label[train_size:len(raw_label)])

#testX = testX[idx,:]
#testY = testY[idx,:]


CNNModel, x = ut._CNNModel(9216)
FlatModel = ut._FlatModel(CNNModel)
DropOut, keep_prob = ut._DropOut(FlatModel)
SoftMaxModel = ut._SoftMax(DropOut)
TrainStep, Accuracy, y_, correct_prediction, cost= ut._SetAccuracy(SoftMaxModel, 2)

#tf.summary.scalar("loss", cost)
#summary=tf.summary.merge_all()

sess.run(tf.global_variables_initializer())
     #Create summary writer
   
#writer=tf.summary.FileWriter(TB_SUMMARY_DIR)
#writer.add_graph(sess.graph)
#global_step=0

#while(True):
for i in range(train_epoch):
  tmp_trainX, tmp_trainY = ut.Nextbatch(TrainX,TrainY,10)
  if i%10 == 0:
    train_accuracy = Accuracy.eval(feed_dict={x:tmp_trainX, y_: tmp_trainY, keep_prob: 0.7})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  TrainStep.run(feed_dict={x: tmp_trainX, y_: tmp_trainY, keep_prob: 0.7})
 # writer.add_summary(cost,global_step=global_step)
  #global_step+=1

print("test accuracy %g"%Accuracy.eval(feed_dict={x:testX[1:test_size,:], y_: testY[1:test_size], keep_prob: 1.0}))

while(True):
    urllib.request.urlretrieve("http://223.194.43.50:8080/stream/snapshot.jpeg?delay_s=0","image.jpeg") #get image from streaming server
    im1 = im.open("image.jpeg")
    img=mpimg.imread('image.jpeg')
    imgplot = plt.imshow(img)
    plt.show()
    im_small = im1.resize((96,96),im.ANTIALIAS)   #resize 96 x 96
    im_small.save("inputimage.jpeg")
    #im1.show()
    inputimage= np.zeros([1, data_shape[1], data_shape[0]])
    inputimage[0,:,:] =np.mean(im_small, -1)
    inputimage = np.reshape(inputimage,[inputimage.shape[0], data_shape[0] * data_shape[1]])
    if Accuracy.eval(feed_dict={x:inputimage, y_: [[1.,0.]], keep_prob: 1.0}) == 1:
        thelove = "깔끔"
        thelovenum = 5+round(cost.eval(feed_dict={x:inputimage, y_: [[1.,0.]], keep_prob: 1.0})*20,2)
    else :
        thelove = "어지러워ㅠㅠㅠㅠ"
        thelovenum = 95-round(cost.eval(feed_dict={x:inputimage, y_: [[1.,0.]], keep_prob: 1.0})*4,2)
    print("thelove :",thelovenum,"%")
    print("input image of thelove ",thelove)
    print(" ")
    time.sleep(1)
    sent = sock.sendto(str(thelovenum).encode(), server_address)
    plt.close()
   # im_small.close()
    
