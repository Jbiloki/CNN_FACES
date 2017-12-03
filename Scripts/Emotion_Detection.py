# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:00:38 2017

@author: Nguyen
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Tensorflow stuff
import tensorflow as tf

#Scikit Learn support
from sklearn.model_selection import train_test_split

import cv2
from PIL import Image

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1],padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def preprocess(data):
    data.drop(labels='usage', axis = 1, inplace= True)

def show_images(arr):
    fig = plt.figure(1, figsize=(20,20))
    #print(arr[1].reshape((48,48)))
    for i in range(9):
        ax = fig.add_subplot(3,3,i+1)
        print(np.array(arr[i]))
        arr = np.asarray(arr[i]).reshape((48,48))
        ax.imshow(arr)

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.Graph().as_default()
    data_frame = pd.read_csv('../Data/train.csv')
    data_frame['Pixels'] = data_frame['Pixels'].apply(lambda x: list(map(int,x.split(' '))))
    #im = Image.fromarray(np.asarray(data_frame['Pixels'].iloc[0]).reshape(48,48))
    #im.show()
    #X_train,X_test,y_train,y_test = train_test_split(data_frame.Pixels, data_frame.Emotion, test_size = 0.33)
    batch_size = 100
    debug_step = 10
    n_samples = data_frame.Emotion.shape[0]
    band = np.array([np.array(band).astype(np.float32) for band in data_frame['Pixels']])
    #show_images(X_train)
    labels = tf.one_hot(indices=tf.cast(data_frame.Emotion, tf.int32), depth=7)
    #labels_test = tf.one_hot(indices=tf.cast(y_test, tf.int32), depth=7)
    #Build our net
    x = tf.placeholder(tf.float32, shape=[None,2304])
    y_ = tf.placeholder(tf.float32, shape=[None, 7])
    
    with tf.name_scope('Model'):
        with tf.name_scope('Conv1'):
            #The shape will be [[patch size], input channels, output channels]
            W_conv1 = weight_variable([3,3,1,32])
            b_conv1 = bias_variable([32])
            x_image = tf.reshape(x, [-1, 48,48,1])
            #Layer 1
            h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)
        with tf.name_scope('Conv2'):
            #Layer 2
            W_conv2 = weight_variable([3,3,32,64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)
        with tf.name_scope('DenseLayer'):
            #Dense layer
            W_fc1 = weight_variable([12*12*64,1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1,12*12*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        with tf.name_scope('Dropout'):
            #Dropout
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        with tf.name_scope('OutputLayer'):
            #Final layer (logits)
            W_fc2 = weight_variable([1024,7])
            b_fc2 = bias_variable([7])
            output_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = output_conv))
    opt = tf.train.AdamOptimizer(.0001).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(output_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    probabilities = tf.nn.softmax(output_conv)
    tf.summary.FileWriterCache.clear()
    saver = tf.train.Saver()
    print('Beginning Session...')
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graph', graph = sess.graph)
        sess.run(tf.global_variables_initializer())
        labels = labels.eval()
        for i in range(30):
            print('Current Epoch: ',i)
            for batch in range(int(n_samples/batch_size)):
                batch_x = band[batch*batch_size : (1+batch) * batch_size]
                batch_y = labels[batch*batch_size : (1+batch) * batch_size]
                sess.run([opt], feed_dict={x:batch_x,y_:batch_y,keep_prob:.4})
            if i % debug_step == 0:
                    train_accuracy = accuracy.eval(feed_dict={x:batch_x,y_:batch_y, keep_prob : 1.0})
                    prob = probabilities.eval(feed_dict={x:batch_x,y_:batch_y, keep_prob : 1.0})
                    print('Probabilities of each class: ', prob)
                    print('Accuracy: ', train_accuracy)
        train_accuracy = accuracy.eval(feed_dict={x:band,y_:labels, keep_prob : 1.0})
        print('Final training accuracy %g' % (train_accuracy))
        save_path = saver.save(sess, '../tmp/')
        print('Model saved in file: %s' % save_path)
    writer.close()
    print('End of program...')