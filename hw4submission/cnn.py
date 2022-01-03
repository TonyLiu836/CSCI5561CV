import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    # TO DO
    
    mini_batch_x = np.array_split(np.asarray(im_train), np.shape(im_train)[1]/batch_size, axis = 1)
    mini_batch_x = np.asarray(mini_batch_x)
    y_onehot = np.zeros([10, np.shape(label_train)[1]])
    
    for i in range(np.shape(label_train)[1]):
        y_onehot[label_train[0,i], i] = 1
    mini_batch_y = np.array_split(np.asarray(y_onehot), np.shape(im_train)[1]/batch_size, axis = 1)
    mini_batch_y = np.asarray(mini_batch_y)
    for i in range(np.shape(mini_batch_x)[0]):
        index = np.random.permutation(np.shape(mini_batch_x)[2])
        mini_batch_x[i,:,:] = np.transpose(mini_batch_x[i,:,index])
        mini_batch_y[i,:,:] = np.transpose(mini_batch_y[i,:,index])
    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    y = np.add(np.matmul(w, np.transpose([x])).reshape(np.shape(b)), b)
    return y


def fc_backward(dl_dy, x, w, b, y):
    dl_dx = np.matmul(dl_dy, w)
    dl_dw = dl_dy * np.transpose([x])
    dl_dw = dl_dw.flatten('F')
    dl_db = dl_dy 
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    l = np.linalg.norm(y - y_tilde)**2
    dl_dy = 2 * (y_tilde - y) 
    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    
    y_tilde = np.exp(x) / np.sum(np.exp(x)) 
    l = np.sum(y * np.log(np.transpose(y_tilde)))
    dl_dy = y_tilde - y
    return l, dl_dy


def relu(x):
    # TO DO
    temp = np.zeros((np.shape(x)))
    y= np.maximum(temp, x)
    return y


def relu_backward(dl_dy, x, y):   
    # TO DO
    
    #dl_dy = 1 x z    **z = size of input(can be tensor, matrix, vector)
    #dz_dz = 1 x z

    dl_dyFlat = dl_dy.flatten()
    dy_drelu = dl_dyFlat
    dy_drelu[dl_dyFlat < 0] = 0
    dy_drelu[dy_drelu > 0] = 1
    dl_dx = dy_drelu.reshape(dl_dy.shape)
    
    return dl_dx


def im2col(matrix, filterH, filterW):
    #pad = 1
    #stride = 1
    
    shiftX = np.shape(matrix)[0] - filterH + 1
    shiftY = np.shape(matrix)[1] - filterW + 1
    
    matrixim2col = np.zeros((filterH * filterW, shiftX * shiftY))
    for w in range(shiftX):
        for h in range(shiftY):
            matrixim2col[:, w * shiftY + h] = matrix[h:h+filterH, w:w+filterW].flatten('F')

    return matrixim2col


def conv(x, w_conv, b_conv):
    # TO DO
    
    #x = HxWxC1
    #w_conv = h x w x C1 x C2
    #b_conv = C2x1
    #stride = 1
    #3x3 convolution and 3 channel output
    
    h, w, c1, c2 = w_conv.shape
    xShape = np.shape([x])    #1x196
    wConvh,wConvw, wConvC1, wConvC2 = np.shape(w_conv)
    y = np.zeros((xShape[0], xShape[1], wConvC2))           #1x196x3
    
    xReshaped = np.reshape(x, (14,14))
    xPadded = np.pad(xReshaped, (1,1), 'constant')          #16x16
    xPadim2col = im2col(xPadded, wConvh, wConvw)            #9x196    
    b_convFlat = np.transpose([b_conv.flatten('F')])        #3x1
    w_convFlat = np.zeros((wConvh*wConvw, wConvC2))
    for i in range(wConvC2):
        w_convFlat[:,i] = np.transpose(w_conv[:,:,:,i].flatten('F'))
    
    y = np.dot(np.transpose(xPadim2col), w_convFlat) + np.transpose(b_convFlat)
    return y



def conv_backward(dl_dy, x, w_conv, b_conv, y):         
    # TO DO
    
    #print("dl_dy shape", np.shape(dl_dy))           #14x14x3
    #print("y shape", np.shape(y))                   #196x3
    #print('x shape', np.shape(x))                   #196
    #print('w_conv shape', np.shape(w_conv))         #3x3x1x3
    #print('b_conv shape', np.shape(b_conv))         #3
    
    wConvh,wConvw, wConvC1, wConvC2 = np.shape(w_conv)
    xReshaped = np.reshape(x, (14,14))
    xPadded = np.pad(xReshaped, (1,1), 'constant')
    xPadim2col = im2col(xPadded, wConvh, wConvw)
    dy_dw = np.transpose(xPadim2col)
    
    newdl_dy = np.zeros((np.shape(dl_dy)[0]**2, np.shape(dl_dy)[2]))
    for i in range(np.shape(dl_dy)[2]):
        newdl_dy[:,i] = dl_dy[:,:,i].flatten("F")
    
    dl_dw = np.dot(np.transpose(newdl_dy), dy_dw)
    dl_dw = np.reshape(dl_dw, (3,3,1,3), order ='F')
    dl_db = np.sum(newdl_dy,axis = 0)
    return dl_dw, dl_db

def pool2x2(x):                                 
    # TO DO
    #stride = 2
    xLen = np.shape(x)[0]
    channels = int(np.shape(x)[1])
    xWidth = xHeight = int(np.sqrt(xLen))           #14
    x = x.reshape((xWidth,xHeight,np.shape(x)[1]))

    newHeight = newWidth = ((xHeight-2)//2) + 1
    y = np.zeros((newHeight, newWidth, channels))       #7x7x3
    
    for c in range(channels):
        for h in range(0,xHeight,2):
            for w in range(0, xWidth,2):
                y[h//2, w//2 ,c] = np.max(x[h:h+2, w:w+2, c])
    return y


def pool2x2_backward(dl_dy, x, y):                  
    # TO DO
    #print('dl_dy shape', np.shape(dl_dy))   #7x7x3
    #print('x shape', np.shape(x))           #196x3
    #print('y shape', np.shape(y))           #7x7x3
    
    x = x.reshape((14,14,3))
    dl_dx = np.zeros((np.shape(x)))
    height = np.shape(x)[0]
    width = np.shape(x)[1]
    channels = np.shape(x)[2]
    
    for c in range(channels):
        for h in range(0, height, 2):
            for w in range(0, width, 2):
                window = x[h: h + 2, w: w + 2, c]
                maxVal= np.amax(window.flatten())
                maxInd = np.argmax(window)  
                delW = (maxInd % 2)
                delH = int(maxInd / 2)
                dl_dx[h+delH, w+delW, c] = maxVal
    return dl_dx

def flattening(x):
    # TO DO
    y = x.flatten('F')
    return y

def flattening_backward(dl_dy, x, y):
    # TO DO
    
    dl_dx = dl_dy.reshape(np.shape(x), order = 'F') 
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    lr = 0.001
    dr= 0.85
    nIters= 1500
    w = np.random.normal(0, 1, size=(10, 196))
    b = np.random.normal(0, 1, size=(10, 1))
    numBatches = len(mini_batch_x)

    for iters in range(nIters):
        
        #print('iters', iters)
        if iters % 250 == 0:
            lr *=  dr
            
        currBatch = iters % numBatches
        
        currBatch_x = mini_batch_x[currBatch]
        currBatch_y = mini_batch_y[currBatch]
        currBatchSize = currBatch_x.shape[1]
        
        batch_dl_dw = np.zeros(10 * 196)
        batch_dl_db = np.zeros(10)
        
        for img in range(currBatchSize ):
            Xi = currBatch_x[:, img]
            y = currBatch_y[:, img]

            y_tilde = fc(Xi.reshape(196, 1), w, b).reshape(-1)

            l, dl_dy = loss_euclidean(y_tilde, y)

            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, Xi, w, b, y_tilde)

            batch_dl_dw += dl_dw
            batch_dl_db += dl_db

        w -= lr * batch_dl_dw.reshape(w.shape)
        b -= lr * batch_dl_db.reshape(b.shape) 

    return w, b

def train_slp(mini_batch_x, mini_batch_y):          #should produce acc > 85%
    
    w = np.random.normal(0,1,(10,196))
    b = np.random.normal(0,1,(10,1))
    
    numBatches = len(mini_batch_x)
    
    lr = 0.5
    dr = 0.95
    nIters = 12500
    
    for iters in range(nIters):
        #print("SLP iters =", iters)
        if iters % 500 == 0:
            lr = lr * dr
        
        currBatch = iters % numBatches
        
        currBatch_x = mini_batch_x[currBatch]
        currBatch_y = mini_batch_y[currBatch]
        batchSize = currBatch_x.shape[1]
        
        dl_dw = np.zeros((10,196))
        dl_db = np.zeros((10,1))
     
        for img in range(batchSize):
            
            Xi = currBatch_x[:, img]
            
            y = currBatch_y[:, img]
            
            y_tilde = fc(Xi, w, b)          #10x1
            
            loss, dl_dy = loss_cross_entropy_softmax(y_tilde.reshape(-1), y)
            
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, Xi, w, b, y)
            
            dl_dw += dl_dw
            dl_db += dl_db
            
        w -= lr * dl_dw.reshape(np.shape(w))
        b -= lr * dl_db.reshape(np.shape(b))   

    return w, b

def train_mlp(mini_batch_x, mini_batch_y):              #should produce acc > 90%
    # TO DO
    
    w1 = np.random.normal(0,1,size=(30,196))
    b1 = np.random.normal(0,1,size=(30,1))
    w2 = np.random.normal(0,1,size=(10,30))
    b2 = np.random.normal(0,1,size=(10,1))
    
    numBatches = len(mini_batch_x)
    
    lr = 0.0005
    dr = 0.96
    nIters = 500
    
    for iters in range(nIters):
        #print("MLP itesr=", iters)
        if iters % 50 == 0:
            lr *= dr
        
        currBatch = iters % numBatches        

        currBatch_x = mini_batch_x[currBatch]
        currBatch_y = mini_batch_y[currBatch]
        batchSize = currBatch_x.shape[1]
        
        batch_dl_dw1 = np.zeros(30 * 196)
        batch_dl_db1 = np.zeros(30)
        batch_dl_dw2 = np.zeros(10 * 30)
        batch_dl_db2 = np.zeros(10)
        
    
        for img in range(batchSize):
            Xi = currBatch_x[:, img].reshape(-1)
            y = currBatch_y[:, img]
            
            a1 = fc(Xi, w1, b1)                  
            
            f1 = relu(a1)                           

            y_tilde = fc(f1, w2, b2)       
            
            loss, dl_dy = loss_cross_entropy_softmax(y_tilde.reshape(-1), y)
            
            dl_dy, dl_dw2, dl_db2 = fc_backward(dl_dy, f1, w2, b2, y_tilde)
            
            dl_dy = relu_backward(dl_dy, a1, f1)
            
            dl_dx, dl_dw1, dl_db1 = fc_backward(dl_dy, Xi, w1, b1, a1)
            
            batch_dl_dw2 += dl_dw2.reshape(-1)
            batch_dl_db2 += dl_db2
            
            batch_dl_dw1 += dl_dw1.reshape(-1)
            batch_dl_db1 += dl_db1
            
        w2 -= lr * batch_dl_dw2.reshape(np.shape(w2))
        b2 -= lr * batch_dl_db2.reshape(np.shape(b2))

        w1 -= lr * batch_dl_dw1.reshape(np.shape(w1))
        b1 -= lr * batch_dl_db1.reshape(np.shape(b1))

    return w1, b1, w2, b2

def train_cnn(mini_batch_x, mini_batch_y):
    
    lr = 0.0004
    dr = 0.95
    nIters = 8000         #0.681 acc
    
    w_conv = np.random.normal(0, 1, size=(3,3,1,3))
    b_conv = np.random.normal(0, 1, size=(3))
    w_fc = np.random.normal(0, 1, size=(10,147))
    b_fc = np.random.normal(0, 1, size=(10, 1))
    numBatches = len(mini_batch_x)
    
    for iters in range(nIters):
        #print('iters', iters)
        if iters % 500 == 0:
            lr *= dr

        currBatch = iters % numBatches        

        currBatch_x = mini_batch_x[currBatch]
        currBatch_y = mini_batch_y[currBatch]
        batchSize = currBatch_x.shape[1]
        
        batch_dl_dwConv = np.zeros((3,3,1,3))
        batch_dl_dbConv= np.zeros((3))
        batch_dl_dwFC= np.zeros(10 * 147)
        batch_dl_dbFC = np.zeros((10))
        
        
        for img in range(batchSize):
            x = currBatch_x[:, img].reshape((14, 14, 1), order='F')
            y = currBatch_y[:, img]

            convLayer= conv(x, w_conv, b_conv)
                
            reluLayer = relu(convLayer)
            
            poolLayer = pool2x2(reluLayer)
            
            flatLayer = flattening(poolLayer)                                               #147x1
            
            y_tilde = fc(flatLayer, w_fc, b_fc)                                             #10x1
            
            loss, dl_dy = loss_cross_entropy_softmax(y_tilde.reshape(-1), y)
            
            
            dl_dy, dl_dwFC, dl_dbFC = fc_backward(dl_dy, flatLayer, w_fc, b_fc, y_tilde)
            
            dl_dy = flattening_backward(dl_dy, poolLayer, flatLayer)    
            
            dl_dy = pool2x2_backward(dl_dy, reluLayer, poolLayer)    
            
            dl_dy = relu_backward(dl_dy, convLayer , reluLayer) 
            
            dl_dwConv, dl_dbConv = conv_backward(dl_dy, x, w_conv, b_conv, convLayer) 

            batch_dl_dwConv += dl_dwConv
            batch_dl_dbConv += dl_dbConv
            batch_dl_dwFC += dl_dwFC
            batch_dl_dbFC  += dl_dbFC
        
        w_conv -= lr * batch_dl_dwConv
        #b_conv -= batch_dl_dbConv * lr         #
        w_fc -= lr * batch_dl_dwFC.reshape(w_fc.shape)
        b_fc -= lr * batch_dl_dbFC.reshape(b_fc.shape) 

    return w_conv, b_conv, w_fc, b_fc

if __name__ == '__main__':
    #main.main_slp_linear()
    #main.main_slp()
    #main.main_mlp()
    main.main_cnn()



