
# coding: utf-8

# ### Casual weaseling

# In[1]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import numpy as np
import os


# # Load data 

# In[2]:

X = np.load("/homeappl/home/austyuzh/data/dae.data.npy")


# In[3]:

print X.shape
plt.imshow(X[1000,0],cmap='gray')


# # Train and test

# In[77]:


train_size = (len(X)*5)/6
print train_size


# In[78]:

X[0]


# In[79]:

X_train = y_train = X[:train_size]
X_val = y_val = X[train_size:]
#X_val = y_val = X[1:]
print "train shapes X:", X_train.shape
print "train shapes X:", X_val.shape



# ### Lasagne part

# In[82]:

import theano
import theano.tensor as T
import lasagne 


input_X = T.tensor4("X cat/dog image")
target_y = T.tensor4('target Y')



#input size, None means "arbitrary
input_shape = [None] + list(X.shape[1:])




# In[83]:

target_y.dtype


# In[98]:

from lasagne.layers import InputLayer,DenseLayer,batch_norm,dropout
from lasagne.layers import Deconv2DLayer, Conv2DLayer
#Input layer
input_layer = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X)

n_hid = 50


nput_layer = lasagne.layers.InputLayer(shape=input_shape, input_var=input_X)

conv1 = Conv2DLayer(input_layer, 16, 3, pad='same', name='conv1')
conv2 = Conv2DLayer(conv1, 1, 3, pad='same', name='conv2')

dense_output = conv2


# In[99]:

#get prediction
y_predicted = lasagne.layers.get_output(dense_output)


# In[100]:

#all weights
all_weights = lasagne.layers.get_all_params(dense_output,trainable=True)
print all_weights


# In[101]:

#loss function
loss = lasagne.objectives.squared_error(y_predicted,target_y).mean()

#maybe regularize

#updates
updates_sgd = lasagne.updates.adadelta(loss,all_weights)



# In[102]:



#train function step
train_fun = theano.function([input_X,target_y],[loss],updates= updates_sgd)



# In[110]:

eval_fun = theano.function([input_X], y_predicted)


# ### Training loop

# In[ ]:




# In[103]:

#Old friend
def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]


# In[106]:

#training

num_epochs = 2

batch_size = 2
import time
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train,batch_size):
        inputs, targets = batch
        print inputs.shape
        print targets.shape
        train_err= train_fun(inputs, targets)
        train_err_batch.append(train_err)
        train_batches += 1


    
    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))

    print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches))
    print("  train accuracy:\t\t{:.2f} %".format(
        train_acc / train_batches * 100))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))


# # UseIT
# Compile a function that returns reconstructed image given original image.
# Use it on several random weasels

# In[107]:

plt.imshow(X[1000][0], cmap='gray')


# In[111]:

plt.imshow(eval_fun(X[None, [1000], 0])[0][0], cmap='gray')


# In[ ]:



