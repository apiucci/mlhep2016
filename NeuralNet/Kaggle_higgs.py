
# coding: utf-8

# In[1]:

#%matplotlib inline


# In[3]:

get_ipython().system(u'pip install --upgrade sklearn')


# In[8]:

get_ipython().system(u'pip install --upgrade pip')


# In[18]:

get_ipython().system(u'pip install numpy')
get_ipython().system(u'pip install root_numpy')


# In[1]:

import random
gpuid = random.randint(0,3)
import os
print "random GPU roll: ",gpuid
os.environ["THEANO_FLAGS"]="device=gpu%i"%gpuid


# In[2]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pandas as pd
from lasagne.layers import InputLayer,DenseLayer,batch_norm,dropout
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
import time
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import roc_curve
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
#import 
#import root_panda


# In[ ]:

#pip install root_numpy


# In[23]:

#!which root


# In[6]:

get_ipython().system(u'cd datasets; wget -O public_train_100000.root -nc --no-check-certificate https://2016.mlhep.yandex.net/data/higgs/public_train_100000.root')


# In[7]:

get_ipython().system(u'cd datasets; wget -O public_test.root -nc --no-check-certificate https://2016.mlhep.yandex.net/data/higgs/public_test.root')


# In[ ]:

#df = pd.DataFrame.from_csv("/homeappl/home/austyuzh/data/train_small.csv",)
#n_features = df.shape[1]-1


# In[3]:

data = pd.DataFrame.from_csv(('/homeappl/home/austyuzh/data/train.csv'))


# In[108]:

test = pd.DataFrame.from_csv(('/homeappl/home/austyuzh/data/public_test.csv'),)


# In[4]:

n_features = data.shape[1]-1
print n_features


# In[5]:

y = data['target'].values==1
X = data[data.columns[1:]].values.astype(theano.config.floatX)


# In[6]:

print y
print X.view
print X.size
data.head()


# In[109]:

test.head()


# In[54]:

hist_params = {'normed': True, 'bins': 60, 'alpha': 0.4}
# create the figure
plt.figure(figsize=(17, 27))
for n, feature in enumerate(data[data.columns[1:]]):
    # add sub plot on our figure
    plt.subplot(n_features // 5 + 11, 3, n+1)
    # define range for histograms by cutting 1% of data from both ends
    min_value, max_value = numpy.percentile(data[feature], [1, 99])
    plt.hist(data.ix[data.target.values == 0, feature].values, range=(min_value, max_value), 
             label='class 0', **hist_params)
    plt.hist(data.ix[data.target.values == 1, feature].values, range=(min_value, max_value), 
             label='class 1', **hist_params)
    plt.legend(loc='best')
    plt.title(feature)


# In[7]:

from sklearn.cross_validation import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=1337)


# In[8]:

input_X = T.matrix('input X')
target_Y = T.ivector('target Y')
print target_Y
print input_X


# In[9]:

l_in = InputLayer([None, n_features],input_X,'input layer')

n_hid = 500

l_1 = DenseLayer((l_in),
                num_units=n_hid,
                name='dense0',
                nonlinearity=lasagne.nonlinearities.tanh) 
l1_penalty = regularize_layer_params(l_1, l2) * 1e-4
l_2 = DenseLayer(batch_norm(l_1),
                num_units=n_hid,
                name='dense1',
                nonlinearity=lasagne.nonlinearities.tanh)
l2_penalty = regularize_layer_params(l_1, l2) * 1e-5
l_3 = DenseLayer(batch_norm(l_2),
                num_units=n_hid,
                name='dense2',
                nonlinearity=lasagne.nonlinearities.tanh)

nn = DenseLayer(batch_norm(l_3),num_units=2,
                name='dense out',
                nonlinearity=lasagne.nonlinearities.softmax,)


#l2_penalty = regularize_layer_params_weighted(nn, l2)
#l1_penalty = regularize_layer_params(l_1, l1) * 1e-4


# In[10]:

weights = lasagne.layers.get_all_params(nn,trainable=True)
weights


# In[11]:

nn_out = lasagne.layers.get_output(nn)
loss = lasagne.objectives.categorical_crossentropy(nn_out, target_Y).mean()+l1_penalty+l2_penalty


# In[12]:

updates =lasagne.updates.adadelta(loss,weights)
train_fun = theano.function([input_X,target_Y],[loss,nn_out[:,1]],updates=updates)


# In[13]:

det_nn_out = lasagne.layers.get_output(nn,deterministic=True) 
det_loss = lasagne.objectives.categorical_crossentropy(det_nn_out,target_Y).mean()
val_fun = theano.function([input_X,target_Y],[det_loss,nn_out[:,1]])
print target_Y
print val_fun.value
print val_fun
print det_loss


# In[14]:

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# In[15]:

train_auc_curve = []
train_acc_curve = []
val_auc_curve = []
val_acc_curve = []


# In[18]:

num_epochs = 20
batch_size=100

import time
from sklearn.metrics import roc_auc_score,accuracy_score

for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    print "here"
    start_time = time.time()
    
    train_err = 0
    Ypred_batches = []
    Ytrue_batches = []
    train_batches = 0
    print "batch iteration train"
    for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
        inputs, targets = batch
        err, y_pred = train_fun(inputs, targets)
        
        Ypred_batches.append(y_pred)
        Ytrue_batches.append(targets)
        
        train_err += err
        train_batches += 1
    
    Ypred_train = np.concatenate(Ypred_batches)
    Ytrue_train = np.concatenate(Ytrue_batches)
    train_acc = accuracy_score(Ytrue_train, Ypred_train>0.5)
    train_auc = roc_auc_score(Ytrue_train, Ypred_train)
    
    train_acc_curve.append(train_acc)
    train_auc_curve.append(train_auc)


    # And a full pass over the validation data:
    val_err = 0
    Ypred_batches = []
    Ytrue_batches = []
    val_batches = 0
    print "batch iteration val"
    for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
        inputs, targets = batch
        err, y_pred = val_fun(inputs, targets)
        Ypred_batches.append(y_pred)
        Ytrue_batches.append(targets)
        
        val_err += err
        val_batches += 1

    Ypred_val = np.concatenate(Ypred_batches)
    Ytrue_val = np.concatenate(Ytrue_batches)
    val_acc = accuracy_score(Ytrue_val, Ypred_val>0.5)
    val_auc = roc_auc_score(Ytrue_val, Ypred_val)
    
    val_acc_curve.append(val_acc)
    val_auc_curve.append(val_auc)


    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  training accuracy:\t\t{:.2f} %".format(
        train_acc * 100))
    print("  training AUCscore:\t\t{:.2f} %".format(
        train_auc * 100))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc  * 100))
    print("  validation AUCscore:\t\t{:.2f} %".format(
        val_auc * 100))


# In[19]:

### Batch normalisation to spread up
plt.plot(val_acc_curve,label='validation accuracy')
plt.plot(val_auc_curve,label='validation auc')
plt.plot(train_acc_curve,label='training accuracy')
plt.plot(train_auc_curve,label='training auc')
plt.xlabel("epochs")
plt.legend(loc='best')
plt.ylim(0.6,0.9);


# In[26]:

from sklearn.metrics import roc_curve
plt.title('Resulting ROC curves')

fpr,tpr,_ = roc_curve(Ytrue_val,Ypred_val)
plt.plot(fpr,tpr,label='validation ROC, auc=%.5f'%(val_auc))
fpr,tpr,_ = roc_curve(Ytrue_train,Ypred_train)
plt.plot(fpr,tpr,label='training ROC, auc=%.5f'%(train_auc))

plt.legend(loc='best')


# In[25]:

#plt.hist(nn.output_shape)



# In[ ]:

# predict test sample
#kaggle_proba = nnpredict_proba(test[high_level_features].astype(np.float64))[:, 1]
#kaggle_ids = test1.event_id


# In[69]:

import cPickle as pickle
def save(nn, filename):
    params = lasagne.layers.get_all_param_values(nn)
    print len(params)
    with open(filename, 'wb') as fout:
        pickle.dump(params, fout, protocol=2)


# In[70]:

save(nn,'baselineNN.csv')


# In[58]:

ls


# In[51]:

y_T = []
X_T = test[test.columns[1:]].values.astype(theano.config.floatX)
type(X_T)


# In[94]:

_, preds = val_fun([data.columns[1:], np.zeros(n_features, dtype=np.int32)
create_solution(data.event_id, preds, 'xs_try.csv')


# In[124]:

predict_fun = theano.function([input_X],nn_out)


# In[121]:

#test.astype(theano.config.floatX)


# In[155]:

kaggle_proba = predict_fun(test.astype(theano.config.floatX))[:,1]


# In[156]:

kaggle_proba


# In[157]:

type(test)
#test[['jet1_pt']]
kaggle_ids = np.arange(1,kaggle_proba.size+1)
print kaggle_ids


# In[161]:

from IPython.display import FileLink
def create_solution(ids, proba, filename='NNSubmission.csv'):
    """saves predictions to file and provides a link for downloading """
    pd.DataFrame({'event_id': ids, 'prediction': proba}).to_csv('{}'.format(filename), index=False)
    return FileLink('{}'.format(filename))

#kaggle_ids = np.range(0,kaggle_proba.size)
create_solution(kaggle_ids, kaggle_proba)


# In[79]:

test_prediction = lasagne.layers.get_output(nn, deterministic=True)
print test_prediction
predict_fn = val_fun(X_T,)
#theano.function([input_X[1:]], T.argmax(test_prediction, axis=1))
#print("Predicted class for first test input: %r" % predict_fn([X_T]))


# In[30]:

from IPython.display import FileLink
def create_solution(ids, proba, filename='baseline.csv'):
    """saves predictions to file and provides a link for downloading """
    pd.DataFrame({'event_id': ids, 'prediction': proba}).to_csv('datasets/{}'.format(filename), index=False)
    return FileLink('datasets/{}'.format(filename))
    
create_solution(y_train, Ypred_val)


# In[ ]:



