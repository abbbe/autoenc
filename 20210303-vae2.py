#!/usr/bin/env python
# coding: utf-8

# Wednesday 3/03/2021
# 
# * DONE: Use grid L0/L1 space for visualization
# * INPROGRESS: Train and run autoencoder for different number of epochs. Run several times with same settings. Assess impact.
# 
# Next:
# 
# * Explore impact of changing filters chain down to (2,2,2)
# * Scatterplot 3D
# * How does it fluctuate depending on network architecture, nlats, training protocol, etc
# * Explore fold area in L0/L1 space. For each frame show:
#  * image of gym robot,
#  * colored L0/L1 scatterplot, with a red cross showing current latent state,
#  * image reconstructed by the autoencoder

# In[26]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

get_ipython().run_line_magic('load_ext', 'tensorboard')


# ## Implementation of dataset load helpers

# In[27]:


import pickle

def load_dataset(filename):
    with open(filename, 'rb') as handle:
        dataset = pickle.load(handle)
        if isinstance(dataset, dict):
            return dataset
        return dataset[0] # dataset-random-100k.mdict.pickle case


# In[28]:


#dataset = load_dataset('dataset-random-100k.mdict.pickle')
#dataset = load_dataset('dataset-grid-10-1000.pickle')


# ### Implementation of env image vizualization

# In[29]:


def _clean_ax(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    
def _imshow(ax, img):
    if img is not None:
        ax.imshow(img, cmap='Greys', origin='lower')
    _clean_ax(ax)
        
def plot_Y(img_array):
    assert(img_array.shape[-1] == 1)
    fig, axs = plt.subplots(figsize=(2, 2))
    _imshow(axs, img_array[..., 0])
    return fig, axs

def plot_Ys(img_array, title=None, ncols=5):
    assert(ncols > 0)

    nimgs = img_array.shape[0]
    if nimgs == 0:
        return None, None
    elif nimgs == 1:
        return plot_Y(img_array[0,...])
    elif nimgs <= ncols:
        ncols = nimgs
        fig, axs = plt.subplots(1, ncols, figsize=(2*ncols, 2))

        for i in range(ncols):
            _imshow(axs[i], img_array[i,...])
    else:
        nrows = int((nimgs-1)/ ncols) + 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))

        for i in range(nrows):
            for j in range(ncols):
                k = i*ncols + j
                if k < nimgs:
                    img = img_array[k,...]
                else:
                    img = None
                _imshow(axs[i][j], img)
                    
    if title is not None:
        fig.suptitle("%s (%s)" % (title, str(img_array.shape)))
        
    return fig, axs


# In[30]:


# _=plot_Ys(dataset['Y'][0:10,...], title="First 10 elements of dataset['Y']", ncols=5)


# ### Implementation of data (in and lat) visualization helpers

# In[31]:


def fine_scatter(data1, x1, i1, data2, x2, i2):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(data1[x1][:,i1], data2[x2][:,i2], 1)
    ax.set_xlabel("%s%d" % (x1, i1))
    ax.set_ylabel("%s%d" % (x2, i2))
    ax.grid()

def fine_scatter_color(data1, x1, i1, data2, x2, i2, c=None, size=10, ax=None):
    if ax is None:
        _fig, ax = plt.subplots(figsize=(size, size))
        
    ax.set_facecolor('xkcd:black')

    x, y = data1[x1][:,i1], data2[x2][:,i2]
    ax.scatter(x, y, 1, c=c)
    ax.set_xlabel("%s%d" % (x1, i1))
    ax.set_ylabel("%s%d" % (x2, i2))
    #ax.grid()

def fine_scatter_sum(data1, x1, i1a, i1b, data2, x2, i2):
    fig, ax = plt.subplots(figsize=(10, 10))

    x = data1[x1][:,i1a] + data1[x1][:,i1b]
    ax.scatter(x, data2[x2][:,i2], 1)
    
    ax.set_xlabel("%s(%d+%d)" % (x1, i1a, i1b))
    ax.set_ylabel("%s%d" % (x2, i2))
    ax.grid()

def fine_scatter_color_sum(data1, x1, i1a, i1b, data2, x2, i2, c=None, size=10, ax=None):
    if ax is None:
        _fig, ax = plt.subplots(figsize=(size, size))
        
    ax.set_facecolor('xkcd:black')

    x = data1[x1][:,i1a] + data1[x1][:,i1b]
    y = data2[x2][:,i2]
    ax.scatter(x, data2[x2][:,i2], 1, c)
    
    ax.set_xlabel("%s(%d+%d)" % (x1, i1a, i1b))
    ax.set_ylabel("%s%d" % (x2, i2))

def fine_scatter_color_sub(data1, x1, i1a, i1b, data2, x2, i2, c=None, size=10, ax=None):
    if ax is None:
        _fig, ax = plt.subplots(figsize=(size, size))
        
    ax.set_facecolor('xkcd:black')

    x = data1[x1][:,i1a] - data1[x1][:,i1b]
    y = data2[x2][:,i2]
    ax.scatter(x, data2[x2][:,i2], 1, c)
    
    ax.set_xlabel("%s(%d-%d)" % (x1, i1a, i1b))
    ax.set_ylabel("%s%d" % (x2, i2))

def display_xvars(data):
    nLat = data['L'].shape[1]
    nVoltages = data['A'].shape[1]

    fig, axs = plt.subplots(nLat+1, nVoltages+1)
    fig.tight_layout()

    for i in range(nLat):
      for j in range(nVoltages):
        title = "volt%d vs lat%d" % (j, i)
        axs[i][j].title.set_text(title)
        axs[i][j].plot(data['L'][:, i], data['A'][:,j], '.')

    axs[nLat][0].title.set_text("volt0 vs volt1")
    axs[nLat][0].plot(data['A'][:,1], data['A'][:,0], '.')

    axs[0][nVoltages].title.set_text("lat1 vs lat0")
    axs[0][nVoltages].plot(data['L'][:,0], data['L'][:,1], '.')

    _clean_ax(axs[1][2])
    _clean_ax(axs[2][1])
    _clean_ax(axs[2][2])

def cycle_autoencoder0(ae, data, prefix="", N=10):
    out_data = dict()
    
    Y = data['Y']
    
    print("Y.shape=" + str(Y.shape))
    YY = ae['ae'].predict(Y)
    YY = np.array(YY)
    print("YY.shape=" + str(YY.shape))
    
    out_data['YY'] = YY

    # - Y
    plot_Ys(Y[0:N,...], title="First %d elements of %sY" % (N, prefix))
   
    # - YY
    plot_Ys(YY[0:N,...], title="First %d elements of %sYY" % (N, prefix))
    
    return out_data

def cycle_autoencoder(ae, data, prefix="", N=10, vae=False, display=False):
    Y = data['Y']
    
    if display:
        print("Y.shape=" + str(Y.shape))
    L = ae['enc'].predict(Y)
    L = np.array(L)
    if display:
        print("L.shape=" + str(L.shape))

    if vae:
        L = np.array(L)[2] # [z_mean, z_log_var, z] - take Z

    YY = ae['dec'].predict(L)
    YY = np.array(YY)
    if display:
        print("YY.shape=" + str(YY.shape))
    
    data['L'] = L # FIXME
    data['YY'] = YY # FIXME

    out_data = dict()
    out_data['L'] = L
    out_data['YY'] = YY

    # - Y
    if display:
        plot_Ys(Y[0:N,...], title="First %d elements of %sY" % (N, prefix))

        # - L
        fig, axs = plt.subplots(1, L.shape[1])
        fig.suptitle("Latent variables %s" % (str(L.shape)))
        for i in range(L.shape[1]):
            axs[i].hist(L[:,i])    

        # - YY
        plot_Ys(YY[0:N,...], title="First %d elements of %sYY" % (N, prefix))

        display_xvars(data)
    
    return out_data    


# ## Model

# ### Implementations of model save and load methods

# In[32]:


def save_models(model, prefix):
    model['ae'].save(prefix + '-ae.h5')
    model['enc'].save(prefix + '-enc.h5')
    model['dec'].save(prefix + '-dec.h5')
    print("Models saved to " + prefix + " ...")

def load_models(prefix):
    model = dict()
    for k in ['ae', 'enc', 'dec']:
        fname = prefix + '-' + k + '.h5'
        model[k] = keras.models.load_model(fname)
    return model

# VAE models cannot be saved by the above methods, can only save weights

def save_models_weights(model, prefix):
    model['ae'].save_weights(prefix + '-ae.h5w')
    model['enc'].save_weights(prefix + '-enc.h5w')
    model['dec'].save_weights(prefix + '-dec.h5w')
    print("Models weights saved to " + prefix + " ...")
    
def load_models_weights(model, prefix):
    model['ae'].load_weights(prefix + '-ae.h5w')
    model['enc'].load_weights(prefix + '-enc.h5w')
    model['dec'].load_weights(prefix + '-dec.h5w')


# ### Visualize L0/L1 space

# In[33]:


def visualize_lat_space(dataset_grid, dataset_grid_out, sheet):
    fig, axs = plt.subplots(2, 2, figsize=(15,12))
    fig.tight_layout()

    if sheet == 1:
        fine_scatter_color(dataset_grid, 'A', 0, dataset_grid_out, 'L' , 0, dataset_grid['A'][:,1], ax=axs[0,0])
        fine_scatter_color(dataset_grid, 'A', 1, dataset_grid_out, 'L' , 1, dataset_grid['A'][:,0], ax=axs[0,1])

        fine_scatter_color(dataset_grid, 'A', 0, dataset_grid_out, 'L' , 1, dataset_grid['A'][:,1], ax=axs[1,0])
        fine_scatter_color(dataset_grid, 'A', 1, dataset_grid_out, 'L' , 0, dataset_grid['A'][:,0], ax=axs[1,1])

    elif sheet == 2:
        fine_scatter_color_sum(dataset_grid, 'A', 0, 1, dataset_grid_out, 'L', 0, dataset_grid_out['L'][:,1], ax=axs[0,0])
        fine_scatter_color_sum(dataset_grid, 'A', 0, 1, dataset_grid_out, 'L', 1, dataset_grid_out['L'][:,0], ax=axs[0,1])

        fine_scatter_color_sub(dataset_grid, 'A', 0, 1, dataset_grid_out, 'L', 0, dataset_grid_out['L'][:,1], ax=axs[1,0])
        fine_scatter_color_sub(dataset_grid, 'A', 0, 1, dataset_grid_out, 'L', 1, dataset_grid_out['L'][:,0], ax=axs[1,1])
    else:
        raise
    
    return fig, axs


# ### Build and train VAE

# In[34]:


from models.vae import build_autoencoder
#autoencoder = build_autoencoder((64, 64, 1), 2)


# In[35]:


#load_models_weights(autoencoder, "20210302-vae") # NB: build the model first


# ### Run

# In[36]:


from models.vae import build_autoencoder

dataset = load_dataset('datasets/dataset-random-100k.mdict.pickle')
dataset_grid = load_dataset('datasets/dataset-grid-10-1000.pickle')

val_dataset = load_dataset('datasets/dataset-random-10k.mdict.pickle')
val_dataset = {'A': val_dataset['A'][:500], 'Y': val_dataset['Y'][:500]} # take a piece


# In[ ]:





# In[38]:


import os, datetime

BASE_DIR = "20210303-vae2/rand_100k-grid_10_1000"
TENSORBOARD_LOGS_DIR =  "%s/tensorboard-logs" % BASE_DIR
TRAINED_MODELS_DIR = "%s/trained-models" % BASE_DIR
IMGS_DIR = "%s/imgs" % BASE_DIR

for dir in [BASE_DIR, TENSORBOARD_LOGS_DIR, TRAINED_MODELS_DIR, IMGS_DIR]:
    if not os.path.isdir(dir):
        os.mkdir(dir)

def train_model(model, dataset, val_dataset, build_params, suffix):
    ae = model['ae']
    
    logdir = os.path.join(("%s/%s" % (TENSORBOARD_LOGS_DIR, suffix)), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir) #, histogram_freq=1)
    
    # FIXME ae.fit(dataset['Y'], validation_data=(val_dataset['Y'], val_dataset['Y']), callbacks=[tensorboard_callback],
    ae.fit(dataset['Y'], callbacks=[tensorboard_callback],
        epochs=build_params['epochs'], batch_size=128, verbose=0)
    
    save_models_weights(model, "%s/%s" % (TRAINED_MODELS_DIR, suffix))
    
def run(dataset, val_dataset, build_params, suffix):
    print("*** run(%s, %s)" % (str(build_params), suffix))
    
    vis_fname = "%s/%s.png" % (IMGS_DIR, suffix)
    vis_fname2 = "%s/%s-2.png" % (IMGS_DIR, suffix)
    
    if os.path.isfile(vis_fname):
        print("%s exists, skipping ..." % vis_fname)
        return
    
    print("building and training vae, build_params=%s" % str(build_params))
    keras.backend.clear_session()
    autoencoder = build_autoencoder((64, 64, 1), 2, build_params)
    train_model(autoencoder, dataset, val_dataset, build_params, suffix)

    print("running vae, saving visualization into %s" % (vis_fname))
    dataset_grid_out = cycle_autoencoder(autoencoder, dataset_grid, vae=True, display=False)
    
    fig, axs = visualize_lat_space(dataset_grid, dataset_grid_out, sheet=1)
    fig.savefig(vis_fname)
    fig.clf()
    
    fig, axs = visualize_lat_space(dataset_grid, dataset_grid_out, sheet=2)
    fig.savefig(vis_fname2)
    fig.clf()


# In[39]:


#run({'epochs': 5, 'convs': [16, 8, 4, 2]}, "test-conv_16_8_4-epochs_%d-%d" % (epochs, i))


# In[ ]:


for epochs in [1, 5, 10, 25, 50]:
    for i in range(5):
        suffix = "convs_16_8_8-epochs_%d-%d" % (epochs, i)
        run(dataset, val_dataset, {'epochs': epochs, 'convs': [16, 8, 8]}, suffix)


# In[ ]:


for epochs in [100]:
    for i in range(5):
        suffix = "convs_16_8_8-epochs_%d-%d" % (epochs, i)
        run(dataset, val_dataset, {'epochs': epochs, 'convs': [16, 8, 8]}, suffix)
        
for epochs in [5, 25, 50, 100]:
    for i in range(5):
        suffix = "convs_16_8_4-epochs_%d-%d" % (epochs, i)
        run(dataset, val_dataset, {'epochs': epochs, 'convs': [16, 8, 4]}, suffix)


# In[ ]:




