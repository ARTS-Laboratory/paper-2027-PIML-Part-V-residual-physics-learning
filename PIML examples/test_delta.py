import numpy as np
import tensorflow as tf
from tensorflow import keras
from numpy.lib.stride_tricks import sliding_window_view
from tensorflow.keras.layers import Dense, Input, Add
import matplotlib.pyplot as plt
"""
testing delta learning for my sanity
"""
#%%
"""
The training generator creates a sort-of virtual array so that passing
over the data per epoch his done optimally.

returns batches
"""
class DatasetGenerator(keras.utils.Sequence):
    
    def __init__(self, x, y, batch_size=32, train_len=50, shuffle=True):
        self.x = x
        self.y = y
        self.train_len= train_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.N = self.x.shape[0]
        self.T = self.x.shape[1] - self.train_len
        self.n_samples = self.N*self.T
        
        self.indices = np.arange(self.n_samples)
        if(self.shuffle):
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.n_samples/self.batch_size))
    
    def __getitem__(self, index):
        inds = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X = np.zeros((self.batch_size, self.train_len))
        Y = np.zeros(self.batch_size)
        for k, ind in enumerate(inds):
            i = ind // self.T
            j = ind % self.T
            X[k] = self.x[i,j:j+self.train_len]
            Y[k] = self.y[i,j+self.train_len]
        
        return X, Y
    
    def on_epoch_end(self):
        if(self.shuffle):
            np.random.shuffle(self.indices)
#%%
# training parameters
train_len = 50
batch_size = 32
epochs = 20
# load data
no_friction_data = np.load('./data/v4/no_friction.npy')
# downsample by a factor of 20 so that sampling rate it 50 S/s
no_friction_data = no_friction_data[:,:-1:20,:]

t_nf = no_friction_data[0,:,0]
x_nf = no_friction_data[:,:,1]
v_nf = no_friction_data[:,:,2]
a_nf = no_friction_data[:,:,3]
k_nf = no_friction_data[:,:,4]
F_nf = no_friction_data[:,:,5]

# the model only uses a and k
x_train_nf = x_nf[:80]; x_test_nf = x_nf[80:]
v_train_nf = v_nf[:80]; v_test_nf = v_nf[80:]
a_train_nf = a_nf[:80]; a_test_nf = a_nf[80:]
k_train_nf = k_nf[:80]; k_test_nf = k_nf[80:]
F_train_nf = F_nf[:80]; F_test_nf = F_nf[80:]

# normalize x and k
a_m = np.mean(a_nf); a_std = np.std(a_nf)
k_m = np.mean(k_nf); k_std = np.std(k_nf)
X_train_nf = (a_train_nf - a_m)/a_std
X_test_nf = (a_test_nf - a_m)/a_std
Y_train_nf = (k_train_nf - k_m)/k_std
Y_test_nf = (k_test_nf - k_m)/k_std
# load data
with_friction_data = np.load('./data/v4/with_friction.npy')
# downsample by a factor of 20 so that sampling rate it 50 S/s
with_friction_data = with_friction_data[:,:-1:20,:]

t_wf = with_friction_data[0,:,0]
x_wf = with_friction_data[:,:,1]
v_wf = with_friction_data[:,:,2]
a_wf = with_friction_data[:,:,3]
k_wf = with_friction_data[:,:,4]
F_wf = with_friction_data[:,:,5]

# the model only uses a and k
# training data: remove second half
midpoint = a_wf.shape[1]//2
x_train_wf = x_wf[:80,:midpoint]; x_test_wf = x_wf[80:]
v_train_wf = v_wf[:80,:midpoint]; v_test_wf = v_wf[80:]
a_train_wf = a_wf[:80,:midpoint]; a_test_wf = a_wf[80:]
k_train_wf = k_wf[:80,:midpoint]; k_test_wf = k_wf[80:]
F_train_wf = F_wf[:80,:midpoint]; F_test_wf = F_wf[80:]

# normalize x and k
X_train_wf = (a_train_wf - a_m)/a_std
X_test_wf = (a_test_wf - a_m)/a_std
Y_train_wf = (k_train_wf - k_m)/k_std
Y_test_wf = (k_test_wf - k_m)/k_std

# load model 1
model_1 = keras.models.load_model('./model_saves/delta_model_1')

# evaluate models on validation datasets
# no friction dataset
k_nf_model_1 = np.zeros((20, k_test_nf.shape[1]-train_len+1))
for i in range(20):
    # model 1
    k_pred = model_1.predict(sliding_window_view(X_test_nf[i], [train_len]))
    # undo scaling
    k_pred = k_pred*k_std + k_m
    k_nf_model_1[i] = k_pred.flatten()
# with friction dataset
k_wf_model_1 = np.zeros((20, k_test_wf.shape[1]-train_len+1))
for i in range(20):
    # model 1
    k_pred = model_1.predict(sliding_window_view(X_test_wf[i], [train_len]))
    # undo scaling
    k_pred = k_pred*k_std + k_m
    k_wf_model_1[i] = k_pred.flatten()

#%%
plt.figure()
for i in range(20):
    plt.plot(t_nf[-k_nf_model_1.shape[1]:], k_nf_model_1[i], c='tab:blue')
    plt.plot(t_wf[-k_wf_model_1.shape[1]:], k_wf_model_1[i], c='tab:orange')
#%%
plt.figure()
plt.xlim((112,120))
plt.ylim((-7,7))
for i in range(20):
    if(i == 4):
        plt.plot(t_nf, a_nf[i], c='tab:blue')
    plt.plot(t_wf, a_wf[i], c='tab:orange')
#%%
j = 5800
X_nf_sample = X_test_nf[4:5,j:j+train_len]
X_wf_sample = X_test_wf[0:1,j:j+train_len]

plt.figure()
plt.plot(X_nf_sample.flatten())
plt.plot(X_wf_sample.flatten())

t_mesh = np.linspace(0, 1, 200).reshape(-1, 1)

X_mesh = (1-t_mesh)*X_nf_sample + t_mesh*X_wf_sample

k_mesh_pred = model_1.predict(X_mesh)
k_mesh_pred = k_mesh_pred*k_std + k_m

plt.figure()
plt.plot(k_mesh_pred)

