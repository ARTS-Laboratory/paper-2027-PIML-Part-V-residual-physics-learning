import numpy as np
import tensorflow as tf
from tensorflow import keras
from numpy.lib.stride_tricks import sliding_window_view
from tensorflow.keras.layers import Dense, Input, Add
"""
Training a model to compare against the delta learning procedure, trained
only on the first half (60 sec) of the with friction dataset tests.
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
def main():
    #%%
    # load data
    with_friction_data = np.load('./data/no_friction.npz')['data']
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
    a_m = np.mean(a_wf); a_std = np.std(a_wf)
    k_m = np.mean(k_wf); k_std = np.std(k_wf)
    X_train_wf = (a_train_wf - a_m)/a_std
    X_test_wf = (a_test_wf - a_m)/a_std
    Y_train_wf = (k_train_wf - k_m)/k_std
    Y_test_wf = (k_test_wf - k_m)/k_std
    
    # training parameters
    train_len = 50
    batch_size = 32
    epochs = 20
    
    training_generator = DatasetGenerator(X_train_wf, Y_train_wf, train_len=train_len, batch_size=batch_size)
    test_generator = DatasetGenerator(X_test_wf, Y_test_wf, train_len=train_len, batch_size=batch_size)
    
    model = keras.models.Sequential([
        keras.layers.Dense(100, activation='sigmoid', input_shape=[train_len,]),
        keras.layers.Dense(100, activation='sigmoid'),
        keras.layers.Dense(100, activation='sigmoid'),
        keras.layers.Dense(1, activation=None)
    ])
    
    opt = keras.optimizers.Adam(
        learning_rate = 0.001,
        beta_1 = 0.9,
        beta_2 = 0.999
    )
    
    model.compile(
        optimizer=opt,
        loss='mse'
    )
    # train the model
    model.fit(
        training_generator,
        epochs=epochs,
        validation_data=test_generator,
    )
    model.save('./model_saves/delta_control')
    
    # with friction dataset
    k_wf_control = np.zeros((20, k_test_wf.shape[1]-train_len+1))
    for i in range(20):
        # model 1
        k_pred = model.predict(sliding_window_view(X_test_wf[i], [train_len]))
        # undo scaling
        k_pred = k_pred*k_std + k_m
        k_wf_control[i] = k_pred.flatten()
    np.save('./model_predictions/delta_learning/with_friction_control.npy', k_wf_control)
    #%%

if __name__ == '__main__':
    main()