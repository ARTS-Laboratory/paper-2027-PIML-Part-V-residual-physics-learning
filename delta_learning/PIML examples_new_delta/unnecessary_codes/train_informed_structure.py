import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv1D, TimeDistributed, GRU, LSTM
from numpy.lib.stride_tricks import sliding_window_view
"""
Informed structure using convolutional and recurrent neural networks.
"""
#%%
"""
The training generator creates a sort-of virtual array so that passing
over the data per epoch his done optimally.

returns batches
"""
class DatasetGenerator(keras.utils.Sequence):
    
    def __init__(self, x, y, kernel_size, batch_size=32, train_len=50, shuffle=True):
        self.x = x
        self.y = y
        self.kernel_size = kernel_size
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
        Y = np.zeros((self.batch_size, self.train_len))
        for k, ind in enumerate(inds):
            i = ind // self.T
            j = ind % self.T
            X[k] = self.x[i,j:j+self.train_len]
            Y[k] = self.y[i,j:j+self.train_len]
        
        return X, Y
    
    def on_epoch_end(self):
        if(self.shuffle):
            np.random.shuffle(self.indices)
#%%
def main():
    #%%
    # load data
    all_data = np.load('./data/with_friction.npz')['data']
    # downsample by a factor of 20 so that sampling rate it 50 S/s
    all_data = all_data[:,:-1:20,:]
    
    t = all_data[0,:,0]
    x = all_data[:,:,1]
    v = all_data[:,:,2]
    a = all_data[:,:,3]
    k = all_data[:,:,4]
    F = all_data[:,:,5]
    
    # the model only uses a and k
    x_train = x[:80]; x_test = x[80:]
    v_train = v[:80]; v_test = v[80:]
    a_train = a[:80]; a_test = a[80:]
    k_train = k[:80]; k_test = k[80:]
    F_train = F[:80]; F_test = F[80:]
    
    # normalize x and k
    a_m = np.mean(a); a_std = np.std(a)
    k_m = np.mean(k); k_std = np.std(k)
    X_train = (a_train - a_m)/a_std
    X_test = (a_test - a_m)/a_std
    Y_train = (k_train - k_m)/k_std
    Y_test = (k_test - k_m)/k_std
    
    # training and model parameters
    train_len = 200
    batch_size = 32
    epochs = 20
    units = [37, 37, 37]
    kernel_size = 12
    
    # define model
    training_generator = DatasetGenerator(X_train, Y_train, kernel_size, train_len=train_len, batch_size=batch_size)
    test_generator = DatasetGenerator(X_test, Y_test, kernel_size, train_len=train_len, batch_size=batch_size)
    
    model = keras.models.Sequential([
        Input([None, 1]),
        Conv1D(units[0], kernel_size, strides=1, padding='causal', activation='sigmoid'),
        Conv1D(units[1], kernel_size, strides=1, padding='causal', activation='sigmoid'),
        GRU(units[2], return_sequences=True),
        TimeDistributed(Dense(1, activation=None))
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
    
    # look at weights and compare with pure nn model
    n_weights = 0
    for layer in model.layers:
        for weights in layer.get_weights():
            n_weights += weights.size
    print('number of model weights:', n_weights)
  
    ############# Commented out per email from Nile #################
    # pure_nn = keras.models.load_model('./model_saves/pure_nn')
    # n_weights_nn = 0
    # for layer in pure_nn.layers:
    #     for weights in layer.get_weights():
    #         n_weights_nn += weights.size
    # print('compared against pure NN:', n_weights_nn)
    
    
    # train the model
    model.fit(
        training_generator,
        batch_size=None,
        epochs=epochs,
        validation_data=test_generator,
    )
    model.save('./model_saves/informed_structure')
    
    # evaluate on testing data
    Y_pred_test = model.predict(X_test)
    k_pred_test = Y_pred_test*k_std + k_m
    np.save('./model_predictions/informed_structure/k_pred.npy', k_pred_test)
    #%%

if __name__ == '__main__':
    main()