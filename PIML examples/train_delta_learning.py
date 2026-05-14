import numpy as np
import tensorflow as tf
from tensorflow import keras
from numpy.lib.stride_tricks import sliding_window_view
from tensorflow.keras.layers import Dense, Input, Add
"""
Delta learning approach

Step 1: Train model_1 on no friction test for 0-120 s.
Step 2: Create model_2 to predict difference between no friction and with
friction tests. Train this model on 0-60 s of friction tests.
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
    # Step 1. This training procedure is basically the same as pure_nn.
    # load data
    no_friction_data = np.load('./data/no_friction.npz')['data']
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
    
    # training parameters
    train_len = 50
    batch_size = 32
    epochs = 20
    
    training_generator = DatasetGenerator(X_train_nf, Y_train_nf, train_len=train_len, batch_size=batch_size)
    test_generator = DatasetGenerator(X_test_nf, Y_test_nf, train_len=train_len, batch_size=batch_size)
    # make model_1
    model_1 = keras.models.Sequential([
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
    
    model_1.compile(
        optimizer=opt,
        loss='mse'
    )
    # train the model
    model_1.fit(
        training_generator,
        epochs=epochs,
        validation_data=test_generator,
    )
    model_1.save('./model_saves/delta_model_1')
    
    # Step 2
    # load data
    with_friction_data = np.load('./data/with_friction.npz')['data']
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
    
    # training parameters
    train_len = 50
    batch_size = 32
    epochs = 20
    
    training_generator = DatasetGenerator(X_train_wf, Y_train_wf, train_len=train_len, batch_size=batch_size)
    test_generator = DatasetGenerator(X_test_wf, Y_test_wf, train_len=train_len, batch_size=batch_size)
    
    # A more economical approach would be to do a full prediction with model_1
    # first then train model_2 on the difference. The implemented model contains
    # model_1 and model_2 to align with inference.
    model_1.trainable = False # no longer train model_1
    input_layer = Input([train_len,])
    model_2 = keras.models.Sequential([
        Dense(100, activation='sigmoid'),
        Dense(100, activation='sigmoid'),
        Dense(100, activation='sigmoid'),
        Dense(1, activation=None)
    ])
    
    model_1_tensor = model_1(input_layer)
    model_2_tensor = model_2(input_layer)
    add_model = Add()([model_1_tensor, model_2_tensor])
    
    combined_model = keras.models.Model(
        inputs = input_layer,
        outputs= add_model
    )
    
    opt = keras.optimizers.Adam(
        learning_rate = 0.001,
        beta_1 = 0.9,
        beta_2 = 0.999
    )
    
    combined_model.compile(
        optimizer=opt,
        loss='mse'
    )
    # train the model
    combined_model.fit(
        training_generator,
        batch_size=None,
        epochs=epochs,
        validation_data=test_generator,
    )
    combined_model.save('./model_saves/delta_combined_model')
    
    # evaluate models on validation datasets
    # no friction dataset
    k_nf_model_1 = np.zeros((20, k_test_nf.shape[1]-train_len+1))
    k_nf_combined_model = np.zeros((20, k_test_nf.shape[1]-train_len+1))
    for i in range(20):
        # model 1
        k_pred = model_1.predict(sliding_window_view(X_test_nf[i], [train_len]))
        # undo scaling
        k_pred = k_pred*k_std + k_m
        k_nf_model_1[i] = k_pred.flatten()
        # combined model
        k_pred = combined_model.predict(sliding_window_view(X_test_nf[i], [train_len]))
        k_pred = k_pred*k_std + k_m
        k_nf_combined_model[i] = k_pred.flatten()
    np.save('./model_predictions/delta_learning/no_friction_model_1.npy', k_nf_model_1)
    np.save('./model_predictions/delta_learning/no_friction_combined_model.npy', k_nf_combined_model)
    # with friction dataset
    k_wf_model_1 = np.zeros((20, k_test_wf.shape[1]-train_len+1))
    k_wf_combined_model = np.zeros((20, k_test_wf.shape[1]-train_len+1))
    for i in range(20):
        # model 1
        k_pred = model_1.predict(sliding_window_view(X_test_wf[i], [train_len]))
        # undo scaling
        k_pred = k_pred*k_std + k_m
        k_wf_model_1[i] = k_pred.flatten()
        # combined model
        k_pred = combined_model.predict(sliding_window_view(X_test_wf[i], [train_len]))
        k_pred = k_pred*k_std + k_m
        k_wf_combined_model[i] = k_pred.flatten()
    np.save('./model_predictions/delta_learning/with_friction_model_1.npy', k_wf_model_1)
    np.save('./model_predictions/delta_learning/with_friction_combined_model.npy', k_wf_combined_model)
    #%%

if __name__ == '__main__':
    main()