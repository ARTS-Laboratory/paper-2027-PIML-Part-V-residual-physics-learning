import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Rescaling, Concatenate
"""
To show the effectiveness of the PINN, training an identical model without
acceleration or physics losses, only training on k data in the first half of
the experiment

The training generator creates a sort-of virtual array so that passing
over the data per epoch his done optimally.

returns batches


"""
class DatasetGenerator(keras.utils.Sequence):
    
    def __init__(self, test_data, batch_size=32, shuffle=True, return_k = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_k = return_k
        self.t = test_data[0]
        x = test_data[1]
        v = test_data[2]
        self.k = test_data[5]
        F = test_data[6]
        
        self.psi = np.array([x[:-1], v[:-1], F[1:]]).T
        self.n_samples = self.psi.shape[0]
        # remove first datapoint for t, k
        self.t = self.t[1:]; self.k = self.k[1:]
        self.indices = np.arange(self.n_samples)
        if(self.shuffle):
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.n_samples/self.batch_size))
    
    def __getitem__(self, index):
        inds = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        t_rtrn = self.t[inds]
        psi_rtrn = self.psi[inds]
        inputs = np.concatenate([t_rtrn.reshape(-1, 1), psi_rtrn], axis=1)
        
        k_rtrn = self.k[inds]
        return inputs, k_rtrn
    
    def on_epoch_end(self):
        if(self.shuffle):
            np.random.shuffle(self.indices) # in place
#%%
def main():
    #%%
    # load data. As PINNs are an interpolation method, only one test can be used
    # per model.
    test_data = np.load('./data/pinn_test_0.npy').T
    # downsample by a factor of 20 so that sampling rate it 50 S/s
    test_data = test_data[:,::20]
    
    # training parameters
    epochs = 100
    batch_size = 32
    
    midpoint = test_data.shape[1]//2
    dataset_generator = DatasetGenerator(test_data[:,:midpoint], batch_size=batch_size, shuffle=True, return_k=True)
    
    # make the model
    inputs = keras.layers.Input([4,])
    sequential_model = keras.models.Sequential([
        Dense(100, activation='sigmoid'),
        Dense(100, activation='sigmoid'),
        Dense(100, activation='sigmoid'),
    ])(inputs)
    k1 = Dense(1, activation=None)(sequential_model)
    k_pred = Rescaling(np.std(test_data[5]), offset=np.mean(test_data[5]))(k1)
    
    model = keras.Model(
        inputs = [inputs],
        outputs = k_pred
    )
    
    opt = keras.optimizers.Adam(
        learning_rate = 0.001,
        beta_1 = 0.9,
        beta_2 = 0.999
    )
    
    model.compile(
        optimizer=opt,
        loss='mse'
    )
    
    model.fit(
        dataset_generator,
        epochs=epochs
    )
    # save model
    model.save('./model_saves/pinn_control_test')
    # run through experiment and save results
    dataset_generator = DatasetGenerator(test_data, batch_size=batch_size, shuffle=False, return_k=True)
    
    pred_out  = model.predict(dataset_generator)
    np.save('./model_predictions/pinn/control_out.npy', pred_out)
    #%%

if __name__ == '__main__':
    main()