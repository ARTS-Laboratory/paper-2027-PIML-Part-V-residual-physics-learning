import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Rescaling, Concatenate
"""
Author: Nile Coble

PINNs as a soft constraint.

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
        self.x = x = test_data[1]
        self.v = v = test_data[2]
        self.a = a = test_data[3]
        j = test_data[4]
        self.k = test_data[5]
        F = test_data[6]
        Fdot = test_data[7]
        Fddot = test_data[8]
        
        self.psi = np.array([x[:-1], v[:-1], F[1:]]).T
        self.psidot = np.array([v[:-1], a[:-1], Fdot[1:]]).T
        self.psiddot = np.array([a[:-1], j[:-1], Fddot[1:]]).T
        self.n_samples = self.psi.shape[0]
        # remove first datapoint for t, a, k
        self.t = self.t[1:]; self.a = self.a[1:]; self.k = self.k[1:]
        self.x = self.x[1:]; self.v = self.v[1:];
        
        self.indices = np.arange(self.n_samples)
        if(self.shuffle):
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.n_samples/self.batch_size))
    
    def __getitem__(self, index):
        inds = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        
        t_rtrn = self.t[inds]
        x_rtrn = self.x[inds]
        v_rtrn = self.v[inds]
        a_rtrn = self.a[inds]
        psi_rtrn = self.psi[inds]
        psidot_rtrn = self.psidot[inds]
        psiddot_rtrn = self.psiddot[inds]
        
        inputs = np.concatenate([t_rtrn.reshape(-1, 1), psi_rtrn], axis=1)
        
        if(self.return_k):
            k_rtrn = self.k[inds]
            return inputs, [psi_rtrn, psidot_rtrn, psiddot_rtrn, x_rtrn, v_rtrn, a_rtrn, k_rtrn]
        return inputs, [psi_rtrn, psidot_rtrn, psiddot_rtrn, x_rtrn, v_rtrn, a_rtrn]
    
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
    
    t = test_data[0]
    x = test_data[1]
    v = test_data[2]
    a = test_data[3]
    k = test_data[5]
    F = test_data[6]
    
    # model parameters
    m = 1.0
    c = .2
    # training parameters
    # rho_k = .01                 # weighting associated with the k prediction
    # rho_x = 100
    # rho_v = 100
    # rho_a = .1                  # weighting associated with the acceleration prediction
    # rho_p = 1                   # weighting associated with physics residual
    
    # training parameters
    rho_k = 1/np.std(k)**2      # weighting associated with the k prediction
    rho_x = 1/np.std(x)**2
    rho_v = 1/np.std(v)**2
    rho_a = 1/np.std(a)**2      # weighting associated with the acceleration prediction
    rho_p = 1                   # weighting associated with physics residual
    
    
    epochs = 100
    batch_size = 32
    
    midpoint = test_data.shape[1]//2
    experimental_generator = DatasetGenerator(test_data[:,:midpoint], batch_size=batch_size, shuffle=True, return_k=True)
    physics_generator = DatasetGenerator(test_data[:,midpoint:], batch_size=batch_size, shuffle=True, return_k=True)
    assert len(experimental_generator) == len(physics_generator)
    n_batches = len(experimental_generator)
    
    # make the model
    inputs = keras.layers.Input([4,])
    sequential_model = keras.models.Sequential([
        Dense(100, activation='sigmoid'),
        Dense(100, activation='sigmoid'),
        Dense(100, activation='sigmoid'),
    ])(inputs)
    k1 = Dense(1, activation=None)(sequential_model)
    k_pred = Rescaling(np.std(test_data[5]), offset=np.mean(test_data[5]))(k1)
    x_pred = Dense(1, activation=None)(sequential_model)
    concat = Concatenate()([x_pred, k_pred])
    
    model = keras.Model(
        inputs = [inputs],
        outputs = concat
    )
    
    opt = keras.optimizers.Adam(
        learning_rate = 0.001,
        beta_1 = 0.9,
        beta_2 = 0.995
    )
    
    model.compile(
        optimizer=opt,
        loss='mse'
    )
    # The PINN methodology is too complicated to use model.fit, so custom
    # training must be used.
    m = tf.constant(m, dtype=tf.float32)
    c = tf.constant(c, dtype=tf.float32)
    rho_k = tf.constant(rho_k, dtype=tf.float32)
    rho_x = tf.constant(rho_x, dtype=tf.float32)
    rho_v = tf.constant(rho_v, dtype=tf.float32)
    rho_a = tf.constant(rho_a, dtype=tf.float32)
    rho_p = tf.constant(rho_p, dtype=tf.float32)
    
    # record error history for each batch
    error_rec = np.zeros((epochs, n_batches, 6))
    
    for epoch in range(epochs):
        running_e_p = 0
        running_e_x = 0
        running_e_v = 0
        running_e_a = 0
        running_e_k = 0
        running_error = 0
        for batch, (exp_data, phys_data) in enumerate(zip(experimental_generator, physics_generator)):
            exp_in = exp_data[0]; phys_in = phys_data[0]
            exp_values = exp_data[1]; phys_values = phys_data[1]
            exp_k = exp_values[-1]
            # concatenate
            all_inputs = np.concatenate([exp_in, phys_in], axis=0)
            psi = np.concatenate([exp_values[0], phys_values[0]], axis=0)
            psidot = np.concatenate([exp_values[1], phys_values[1]], axis=0)
            psiddot = np.concatenate([exp_values[2], phys_values[2]], axis=0)
            x_true = np.concatenate([exp_values[3], phys_values[3]], axis=0)
            v_true = np.concatenate([exp_values[4], phys_values[4]], axis=0)
            a_true = np.concatenate([exp_values[5], phys_values[5]], axis=0)
            # cast to tf tensors
            all_inputs = tf.Variable(all_inputs, dtype=tf.float32)
            psi = tf.constant(psi, dtype=tf.float32)
            psidot = tf.constant(psidot, dtype=tf.float32)
            psiddot = tf.constant(psiddot, dtype=tf.float32)
            x_true = tf.constant(x_true, dtype=tf.float32)
            v_true = tf.constant(v_true, dtype=tf.float32)
            a_true = tf.constant(a_true, dtype=tf.float32)
            exp_k = tf.constant(exp_k, dtype=tf.float32)
            with tf.GradientTape() as tape1: # tape for error w.r.t. weights
                with tf.GradientTape() as tape2: # tape for second derivatives
                    with tf.GradientTape() as tape3: # tape for first derivatives
                        all_out = model(all_inputs)
                        x_pred = all_out[:,0]
                        k_pred = all_out[:,1]
                    d1 = tape3.batch_jacobian(all_out, all_inputs)
                    dx1 = d1[:,0:1] # gradients of just x
                    pax_t = dx1[:,:,0]# partial x w.r.t. t
                    dx_dt = pax_t[:,0] + tf.einsum('...j,...j->...', dx1[:,0,1:], psidot)
                dx2 = tape2.batch_jacobian(dx1, all_inputs)
                dx2 = tf.squeeze(dx2) # remove extraneous dimension
                pa2x_t = dx2[:,0,0] # second partial x w.r.t. t
                # assemble total derivative
                d2x_dt2 = pa2x_t + 2*tf.einsum('...j,...j->...', dx2[:,0,1:], psidot)
                d2x_dt2 += tf.einsum('...j,...j->...', dx1[:,0,1:], psiddot)
                d2x_dt2 += tf.einsum('...ij,...i,...j->...', dx2[:,1:,1:], psidot, psidot)
                F = psi[:,-1]
                e_p = rho_p*tf.reduce_mean(tf.square(F - m*a_true - c*v_true - k_pred*x_true)) # physics residual
                e_x = rho_x*tf.reduce_mean(tf.square(x_pred - x_true))
                e_v = rho_v*tf.reduce_mean(tf.square(dx_dt - v_true))
                e_a = rho_a*tf.reduce_mean(tf.square(d2x_dt2 - a_true))
                e_k = rho_k*tf.reduce_mean(tf.square(k_pred[:k_pred.shape[0]//2] - exp_k))
                error = e_p + e_x + e_v + e_a + e_k
            grads = tape1.gradient(error, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            e_p = float(e_p)
            e_x = float(e_x)
            e_v = float(e_v)
            e_a = float(e_a)
            e_k = float(e_k)
            error = float(error)
            
            error_rec[epoch, batch] = [e_p, e_x, e_v, e_a, e_k, error]
            
            running_e_p = (running_e_p*batch + e_p)/(batch+1)
            running_e_x = (running_e_x*batch + e_x)/(batch+1)
            running_e_v = (running_e_v*batch + e_v)/(batch+1)
            running_e_a = (running_e_a*batch + e_a)/(batch + 1)
            running_e_k = (running_e_k*batch + e_k)/(batch+1)
            running_error = (running_error*batch + error)/(batch + 1)
            percent_complete = batch/n_batches*100
            
            print('\r', 'epoch: %d, %.2f percent complete. error: %.3f (p: %.3f, x: %.3f, v: %.3f, a: %.3f, k: %.3f)'\
                  %(epoch+1, percent_complete, running_error, running_e_p,\
                    running_e_x, running_e_v, running_e_a, running_e_k), end='')
        experimental_generator.on_epoch_end()
        physics_generator.on_epoch_end()
    # save model
    model.save('./model_saves/pinn')
    # save training history
    np.save('./model_predictions/pinn/error_rec.npy', error_rec)
    # run through experiment and save results
    dataset_generator = DatasetGenerator(test_data, batch_size=batch_size, shuffle=False, return_k=True)
    
    pred_out  = model.predict(dataset_generator)
    np.save('./model_predictions/pinn/pred_out.npy', pred_out)
    

if __name__ == '__main__':
    main()