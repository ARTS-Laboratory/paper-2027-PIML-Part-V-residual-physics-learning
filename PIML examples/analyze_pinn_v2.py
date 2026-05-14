import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
plt.close('all')


################## fixed console crashing #######################
# The console crahsed when loadking model
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#################################################################



"""
Author: Nile Coble

look closer at the results of PINN training

"""

#%% 
pinn_pred = np.load('./model_predictions/pinn/pred_out.npy')
control_pred = np.load('./model_predictions/pinn/control_out.npy').flatten()
test_data = np.load('./data/pinn_test_0.npy').T
# downsample by a factor of 20 so that sampling rate it 50 S/s
test_data = test_data[:,::20]

t = test_data[0]
k = test_data[5]
k_pinn = pinn_pred[:,1]
k_control = control_pred

plt.figure(figsize=(6,2.3))
plt.plot(t[1:], k_control, c='tab:green', label='control')
plt.plot(t[1:], k_pinn, c='tab:orange', label='PINN')
plt.plot(t, k, c='tab:blue', label='true')
plt.xlim((0, 120))
plt.ylim((300, 1550))
plt.xlabel('time (s)')
plt.ylabel('stiffness (N/m)')
plt.legend()
plt.tight_layout()


#%% plot pinn prediction of x
pinn_pred = np.load('./model_predictions/pinn/pred_out.npy')
control_pred = np.load('./model_predictions/pinn/control_out.npy').flatten()
test_data = np.load('./data/pinn_test_0.npy').T
# downsample by a factor of 20 so that sampling rate it 50 S/s
test_data = test_data[:,::20]

t = test_data[0]
x = test_data[1]
k = test_data[5]
k_pinn = pinn_pred[:,1]
x_pred = pinn_pred[:,0]
k_control = control_pred

plt.figure(figsize=(5,2.2))
plt.plot(t[1:], x_pred, c='tab:orange', label='PINN')
plt.plot(t, x, c='tab:blue', label='true')
# plt.xlim((40, 45))
plt.xlabel('time (s)')
plt.ylabel('stiffness (N/m)')
plt.legend()
plt.tight_layout()
plt.savefig('./plots/pinn_x_pred.png', dpi=300)

#%% error history of v3
error_rec = np.load('./model_predictions/pinn/error_rec.npy')
e_epoch = np.mean(error_rec, axis=1)

plt.figure()
plt.plot(e_epoch[:,-1])

#%% plot test data for physical consistency
# test_data = np.load('./data/pinn_data/test_0.npy').T # Line changed to that below by Austin Downey
test_data = np.load('./data/pinn_test_0.npy').T
x = test_data[1]
v = test_data[2]
a = test_data[3]
k = test_data[5]
F = test_data[6]
m = 1
c = 0.2

i = 1
# x = x[:-i]
x = x[i:]
v = v[i:]
a = a[i:]
k = k[i:]
# F = F[:-i]
F = F[i:]

residual = F - m*a - c*v - k*x

plt.figure()
plt.plot(residual, label='residual')
# plt.plot(F, label='force')
# plt.plot(m*a, label='inertial force')
# plt.plot(c*v, label='damping')
# plt.plot(k*x, label='spring force')
plt.legend()


#%% test out model
class DatasetGenerator(keras.utils.Sequence):
    
    def __init__(self, test_data, batch_size=32, shuffle=True, return_k = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_k = return_k
        
        self.t = test_data[0]
        x = test_data[1]
        v = test_data[2]
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
        self.indices = np.arange(self.n_samples)
        if(self.shuffle):
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.n_samples/self.batch_size))
    
    def __getitem__(self, index):
        inds = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        
        t_rtrn = self.t[inds]
        a_rtrn = self.a[inds]

        psi_rtrn = self.psi[inds]
        psidot_rtrn = self.psidot[inds]
        psiddot_rtrn = self.psiddot[inds]
        
        inputs = np.concatenate([t_rtrn.reshape(-1, 1), psi_rtrn], axis=1)
        
        if(self.return_k):
            k_rtrn = self.k[inds]
            return inputs, [psi_rtrn, psidot_rtrn, psiddot_rtrn, a_rtrn, k_rtrn]
        return inputs, [psi_rtrn, psidot_rtrn, psiddot_rtrn, a_rtrn]
    
    def on_epoch_end(self):
        if(self.shuffle):
            np.random.shuffle(self.indices) # in place

#%% Set up loop

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
rho_k = .1 # weighting associated with the k prediction
rho_a = 1 # weighting associated with the acceleration prediction
rho_p = 10 # weighting associated with physics residual
batch_size = 32

dataset_generator = DatasetGenerator(test_data, batch_size=batch_size, shuffle=False, return_k=True)

#%% Kills the console unless os.environ['KMP_DUPLICATE_LIB_OK']='True'
# this may be an issue with my install. Also, ths line may cause bad results to be returned.  

model = keras.models.load_model('./model_saves/pinn')


#%% 

a_tot = np.zeros((len(dataset_generator), batch_size))
v_tot = np.zeros((len(dataset_generator), batch_size))
x_tot = np.zeros((len(dataset_generator), batch_size))

#%% Run loop
for batch, data in enumerate(dataset_generator):
    inputs = tf.Variable(data[0])
    psi = tf.constant(data[1][0])
    psidot = tf.constant(data[1][1])
    psiddot = tf.constant(data[1][2])
    
    with tf.GradientTape() as tape2: # tape for second derivatives
        with tf.GradientTape() as tape3: # tape for first derivatives
            all_out = model(inputs)
            x_pred = all_out[:,0]
            k_pred = all_out[:,1]
        d1 = tape3.batch_jacobian(all_out, inputs)
        dx1 = d1[:,0:1] # gradients of just x
        pax_t = dx1[:,:,0] # partial x w.r.t. t
        dx_dt = pax_t[:,0] + tf.einsum('...j,...j->...', dx1[:,0,1:], psidot)
    dx2 = tape2.batch_jacobian(dx1, inputs)
    dx2 = tf.squeeze(dx2) # remove extraneous dimension
    pa2x_t = dx2[:,0,0] # second partial x w.r.t. t
    # assemble total derivative
    d2x_dt2 = pa2x_t + 2*tf.einsum('...j,...j->...', dx2[:,0,1:], psidot)
    d2x_dt2 += tf.einsum('...j,...j->...', dx1[:,0,1:], psiddot)
    d2x_dt2 += tf.einsum('...ij,...i,...j->...', dx2[:,1:,1:], psidot, psidot)
    F = psi[:,-1]
    
    a_batch = d2x_dt2.numpy()
    v_batch = dx_dt.numpy()
    x_batch = x_pred.numpy()
    
    # pad batch to always contain batch_size elements (should only matter for last batch)
    a_batch = np.pad(a_batch, (0, batch_size - a_batch.size))
    v_batch = np.pad(v_batch, (0, batch_size - v_batch.size))
    x_batch = np.pad(x_batch, (0, batch_size - x_batch.size))
    
    
    a_tot[batch] = a_batch
    v_tot[batch] = v_batch
    x_tot[batch] = x_batch
    print('finished batch', batch)

#%% 

a_pred = a_tot.flatten()[:t.size-1]
v_pred = v_tot.flatten()[:t.size-1]
x_pred = x_tot.flatten()[:t.size-1]

#%% plot time vs acceleration
plt.figure()
plt.title('time vs acceleration')
plt.xlabel('time (s)')
plt.ylabel(r'acceleration (m/s$^2%?)')
plt.plot(t, a, label='true acc.')
plt.plot(t[1:], a_pred, marker='.', linewidth=0, label='pred. acc.')
plt.legend()
plt.tight_layout()

#%% plot time vs velocity
plt.figure()
plt.title('time vs velocity')
plt.xlabel('time (s)')
plt.ylabel('velocity (m/s?)')
plt.plot(t, v, label='true v')
plt.plot(t[1:], v_pred, marker='.', linewidth=0, label='pred. v')
plt.legend()
plt.tight_layout()

#%% plot time vs displacement
plt.figure()
plt.title('time vs displacement')
plt.xlabel('time (s)')
plt.ylabel('displacement (m?)')
plt.plot(t, x, label='true x')
plt.plot(t[1:], x_pred, marker='.', linewidth=0, label='pred. x')
plt.legend()
plt.tight_layout()
