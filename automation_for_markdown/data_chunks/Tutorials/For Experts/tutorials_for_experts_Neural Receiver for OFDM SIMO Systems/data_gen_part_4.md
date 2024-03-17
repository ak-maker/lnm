# Neural Receiver for OFDM SIMO Systems<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Neural_Receiver.html#Neural-Receiver-for-OFDM-SIMO-Systems" title="Permalink to this headline"></a>
    
In this notebook, you will learn how to train a neural receiver that implements OFDM detection. The considered setup is shown in the figure below. As one can see, the neural receiver substitutes channel estimation, equalization, and demapping. It takes as input the post-DFT (discrete Fourier transform) received samples, which form the received resource grid, and computes log-likelihood ratios (LLRs) on the transmitted coded bits. These LLRs are then fed to the outer decoder to reconstruct the
transmitted information bits.
    
    
Two baselines are considered for benchmarking, which are shown in the figure above. Both baselines use linear minimum mean square error (LMMSE) equalization and demapping assuming additive white Gaussian noise (AWGN). They differ by how channel estimation is performed:
 
- **Pefect CSI**: Perfect channel state information (CSI) knowledge is assumed.
- **LS estimation**: Uses the transmitted pilots to perform least squares (LS) estimation of the channel with nearest-neighbor interpolation.

    
All the considered end-to-end systems use an LDPC outer code from the 5G NR specification, QPSK modulation, and a 3GPP CDL channel model simulated in the frequency domain.
# Table of Content
## GPU Configuration and Imports
## Training the Neural Receiver
## Evaluation of the Neural Receiver
## Pre-computed Results
## References
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Neural_Receiver.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

```python
[1]:
```

```python
import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
```
```python
[2]:
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu
from sionna.channel.tr38901 import Antenna, AntennaArray, CDL
from sionna.channel import OFDMChannel
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber
from sionna.utils import sim_ber
```

## Training the Neural Receiver<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Neural_Receiver.html#Training-the-Neural-Receiver" title="Permalink to this headline"></a>
    
In the next cell, one forward pass is performed within a <em>gradient tape</em>, which enables the computation of gradient and therefore the optimization of the neural network through stochastic gradient descent (SGD).
    
**Note:** For an introduction to the implementation of differentiable communication systems and their optimization through SGD and backpropagation with Sionna, please refer to <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part2.html">the Part 2 of the Sionna tutorial for Beginners</a>.

```python
[15]:
```

```python
# The end-to-end system equipped with the neural receiver is instantiated for training.
# When called, it therefore returns the estimated BMD rate
model = E2ESystem('neural-receiver', training=True)
# Sampling a batch of SNRs
ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
# Forward pass
with tf.GradientTape() as tape:
    rate = model(training_batch_size, ebno_db)
    # Tensorflow optimizers only know how to minimize loss function.
    # Therefore, a loss function is defined as the additive inverse of the BMD rate
    loss = -rate
```

    
Next, one can perform one step of stochastic gradient descent (SGD). The Adam optimizer is used

```python
[16]:
```

```python
optimizer = tf.keras.optimizers.Adam()
# Computing and applying gradients
weights = model.trainable_weights
grads = tape.gradient(loss, weights)
optimizer.apply_gradients(zip(grads, weights))
```
```python
[16]:
```
```python
<tf.Variable 'UnreadVariable' shape=() dtype=int64, numpy=1>
```

    
Training consists in looping over SGD steps. The next cell implements a training loop.
    
At each iteration: - A batch of SNRs $E_b/N_0$ is sampled - A forward pass through the end-to-end system is performed within a gradient tape - The gradients are computed using the gradient tape, and applied using the Adam optimizer - The achieved BMD rate is periodically shown
    
After training, the weights of the models are saved in a file
    
**Note:** Training can take a while. Therefore, <a class="reference external" href="https://drive.google.com/file/d/1W9WkWhup6H_vXx0-CojJHJatuPmHJNRF/view?usp=sharing">we have made pre-trained weights available</a>. Do not execute the next cell if you don’t want to train the model from scratch.

```python
[ ]:
```

```python
model = E2ESystem('neural-receiver', training=True)
optimizer = tf.keras.optimizers.Adam()
for i in range(num_training_iterations):
    # Sampling a batch of SNRs
    ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
    # Forward pass
    with tf.GradientTape() as tape:
        rate = model(training_batch_size, ebno_db)
        # Tensorflow optimizers only know how to minimize loss function.
        # Therefore, a loss function is defined as the additive inverse of the BMD rate
        loss = -rate
    # Computing and applying gradients
    weights = model.trainable_weights
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))
    # Periodically printing the progress
    if i % 100 == 0:
        print('Iteration {}/{}  Rate: {:.4f} bit'.format(i, num_training_iterations, rate.numpy()), end='\r')
# Save the weights in a file
weights = model.get_weights()
with open(model_weights_path, 'wb') as f:
    pickle.dump(weights, f)
```

## Evaluation of the Neural Receiver<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Neural_Receiver.html#Evaluation-of-the-Neural-Receiver" title="Permalink to this headline"></a>
    
The next cell evaluates the neural receiver.
    
**Note:** Evaluation of the system can take a while and requires having the trained weights of the neural receiver. Therefore, we provide pre-computed results at the end of this notebook.

```python
[17]:
```

```python
model = E2ESystem('neural-receiver')
# Run one inference to build the layers and loading the weights
model(1, tf.constant(10.0, tf.float32))
with open(model_weights_path, 'rb') as f:
    weights = pickle.load(f)
model.set_weights(weights)
# Evaluations
_,bler = sim_ber(model, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=100)
BLER['neural-receiver'] = bler.numpy()

```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -5.0 | 2.5993e-01 | 1.0000e+00 |       46314 |      178176 |          128 |         128 |         0.2 |reached target block errors
     -4.5 | 2.4351e-01 | 1.0000e+00 |       43387 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -4.0 | 2.2642e-01 | 1.0000e+00 |       40343 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -3.5 | 2.0519e-01 | 1.0000e+00 |       36560 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -3.0 | 1.7735e-01 | 1.0000e+00 |       31600 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -2.5 | 1.2847e-01 | 1.0000e+00 |       22890 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -2.0 | 4.3592e-02 | 7.9688e-01 |        7767 |      178176 |          102 |         128 |         0.1 |reached target block errors
     -1.5 | 3.0379e-03 | 1.1830e-01 |        3789 |     1247232 |          106 |         896 |         0.9 |reached target block errors
     -1.0 | 4.8306e-04 | 7.4219e-03 |        8607 |    17817600 |           95 |       12800 |        13.3 |reached max iter
     -0.5 | 2.4481e-04 | 2.1875e-03 |        4362 |    17817600 |           28 |       12800 |        13.2 |reached max iter
      0.0 | 1.9026e-04 | 1.4844e-03 |        3390 |    17817600 |           19 |       12800 |        13.2 |reached max iter
      0.5 | 7.0436e-05 | 5.4688e-04 |        1255 |    17817600 |            7 |       12800 |        13.3 |reached max iter
      1.0 | 4.5405e-05 | 3.1250e-04 |         809 |    17817600 |            4 |       12800 |        13.2 |reached max iter
      1.5 | 3.0083e-05 | 3.1250e-04 |         536 |    17817600 |            4 |       12800 |        13.2 |reached max iter
      2.0 | 5.8145e-05 | 3.1250e-04 |        1036 |    17817600 |            4 |       12800 |        13.3 |reached max iter
      2.5 | 1.6276e-05 | 1.5625e-04 |         290 |    17817600 |            2 |       12800 |        13.2 |reached max iter
      3.0 | 0.0000e+00 | 0.0000e+00 |           0 |    17817600 |            0 |       12800 |        13.2 |reached max iter
Simulation stopped as no error occurred @ EbNo = 3.0 dB.

```

    
Finally, we plots the BLERs

```python
[18]:
```

```python
plt.figure(figsize=(10,6))
# Baseline - Perfect CSI
plt.semilogy(ebno_dbs, BLER['baseline-perfect-csi'], 'o-', c=f'C0', label=f'Baseline - Perfect CSI')
# Baseline - LS Estimation
plt.semilogy(ebno_dbs, BLER['baseline-ls-estimation'], 'x--', c=f'C1', label=f'Baseline - LS Estimation')
# Neural receiver
plt.semilogy(ebno_dbs, BLER['neural-receiver'], 's-.', c=f'C2', label=f'Neural receiver')
#
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()
```


## Pre-computed Results<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Neural_Receiver.html#Pre-computed-Results" title="Permalink to this headline"></a>

```python
[ ]:
```

```python
pre_computed_results = "{'baseline-perfect-csi': [1.0, 1.0, 1.0, 1.0, 1.0, 0.9916930379746836, 0.5367080479452054, 0.0285078125, 0.0017890625, 0.0006171875, 0.0002265625, 9.375e-05, 2.34375e-05, 7.8125e-06, 1.5625e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'baseline-ls-estimation': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9998022151898734, 0.9199448529411764, 0.25374190938511326, 0.0110234375, 0.002078125, 0.0008359375, 0.0004375, 0.000171875, 9.375e-05, 4.6875e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'neural-receiver': [1.0, 1.0, 1.0, 1.0, 1.0, 0.9984177215189873, 0.7505952380952381, 0.10016025641025642, 0.00740625, 0.0021640625, 0.000984375, 0.0003671875, 0.000203125, 0.0001484375, 3.125e-05, 2.34375e-05, 7.8125e-06, 7.8125e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"
BLER = eval(pre_computed_results)
```

## References<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Neural_Receiver.html#References" title="Permalink to this headline"></a>
    
[1] M. Honkala, D. Korpi and J. M. J. Huttunen, “DeepRx: Fully Convolutional Deep Learning Receiver,” in IEEE Transactions on Wireless Communications, vol. 20, no. 6, pp. 3925-3940, June 2021, doi: 10.1109/TWC.2021.3054520.
    
[2] F. Ait Aoudia and J. Hoydis, “End-to-end Learning for OFDM: From Neural Receivers to Pilotless Communication,” in IEEE Transactions on Wireless Communications, doi: 10.1109/TWC.2021.3101364.
    
[3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, “Deep Residual Learning for Image Recognition”, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778
    
[4] G. Böcherer, “Achievable Rates for Probabilistic Shaping”, arXiv:1707.01134, 2017.
<script type="application/vnd.jupyter.widget-state+json">
{"state": {}, "version_major": 2, "version_minor": 0}
</script>