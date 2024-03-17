# End-to-end Learning with Autoencoders<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Autoencoder.html#End-to-end-Learning-with-Autoencoders" title="Permalink to this headline"></a>
    
In this notebook, you will learn how to implement an end-to-end communication system as an autoencoder [1]. The implemented system is shown in the figure below. An additive white Gaussian noise (AWGN) channel is considered. On the transmitter side, joint training of the constellation geometry and bit-labeling is performed, as in [2]. On the receiver side, a neural network-based demapper that computes log-likelihood ratios (LLRs) on the transmitted bits from the received samples is optimized. The
considered autoencoder is benchmarked against a quadrature amplitude modulation (QAM) with Gray labeling and the optimal AWGN demapper.
    
    
Two algorithms for training the autoencoder are implemented in this notebook:
 
- Conventional stochastic gradient descent (SGD) with backpropagation, which assumes a differentiable channel model and therefore optimizes the end-to-end system by backpropagating the gradients through the channel (see, e.g., [1]).
- The training algorithm from [3], which does not assume a differentiable channel model, and which trains the end-to-end system by alternating between conventional training of the receiver and reinforcement learning (RL)-based training of the transmitter. Compared to [3], an additional step of fine-tuning of the receiver is performed after alternating training.

    
**Note:** For an introduction to the implementation of differentiable communication systems and their optimization through SGD and backpropagation with Sionna, please refer to <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part2.html">the Part 2 of the Sionna tutorial for Beginners</a>.
# Table of Content
## GPU Configuration and Imports
## Evaluation
## Visualizing the Learned Constellations
## References
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Autoencoder.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

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
from sionna.channel import AWGN
from sionna.utils import BinarySource, ebnodb2no, log10, expand_to_rank, insert_dims
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper, Constellation
from sionna.utils import sim_ber
```
```python
[2]:
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pickle
```
```python
[3]:
```

```python
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
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
```

## Evaluation<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Autoencoder.html#Evaluation" title="Permalink to this headline"></a>
    
The following cell implements a baseline which uses QAM with Gray labeling and conventional demapping for AWGN channel.

```python
[13]:
```

```python
class Baseline(Model):
    def __init__(self):
        super().__init__()
        ################
        ## Transmitter
        ################
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
        constellation = Constellation("qam", num_bits_per_symbol, trainable=False)
        self.constellation = constellation
        self._mapper = Mapper(constellation=constellation)
        ################
        ## Channel
        ################
        self._channel = AWGN()
        ################
        ## Receiver
        ################
        self._demapper = Demapper("app", constellation=constellation)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db, perturbation_variance=tf.constant(0.0, tf.float32)):
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)
        ################
        ## Transmitter
        ################
        b = self._binary_source([batch_size, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c) # x [batch size, num_symbols_per_codeword]
        ################
        ## Channel
        ################
        y = self._channel([x, no]) # [batch size, num_symbols_per_codeword]
        ################
        ## Receiver
        ################
        llr = self._demapper([y, no])
        # Outer decoding
        b_hat = self._decoder(llr)
        return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation
```
```python
[14]:
```

```python
# Range of SNRs over which the systems are evaluated
ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                     ebno_db_max, # Max SNR for evaluation
                     0.5) # Step
```
```python
[15]:
```

```python
# Utility function to load and set weights of a model
def load_weights(model, model_weights_path):
    model(1, tf.constant(10.0, tf.float32))
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
```

    
The next cell evaluate the baseline and the two autoencoder-based communication systems, trained with different method. The results are stored in the dictionary `BLER`.

```python
[16]:
```

```python
# Dictionnary storing the results
BLER = {}
model_baseline = Baseline()
_,bler = sim_ber(model_baseline, ebno_dbs, batch_size=128, num_target_block_errors=1000, max_mc_iter=1000)
BLER['baseline'] = bler.numpy()
model_conventional = E2ESystemConventionalTraining(training=False)
load_weights(model_conventional, model_weights_path_conventional_training)
_,bler = sim_ber(model_conventional, ebno_dbs, batch_size=128, num_target_block_errors=1000, max_mc_iter=1000)
BLER['autoencoder-conv'] = bler.numpy()
model_rl = E2ESystemRLTraining(training=False)
load_weights(model_rl, model_weights_path_rl_training)
_,bler = sim_ber(model_rl, ebno_dbs, batch_size=128, num_target_block_errors=1000, max_mc_iter=1000)
BLER['autoencoder-rl'] = bler.numpy()
with open(results_filename, 'wb') as f:
    pickle.dump((ebno_dbs, BLER), f)
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      4.0 | 1.2364e-01 | 1.0000e+00 |       94957 |      768000 |         1024 |        1024 |         3.2 |reached target block errors
      4.5 | 9.7535e-02 | 9.9805e-01 |       74907 |      768000 |         1022 |        1024 |         0.1 |reached target block errors
      5.0 | 5.7527e-02 | 9.0712e-01 |       49703 |      864000 |         1045 |        1152 |         0.1 |reached target block errors
      5.5 | 1.9050e-02 | 5.1562e-01 |       29261 |     1536000 |         1056 |        2048 |         0.2 |reached target block errors
      6.0 | 2.3017e-03 | 1.0621e-01 |       16351 |     7104000 |         1006 |        9472 |         0.7 |reached target block errors
      6.5 | 1.2964e-04 | 9.6213e-03 |       10106 |    77952000 |         1000 |      103936 |         7.6 |reached target block errors
      7.0 | 7.8333e-06 | 7.2656e-04 |         752 |    96000000 |           93 |      128000 |         9.3 |reached max iter
      7.5 | 1.4583e-07 | 3.1250e-05 |          14 |    96000000 |            4 |      128000 |         9.4 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      4.0 | 1.0696e-01 | 9.9707e-01 |       82149 |      768000 |         1021 |        1024 |         0.9 |reached target block errors
      4.5 | 6.9547e-02 | 9.3142e-01 |       60089 |      864000 |         1073 |        1152 |         0.1 |reached target block errors
      5.0 | 2.3789e-02 | 5.4010e-01 |       34256 |     1440000 |         1037 |        1920 |         0.1 |reached target block errors
      5.5 | 4.2181e-03 | 1.5472e-01 |       20652 |     4896000 |         1010 |        6528 |         0.5 |reached target block errors
      6.0 | 2.4640e-04 | 1.6292e-02 |       11354 |    46080000 |         1001 |       61440 |         4.3 |reached target block errors
      6.5 | 1.2156e-05 | 9.3750e-04 |        1167 |    96000000 |          120 |      128000 |         9.1 |reached max iter
      7.0 | 1.1667e-06 | 7.0312e-05 |         112 |    96000000 |            9 |      128000 |         9.1 |reached max iter
      7.5 | 8.7500e-07 | 3.9063e-05 |          84 |    96000000 |            5 |      128000 |         9.1 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      4.0 | 1.0489e-01 | 9.9805e-01 |       80553 |      768000 |         1022 |        1024 |         1.1 |reached target block errors
      4.5 | 6.4516e-02 | 9.2101e-01 |       55742 |      864000 |         1061 |        1152 |         0.1 |reached target block errors
      5.0 | 2.3047e-02 | 5.2812e-01 |       33187 |     1440000 |         1014 |        1920 |         0.1 |reached target block errors
      5.5 | 3.7078e-03 | 1.4318e-01 |       19577 |     5280000 |         1008 |        7040 |         0.5 |reached target block errors
      6.0 | 2.2505e-04 | 1.4167e-02 |       11926 |    52992000 |         1001 |       70656 |         5.0 |reached target block errors
      6.5 | 8.1771e-06 | 8.5938e-04 |         785 |    96000000 |          110 |      128000 |         9.2 |reached max iter
      7.0 | 7.0833e-07 | 5.4688e-05 |          68 |    96000000 |            7 |      128000 |         9.1 |reached max iter
      7.5 | 1.1458e-07 | 1.5625e-05 |          11 |    96000000 |            2 |      128000 |         9.1 |reached max iter
```
```python
[17]:
```

```python
plt.figure(figsize=(10,8))
# Baseline - Perfect CSI
plt.semilogy(ebno_dbs, BLER['baseline'], 'o-', c=f'C0', label=f'Baseline')
# Autoencoder - conventional training
plt.semilogy(ebno_dbs, BLER['autoencoder-conv'], 'x-.', c=f'C1', label=f'Autoencoder - conventional training')
# Autoencoder - RL-based training
plt.semilogy(ebno_dbs, BLER['autoencoder-rl'], 'o-.', c=f'C2', label=f'Autoencoder - RL-based training')
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()
```


## Visualizing the Learned Constellations<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Autoencoder.html#Visualizing-the-Learned-Constellations" title="Permalink to this headline"></a>

```python
[18]:
```

```python
model_conventional = E2ESystemConventionalTraining(training=True)
load_weights(model_conventional, model_weights_path_conventional_training)
fig = model_conventional.constellation.show()
fig.suptitle('Conventional training');
```

```python
[19]:
```

```python
model_rl = E2ESystemRLTraining(training=False)
load_weights(model_rl, model_weights_path_rl_training)
fig = model_rl.constellation.show()
fig.suptitle('RL-based training');
```


## References<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Autoencoder.html#References" title="Permalink to this headline"></a>
    
[1] T. O’Shea and J. Hoydis, “An Introduction to Deep Learning for the Physical Layer,” in IEEE Transactions on Cognitive Communications and Networking, vol. 3, no. 4, pp. 563-575, Dec. 2017, doi: 10.1109/TCCN.2017.2758370.
    
[2] S. Cammerer, F. Ait Aoudia, S. Dörner, M. Stark, J. Hoydis and S. ten Brink, “Trainable Communication Systems: Concepts and Prototype,” in IEEE Transactions on Communications, vol. 68, no. 9, pp. 5489-5503, Sept. 2020, doi: 10.1109/TCOMM.2020.3002915.
    
[3] F. Ait Aoudia and J. Hoydis, “Model-Free Training of End-to-End Communication Systems,” in IEEE Journal on Selected Areas in Communications, vol. 37, no. 11, pp. 2503-2516, Nov. 2019, doi: 10.1109/JSAC.2019.2933891.
<script type="application/vnd.jupyter.widget-state+json">
{"state": {}, "version_major": 2, "version_minor": 0}
</script>