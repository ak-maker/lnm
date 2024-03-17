I'm building a dataset for model training focused on the "sionna" Python package, based on the given following markdown context. Cover the markdown context as much as you can.Your role is to generate clear, concise instructions (as user questions) that will guide the model in mastering Sionna coding.Start each instruction with "INSTRUCTION:" and tailor it to fit the provided context, ensuring to cover as much of the context's information as possible for comprehensive learning.

<CONTEXT>
# Basic MIMO Simulations<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Simple_MIMO_Simulation.html#Basic-MIMO-Simulations" title="Permalink to this headline"></a>
    
In this notebook, you will learn how to setup simulations of MIMO transmissions over a flat-fading channel.
    
Here is a schematic diagram of the system model with all required components:
    
    
You will learn how to:
 
- Use the FastFadingChannel class
- Apply spatial antenna correlation
- Implement LMMSE detection with perfect channel knowledge
- Run BER/SER simulations

    
We will first walk through the configuration of all components of the system model, before building a general Keras model which will allow you to run efficiently simulations with different parameter settings.

# Table of Content
## GPU Configuration and Imports
## Simple uncoded transmission
### Adding spatial correlation
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Simple_MIMO_Simulation.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

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
import sys
from sionna.utils import BinarySource, QAMSource, ebnodb2no, compute_ser, compute_ber, PlotBER
from sionna.channel import FlatFadingChannel, KroneckerModel
from sionna.channel.utils import exp_corr_mat
from sionna.mimo import lmmse_equalizer
from sionna.mapping import SymbolDemapper, Mapper, Demapper
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
```

## Simple uncoded transmission<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Simple_MIMO_Simulation.html#Simple-uncoded-transmission" title="Permalink to this headline"></a>
    
We will consider point-to-point transmissions from a transmitter with `num_tx_ant` antennas to a receiver with `num_rx_ant` antennas. The transmitter applies no precoding and sends independent data stream from each antenna.
    
Let us now generate a batch of random transmit vectors of random 16QAM symbols:

```python
[3]:
```

```python
num_tx_ant = 4
num_rx_ant = 16
num_bits_per_symbol = 4
batch_size = 1024
qam_source = QAMSource(num_bits_per_symbol)
x = qam_source([batch_size, num_tx_ant])
print(x.shape)
```


```python
(1024, 4)
```

    
Next, we will create an instance of the `FlatFadingChannel` class to simulate transmissions over an i.i.d. Rayleigh fading channel. The channel will also add AWGN with variance `no`. As we will need knowledge of the channel realizations for detection, we activate the `return_channel` flag.

```python
[4]:
```

```python
channel = FlatFadingChannel(num_tx_ant, num_rx_ant, add_awgn=True, return_channel=True)
no = 0.2 # Noise variance of the channel
# y and h are the channel output and channel realizations, respectively.
y, h = channel([x, no])
print(y.shape)
print(h.shape)
```


```python
(1024, 16)
(1024, 16, 4)
```

    
Using the perfect channel knowledge, we can now implement an LMMSE equalizer to compute soft-symbols. The noise covariance matrix in this example is just a scaled identity matrix which we need to provide to the `lmmse_equalizer`.

```python
[5]:
```

```python
s = tf.cast(no*tf.eye(num_rx_ant, num_rx_ant), y.dtype)
x_hat, no_eff = lmmse_equalizer(y, h, s)
```

    
Let us know have a look at the transmitted and received constellations:

```python
[6]:
```

```python
plt.axes().set_aspect(1.0)
plt.scatter(np.real(x_hat), np.imag(x_hat));
plt.scatter(np.real(x), np.imag(x));
```


    
As expected, the soft symbols `x_hat` are scattered around the 16QAM constellation points. The equalizer output `no_eff` provides an estimate of the effective noise variance for each soft-symbol.

```python
[7]:
```

```python
print(no_eff.shape)
```


```python
(1024, 4)
```

    
One can confirm that this estimate is correct by comparing the MSE between the transmitted and equalized symbols against the average estimated effective noise variance:

```python
[8]:
```

```python
noise_var_eff = np.var(x-x_hat)
noise_var_est = np.mean(no_eff)
print(noise_var_eff)
print(noise_var_est)
```


```python
0.016722694
0.016684469
```

    
The last step is to make hard decisions on the symbols and compute the SER:

```python
[9]:
```

```python
symbol_demapper = SymbolDemapper("qam", num_bits_per_symbol, hard_out=True)
# Get symbol indices for the transmitted symbols
x_ind = symbol_demapper([x, no])
# Get symbol indices for the received soft-symbols
x_ind_hat = symbol_demapper([x_hat, no])
compute_ser(x_ind, x_ind_hat)
```
```python
[9]:
```
```python
<tf.Tensor: shape=(), dtype=float64, numpy=0.002197265625>
```
### Adding spatial correlation<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Simple_MIMO_Simulation.html#Adding-spatial-correlation" title="Permalink to this headline"></a>
    
It is very easy add spatial correlation to the `FlatFadingChannel` using the `SpatialCorrelation` class. We can, e.g., easily setup a Kronecker (`KroneckerModel`) (or two-sided) correlation model using exponetial correlation matrices (`exp_corr_mat`).

```python
[10]:
```

```python
# Create transmit and receive correlation matrices
r_tx = exp_corr_mat(0.4, num_tx_ant)
r_rx = exp_corr_mat(0.9, num_rx_ant)
# Add the spatial correlation model to the channel
channel.spatial_corr = KroneckerModel(r_tx, r_rx)
```

    
Next, we can validate that the channel model applies the desired spatial correlation by creating a large batch of channel realizations from which we compute the empirical transmit and receiver covariance matrices:

```python
[11]:
```

```python
h = channel.generate(1000000)
# Compute empirical covariance matrices
r_tx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_a=True), 0)/num_rx_ant
r_rx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_b=True), 0)/num_tx_ant
# Test that the empirical results match the theory
assert(np.allclose(r_tx, r_tx_hat, atol=1e-2))
assert(np.allclose(r_rx, r_rx_hat, atol=1e-2))
```

    
Now, we can transmit the same symbols `x` over the channel with spatial correlation and compute the SER:

```python
[12]:
```

```python
y, h = channel([x, no])
x_hat, no_eff = lmmse_equalizer(y, h, s)
x_ind_hat = symbol_demapper([x_hat, no])
compute_ser(x_ind, x_ind_hat)
```
```python
[12]:
```
```python
<tf.Tensor: shape=(), dtype=float64, numpy=0.115234375>
```

    
The result cleary show the negative effect of spatial correlation in this setting. You can play around with the `a` parameter defining the exponential correlation matrices and see its impact on the SER.

</CONTEXT>

These instructions are crucial for teaching the model to effectively understand and apply Sionna's code and APIs, tailored to real-world programming scenarios.