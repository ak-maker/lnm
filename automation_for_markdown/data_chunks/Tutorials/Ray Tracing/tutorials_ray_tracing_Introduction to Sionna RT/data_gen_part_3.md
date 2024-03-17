# Introduction to Sionna RT<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#Introduction-to-Sionna-RT" title="Permalink to this headline"></a>
    
In this notebook, you will
 
- Discover the basic functionalities of Sionna’s <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html">ray tracing (RT) module</a>
- Learn how to compute coverage maps
- Use ray-traced channels for link-level simulations instead of stochastic channel models
# Table of Content
## GPU Configuration and Imports
## From Paths to Channel Impulse Responses
## BER Evaluation
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

```python
[1]:
```

```python
import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Colab does currently not support the latest version of ipython.
# Thus, the preview does not work in Colab. However, whenever possible we
# strongly recommend to use the scene preview mode.
try: # detect if the notebook runs in Colab
    import google.colab
    colab_compat = True # deactivate preview
except:
    colab_compat = False
resolution = [480,320] # increase for higher quality of renderings
# Allows to exit cell execution in Jupyter
class ExitCell(Exception):
    def _render_traceback_(self):
        pass
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
tf.random.set_seed(1) # Set global random seed for reproducibility

```
```python
[2]:
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time
# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement

```

## From Paths to Channel Impulse Responses<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#From-Paths-to-Channel-Impulse-Responses" title="Permalink to this headline"></a>
    
Once paths are computed, they can be transformed into channel impulse responses (CIRs). The class method <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html#Paths.apply_doppler">apply_doppler</a> can simulate time evolution of the CIR based on arbitrary velocity vectors of all transmitters and receivers for a desired sampling frequency and number of time steps. The class method <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html#Paths.cir">cir</a> generates the channel impulse responses which can be used by
other components for link-level simulations in either time or frequency domains. The method also allows you to only consider certain types of paths, e.g., line-of-sight, reflections, etc.

```python
[14]:
```

```python
# Default parameters in the PUSCHConfig
subcarrier_spacing = 15e3
fft_size = 48

```
```python
[15]:
```

```python
# Print shape of channel coefficients before the application of Doppler shifts
# The last dimension corresponds to the number of time steps which defaults to one
# as there is no mobility
print("Shape of `a` before applying Doppler shifts: ", paths.a.shape)
# Apply Doppler shifts
paths.apply_doppler(sampling_frequency=subcarrier_spacing, # Set to 15e3 Hz
                    num_time_steps=14, # Number of OFDM symbols
                    tx_velocities=[3.,0,0], # We can set additional tx speeds
                    rx_velocities=[0,7.,0]) # Or rx speeds
print("Shape of `a` after applying Doppler shifts: ", paths.a.shape)
a, tau = paths.cir()
print("Shape of tau: ", tau.shape)

```


```python
Shape of `a` before applying Doppler shifts:  (1, 1, 2, 1, 1, 13, 1)
Shape of `a` after applying Doppler shifts:  (1, 1, 2, 1, 1, 13, 14)
Shape of tau:  (1, 1, 1, 13)
```

    
Let us have a look at the channel impulse response for the 14 incoming paths from the simulation above.

```python
[16]:
```

```python
t = tau[0,0,0,:]/1e-9 # Scale to ns
a_abs = np.abs(a)[0,0,0,0,0,:,0]
a_max = np.max(a_abs)
# Add dummy entry at start/end for nicer figure
t = np.concatenate([(0.,), t, (np.max(t)*1.1,)])
a_abs = np.concatenate([(np.nan,), a_abs, (np.nan,)])
# And plot the CIR
plt.figure()
plt.title("Channel impulse response realization")
plt.stem(t, a_abs)
plt.xlim([0, np.max(t)])
plt.ylim([-2e-6, a_max*1.1])
plt.xlabel(r"$\tau$ [ns]")
plt.ylabel(r"$|a|$");

```


    
Note that the delay of the first arriving path is normalized to zero. This behavior can be changed using the Paths’ call property `normalize_delays`. For link-level simulations, it is recommended to work with normalized delays, unless perfect synchronization is explicitly desired.

```python
[17]:
```

```python
# Disable normalization of delays
paths.normalize_delays = False
# Get only the LoS path
_, tau = paths.cir(los=True, reflection=False)
print("Delay of first path without normalization: ", np.squeeze(tau))
paths.normalize_delays = True
_, tau = paths.cir(los=True, reflection=False)
print("Delay of first path with normalization: ", np.squeeze(tau))

```


```python
Delay of first path without normalization:  2.739189e-07
Delay of first path with normalization:  0.0
```

    
The CIRs can now be loaded either in the time-domain or frequency-domain channel models, respectively. Please see <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_ofdm_channel">cir_to_ofdm_channel</a> and <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_time_channel">cir_to_time_channel</a> for further details.

```python
[18]:
```

```python
# Compute frequencies of subcarriers and center around carrier frequency
frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing)
# Compute the frequency response of the channel at frequencies.
h_freq = cir_to_ofdm_channel(frequencies,
                             a,
                             tau,
                             normalize=True) # Non-normalized includes path-loss
# Verify that the channel power is normalized
h_avg_power = tf.reduce_mean(tf.abs(h_freq)**2).numpy()
print("Shape of h_freq: ", h_freq.shape)
print("Average power h_freq: ", h_avg_power) # Channel is normalized

```


```python
Shape of h_freq:  (1, 1, 2, 1, 1, 14, 48)
Average power h_freq:  1.0000001
```

    
The frequency responses `h_freq` are now ready to be processed by the <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyOFDMChannel">ApplyOFDMChannel</a> Layer.

```python
[19]:
```

```python
# Placeholder for tx signal of shape
# [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
x = tf.zeros([h_freq.shape.as_list()[i] for i in [0,3,4,5,6]], tf.complex64)
no = 0.1 # noise variance
# Init channel layer
channel = ApplyOFDMChannel(add_awgn=True)
# Apply channel
y = channel([x, h_freq, no])
# [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
print(y.shape)

```


```python
(1, 1, 2, 14, 48)
```
## BER Evaluation<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#BER-Evaluation" title="Permalink to this headline"></a>
    
We now initialize a transmitter and receiver from the <a class="reference external" href="https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html">5G NR PUSCH Tutorial</a> notebook. These components could be replaced by your own transceiver implementations. Then we simulate PUSCH transmissions over the ray-traced CIRs that we generated in the previous cells.

```python
[20]:
```

```python
# Init pusch_transmitter
pusch_config = PUSCHConfig()
# Instantiate a PUSCHTransmitter from the PUSCHConfig
pusch_transmitter = PUSCHTransmitter(pusch_config)
# Create a PUSCHReceiver using the PUSCHTransmitter
pusch_receiver = PUSCHReceiver(pusch_transmitter)

```
```python
[21]:
```

```python
# Simulate transmissions over the
batch_size = 100 # h_freq is broadcast, i.e., same CIR for all samples but different AWGN realizations
ebno_db = 2. # SNR in dB
no = ebnodb2no(ebno_db,
               pusch_transmitter._num_bits_per_symbol,
               pusch_transmitter._target_coderate,
               pusch_transmitter.resource_grid)
x, b = pusch_transmitter(batch_size) # Generate transmit signal and info bits
y = channel([x, h_freq, no]) # Simulate channel output
b_hat = pusch_receiver([y, no]) # Recover the info bits
# Compute BER
print(f"BER: {compute_ber(b, b_hat).numpy():.5f}")

```


```python
BER: 0.07575
```

    
**Remark** Contrary to other Sionna components, ray tracing does not have a dedicated batch dimension. However, multiple transmitter and receivers can be simulated in parallel, which effectively equals a batch-dimension.
    
Note that simulating multiple receivers in the ray tracer comes at little additional overhead. However, the complexity increases significantly for multiple transmitters as an individual ray tracing step is required for each transmitter. As the total number of rays is fixed, an increased number of transmitter requires also an increased number of rays in the `compute_paths` step for the same overall precision.

