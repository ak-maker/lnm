# MIMO OFDM Transmissions over the CDL Channel Model<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#MIMO-OFDM-Transmissions-over-the-CDL-Channel-Model" title="Permalink to this headline"></a>
    
In this notebook, you will learn how to setup a realistic simulation of a MIMO point-to-point link between a mobile user terminal (UT) and a base station (BS). Both, uplink and downlink directions are considered. Here is a schematic diagram of the system model with all required components:
    
    
The setup includes:
 
- 5G LDPC FEC
- QAM modulation
- OFDM resource grid with configurabel pilot pattern
- Multiple data streams
- 3GPP 38.901 CDL channel models and antenna patterns
- ZF Precoding with perfect channel state information
- LS Channel estimation with nearest-neighbor interpolation as well as perfect CSI
- LMMSE MIMO equalization

    
You will learn how to simulate the channel in the time and frequency domains and understand when to use which option.
    
In particular, you will investigate:
 
- The performance over different CDL models
- The impact of imperfect CSI
- Channel aging due to mobility
- Inter-symbol interference due to insufficient cyclic prefix length

    
We will first walk through the configuration of all components of the system model, before simulating some simple transmissions in the time and frequency domain. Then, we will build a general Keras model which will allow us to run efficiently simulations with different parameter settings.
    
This is a notebook demonstrating a fairly advanced use of the Sionna library. It is recommended that you familiarize yourself with the API documentation of the <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html">Channel</a> module and understand the difference between time- and frequency-domain modeling. Some of the simulations take some time, especially when you have no GPU available. For this reason, we provide the simulation results within the cells generating the figures. If you want to
visualize your own results, just comment the corresponding line.

# Table of Content
## System Setup
### CDL Channel Model
#### CIR Sampling Process
#### Generate the Channel Frequency Response
#### Generate the Discrete-Time Channel Impulse Response
### Other Physical Layer Components
## Simulations
  
  

### GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

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
import time
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers
from sionna.channel.tr38901 import AntennaArray, CDL, Antenna
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource, ebnodb2no, sim_ber
from sionna.utils.metrics import compute_ber
```

#### CIR Sampling Process<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#CIR-Sampling-Process" title="Permalink to this headline"></a>
    
The instance `cdl` of the <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#clustered-delay-line-cdl">CDL</a> <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#channel-model-interface">ChannelModel</a> can be used to generate batches of random realizations of continuous-time channel impulse responses, consisting of complex gains `a` and delays `tau` for each path. To account for time-varying channels, a channel impulse responses is sampled at the `sampling_frequency` for
`num_time_samples` samples. For more details on this, please have a look at the <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html">API documentation</a> of the channel models.
    
In order to model the channel in the frequency domain, we need `num_ofdm_symbols` samples that are taken once per `ofdm_symbol_duration`, which corresponds to the length of an OFDM symbol plus the cyclic prefix.

```python
[10]:
```

```python
a, tau = cdl(batch_size=32, num_time_steps=rg.num_ofdm_symbols, sampling_frequency=1/rg.ofdm_symbol_duration)
```
The path gains `a` have shape
`[batch` `size,` `num_rx,` `num_rx_ant,` `num_tx,` `num_tx_ant,` `num_paths,` `num_time_steps]`
and the delays `tau` have shape
`[batch_size,` `num_rx,` `num_tx,` `num_paths]`.

```python
[11]:
```

```python
print("Shape of the path gains: ", a.shape)
print("Shape of the delays:", tau.shape)
```


```python
Shape of the path gains:  (32, 1, 8, 1, 4, 23, 14)
Shape of the delays: (32, 1, 1, 23)
```

    
The delays are assumed to be static within the time-window of interest. Only the complex path gains change over time. The following two figures depict the channel impulse response at a particular time instant and the time-evolution of the gain of one path, respectively.

```python
[12]:
```

```python
plt.figure()
plt.title("Channel impulse response realization")
plt.stem(tau[0,0,0,:]/1e-9, np.abs(a)[0,0,0,0,0,:,0])
plt.xlabel(r"$\tau$ [ns]")
plt.ylabel(r"$|a|$")

plt.figure()
plt.title("Time evolution of path gain")
plt.plot(np.arange(rg.num_ofdm_symbols)*rg.ofdm_symbol_duration/1e-6, np.real(a)[0,0,0,0,0,0,:])
plt.plot(np.arange(rg.num_ofdm_symbols)*rg.ofdm_symbol_duration/1e-6, np.imag(a)[0,0,0,0,0,0,:])
plt.legend(["Real part", "Imaginary part"])
plt.xlabel(r"$t$ [us]")
plt.ylabel(r"$a$");
```


#### Generate the Channel Frequency Response<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#Generate-the-Channel-Frequency-Response" title="Permalink to this headline"></a>
    
If we want to use the continuous-time channel impulse response to simulate OFDM transmissions under ideal conditions, i.e., no inter-symbol interference, inter-carrier interference, etc., we need to convert it to the frequency domain.
    
This can be done with the function <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#cir-to-ofdm-channel">cir_to_ofdm_channel</a> that computes the Fourier transform of the continuous-time channel impulse response at a set of `frequencies`, corresponding to the different subcarriers. The frequencies can be obtained with the help of the convenience function <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#subcarrier-frequencies">subcarrier_frequencies</a>.

```python
[13]:
```

```python
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
```

    
Let us have a look at the channel frequency response at a given time instant:

```python
[14]:
```

```python
plt.figure()
plt.title("Channel frequency response")
plt.plot(np.real(h_freq[0,0,0,0,0,0,:]))
plt.plot(np.imag(h_freq[0,0,0,0,0,0,:]))
plt.xlabel("OFDM Symbol Index")
plt.ylabel(r"$h$")
plt.legend(["Real part", "Imaginary part"]);
```


    
We can apply the channel frequency response to a given input with the <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#applyofdmchannel">ApplyOFDMChannel</a> layer. This layer can also add additive white Gaussian noise (AWGN) to the channel output.

```python
[15]:
```

```python
# Function that will apply the channel frequency response to an input signal
channel_freq = ApplyOFDMChannel(add_awgn=True)
```

#### Generate the Discrete-Time Channel Impulse Response<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#Generate-the-Discrete-Time-Channel-Impulse-Response" title="Permalink to this headline"></a>
    
In the same way as we have created the frequency channel impulse response from the continuous-time response, we can use the latter to compute a discrete-time impulse response. This can then be used to model the channel in the time-domain through discrete convolution with an input signal. Time-domain channel modeling is necessary whenever we want to deviate from the perfect OFDM scenario, e.g., OFDM without cyclic prefix, inter-subcarrier interference due to carrier-frequency offsets, phase
noise, or very high Doppler spread scenarios, as well as other single or multicarrier waveforms (OTFS, FBMC, UFMC, etc).
    
A discrete-time impulse response can be obtained with the help of the function <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#cir-to-time-channel">cir_to_time_channel</a> that requires a `bandwidth` parameter. This function first applies a perfect low-pass filter of the provided `bandwith` to the continuous-time channel impulse response and then samples the filtered response at the Nyquist rate. The resulting discrete-time impulse response is then truncated to finite length, depending on
the delay spread. `l_min` and `l_max` denote truncation boundaries and the resulting channel has `l_tot=l_max-l_min+1` filter taps. A detailed mathematical description of this process is provided in the API documentation of the channel models. You can freely chose both parameters if you do not want to rely on the default values.
    
In order to model the channel in the domain, the continuous-time channel impulse response must be sampled at the Nyquist rate. We also need now `num_ofdm_symbols` `x` `(fft_size` `+` `cyclic_prefix_length)` `+` `l_tot-1` samples in contrast to `num_ofdm_symbols` samples for modeling in the frequency domain. This implies that the memory requirements of time-domain channel modeling is significantly higher. We therefore recommend to only use this feature if it is really necessary. Simulations with many
transmitters, receivers, and/or large antenna arrays become otherwise quickly prohibitively complex.

```python
[16]:
```

```python
# The following values for truncation are recommended.
# Please feel free to tailor them to you needs.
l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
l_tot = l_max-l_min+1
a, tau = cdl(batch_size=2, num_time_steps=rg.num_time_samples+l_tot-1, sampling_frequency=rg.bandwidth)
```
```python
[17]:
```

```python
h_time = cir_to_time_channel(rg.bandwidth, a, tau, l_min=l_min, l_max=l_max, normalize=True)
```
```python
[18]:
```

```python
plt.figure()
plt.title("Discrete-time channel impulse response")
plt.stem(np.abs(h_time[0,0,0,0,0,0]))
plt.xlabel(r"Time step $\ell$")
plt.ylabel(r"$|\bar{h}|$");
```


    
We can apply the discrete-time impulse response to a given input with the <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#applytimechannel">ApplyTimeChannel</a> layer. This layer can also add additive white Gaussian noise (AWGN) to the channel output.

```python
[19]:
```

```python
# Function that will apply the discrete-time channel impulse response to an input signal
channel_time = ApplyTimeChannel(rg.num_time_samples, l_tot=l_tot, add_awgn=True)
```

### Other Physical Layer Components<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#Other-Physical-Layer-Components" title="Permalink to this headline"></a>
    
Finally, we create instances of all other physical layer components we need. Most of these layers are self-explanatory. For more information, please have a look at the API documentation.

```python
[20]:
```

```python
num_bits_per_symbol = 2 # QPSK modulation
coderate = 0.5 # Code rate
n = int(rg.num_data_symbols*num_bits_per_symbol) # Number of coded bits
k = int(n*coderate) # Number of information bits
# The binary source will create batches of information bits
binary_source = BinarySource()
# The encoder maps information bits to coded bits
encoder = LDPC5GEncoder(k, n)
# The mapper maps blocks of information bits to constellation symbols
mapper = Mapper("qam", num_bits_per_symbol)
# The resource grid mapper maps symbols onto an OFDM resource grid
rg_mapper = ResourceGridMapper(rg)
# The zero forcing precoder precodes the transmit stream towards the intended antennas
zf_precoder = ZFPrecoder(rg, sm, return_effective_channel=True)
# OFDM modulator and demodulator
modulator = OFDMModulator(rg.cyclic_prefix_length)
demodulator = OFDMDemodulator(rg.fft_size, l_min, rg.cyclic_prefix_length)
# This function removes nulled subcarriers from any tensor having the shape of a resource grid
remove_nulled_scs = RemoveNulledSubcarriers(rg)
# The LS channel estimator will provide channel estimates and error variances
ls_est = LSChannelEstimator(rg, interpolation_type="nn")
# The LMMSE equalizer will provide soft symbols together with noise variance estimates
lmmse_equ = LMMSEEqualizer(rg, sm)
# The demapper produces LLR for all coded bits
demapper = Demapper("app", "qam", num_bits_per_symbol)
# The decoder provides hard-decisions on the information bits
decoder = LDPC5GDecoder(encoder, hard_out=True)
```

## Simulations<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#Simulations" title="Permalink to this headline"></a>

