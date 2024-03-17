# Realistic Multiuser MIMO OFDM Simulations<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Realistic_Multiuser_MIMO_Simulations.html#Realistic-Multiuser-MIMO-OFDM-Simulations" title="Permalink to this headline"></a>
    
In this notebook, you will learn how to setup realistic simulations of multiuser MIMO uplink transmissions. Multiple user terminals (UTs) are randomly distributed in a cell sector and communicate with a multi-antenna base station.
    
    
The block-diagramm of the system model looks as follows:
    
    
It includes the following components:
 
- 5G LDPC FEC
- QAM modulation
- OFDM resource grid with configurable pilot pattern
- Multiple single-antenna transmitters and a multi-antenna receiver
- 3GPP 38.901 UMi, UMa, and RMa channel models and antenna patterns
- LS Channel estimation with nearest-neighbor interpolation as well as perfect CSI
- LMMSE MIMO equalization

    
You will learn how to setup the topologies required to simulate such scenarios and investigate
 
- the performance over different models, and
- the impact of imperfect CSI.

    
We will first walk through the configuration of all components of the system model, before simulating some simple uplink transmissions in the frequency domain. We will then simulate CDFs of the channel condition number and look into frequency-selectivity of the different channel models to understand the reasons for the observed performance differences.
    
It is recommended that you familiarize yourself with the <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html">API documentation</a> of the `Channel` module and, in particular, the 3GPP 38,901 models that require a substantial amount of configuration. The last set of simulations in this notebook take some time, especially when you have no GPU available. For this reason, we provide the simulation results directly in the cells generating the figures. Simply uncomment the corresponding lines to show
this results.

# Table of Content
## GPU Configuration and Imports
## Uplink Transmissions in the Frequency Domain
### Understand the Difference Between the Channel Models
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Realistic_Multiuser_MIMO_Simulations.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

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
import time
import pickle
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers
from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, UMi, UMa, RMa
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource, ebnodb2no, sim_ber, QAMSource
from sionna.utils.metrics import compute_ber
```

### Understand the Difference Between the Channel Models<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Realistic_Multiuser_MIMO_Simulations.html#Understand-the-Difference-Between-the-Channel-Models" title="Permalink to this headline"></a>
    
Before we proceed with more advanced simulations, it is important to understand the differences between the UMi, UMa, and RMa models. In the following code snippet, we compute the empirical cummulative distribution function (CDF) of the condition number of the channel frequency response matrix between all receiver and transmit antennas.

```python
[10]:
```

```python
def cond_hist(scenario):
    """Generates a histogram of the channel condition numbers"""
    # Setup a CIR generator
    if scenario == "umi":
        channel_model = UMi(carrier_frequency=carrier_frequency,
                                      o2i_model="low",
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    elif scenario == "uma":
        channel_model = UMa(carrier_frequency=carrier_frequency,
                                      o2i_model="low",
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    elif scenario == "rma":
        channel_model = RMa(carrier_frequency=carrier_frequency,
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    topology = gen_topology(1024, num_ut, scenario)
    # Set the topology
    channel_model.set_topology(*topology)
    # Generate random CIR realizations
    # As we nned only a single sample in time, the sampling_frequency
    # does not matter.
    cir = channel_model(1, 1)
    # Compute the frequency response
    h = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
    h = tf.squeeze(h)
    h = tf.transpose(h, [0,3,1,2])
    # Compute condition number
    c = np.reshape(np.linalg.cond(h), [-1])
    # Compute normalized histogram
    hist, bins = np.histogram(c, 100, (1, 100))
    hist = hist/np.sum(hist)
    return bins[:-1], hist
plt.figure()
for cdl_model in ["umi", "uma", "rma"]:
    bins, hist = cond_hist(cdl_model)
    plt.plot(bins, np.cumsum(hist))
plt.xlim([0,40])
plt.legend(["UMi", "UMa", "RMa"]);
plt.xlabel("Channel Condition Number")
plt.ylabel("CDF")
plt.title("CDF of the channel condition number");
```


    
From the figure above, you can observe that the UMi and UMa models are substantially better conditioned than the RMa models. This makes them more suitable for MIMO transmissions as we will observe in the next section.
    
It is also interesting to look at the channel frequency responses of these different models, as done in the next cell:

```python
[11]:
```

```python
def freq_response(scenario):
    """Generates an example frequency response"""
    tf.random.set_seed(2)
    # Setup a CIR generator
    if scenario == "umi":
        channel_model = UMi(carrier_frequency=carrier_frequency,
                                      o2i_model="low",
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    elif scenario == "uma":
        channel_model = UMa(carrier_frequency=carrier_frequency,
                                      o2i_model="low",
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    elif scenario == "rma":
        channel_model = RMa(carrier_frequency=carrier_frequency,
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    topology = gen_topology(1, num_ut, scenario)
    # Set the topology
    channel_model.set_topology(*topology)
    # Generate random CIR realizations
    # As we nned only a single sample in time, the sampling_frequency
    # does not matter.
    cir = channel_model(1, 1)
    # Compute the frequency response
    h = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
    h = tf.squeeze(h)
    return h[0,0]
plt.figure()
for cdl_model in ["umi", "uma", "rma"]:
    h = freq_response(cdl_model)
    plt.plot(np.real(h))
plt.legend(["UMi", "UMa", "RMa"]);
plt.xlabel("Subcarrier Index")
plt.ylabel(r"$\Re(h)$")
plt.title("Channel frequency response");
```


    
The RMa model has significantly less frequency selectivity than the other models which makes channel estimation easier.

