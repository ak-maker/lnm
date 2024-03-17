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
## System Setup
## Uplink Transmissions in the Frequency Domain
### Compare Estimated and Actual Frequency Responses
  
  

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

## System Setup<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Realistic_Multiuser_MIMO_Simulations.html#System-Setup" title="Permalink to this headline"></a>
    
We will now configure all components of the system model step-by-step.

```python
[3]:
```

```python
scenario = "umi"
carrier_frequency = 3.5e9
direction = "uplink"
num_ut = 4
batch_size = 32
```
```python
[4]:
```

```python
tf.random.set_seed(1)
# Define the UT antenna array
ut_array = Antenna(polarization="single",
                   polarization_type="V",
                   antenna_pattern="omni",
                   carrier_frequency=carrier_frequency)
# Define the BS antenna array
bs_array = AntennaArray(num_rows=1,
                        num_cols=4,
                        polarization="dual",
                        polarization_type="VH",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
# Create channel model
channel_model = UMi(carrier_frequency=carrier_frequency,
                    o2i_model="low",
                    ut_array=ut_array,
                    bs_array=bs_array,
                    direction=direction,
                    enable_pathloss=False,
                    enable_shadow_fading=False)
# Generate the topology
topology = gen_topology(batch_size, num_ut, scenario)
# Set the topology
channel_model.set_topology(*topology)
# Visualize the topology
channel_model.show_topology()
```

```python
[5]:
```

```python
# The number of transmitted streams is equal to the number of UT antennas
num_streams_per_tx = 1
# Create an RX-TX association matrix
# rx_tx_association[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change. However, as we have only a single
# transmitter and receiver, this does not matter:
rx_tx_association = np.zeros([1, num_ut])
rx_tx_association[0, :] = 1
# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly simple. However, it can get complicated
# for simulations with many transmitters and receivers.
sm = StreamManagement(rx_tx_association, num_streams_per_tx)
```
```python
[6]:
```

```python
rg = ResourceGrid(num_ofdm_symbols=14,
                  fft_size=128,
                  subcarrier_spacing=30e3,
                  num_tx=num_ut,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=20,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[2,11])
rg.show();
```

```python
[7]:
```

```python
num_bits_per_symbol = 2 # QPSK modulation
coderate = 0.5 # The code rate
n = int(rg.num_data_symbols*num_bits_per_symbol) # Number of coded bits
k = int(n*coderate) # Number of information bits
# The binary source will create batches of information bits
binary_source = BinarySource()
qam_source = QAMSource(num_bits_per_symbol)
# The encoder maps information bits to coded bits
encoder = LDPC5GEncoder(k, n)
# The mapper maps blocks of information bits to constellation symbols
mapper = Mapper("qam", num_bits_per_symbol)
# The resource grid mapper maps symbols onto an OFDM resource grid
rg_mapper = ResourceGridMapper(rg)
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
# OFDM CHannel
ofdm_channel = OFDMChannel(channel_model, rg, add_awgn=True, normalize_channel=False, return_channel=True)
channel_freq = ApplyOFDMChannel(add_awgn=True)
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
```

## Uplink Transmissions in the Frequency Domain<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Realistic_Multiuser_MIMO_Simulations.html#Uplink-Transmissions-in-the-Frequency-Domain" title="Permalink to this headline"></a>
    
We now simulate a batch of uplink transmissions. We keep references to the estimated and actual channel frequency responses.

```python
[8]:
```

```python
ebno_db = 10
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
b = binary_source([batch_size, num_ut, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)
a, tau = channel_model(num_time_samples=rg.num_ofdm_symbols, sampling_frequency=1/rg.ofdm_symbol_duration)
h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
y = channel_freq([x_rg, h_freq, no])
h_hat, err_var = ls_est ([y, no])
x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])
llr = demapper([x_hat, no_eff])
b_hat = decoder(llr)
print("BER: {}".format(compute_ber(b, b_hat).numpy()))
```


```python
BER: 6.103515625e-05
```
### Compare Estimated and Actual Frequency Responses<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Realistic_Multiuser_MIMO_Simulations.html#Compare-Estimated-and-Actual-Frequency-Responses" title="Permalink to this headline"></a>
    
We can now compare the estimated frequency responses and ground truth:

```python
[9]:
```

```python
# In the example above, we assumed perfect CSI, i.e.,
# h_hat correpsond to the exact ideal channel frequency response.
h_perf = remove_nulled_scs(h_freq)[0,0,0,0,0,0]
# We now compute the LS channel estimate from the pilots.
h_est = h_hat[0,0,0,0,0,0]
plt.figure()
plt.plot(np.real(h_perf))
plt.plot(np.imag(h_perf))
plt.plot(np.real(h_est), "--")
plt.plot(np.imag(h_est), "--")
plt.xlabel("Subcarrier index")
plt.ylabel("Channel frequency response")
plt.legend(["Ideal (real part)", "Ideal (imaginary part)", "Estimated (real part)", "Estimated (imaginary part)"]);
plt.title("Comparison of channel frequency responses");
```


