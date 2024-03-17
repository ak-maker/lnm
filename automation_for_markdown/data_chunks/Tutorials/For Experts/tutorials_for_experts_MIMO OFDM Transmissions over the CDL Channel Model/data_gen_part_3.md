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
## Simulations
### Uplink Transmission in the Frequency Domain
### Uplink Transmission in the Time Domain
### Downlink Transmission in the Frequency Domain
  
  

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

### Uplink Transmission in the Frequency Domain<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#Uplink-Transmission-in-the-Frequency-Domain" title="Permalink to this headline"></a>
    
Now, we will simulate our first uplink transmission! Inspect the code to understand how perfect CSI at the receiver can be simulated.

```python
[21]:
```

```python
batch_size = 32 # Depending on the memory of your GPU (or system when a CPU is used),
                # you can in(de)crease the batch size. The larger the batch size, the
                # more memory is required. However, simulations will also run much faster.
ebno_db = 40
perfect_csi = False # Change to switch between perfect and imperfect CSI
# Compute the noise power for a given Eb/No value.
# This takes not only the coderate but also the overheads related pilot
# transmissions and nulled carriers
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)
# As explained above, we generate random batches of CIR, transform them
# in the frequency domain and apply them to the resource grid in the
# frequency domain.
cir = cdl(batch_size, rg.num_ofdm_symbols, 1/rg.ofdm_symbol_duration)
h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
y = channel_freq([x_rg, h_freq, no])
if perfect_csi:
    # For perfect CSI, the receiver gets the channel frequency response as input
    # However, the channel estimator only computes estimates on the non-nulled
    # subcarriers. Therefore, we need to remove them here from `h_freq`.
    # This step can be skipped if no subcarriers are nulled.
    h_hat, err_var = remove_nulled_scs(h_freq), 0.
else:
    h_hat, err_var = ls_est ([y, no])
x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])
llr = demapper([x_hat, no_eff])
b_hat = decoder(llr)
ber = compute_ber(b, b_hat)
print("BER: {}".format(ber))
```


```python
BER: 0.0
```

    
An alternative approach to simulations in the frequency domain is to use the convenience function <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#ofdmchannel">OFDMChannel</a> that jointly generates and applies the channel frequency response. Using this function, we could have used the following code:

```python
[22]:
```

```python
ofdm_channel = OFDMChannel(cdl, rg, add_awgn=True, normalize_channel=True, return_channel=True)
y, h_freq = ofdm_channel([x_rg, no])
```

### Uplink Transmission in the Time Domain<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#Uplink-Transmission-in-the-Time-Domain" title="Permalink to this headline"></a>
    
In the previous example, OFDM modulation/demodulation were not needed as the entire system was simulated in the frequency domain. However, this modeling approach is not able to capture many realistic effects.
    
With the following modifications, the system can be modeled in the time domain.
    
Have a careful look at how perfect CSI of the channel frequency response is simulated here.

```python
[23]:
```

```python
batch_size = 4 # We pick a small batch_size as executing this code in Eager mode could consume a lot of memory
ebno_db = 30
perfect_csi = True
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)
# The CIR needs to be sampled every 1/bandwith [s].
# In contrast to frequency-domain modeling, this implies
# that the channel can change over the duration of a single
# OFDM symbol. We now also need to simulate more
# time steps.
cir = cdl(batch_size, rg.num_time_samples+l_tot-1, rg.bandwidth)
# OFDM modulation with cyclic prefix insertion
x_time = modulator(x_rg)
# Compute the discrete-time channel impulse reponse
h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min, l_max, normalize=True)
# Compute the channel output
# This computes the full convolution between the time-varying
# discrete-time channel impulse reponse and the discrete-time
# transmit signal. With this technique, the effects of an
# insufficiently long cyclic prefix will become visible. This
# is in contrast to frequency-domain modeling which imposes
# no inter-symbol interfernce.
y_time = channel_time([x_time, h_time, no])
# OFDM demodulation and cyclic prefix removal
y = demodulator(y_time)
if perfect_csi:
    a, tau = cir
    # We need to sub-sample the channel impulse reponse to compute perfect CSI
    # for the receiver as it only needs one channel realization per OFDM symbol
    a_freq = a[...,rg.cyclic_prefix_length:-1:(rg.fft_size+rg.cyclic_prefix_length)]
    a_freq = a_freq[...,:rg.num_ofdm_symbols]
    # Compute the channel frequency response
    h_freq = cir_to_ofdm_channel(frequencies, a_freq, tau, normalize=True)
    h_hat, err_var = remove_nulled_scs(h_freq), 0.
else:
    h_hat, err_var = ls_est ([y, no])
x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])
llr = demapper([x_hat, no_eff])
b_hat = decoder(llr)
ber = compute_ber(b, b_hat)
print("BER: {}".format(ber))
```


```python
BER: 0.0
```

    
An alternative approach to simulations in the time domain is to use the convenience function <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#timechannel">TimeChannel</a> that jointly generates and applies the discrete-time channel impulse response. Using this function, we could have used the following code:

```python
[24]:
```

```python
time_channel = TimeChannel(cdl, rg.bandwidth, rg.num_time_samples,
                           l_min=l_min, l_max=l_max, normalize_channel=True,
                           add_awgn=True, return_channel=True)
y_time, h_time = time_channel([x_time, no])
```

    
Next, we will compare the perfect CSI that we computed above using the ideal channel frequency response and the estimated channel response that we obtain from pilots with nearest-neighbor interpolation based on simulated transmissions in the time domain.

```python
[25]:
```

```python
# In the example above, we assumed perfect CSI, i.e.,
# h_hat correpsond to the exact ideal channel frequency response.
h_perf = h_hat[0,0,0,0,0,0]
# We now compute the LS channel estimate from the pilots.
h_est, _ = ls_est ([y, no])
h_est = h_est[0,0,0,0,0,0]
```
```python
[26]:
```

```python
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


### Downlink Transmission in the Frequency Domain<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#Downlink-Transmission-in-the-Frequency-Domain" title="Permalink to this headline"></a>
    
We will now simulate a simple downlink transmission in the frequency domain. In contrast to the uplink, the transmitter is now assumed to precode independent data streams to each antenna of the receiver based on perfect CSI.
    
The receiver can either estimate the channel or get access to the effective channel after precoding.
    
The first thing to do, is to change the `direction` within the CDL model. This makes the BS the transmitter and the UT the receiver.

```python
[27]:
```

```python
direction = "downlink"
cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=speed)
```

    
The following code shows the other necessary modifications:

```python
[28]:
```

```python
perfect_csi = True # Change to switch between perfect and imperfect CSI
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)
cir = cdl(batch_size, rg.num_ofdm_symbols, 1/rg.ofdm_symbol_duration)
h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
# Precode the transmit signal in the frequency domain
# It is here assumed that the transmitter has perfect knowledge of the channel
# One could here reduce this to perfect knowledge of the channel for the first
# OFDM symbol, or a noisy version of it to take outdated transmit CSI into account.
# `g` is the post-beamforming or `effective channel` that can be
# used to simulate perfect CSI at the receiver.
x_rg, g = zf_precoder([x_rg, h_freq])
y = channel_freq([x_rg, h_freq, no])
if perfect_csi:
    # The receiver gets here the effective channel after precoding as CSI
    h_hat, err_var = g, 0.
else:
    h_hat, err_var = ls_est ([y, no])
x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])
llr = demapper([x_hat, no_eff])
b_hat = decoder(llr)
ber = compute_ber(b, b_hat)
print("BER: {}".format(ber))
```


```python
BER: 0.0
```

    
We do not explain here on purpose how to model the downlink transmission in the time domain as it is a good exercise for the reader to do it her/himself. The key steps are:
 
- Sample the channel impulse response at the Nyquist rate.
- Downsample it to the OFDM symbol (+ cyclic prefix) rate (look at the uplink example).
- Convert the downsampled CIR to the frequency domain.
- Give this CSI to the transmitter for precoding.
- Convert the CIR to discrete-time to compute the channel output in the time domain.
