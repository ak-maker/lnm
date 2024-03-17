# 5G NR PUSCH Tutorial<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html#5G-NR-PUSCH-Tutorial" title="Permalink to this headline"></a>
    
This notebook provides an introduction to Sionna’s <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html">5G New Radio (NR) module</a> and, in particular, the <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#pusch">physical uplink shared channel (PUSCH)</a>. This module provides implementations of a small subset of the physical layer functionalities as described in the 3GPP specifications <a class="reference external" href="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3213">38.211</a>,
<a class="reference external" href="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3214">38.212</a> and <a class="reference external" href="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3216">38.214</a>.
    
You will
 
- Get an understanding of the different components of a PUSCH configuration, such as the carrier, DMRS, and transport block,
- Learn how to rapidly simulate PUSCH transmissions for multiple transmitters,
- Modify the PUSCHReceiver to use a custom MIMO Detector.
# Table of Content
## GPU Configuration and Imports
## Looking into the PUSCHTransmitter
## Components of the PUSCHReceiver
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

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
# Load the required Sionna components
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.channel import AWGN, RayleighBlockFading, OFDMChannel, TimeChannel, time_lag_discrete_time_channel
from sionna.channel.tr38901 import AntennaArray, UMi, UMa, RMa
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.utils import compute_ber, ebnodb2no, sim_ber
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement
```
```python
[2]:
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time
```
```python
[3]:
```

```python
import tensorflow as tf
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
```

## Looking into the PUSCHTransmitter<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html#Looking-into-the-PUSCHTransmitter" title="Permalink to this headline"></a>
    
We have used the `PUSCHTransmitter` class already multiple times without speaking about what it actually does. In short, it generates for every configured transmitter a batch of random information bits of length `pusch_config.tb_size` and outputs either a frequency fo time-domain representation of the transmitted OFDM waveform from each of the antenna ports of each transmitter.
    
However, under the hood it implements the sequence of layers shown in the following figure:
    
    
Information bits are either randomly generated or provided as input and then encoded into a transport block by the <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder">TBEncoder</a>. The encoded bits are then mapped to QAM constellation symbols by the <a class="reference external" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Mapper">Mapper</a>. The <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper">LayerMapper</a> splits the modulated symbols into different layers which are
then mapped onto OFDM resource grids by the <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGridMapper">ResourceGridMapper</a>. If precoding is enabled in the <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig">PUSCHConfig</a>, the resource grids are further precoded by the <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPrecoder">PUSCHPrecoder</a> so that there is one for each transmitter and antenna port. If `output_domain` equals “freq”, these
are the ouputs x . If `output_domain` is chosen to be “time”, the resource grids are transformed into time-domain signals by the <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMModulator">OFDMModulator</a>.
    
Let us configure a `PUSCHTransmitter` from a list of two `PUSCHConfig` and inspect the output shapes:

```python
[29]:
```

```python
pusch_config = PUSCHConfig()
pusch_config.num_antenna_ports = 4
pusch_config.num_layers = 2
pusch_config.dmrs.dmrs_port_set = [0,1]
pusch_config.precoding = "codebook"
pusch_config.tpmi = 7
pusch_config_1 = pusch_config.clone()
pusch_config.dmrs.dmrs_port_set = [2,3]
pusch_transmitter = PUSCHTransmitter([pusch_config, pusch_config_1])
batch_size = 32
x, b = pusch_transmitter(batch_size)
# b has shape [batch_size, num_tx, tb_size]
print("Shape of b:", b.shape)
# x has shape [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
print("Shape of x:", x.shape)
```


```python
Shape of b: (32, 2, 2728)
Shape of x: (32, 2, 4, 14, 48)
```

    
If you want to transmit a custom payload, you simply need to deactive the `return_bits` flag when creating the transmitter:

```python
[30]:
```

```python
pusch_transmitter = PUSCHTransmitter([pusch_config, pusch_config_1], return_bits=False)
x_2 = pusch_transmitter(b)
assert np.array_equal(x, x_2) # Check that we get the same output for the payload b generated above
```

    
By default, the `PUSCHTransmitter` generates frequency-domain outputs. If you want to make time-domain simulations, you need to configure the `output_domain` during the initialization:

```python
[31]:
```

```python
pusch_transmitter = PUSCHTransmitter([pusch_config, pusch_config_1], output_domain="time", return_bits=False)
x_time = pusch_transmitter(b)
# x has shape [batch_size, num_tx, num_tx_ant, num_time_samples]
print("Shape of x:", x_time.shape)
```


```python
Shape of x: (32, 2, 4, 728)
```

    
The last dimension of the output signal correspond to the total number of time-domain samples which can be computed in the following way:

```python
[32]:
```

```python
(pusch_transmitter.resource_grid.cyclic_prefix_length  \
 + pusch_transmitter.resource_grid.fft_size) \
* pusch_transmitter.resource_grid.num_ofdm_symbols
```
```python
[32]:
```
```python
728
```
## Components of the PUSCHReceiver<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html#Components-of-the-PUSCHReceiver" title="Permalink to this headline"></a>
    
The <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHReceiver">PUSCHReceiver</a> is the counter-part to the <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter">PUSCHTransmitter</a> as it <em>simply</em> recovers the transmitted information bits from received waveform. It combines multiple processing blocks in a single layer as shown in the following figure:
    
    
If the `input_domain` equals “time”, the inputs $\mathbf{y}$ are first transformed to resource grids with the <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMModulator">OFDMDemodulator</a>. Then channel estimation is performed, e.g., with the help of the <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHLSChannelEstimator">PUSCHLSChannelEstimator</a>. If `channel_estimator` is chosen to be “perfect”, this step is skipped and the input $\mathbf{h}$ is used
instead. Next, MIMO detection is carried out with an arbitrary <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMDetector">OFDMDetector</a>. The resulting LLRs for each layer are then combined to transport blocks with the help of the <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerDemapper">LayerDemapper</a>. Finally, the transport blocks are decoded with the <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder">TBDecoder</a>.
    
If we instantiate a `PUSCHReceiver` as done in the next cell, default implementations of all blocks as described in the <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHReceiver">API documentation</a> are used.

```python
[33]:
```

```python
pusch_receiver = PUSCHReceiver(pusch_transmitter)
pusch_receiver._mimo_detector
```
```python
[33]:
```
```python
<sionna.ofdm.detection.LinearDetector at 0x7f86f28c72b0>
```

    
We can also provide custom implementations for each block by providing them as keyword arguments during initialization. In the folllwing code snippet, we first create an instance of the <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.KBestDetector">KBestDetector</a>, which is then used as MIMO detector in the `PUSCHReceiver`.

```python
[34]:
```

```python
# Create a new PUSCHTransmitter
pusch_transmitter = PUSCHTransmitter([pusch_config, pusch_config_1])
# Create a StreamManagement instance
rx_tx_association = np.ones([1, pusch_transmitter.resource_grid.num_tx], bool)
stream_management = StreamManagement(rx_tx_association,
                                     pusch_config.num_layers)
# Get relevant parameters for the detector
num_streams = pusch_transmitter.resource_grid.num_tx \
              * pusch_transmitter.resource_grid.num_streams_per_tx
k = 32 # Number of canditates for K-Best detection
k_best = KBestDetector("bit", num_streams, k,
                       pusch_transmitter.resource_grid,
                       stream_management,
                       "qam", pusch_config.tb.num_bits_per_symbol)
# Create a PUSCHReceiver using the KBest detector
pusch_receiver = PUSCHReceiver(pusch_transmitter, mimo_detector=k_best)
```

    
Next, we test if this receiver works over a simple Rayleigh block fading channel:

```python
[35]:
```

```python
num_rx_ant = 16
rayleigh = RayleighBlockFading(num_rx=1,
                               num_rx_ant=num_rx_ant,
                               num_tx=pusch_transmitter.resource_grid.num_tx,
                               num_tx_ant=pusch_config.num_antenna_ports)
channel = OFDMChannel(rayleigh,
                      pusch_transmitter.resource_grid,
                      add_awgn=True,
                      normalize_channel=True)
x, b = pusch_transmitter(32)
no = 0.1
y = channel([x, no])
b_hat = pusch_receiver([y, no])
print("BER:", compute_ber(b, b_hat).numpy())
```


```python
BER: 0.0
```
