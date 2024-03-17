# Using the DeepMIMO Dataset with Sionna<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/DeepMIMO.html#Using-the-DeepMIMO-Dataset-with-Sionna" title="Permalink to this headline"></a>
    
In this example, you will learn how to use the ray-tracing based DeepMIMO dataset.
    
<a class="reference external" href="https://deepmimo.net/">DeepMIMO</a> is a generic dataset that enables a wide range of machine/deep learning applications for MIMO systems. It takes as input a set of parameters (such as antenna array configurations and time-domain/OFDM parameters) and generates MIMO channel realizations, corresponding locations, angles of arrival/departure, etc., based on these parameters and on a ray-tracing scenario selected <a class="reference external" href="https://deepmimo.net/scenarios/">from those available in DeepMIMO</a>.

# Table of Content
## GPU Configuration and Imports
## Link-level Simulations using Sionna and DeepMIMO
## DeepMIMO License and Citation
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/DeepMIMO.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

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
```
```python
[2]:
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
```
```python
[3]:
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import os
# Load the required Sionna components
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

## Link-level Simulations using Sionna and DeepMIMO<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/DeepMIMO.html#Link-level-Simulations-using-Sionna-and-DeepMIMO" title="Permalink to this headline"></a>
    
In the following cell, we define a Sionna model implementing the end-to-end link.
    
**Note:** The Sionna CIRDataset object shuffles the DeepMIMO channels provided by the adapter. Therefore, channel samples are passed through the model in a random order.

```python
[7]:
```

```python
class LinkModel(tf.keras.Model):
    def __init__(self,
                 DeepMIMO_Sionna_adapter,
                 carrier_frequency,
                 cyclic_prefix_length,
                 pilot_ofdm_symbol_indices,
                 subcarrier_spacing = 60e3,
                 batch_size = 64
                ):
        super().__init__()
        self._batch_size = batch_size
        self._cyclic_prefix_length = cyclic_prefix_length
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        # CIRDataset to parse the dataset
        self._CIR = sionna.channel.CIRDataset(DeepMIMO_Sionna_adapter,
                                              self._batch_size,
                                              DeepMIMO_Sionna_adapter.num_rx,
                                              DeepMIMO_Sionna_adapter.num_rx_ant,
                                              DeepMIMO_Sionna_adapter.num_tx,
                                              DeepMIMO_Sionna_adapter.num_tx_ant,
                                              DeepMIMO_Sionna_adapter.num_paths,
                                              DeepMIMO_Sionna_adapter.num_time_steps)
        # System parameters
        self._carrier_frequency = carrier_frequency
        self._subcarrier_spacing = subcarrier_spacing
        self._fft_size = 76
        self._num_ofdm_symbols = 14
        self._num_streams_per_tx = DeepMIMO_Sionna_adapter.num_rx
        self._dc_null = False
        self._num_guard_carriers = [0, 0]
        self._pilot_pattern = "kronecker"
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self._num_bits_per_symbol = 4
        self._coderate = 0.5
        # Setup the OFDM resource grid and stream management
        self._sm = StreamManagement(np.ones([DeepMIMO_Sionna_adapter.num_rx, 1], int), self._num_streams_per_tx)
        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing = self._subcarrier_spacing,
                                num_tx=DeepMIMO_Sionna_adapter.num_tx,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers,
                                dc_null=self._dc_null,
                                pilot_pattern=self._pilot_pattern,
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)
        # Components forming the link
        # Codeword length
        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        # Number of information bits per codeword
        self._k = int(self._n * self._coderate)
        # OFDM channel
        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)
        self._ofdm_channel = sionna.channel.GenerateOFDMChannel(self._CIR, self._rg, normalize_channel=True)
        self._channel_freq = ApplyOFDMChannel(add_awgn=True)
        # Transmitter
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)
        self._zf_precoder = ZFPrecoder(self._rg, self._sm, return_effective_channel=True)
        # Receiver
        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="lin_time_avg")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)
    @tf.function
    def call(self, batch_size, ebno_db):
        # Transmitter
        b = self._binary_source([self._batch_size, 1, self._num_streams_per_tx, self._k])
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)
        # Generate the OFDM channel
        h_freq = self._ofdm_channel()
        # Precoding
        x_rg, g = self._zf_precoder([x_rg, h_freq])
        # Apply OFDM channel
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        y = self._channel_freq([x_rg, h_freq, no])
        # Receiver
        h_hat, err_var = self._ls_est ([y, no])
        x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no])
        llr = self._demapper([x_hat, no_eff])
        b_hat = self._decoder(llr)
        return b, b_hat
```

    
We next evaluate the setup with different $E_b/N_0$ values to obtain BLER curves.

```python
[8]:
```

```python
sim_params = {
              "ebno_db": np.linspace(-7, -5.25, 10),
              "cyclic_prefix_length" : 0,
              "pilot_ofdm_symbol_indices" : [2, 11],
              }
batch_size = 64
model = LinkModel(DeepMIMO_Sionna_adapter=DeepMIMO_Sionna_adapter,
                  carrier_frequency=DeepMIMO_params['scenario_params']['carrier_freq'],
                  cyclic_prefix_length=sim_params["cyclic_prefix_length"],
                  pilot_ofdm_symbol_indices=sim_params["pilot_ofdm_symbol_indices"])
ber, bler = sim_ber(model,
                    sim_params["ebno_db"],
                    batch_size=batch_size,
                    max_mc_iter=100,
                    num_target_block_errors=100)
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -7.0 | 1.2077e-01 | 1.0000e+00 |       28196 |      233472 |          128 |         128 |         5.3 |reached target block errors
   -6.806 | 9.1514e-02 | 1.0000e+00 |       21366 |      233472 |          128 |         128 |         0.2 |reached target block errors
   -6.611 | 5.5651e-02 | 9.6094e-01 |       12993 |      233472 |          123 |         128 |         0.2 |reached target block errors
   -6.417 | 2.3723e-02 | 7.2396e-01 |        8308 |      350208 |          139 |         192 |         0.3 |reached target block errors
   -6.222 | 5.3968e-03 | 3.5938e-01 |        3150 |      583680 |          115 |         320 |         0.4 |reached target block errors
   -6.028 | 8.9899e-04 | 9.2014e-02 |        1889 |     2101248 |          106 |        1152 |         1.6 |reached target block errors
   -5.833 | 9.4144e-05 | 1.3437e-02 |        1099 |    11673600 |           86 |        6400 |         8.6 |reached max iter
   -5.639 | 4.2832e-07 | 3.1250e-04 |           5 |    11673600 |            2 |        6400 |         8.6 |reached max iter
   -5.444 | 1.1993e-06 | 1.5625e-04 |          14 |    11673600 |            1 |        6400 |         8.5 |reached max iter
    -5.25 | 0.0000e+00 | 0.0000e+00 |           0 |    11673600 |            0 |        6400 |         8.5 |reached max iter
Simulation stopped as no error occurred @ EbNo = -5.2 dB.

```
```python
[9]:
```

```python
plt.figure(figsize=(12,8))
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.semilogy(sim_params["ebno_db"], bler)
```
```python
[9]:
```
```python
[<matplotlib.lines.Line2D at 0x7fe0b00c13a0>]
```


## DeepMIMO License and Citation<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/DeepMIMO.html#DeepMIMO-License-and-Citation" title="Permalink to this headline"></a>
<ol class="upperalpha simple">
- Alkhateeb, “<a class="reference external" href="https://arxiv.org/pdf/1902.06435.pdf">DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications</a>,” in Proc. of Information Theory and Applications Workshop (ITA), San Diego, CA, Feb. 2019.
</ol>
    
To use the DeepMIMO dataset, please check the license information <a class="reference external" href="https://deepmimo.net/license/">here</a>.