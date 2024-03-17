# OFDM MIMO Channel Estimation and Detection<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/OFDM_MIMO_Detection.html#OFDM-MIMO-Channel-Estimation-and-Detection" title="Permalink to this headline"></a>
    
In this notebook, we will evaluate some of the OFDM channel estimation and MIMO detection algorithms available in Sionna.
    
We will start by evaluating the mean square error (MSE) preformance of various channel estimation and interpolation methods.
    
Then, we will compare some of the MIMO detection algorithms under both perfect and imperfect channel state information (CSI) in terms of uncoded symbol error rate (SER) and coded bit error rate (BER).
    
The developed end-to-end Keras models in this notebook are a great tool for benchmarking of MIMO receivers under realistic conditions. They can be easily extended to new channel estimation methods or MIMO detection algorithms.
    
For MSE evaluations, the block diagram of the system looks as follows:
    
    
where the channel estimation module is highlighted as it is the focus of this evaluation. The channel covariance matrices are required for linear minimum mean square error (LMMSE) channel interpolation.
    
For uncoded SER evaluations, the block diagram of the system looks as follows:
    
    
where the channel estimation and detection modules are highlighted as they are the focus of this evaluation.
    
Finally, for coded BER evaluations, the block diagram of the system looks as follows:
    

# Table of Content
## GPU Configuration and Imports
## Simulation parameters
## Estimation of the channel time, frequency, and spatial covariance matrices
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/OFDM_MIMO_Detection.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

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
from sionna.mimo import StreamManagement
from sionna.utils import QAMSource, compute_ser, BinarySource, sim_ber, ebnodb2no, QAMSource
from sionna.mapping import Mapper
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEInterpolator, LinearDetector, KBestDetector, EPDetector, MMSEPICDetector
from sionna.channel import GenerateOFDMChannel, OFDMChannel, gen_single_sector_topology
from sionna.channel.tr38901 import UMi, Antenna, PanelArray
from sionna.fec.ldpc import LDPC5GEncoder
from sionna.fec.ldpc import LDPC5GDecoder
```

## Simulation parameters<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/OFDM_MIMO_Detection.html#Simulation-parameters" title="Permalink to this headline"></a>
    
The next cell defines the simulation parameters used throughout this notebook.
    
This includes the OFDM waveform parameters, <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray">antennas geometries and patterns</a>, and the <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi">3GPP UMi channel model</a>.

```python
[3]:
```

```python
NUM_OFDM_SYMBOLS = 14
FFT_SIZE = 12*4 # 4 PRBs
SUBCARRIER_SPACING = 30e3 # Hz
CARRIER_FREQUENCY = 3.5e9 # Hz
SPEED = 3. # m/s
# The user terminals (UTs) are equipped with a single antenna
# with vertial polarization.
UT_ANTENNA = Antenna(polarization='single',
                     polarization_type='V',
                     antenna_pattern='omni', # Omnidirectional antenna pattern
                     carrier_frequency=CARRIER_FREQUENCY)
# The base station is equipped with an antenna
# array of 8 cross-polarized antennas,
# resulting in a total of 16 antenna elements.
NUM_RX_ANT = 16
BS_ARRAY = PanelArray(num_rows_per_panel=4,
                      num_cols_per_panel=2,
                      polarization='dual',
                      polarization_type='cross',
                      antenna_pattern='38.901', # 3GPP 38.901 antenna pattern
                      carrier_frequency=CARRIER_FREQUENCY)
# 3GPP UMi channel model is considered
CHANNEL_MODEL = UMi(carrier_frequency=CARRIER_FREQUENCY,
                    o2i_model='low',
                    ut_array=UT_ANTENNA,
                    bs_array=BS_ARRAY,
                    direction='uplink',
                    enable_shadow_fading=False,
                    enable_pathloss=False)
```

## Estimation of the channel time, frequency, and spatial covariance matrices<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/OFDM_MIMO_Detection.html#Estimation-of-the-channel-time,-frequency,-and-spatial-covariance-matrices" title="Permalink to this headline"></a>
    
The linear minimum mean square (LMMSE) interpolation method requires knowledge of the time (i.e., across OFDM symbols), frequency (i.e., across sub-carriers), and spatial (i.e., across receive antennas) covariance matrices of the channel frequency response.
    
These are estimated in this section using Monte Carlo sampling.
    
We explain below how this is achieved for the frequency covariance matrix. The same approach is used for the time and spatial covariance matrices.
    
Let $N$ be the number of sub-carriers. The first step for estimating the frequency covariance matrix is to sample the channel model in order to build a set of frequency-domain channel realizations $\left\{ \mathbf{h}_k \right\}, 1 \leq k \leq K$, where $K$ is the number of samples and $\mathbf{h}_k \in \mathbb{C}^{N}$ are complex-valued samples of the channel frequency response.
    
The frequency covariance matrix $\mathbf{R}^{(f)} \in \mathbb{C}^{N \times N}$ is then estimated by
    
\begin{equation}
\mathbf{R}^{(f)} \approx \frac{1}{K} \sum_{k = 1}^K \mathbf{h}_k \mathbf{h}_k^{\mathrm{H}}
\end{equation}
    
where we assume that the frequency-domain channel response has zero mean.
    
The following cells implement this process for all three dimensions (frequency, time, and space).
    
The next cell defines a <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid">resource grid</a> and an <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateOFDMChannel">OFDM channel generator</a> for sampling the channel in the frequency domain.

```python
[4]:
```

```python
rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS,
                  fft_size=FFT_SIZE,
                  subcarrier_spacing=SUBCARRIER_SPACING)
channel_sampler = GenerateOFDMChannel(CHANNEL_MODEL, rg)
```

    
Then, a function that samples the channel is defined. It randomly samples a network topology for every batch and for every batch example using the <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.gen_single_sector_topology">appropriate utility function</a>.

```python
[5]:
```

```python
def sample_channel(batch_size):
    # Sample random topologies
    topology = gen_single_sector_topology(batch_size, 1, 'umi', min_ut_velocity=SPEED, max_ut_velocity=SPEED)
    CHANNEL_MODEL.set_topology(*topology)
    # Sample channel frequency responses
    # [batch size, 1, num_rx_ant, 1, 1, num_ofdm_symbols, fft_size]
    h_freq = channel_sampler(batch_size)
    # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
    h_freq = h_freq[:,0,:,0,0]
    return h_freq
```

    
We now define a function that estimates the frequency, time, and spatial covariance matrcies using Monte Carlo sampling.

```python
[6]:
```

```python
@tf.function(jit_compile=True) # Use XLA for speed-up
def estimate_covariance_matrices(num_it, batch_size):
    freq_cov_mat = tf.zeros([FFT_SIZE, FFT_SIZE], tf.complex64)
    time_cov_mat = tf.zeros([NUM_OFDM_SYMBOLS, NUM_OFDM_SYMBOLS], tf.complex64)
    space_cov_mat = tf.zeros([NUM_RX_ANT, NUM_RX_ANT], tf.complex64)
    for _ in tf.range(num_it):
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        h_samples = sample_channel(batch_size)
        #################################
        # Estimate frequency covariance
        #################################
        # [batch size, num_rx_ant, fft_size, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0,1,3,2])
        # [batch size, num_rx_ant, fft_size, fft_size]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [fft_size, fft_size]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0,1))
        # [fft_size, fft_size]
        freq_cov_mat += freq_cov_mat_
        ################################
        # Estimate time covariance
        ################################
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0,1))
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat += time_cov_mat_
        ###############################
        # Estimate spatial covariance
        ###############################
        # [batch size, num_ofdm_symbols, num_rx_ant, fft_size]
        h_samples_ = tf.transpose(h_samples, [0,2,1,3])
        # [batch size, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=(0,1))
        # [num_rx_ant, num_rx_ant]
        space_cov_mat += space_cov_mat_
    freq_cov_mat /= tf.complex(tf.cast(NUM_OFDM_SYMBOLS*num_it, tf.float32), 0.0)
    time_cov_mat /= tf.complex(tf.cast(FFT_SIZE*num_it, tf.float32), 0.0)
    space_cov_mat /= tf.complex(tf.cast(FFT_SIZE*num_it, tf.float32), 0.0)
    return freq_cov_mat, time_cov_mat, space_cov_mat
```

    
We then compute the estimates by executing the function defined in the previous cell.
    
The batch size and number of iterations determine the total number of samples, i.e.,
```python
number of samples = batch_size x num_iterations
```

    
and hence control the tradeoff between the accuracy of the estimates and the time needed for their computation.

```python
[7]:
```

```python
batch_size = 1000
num_iterations = 100
sionna.Config.xla_compat = True # Enable Sionna's support of XLA
FREQ_COV_MAT, TIME_COV_MAT, SPACE_COV_MAT = estimate_covariance_matrices(batch_size, num_iterations)
sionna.Config.xla_compat = False # Disable Sionna's support of XLA
```

    
Finally, the estimated matrices are saved (as numpy arrays) for future use.

```python
[8]:
```

```python
# FREQ_COV_MAT : [fft_size, fft_size]
# TIME_COV_MAT : [num_ofdm_symbols, num_ofdm_symbols]
# SPACE_COV_MAT : [num_rx_ant, num_rx_ant]
np.save('freq_cov_mat', FREQ_COV_MAT.numpy())
np.save('time_cov_mat', TIME_COV_MAT.numpy())
np.save('space_cov_mat', SPACE_COV_MAT.numpy())
```

