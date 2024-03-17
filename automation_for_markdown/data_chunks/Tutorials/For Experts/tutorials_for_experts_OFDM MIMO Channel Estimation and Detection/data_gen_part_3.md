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
## Comparison of OFDM estimators
  
  

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

## Comparison of OFDM estimators<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/OFDM_MIMO_Detection.html#Comparison-of-OFDM-estimators" title="Permalink to this headline"></a>
    
This section focuses on comparing the available OFDM channel estimators in Sionna for the considered setup.
    
OFDM channel estimation consists of two steps:
<ol class="arabic simple">
- Channel estimation at pilot-carrying resource elements using <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LSChannelEstimator">least-squares (LS)</a>.
- Interpolation for data-carrying resource elements, for which three methods are available in Sionna:
</ol>
 
- <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.NearestNeighborInterpolator">Nearest-neighbor</a>, which uses the channel estimate of the nearest pilot
- <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LinearInterpolator">Linear</a>, with optional averaging over the OFDM symbols (time dimension) for low mobility scenarios
- <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEInterpolator">LMMSE</a>, which requires knowledge of the time and frequency covariance matrices

    
The LMMSE interpolator also features optional spatial smoothin, which requires the spatial covarance matrix. The <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEInterpolator">API documentation</a> explains in more detail how this interpolator operates.

### End-to-end model<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/OFDM_MIMO_Detection.html#End-to-end-model" title="Permalink to this headline"></a>
    
In the next cell, we will create a Keras model which uses the interpolation method specified at initialization.
    
It computes the mean square error (MSE) for a specified batch size and signal-to-noise ratio (SNR) (in dB).
    
The following interpolation methods are available (set through the `int_method` parameter):
 
- `"nn"` : Nearest-neighbor interpolation
- `"lin"` : Linear interpolation
- `"lmmse"` : LMMSE interpolation

    
When LMMSE interpolation is used, it is required to specified the order in which interpolation and optional spatial smoothing is performed. This is achieved using the `lmmse_order` parameter. For example, setting this parameter to `"f-t"` leads to frequency interpolation being performed first followed by time interpolation, and no spatial smoothing. Setting it to `"t-f-s"` leads to time interpolation being performed first, followed by frequency interpolation, and finally spatial smoothing.

```python
[11]:
```

```python
class MIMOOFDMLink(Model):
    def __init__(self, int_method, lmmse_order=None, **kwargs):
        super().__init__(kwargs)
        assert int_method in ('nn', 'lin', 'lmmse')

        # Configure the resource grid
        rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS,
                          fft_size=FFT_SIZE,
                          subcarrier_spacing=SUBCARRIER_SPACING,
                          num_tx=1,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2,11])
        self.rg = rg
        # Stream management
        # Only a sinlge UT is considered for channel estimation
        sm = StreamManagement([[1]], 1)
        ##################################
        # Transmitter
        ##################################
        self.qam_source = QAMSource(num_bits_per_symbol=2) # Modulation order does not impact the channel estimation. Set to QPSK
        self.rg_mapper = ResourceGridMapper(rg)
        ##################################
        # Channel
        ##################################
        self.channel = OFDMChannel(CHANNEL_MODEL, rg, return_channel=True)
        ###################################
        # Receiver
        ###################################
        # Channel estimation
        freq_cov_mat = tf.constant(FREQ_COV_MAT, tf.complex64)
        time_cov_mat = tf.constant(TIME_COV_MAT, tf.complex64)
        space_cov_mat = tf.constant(SPACE_COV_MAT, tf.complex64)
        if int_method == 'nn':
            self.channel_estimator = LSChannelEstimator(rg, interpolation_type='nn')
        elif int_method == 'lin':
            self.channel_estimator = LSChannelEstimator(rg, interpolation_type='lin')
        elif int_method == 'lmmse':
            lmmse_int_freq_first = LMMSEInterpolator(rg.pilot_pattern, time_cov_mat, freq_cov_mat, space_cov_mat, order=lmmse_order)
            self.channel_estimator = LSChannelEstimator(rg, interpolator=lmmse_int_freq_first)
    @tf.function
    def call(self, batch_size, snr_db):

        ##################################
        # Transmitter
        ##################################
        x = self.qam_source([batch_size, 1, 1, self.rg.num_data_symbols])
        x_rg = self.rg_mapper(x)
        ##################################
        # Channel
        ##################################
        no = tf.pow(10.0, -snr_db/10.0)
        topology = gen_single_sector_topology(batch_size, 1, 'umi', min_ut_velocity=SPEED, max_ut_velocity=SPEED)
        CHANNEL_MODEL.set_topology(*topology)
        y_rg, h_freq = self.channel((x_rg, no))
        ###################################
        # Channel estimation
        ###################################
        h_hat,_ = self.channel_estimator((y_rg,no))
        ###################################
        # MSE
        ###################################
        mse = tf.reduce_mean(tf.square(tf.abs(h_freq-h_hat)))
        return mse
```

    
The next cell defines a function for evaluating the mean square error (MSE) of a `model` over a range of SNRs (`snr_dbs`).
    
The `batch_size` and `num_it` parameters control the number of samples used to compute the MSE for each SNR value.

```python
[12]:
```

```python
def evaluate_mse(model, snr_dbs, batch_size, num_it):
    # Casting model inputs to TensorFlow types to avoid
    # re-building of the graph
    snr_dbs = tf.cast(snr_dbs, tf.float32)
    batch_size = tf.cast(batch_size, tf.int32)
    mses = []
    for snr_db in snr_dbs:
        mse_ = 0.0
        for _ in range(num_it):
            mse_ += model(batch_size, snr_db).numpy()
        # Averaging over the number of iterations
        mse_ /= float(num_it)
        mses.append(mse_)
    return mses
```

    
The next cell defines the evaluation parameters.

```python
[13]:
```

```python
# Range of SNR (in dB)
SNR_DBs = np.linspace(-10.0, 20.0, 20)
# Number of iterations and batch size.
# These parameters control the number of samples used to compute each SNR value.
# The higher the number of samples is, the more accurate the MSE estimation is, at
# the cost of longer compute time.
BATCH_SIZE = 512
NUM_IT = 10
# Interpolation/filtering order for the LMMSE interpolator.
# All valid configurations are listed.
# Some are commented to speed-up simulations.
# Uncomment configurations to evaluate them!
ORDERS = ['s-t-f', # Space - time - frequency
          #'s-f-t', # Space - frequency - time
          #'t-s-f', # Time - space - frequency
          't-f-s', # Time - frequency - space
          #'f-t-s', # Frequency - time - space
          #'f-s-t', # Frequency - space- time
          #'f-t',   # Frequency - time (no spatial smoothing)
          't-f'   # Time - frequency (no spatial smoothing)
          ]
```

    
The next cell evaluates the nearest-neighbor, linear, and LMMSE interpolator. For the LMMSE interpolator, we loop through the configuration listed in `ORDERS`.

```python
[14]:
```

```python
MSES = {}
# Nearest-neighbor interpolation
e2e = MIMOOFDMLink("nn")
MSES['nn'] = evaluate_mse(e2e, SNR_DBs, BATCH_SIZE, NUM_IT)
# Linear interpolation
e2e = MIMOOFDMLink("lin")
MSES['lin'] = evaluate_mse(e2e, SNR_DBs, BATCH_SIZE, NUM_IT)
# LMMSE
for order in ORDERS:
    e2e = MIMOOFDMLink("lmmse", order)
    MSES[f"lmmse: {order}"] = evaluate_mse(e2e, SNR_DBs, BATCH_SIZE, NUM_IT)

```


```python
WARNING:tensorflow:From /home/faycal/.local/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:1176: calling gather (from tensorflow.python.ops.array_ops) with validate_indices is deprecated and will be removed in a future version.
Instructions for updating:
The `validate_indices` argument has no effect. Indices are always validated on CPU and never validated on GPU.
```

    
Finally, we plot the MSE.

```python
[15]:
```

```python
plt.figure(figsize=(8,6))
for est_label in MSES:
    plt.semilogy(SNR_DBs, MSES[est_label], label=est_label)
plt.xlabel(r"SNR (dB)")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
```


    
Unsurprisingly, the LMMSE interpolator leads to more accurate estimates compared to the two other methods, as it leverages knowledge of the the channel statistics. Moreover, the order in which the LMMSE interpolation steps are performed strongly impacts the accuracy of the estimator. This is because the LMMSE interpolation operates in one dimension at a time which is not equivalent to full-blown LMMSE estimation across all dimensions at one.
    
Also note that the order that leads to the best accuracy depends on the channel statistics. As a rule of thumb, it might be good to start with the dimension that is most strongly correlated (i.e., time in our example).
