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
## Loading the channel covariance matrices
  
  

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

## Loading the channel covariance matrices<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/OFDM_MIMO_Detection.html#Loading-the-channel-covariance-matrices" title="Permalink to this headline"></a>
    
The next cell loads saved estimates of the time, frequency, and space covariance matrices.

```python
[9]:
```

```python
FREQ_COV_MAT = np.load('freq_cov_mat.npy')
TIME_COV_MAT = np.load('time_cov_mat.npy')
SPACE_COV_MAT = np.load('space_cov_mat.npy')
```

    
We then visualize the loaded matrices.
    
As one can see, the frequency correlation slowly decays with increasing spectral distance.
    
The time-correlation is much stronger as the mobility low. The covariance matrix is hence very badly conditioned with rank almost equal to one.
    
The spatial covariance matrix has a regular structure which is determined by the array geometry and polarization of its elements.

```python
[10]:
```

```python
fig, ax = plt.subplots(3,2, figsize=(10,12))
fig.suptitle("Time and frequency channel covariance matrices")
ax[0,0].set_title("Freq. cov. Real")
im = ax[0,0].imshow(FREQ_COV_MAT.real, vmin=-0.3, vmax=1.8)
ax[0,1].set_title("Freq. cov. Imag")
im = ax[0,1].imshow(FREQ_COV_MAT.imag, vmin=-0.3, vmax=1.8)
ax[1,0].set_title("Time cov. Real")
im = ax[1,0].imshow(TIME_COV_MAT.real, vmin=-0.3, vmax=1.8)
ax[1,1].set_title("Time cov. Imag")
im = ax[1,1].imshow(TIME_COV_MAT.imag, vmin=-0.3, vmax=1.8)
ax[2,0].set_title("Space cov. Real")
im = ax[2,0].imshow(SPACE_COV_MAT.real, vmin=-0.3, vmax=1.8)
ax[2,1].set_title("Space cov. Imag")
im = ax[2,1].imshow(SPACE_COV_MAT.imag, vmin=-0.3, vmax=1.8)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax);
```
```python
[10]:
```
```python
<matplotlib.colorbar.Colorbar at 0x7effa863f0d0>
```


