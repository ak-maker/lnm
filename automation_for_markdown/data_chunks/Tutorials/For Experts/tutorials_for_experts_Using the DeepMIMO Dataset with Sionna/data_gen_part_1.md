# Using the DeepMIMO Dataset with Sionna<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/DeepMIMO.html#Using-the-DeepMIMO-Dataset-with-Sionna" title="Permalink to this headline"></a>
    
In this example, you will learn how to use the ray-tracing based DeepMIMO dataset.
    
<a class="reference external" href="https://deepmimo.net/">DeepMIMO</a> is a generic dataset that enables a wide range of machine/deep learning applications for MIMO systems. It takes as input a set of parameters (such as antenna array configurations and time-domain/OFDM parameters) and generates MIMO channel realizations, corresponding locations, angles of arrival/departure, etc., based on these parameters and on a ray-tracing scenario selected <a class="reference external" href="https://deepmimo.net/scenarios/">from those available in DeepMIMO</a>.

# Table of Content
## GPU Configuration and Imports
## Configuration of DeepMIMO
## Using DeepMIMO with Sionna
  
  

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

## Configuration of DeepMIMO<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/DeepMIMO.html#Configuration-of-DeepMIMO" title="Permalink to this headline"></a>
    
DeepMIMO provides multiple <a class="reference external" href="https://deepmimo.net/scenarios/">scenarios</a> that one can select from. In this example, we use the O1 scenario with the carrier frequency set to 60 GHz (O1_60). To run this example, please download the “O1_60” data files <a class="reference external" href="https://deepmimo.net/scenarios/o1-scenario/">from this page</a>. The downloaded zip file should be extracted into a folder, and the parameter `DeepMIMO_params['dataset_folder']` should be set to point to this folder, as done below.
    
To use DeepMIMO with Sionna, the DeepMIMO dataset first needs to be generated. The generated DeepMIMO dataset contains channels for different locations of the users and basestations. The layout of the O1 scenario is shown in the figure below.
    
    
In this example, we generate a dataset that consists of channels for the links from the basestation 6 to the users located on the rows 400 to 450. Each of these rows consists of 181 user locations, resulting in $51 \times 181 = 9231$ basestation-user channels.
    
The antenna arrays in the DeepMIMO dataset are defined through the x-y-z axes. In the following example, a single-user MISO downlink is considered. The basestation is equipped with a uniform linear array of 16 elements spread along the x-axis. The users are each equipped with a single antenna. These parameters can be configured using the code below (for more information about the DeepMIMO parameters, please check <a class="reference external" href="https://deepmimo.net/versions/v2-python/">the DeepMIMO configurations</a>).

```python
[4]:
```

```python
# Import DeepMIMO
try:
    import DeepMIMO
except ImportError as e:
    # Install DeepMIMO if package is not already installed
    import os
    os.system("pip install DeepMIMO")
    import DeepMIMO
# Channel generation
DeepMIMO_params = DeepMIMO.default_params() # Load the default parameters
DeepMIMO_params['dataset_folder'] = r'./scenarios' # Path to the downloaded scenarios
DeepMIMO_params['scenario'] = 'O1_60' # DeepMIMO scenario
DeepMIMO_params['num_paths'] = 10 # Maximum number of paths
DeepMIMO_params['active_BS'] = np.array([6]) # Basestation indices to be included in the dataset
# Selected rows of users, whose channels are to be generated.
DeepMIMO_params['user_row_first'] = 400 # First user row to be included in the dataset
DeepMIMO_params['user_row_last'] = 450 # Last user row to be included in the dataset
# Configuration of the antenna arrays
DeepMIMO_params['bs_antenna']['shape'] = np.array([16, 1, 1]) # BS antenna shape through [x, y, z] axes
DeepMIMO_params['ue_antenna']['shape'] = np.array([1, 1, 1]) # UE antenna shape through [x, y, z] axes
# The OFDM_channels parameter allows choosing between the generation of channel impulse
# responses (if set to 0) or frequency domain channels (if set to 1).
# It is set to 0 for this simulation, as the channel responses in frequency domain
# will be generated using Sionna.
DeepMIMO_params['OFDM_channels'] = 0
# Generates a DeepMIMO dataset
DeepMIMO_dataset = DeepMIMO.generate_data(DeepMIMO_params)
```


```python

Basestation 6
UE-BS Channels
```

```python
Reading ray-tracing: 100%|██████████| 81450/81450 [00:00<00:00, 129737.29it/s]
Generating channels: 100%|██████████| 9231/9231 [00:00<00:00, 17426.44it/s]
```

```python

BS-BS Channels
```

```python
Reading ray-tracing: 100%|██████████| 6/6 [00:00<00:00, 33509.75it/s]
Generating channels: 100%|██████████| 1/1 [00:00<00:00, 2589.08it/s]
```
### Visualization of the dataset<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/DeepMIMO.html#Visualization-of-the-dataset" title="Permalink to this headline"></a>
    
To provide a better understanding of the user and basestation locations, we next visualize the locations of the users, highlighting the first active row of users (row 400), and basestation 6.

```python
[5]:
```

```python
plt.figure(figsize=(12,8))
## User locations
active_bs_idx = 0 # Select the first active basestation in the dataset
plt.scatter(DeepMIMO_dataset[active_bs_idx]['user']['location'][:, 1], # y-axis location of the users
         DeepMIMO_dataset[active_bs_idx]['user']['location'][:, 0], # x-axis location of the users
         s=1, marker='x', c='C0', label='The users located on the rows %i to %i (R%i to R%i)'%
           (DeepMIMO_params['user_row_first'], DeepMIMO_params['user_row_last'],
           DeepMIMO_params['user_row_first'], DeepMIMO_params['user_row_last']))
# First 181 users correspond to the first row
plt.scatter(DeepMIMO_dataset[active_bs_idx]['user']['location'][0:181, 1],
         DeepMIMO_dataset[active_bs_idx]['user']['location'][0:181, 0],
         s=1, marker='x', c='C1', label='First row of users (R%i)'% (DeepMIMO_params['user_row_first']))
## Basestation location
plt.scatter(DeepMIMO_dataset[active_bs_idx]['location'][1],
         DeepMIMO_dataset[active_bs_idx]['location'][0],
         s=50.0, marker='o', c='C2', label='Basestation')
plt.gca().invert_xaxis() # Invert the x-axis to align the figure with the figure above
plt.ylabel('x-axis')
plt.xlabel('y-axis')
plt.grid()
plt.legend();
```


## Using DeepMIMO with Sionna<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/DeepMIMO.html#Using-DeepMIMO-with-Sionna" title="Permalink to this headline"></a>
    
The DeepMIMO Python package provides <a class="reference external" href="https://nvlabs.github.io/sionna/examples/CIR_Dataset.html#Generators">a Sionna-compliant channel impulse response generator</a> that adapts the structure of the DeepMIMO dataset to be consistent with Sionna.
    
An adapter is instantiated for a given DeepMIMO dataset. In addition to the dataset, the adapter takes the indices of the basestations and users, to generate the channels between these basestations and users:
    
`DeepMIMOSionnaAdapter(DeepMIMO_dataset,` `bs_idx,` `ue_idx)`
    
**Note:** `bs_idx` and `ue_idx` set the links from which the channels are drawn. For instance, if `bs_idx` `=` `[0,` `1]` and `ue_idx` `=` `[2,` `3]`, the adapter then outputs the 4 channels formed by the combination of the first and second basestations with the third and fourth users.
    
The default behavior for `bs_idx` and `ue_idx` are defined as follows: - If value for `bs_idx` is not given, it will be set to `[0]` (i.e., the first basestation in the `DeepMIMO_dataset`). - If value for `ue_idx` is not given, then channels are provided for the links between the `bs_idx` and all users (i.e., `ue_idx=range(len(DeepMIMO_dataset[0]['user']['channel']))`. - If the both `bs_idx` and `ue_idx` are not given, the channels between the first basestation and all the
users are provided by the adapter. For this example, `DeepMIMOSionnaAdapter(DeepMIMO_dataset)` returns the channels from the basestation 6 and the 9231 available user locations.
    
**Note:** The adapter assumes basestations are transmitters and users are receivers. Uplink channels can be obtained using (transpose) reciprocity.

### Random Sampling of Multi-User Channels<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/DeepMIMO.html#Random-Sampling-of-Multi-User-Channels" title="Permalink to this headline"></a>
    
When considering multiple basestations, `bs_idx` can be set to a 2D numpy matrix of shape $($ # of samples $\times$ # of basestations per sample $)$. In this case, for each sample of basestations, the `DeepMIMOSionnaAdapter` returns a set of $($ # of basestations per sample $\times$ # of users $)$ channels, which can be provided as a multi-transmitter sample for the Sionna model. For example, `bs_idx` `=` `np.array([[0,` `1],` `[2,` `3],` `[4,` `5]])` provides three
sets of $($ 2 basestations $\times$ # of users $)$ channels. These three channel sets are from the basestation sets `[0,` `1]`, `[2,` `3]`, and `[4,` `5]`, respectively, to the users.
    
To use the adapter for multi-user channels, `ue_idx` can be set to a 2D numpy matrix of shape $($ # of samples $\times$ # of users per sample $)$. In this case, for each sample of users, the `DeepMIMOSionnaAdapter` returns a set of $($ # of basestations $\times$ # of users per sample $)$ channels, which can be provided as a multi-receiver sample for the Sionna model. For example, `ue_idx` `=` `np.array([[0,` `1` `,2],` `[4,` `5,` `6]])` provides two sets of $($
# of basestations $\times$ 3 users $)$ channels. These two channel sets are from the basestations to the user sets `[0,` `1,` `2]` and `[4,` `5,` `6]`, respectively.
    
In order to randomly sample channels from all the available user locations considering `num_rx` users, one may set `ue_idx` as in the following cell. In this example, the channels will be randomly chosen from the links between the basestation 6 and the 9231 available user locations.

```python
[6]:
```

```python
from DeepMIMO import DeepMIMOSionnaAdapter
# Number of receivers for the Sionna model.
# MISO is considered here.
num_rx = 1
# The number of UE locations in the generated DeepMIMO dataset
num_ue_locations = len(DeepMIMO_dataset[0]['user']['channel']) # 9231
# Pick the largest possible number of user locations that is a multiple of ``num_rx``
ue_idx = np.arange(num_rx*(num_ue_locations//num_rx))
# Optionally shuffle the dataset to not select only users that are near each others
np.random.shuffle(ue_idx)
# Reshape to fit the requested number of users
ue_idx = np.reshape(ue_idx, [-1, num_rx]) # In the shape of (floor(9231/num_rx) x num_rx)
DeepMIMO_Sionna_adapter = DeepMIMOSionnaAdapter(DeepMIMO_dataset, ue_idx=ue_idx)
```


