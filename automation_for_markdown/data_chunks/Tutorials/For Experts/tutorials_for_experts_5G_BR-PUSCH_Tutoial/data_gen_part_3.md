# 5G NR PUSCH Tutorial<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html#5G-NR-PUSCH-Tutorial" title="Permalink to this headline"></a>
    
This notebook provides an introduction to Sionna’s <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html">5G New Radio (NR) module</a> and, in particular, the <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#pusch">physical uplink shared channel (PUSCH)</a>. This module provides implementations of a small subset of the physical layer functionalities as described in the 3GPP specifications <a class="reference external" href="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3213">38.211</a>,
<a class="reference external" href="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3214">38.212</a> and <a class="reference external" href="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3216">38.214</a>.
    
You will
 
- Get an understanding of the different components of a PUSCH configuration, such as the carrier, DMRS, and transport block,
- Learn how to rapidly simulate PUSCH transmissions for multiple transmitters,
- Modify the PUSCHReceiver to use a custom MIMO Detector.
# Table of Content
## GPU Configuration and Imports
## Understanding the DMRS Configuration
### How to control the number of available DMRS ports?
## Transport Blocks and MCS
  
  

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

### How to control the number of available DMRS ports?<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html#How-to-control-the-number-of-available-DMRS-ports?" title="Permalink to this headline"></a>
    
There are two factors that determine the available number of DMRS ports, i.e., layers, that can be transmitted. The first is the DMRS Configuration and the second the length of a DMRS symbol. Both parameters can take to values so that there are four options in total. In the previous example, the DMRS Configuration Type 1 was used. In this case, there are two CDM groups and each groups uses either odd or even subcarriers. This leads to four available DMRS ports. With DMRS Configuration Type 2,
there are three CDM groups and each group uses two pairs of adjacent subcarriers per PRB, i.e., four pilot-carrying subcarriers. That means that there are six available DMRS ports.

```python
[19]:
```

```python
pusch_config.dmrs.config_type = 2
PUSCHTransmitter(pusch_config).pilot_pattern.show(stream_ind = 0);
print("Available DMRS ports:", pusch_config.dmrs.allowed_dmrs_ports)
```


```python
Available DMRS ports: [0, 1, 2, 3]
```


    
In the above figure, you can see that the pilot pattern has become sparser in the frequency domain. However, there are still only four available DMRS ports. This is because we now need to mask also the resource elements that are used by the third CDM group. This can be done by setting the parameter `NumCDMGroupsWithoutData` equal to three.

```python
[20]:
```

```python
pusch_config.dmrs.num_cdm_groups_without_data = 3
PUSCHTransmitter(pusch_config).pilot_pattern.show(stream_ind = 0);
print("Available antenna ports:", pusch_config.dmrs.allowed_dmrs_ports)
```


```python
Available antenna ports: [0, 1, 2, 3, 4, 5]
```


    
The second parameter that controls the number of available DMRS ports is the `length`, which can be equal to either one or two. Let’s see what happens when we change it to two.

```python
[21]:
```

```python
pusch_config.n_size_bwp = 1 # We reduce the bandwidth to one PRB for better visualization
pusch_config.dmrs.length = 2
PUSCHTransmitter(pusch_config).pilot_pattern.show(stream_ind = 0);
print("Available DMRS ports:", pusch_config.dmrs.allowed_dmrs_ports)
```


```python
Available DMRS ports: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```


    
The pilot pattern is now composed of four 2x2 blocks within a PRB. These blocks are used by the four DMRS ports within the same CDM group. This means that we can now support up to twelve layers!
    
Let’s create a setup with three transmitters, each sending four layers using four antenna ports. We choose the DMRS ports for each transmitters such that they belong to the CDM group. This is not necessary and you are free to choose any desired allocation. It is however important to understand, thet for channel estimation to work, the channel is supposed to be static over 2x2 blocks of resource elements. This is in general the case for low mobility scenarios and channels with not too large delay
spread. You can see from the results below that the pilot sequences of the DMRS ports in the same CDM group are indeed orthogonal over the 2x2 blocks.

```python
[22]:
```

```python
pusch_config = PUSCHConfig()
pusch_config.n_size_bwp = 1
pusch_config.dmrs.config_type = 2
pusch_config.dmrs.length = 2
pusch_config.dmrs.additional_position = 1
pusch_config.dmrs.num_cdm_groups_without_data = 3
pusch_config.num_antenna_ports = 4
pusch_config.num_layers = 4
pusch_config.dmrs.dmrs_port_set = [0,1,6,7]
pusch_config.precoding = "codebook"
pusch_config.tpmi = 4
pusch_config_1 = pusch_config.clone()
pusch_config_1.dmrs.dmrs_port_set = [2,3,8,9]
pusch_config_2 = pusch_config.clone()
pusch_config_2.dmrs.dmrs_port_set = [4,5,10,11]
pusch_transmitter_multi = PUSCHTransmitter([pusch_config, pusch_config_1, pusch_config_2])
```
```python
[23]:
```

```python
# Extract the first 2x2 block of pilot symbols for all DMRS ports of the first transmitter
p = pusch_transmitter_multi.pilot_pattern.pilots[0].numpy()
p = np.matrix(p[:, [0,1,12,13]])
# Test that these pilot sequences are mutually orthogonal
# The result should be a boolean identity matrix
np.abs(p*p.getH())>1e-6
```
```python
[23]:
```
```python
matrix([[ True, False, False, False],
        [False,  True, False, False],
        [False, False,  True, False],
        [False, False, False,  True]])
```

    
There are several other parameters that impact the pilot patterns. The full DMRS configuration can be displayed with the following command. We refer to the <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#puschdmrsconfig">API documentation of the PUSCHDMRSConfig class</a> for further details.

```python
[24]:
```

```python
pusch_config.dmrs.show()
```


```python
PUSCH DMRS Configuration
========================
additional_position : 1
allowed_dmrs_ports : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
beta : 1.7320508075688772
cdm_groups : [0, 0, 0, 0]
config_type : 2
deltas : [0, 0, 0, 0]
dmrs_port_set : [0, 1, 6, 7]
length : 2
n_id : None
n_scid : 0
num_cdm_groups_without_data : 3
type_a_position : 2
w_f : [[ 1  1  1  1]
 [ 1 -1  1 -1]]
w_t : [[ 1  1  1  1]
 [ 1  1 -1 -1]]

```

## Transport Blocks and MCS<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html#Transport-Blocks-and-MCS" title="Permalink to this headline"></a>
    
The modulation and coding scheme (MCS) is set in 5G NR via the MCS index and MCS table which are properties of transport block configuration <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig">TBConfig</a>. When you create an instance of `PUSCHConfig`, a default instance of `TBConfig` is created. It can be accessed via the following command:

```python
[25]:
```

```python
pusch_config = PUSCHConfig()
pusch_config.tb.show()
```


```python
Transport Block Configuration
=============================
channel_type : PUSCH
mcs_index : 14
mcs_table : 1
n_id : None
num_bits_per_symbol : 4
target_coderate : 0.5400390625
tb_scaling : 1.0

```

    
You can see that the current MCS Table is 1 and the MCS index is 14. Looking at the corresponding table in the API documentation of <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig">TBConfig</a>, you can see that we should have a 16QAM modulation (i.e., 4 bits per symbol) and a target coderate of 553/1024=0.54 which matches the values above. The data scrambling ID $n_\text{ID}$ is set to `None` which implies that the physical layer cell id $N^\text{cell}_\text{ID}$
will be used instead.
    
We can change the MCS index and table as follows:

```python
[26]:
```

```python
pusch_config.tb.mcs_index = 26
pusch_config.tb.mcs_table = 2
pusch_config.tb.show()
```


```python
Transport Block Configuration
=============================
channel_type : PUSCH
mcs_index : 26
mcs_table : 2
n_id : None
num_bits_per_symbol : 8
target_coderate : 0.89501953125
tb_scaling : 1.0

```

    
The transport block segmentation allows the PUSCH transmitter to fill resource grids of almost arbitrary size and with any of the possible DMRS configurations. The number of information bits transmitted in a single slot is given by the property `tb_size` of the `PUSCHConfig`.

```python
[27]:
```

```python
# Adding more PRBs will increase the TB size
pusch_config.carrier.n_size_grid = 273
pusch_config.tb_size
```
```python
[27]:
```
```python
303240
```
```python
[28]:
```

```python
# Adding more layers will increase the TB size
pusch_config.num_antenna_ports = 4
pusch_config.num_layers = 4
pusch_config.tb_size
```
```python
[28]:
```
```python
1213032
```

    
For more details about how the transportblock encoding/decoding works, we refer to the API documentation of the <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder">TBEncoder</a>.

