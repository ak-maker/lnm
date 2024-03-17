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
### Configuring Multiple Layers
### Controlling the Number of DMRS Symbols in a Slot
  
  

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

## Understanding the DMRS Configuration<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html#Understanding-the-DMRS-Configuration" title="Permalink to this headline"></a>
    
We can learn more about the structure of the resoure grid by having a look at the pilot pattern in the next section.

```python
[9]:
```

```python
pusch_transmitter.pilot_pattern.show();
```


    
From the figure above, we can see that there is a single transmitter sending a single stream (or so-called layer). DMRS are only sent on even subcarriers while odd subcarriers are masked, i.e., blocked for data transmission. This corresponds to the DMRS Configuration Type 1 with the parameter `NumCDMGroupsWithoutData` set to 2. We will explain what that means later.
    
In 5G NR, one can configure many different pilot patterns to adapt to different channel conditions and to allow for spatial multiplexing of up to twelve layers. Each transmitted layer is identified by a DMRS port, i.e., a distinct pilot pattern. In our running example, the transmitter uses the DMRS port 0.
    
With the current PUSCH configuration, four different DMRS ports 0,1,2,3 are available. This can be verified with the following command:

```python
[10]:
```

```python
pusch_config.dmrs.allowed_dmrs_ports
```
```python
[10]:
```
```python
[0, 1, 2, 3]
```

    
Next, we configure three other transmitters using each one of the remaing ports. Then, we create a new PUSCHTransmitter instance from the list of PUSCH configurations which is able to generate transmit signals for all four transmitters in parallel.

```python
[11]:
```

```python
# Clone the original PUSCHConfig and change the DMRS port set
pusch_config_1 = pusch_config.clone()
pusch_config_1.dmrs.dmrs_port_set = [1]
pusch_config_2 = pusch_config.clone()
pusch_config_2.dmrs.dmrs_port_set = [2]
pusch_config_3 = pusch_config.clone()
pusch_config_3.dmrs.dmrs_port_set = [3]
# Create a PUSCHTransmitter from the list of PUSCHConfigs
pusch_transmitter_multi = PUSCHTransmitter([pusch_config, pusch_config_1, pusch_config_2, pusch_config_3])
# Generate a batch of random transmit signals
x, b  = pusch_transmitter_multi(batch_size)
# x has shape [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
print("Shape of x:", x.shape)
```


```python
Shape of x: (16, 4, 1, 14, 48)
```

    
Inspecting the shape of x reveals that we have indeed four single-antenna transmitters. Let us now have a look at the resuling pilot pattern for each of them:

```python
[12]:
```

```python
pusch_transmitter_multi.pilot_pattern.show();
```





    
As before, all transmitters send pilots only on the third OFDM symbol. Transmitter 0 and 1 (using DMRS port 0 and 1, respectively) send pilots on all even subcarriers, while Transmitter 2 and 3 (using DMRS port 2 and 3, respectively), send pilots on the odd subcarriers. This means that the pilots signals of DMRS port 0 and 1 (as well as 2 and 3) interfere with each other as they occupy the same resource elements. So how can we estimate the channel coefficients for both transmitters individually
without pilot contamination?
    
The solution to this problem are the so-called code division multiplexing (CDM) groups in 5G NR. DMRS ports 0,1 belong to CDM group 0, while DMRS ports 2,3 belong to CDM group 1.
    
The pilot signals belonging to the same CDM group are multiplied by orthogonal cover codes which allow separating them during channel estimation. The way this works is as follows. Denote by $\mathbf{p_0} = [s_1, s_2]^\textsf{T}$ a pair of two adjacent pilot symbols, e.g., those on subcarrier 0 and 2, of DMRS port 0. DMRS port 1 will simply send $\mathbf{p_1} = [s_1, -s_2]^\textsf{T}$. If we assume that the channel is constant over both subcarriers, we get the following received pilot
signal at the receiver (we look only at a single antenna here):
    
\begin{align}
\mathbf{y} = h_0\mathbf{p}_0 + h_1\mathbf{p}_1 + \mathbf{n}
\end{align}
    
where $\mathbf{y}\in\mathbb{C}^2$ is the received signal on both subcarriers, $h_0, h_1$ are the channel coefficients for both users, and $\mathbf{n}\in\mathbb{C}^2$ is a noise vector.
    
We can now obtain channel estimates for both transmitters by projecting $\mathbf{y}$ onto their respective pilot sequences:
    
\begin{align}
\hat{h}_0 &= \frac{\mathbf{p}_0^\mathsf{H}}{\lVert \mathbf{p}_0 \rVert|^2} \mathbf{y} = h_0 + \frac{|s_1|^2-|s_2|^2}{\lVert \mathbf{p}_0 \rVert|^2} h_1 + \frac{\mathbf{p}_0^\mathsf{H}}{\lVert \mathbf{p}_0 \rVert|^2} \mathbf{n} = h_0 + n_0 \\
\hat{h}_1 &= \frac{\mathbf{p}_1^\mathsf{H}}{\lVert \mathbf{p}_1 \rVert|^2} \mathbf{y} = \frac{|s_1|^2-|s_2|^2}{\lVert \mathbf{p}_1 \rVert|^2} h_0 + h_1 +\frac{\mathbf{p}_1^\mathsf{H}}{\lVert \mathbf{p}_1 \rVert|^2} \mathbf{n} = h_1 + n_1.
\end{align}
    
Since the pilot symbols have the same amplitude, we have $|s_1|^2-|s_2|^2=0$, i.e., the interference between both pilot sequence is zero. Moreover, due to an implict averaging of the channel estimates for both subcarriers, the effective noise variance is reduced by a factor of 3dB since
    
\begin{align}
\mathbb{E}\left[ |n_0|^2 \right] = \mathbb{E}\left[ |n_1|^2 \right] = \frac{\sigma^2}{\lVert \mathbf{p}_1 \rVert|^2} = \frac{\sigma^2}{2 |s_0|^2}.
\end{align}
    
We can access the actual pilot sequences that are transmitted as follows:

```python
[13]:
```

```python
# pilots has shape [num_tx, num_layers, num_pilots]
pilots = pusch_transmitter_multi.pilot_pattern.pilots
print("Shape of pilots:", pilots.shape)
# Select only the non-zero subcarriers for all sequence
p_0 = pilots[0,0,::2] # Pilot sequence of TX 0 on even subcarriers
p_1 = pilots[1,0,::2] # Pilot sequence of TX 1 on even subcarriers
p_2 = pilots[2,0,1::2] # Pilot sequence of TX 2 on odd subcarriers
p_3 = pilots[3,0,1::2] # Pilot sequence of TX 3 on odd subcarriers
```


```python
Shape of pilots: (4, 1, 48)
```

    
Each pilot pattern consists of 48 symbols that are transmitted on the third OFDM symbol with 4PRBs, i.e., 48 subcarriers. Let us now verify that pairs of two adjacent pilot symbols in `p_0` and `p_1` as well as in `p_2` and `p_3` are orthogonal.

```python
[14]:
```

```python
print(np.sum(np.reshape(p_0, [-1,2]) * np.reshape(np.conj(p_1), [-1,2]), axis=1))
print(np.sum(np.reshape(p_2, [-1,2]) * np.reshape(np.conj(p_3), [-1,2]), axis=1))
```


```python
[0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j
 0.+0.j 0.+0.j]
[0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j
 0.+0.j 0.+0.j]
```

    
Let us now come back to the masked resource elements in each pilot pattern. The parameter `NumCDMGroupsWithoutData` mentioned earlier determines which resource elements in a DMRS-carrying OFDM symbol are masked for data transmissions. This is to avoid inference with pilots from other DMRS groups.
    
In our example, `NumCDMGroupsWithoutData` is set to two. This means that no data can be transmitted on any of the resource elements occupied by both DMRS groups. However, if we would have set `NumCDMGroupsWithoutData` equal to one, data and pilots would be frequency multiplexed. This can be useful, if we only schedule transmissions from DMRS ports in the same CDM group.
    
Here is an example of such a configuration:

```python
[15]:
```

```python
pusch_config = PUSCHConfig()
pusch_config.dmrs.num_cdm_groups_without_data = 1
pusch_config.dmrs.dmrs_port_set = [0]
pusch_config_1 = pusch_config.clone()
pusch_config_1.dmrs.dmrs_port_set = [1]
PUSCHTransmitter([pusch_config, pusch_config_1]).pilot_pattern.show();
```



    
The DRMS ports 0 and 1 belong both to CDM group 0 so that the resource elements of CDM group 1 do not need to be masked and can be used for data transmission. One can see in the above figure that data and pilots are now indeed multiplexed in the frequency domain.

### Configuring Multiple Layers<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html#Configuring-Multiple-Layers" title="Permalink to this headline"></a>
    
In 5G NR, a transmitter can be equipped with 1,2, or 4 antenna ports, i.e., physical antennas that are fed with an individual transmit signal. It can transmit 1,2,3 or 4 layers, i.e., spatial streams, as long as the number of layers does not exceed the number of antenna ports. Using codebook-based precoding, a number of layers can be mapped onto a larger number of antenna ports, e.g., 2 layers using 4 antenna ports. If no precoding is used, each layer is simply mapped to one of the antenna
ports.
    
It is important to understand that each layer is transmitted using a different DMRS port. That means that the number of DMRS ports is independent of the number of antenna ports.
    
In the next cell, we will configure a single transmitter with four antenna ports, sending two layers on DMRS ports 0 and 1. We can then choose among different precoding matrices with the help of the transmit transmit precoding matrix identifier (TPMI).

```python
[16]:
```

```python
pusch_config = PUSCHConfig()
pusch_config.num_antenna_ports = 4
pusch_config.num_layers = 2
pusch_config.dmrs.dmrs_port_set = [0,1]
pusch_config.precoding = "codebook"
pusch_config.tpmi = 7
# Show the precoding matrix
pusch_config.precoding_matrix
```
```python
[16]:
```
```python
array([[0.5+0.j , 0. +0.j ],
       [0. +0.j , 0.5+0.j ],
       [0.5+0.j , 0. +0.j ],
       [0. +0.j , 0. +0.5j]])
```
```python
[17]:
```

```python
PUSCHTransmitter(pusch_config).pilot_pattern.show();
```



    
We can see from the pilot patterns above, that we have now a single transmitter sending two streams. Both streams will be precoded and transmit over four antenna ports. From a channel estimation perspective at the receiver, however, this scenario is identical to the previous one with two single-antenna transmitters. The receiver will simply estimate the effective channel (including precoding) for every configured DMRS port.

### Controlling the Number of DMRS Symbols in a Slot<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html#Controlling-the-Number-of-DMRS-Symbols-in-a-Slot" title="Permalink to this headline"></a>
    
How can we add additional DMRS symbols to the resource grid to enable channel estimation for high-speed scenarios?
    
This can be controlled with the parameter `DMRS.additional_position`. In the next cell, we configure one additional DMRS symbol to the pattern and visualize it. You can try setting it to different values and see the impact.

```python
[18]:
```

```python
pusch_config.dmrs.additional_position = 1
# In order to reduce the number of figures, we only limit us here
# to the pilot pattern of the first stream
PUSCHTransmitter(pusch_config).pilot_pattern.show(stream_ind = 0);
```

