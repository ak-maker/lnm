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
## System Setup
## System Setup
### Stream Management
### OFDM Resource Grid & Pilot Pattern
### Antenna Arrays
### CDL Channel Model
  
  

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

## System Setup<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#System-Setup" title="Permalink to this headline"></a>
    
We will now configure all components of the system model step-by-step.

### Stream Management<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#Stream-Management" title="Permalink to this headline"></a>
    
For any type of MIMO simulations, it is useful to setup a <a class="reference external" href="https://nvlabs.github.io/sionna/api/mimo.html#stream-management">StreamManagement</a> object. It determines which transmitters and receivers communicate data streams with each other. In our scenario, we will configure a single UT and BS with multiple antennas each. Whether the UT or BS is considered as a transmitter depends on the `direction`, which can be either uplink or downlink. The
<a class="reference external" href="https://nvlabs.github.io/sionna/api/mimo.html#stream-management">StreamManagement</a> has many properties that are used by other components, such as precoding and equalization.
    
We will configure the system here such that the number of streams per transmitter (in both uplink and donwlink) is equal to the number of UT antennas.

```python
[3]:
```

```python
# Define the number of UT and BS antennas.
# For the CDL model, that will be used in this notebook, only
# a single UT and BS are supported.
num_ut = 1
num_bs = 1
num_ut_ant = 4
num_bs_ant = 8
# The number of transmitted streams is equal to the number of UT antennas
# in both uplink and downlink
num_streams_per_tx = num_ut_ant
# Create an RX-TX association matrix
# rx_tx_association[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change. However, as we have only a single
# transmitter and receiver, this does not matter:
rx_tx_association = np.array([[1]])
# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly easy. However, it can get more involved
# for simulations with many transmitters and receivers.
sm = StreamManagement(rx_tx_association, num_streams_per_tx)
```

### OFDM Resource Grid & Pilot Pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#OFDM-Resource-Grid-&-Pilot-Pattern" title="Permalink to this headline"></a>
    
Next, we configure an OFDM <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#resource-grid">ResourceGrid</a> spanning multiple OFDM symbols. The resource grid contains data symbols and pilots and is equivalent to a <em>slot</em> in 4G/5G terminology. Although it is not relevant for our simulation, we null the DC subcarrier and a few guard carriers to the left and right of the spectrum. Also a cyclic prefix is added.
    
During the creation of the <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#resource-grid">ResourceGrid</a>, a <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#pilot-pattern">PilotPattern</a> is automatically generated. We could have alternatively created a <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#pilot-pattern">PilotPattern</a> first and then provided it as initialization parameter.

```python
[4]:
```

```python
rg = ResourceGrid(num_ofdm_symbols=14,
                  fft_size=76,
                  subcarrier_spacing=15e3,
                  num_tx=1,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=6,
                  num_guard_carriers=[5,6],
                  dc_null=True,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[2,11])
rg.show();
```


    
As can be seen in the figure above, the resource grid spans 76 subcarriers over 14 OFDM symbols. A DC guard carrier as well as some guard carriers to the left and right of the spectrum are nulled. The third and twelfth OFDM symbol are dedicated to pilot transmissions.
    
Let us now have a look at the pilot pattern used by the transmitter.

```python
[5]:
```

```python
rg.pilot_pattern.show();
```





    
The pilot patterns are defined over the resource grid of <em>effective subcarriers</em> from which the nulled DC and guard carriers have been removed. This leaves us in our case with 76 - 1 (DC) - 5 (left guards) - 6 (right guards) = 64 effective subcarriers.
    
While the resource grid only knows which resource elements are reserved for pilots, it is the pilot pattern that defines what is actually transmitted on them. In our scenario, we have four transmit streams and configured the <a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#kroneckerpilotpattern">KroneckerPilotPattern</a>. All streams use orthogonal pilot sequences, i.e., one pilot on every fourth subcarrier. You have full freedom to configure your own
<a class="reference external" href="https://nvlabs.github.io/sionna/api/ofdm.html#pilotpattern">PilotPattern</a>.
    
Let us now have a look at the actual pilot sequences for all streams which consists of random QPSK symbols. By default, the pilot sequences are normalized, such that the average power per pilot symbol is equal to one. As only every fourth pilot symbol in the sequence is used, their amplitude is scaled by a factor of two.

```python
[6]:
```

```python
plt.figure()
plt.title("Real Part of the Pilot Sequences")
for i in range(num_streams_per_tx):
    plt.stem(np.real(rg.pilot_pattern.pilots[0, i]),
             markerfmt="C{}.".format(i), linefmt="C{}-".format(i),
             label="Stream {}".format(i))
plt.legend()
print("Average energy per pilot symbol: {:1.2f}".format(np.mean(np.abs(rg.pilot_pattern.pilots[0,0])**2)))
```


```python
Average energy per pilot symbol: 1.00
```


### Antenna Arrays<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#Antenna-Arrays" title="Permalink to this headline"></a>
    
Next, we need to configure the antenna arrays used by the UT and BS. This can be ignored for simple channel models, such as <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#awgn">AWGN</a>, <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#flat-fading-channel">flat-fading</a>, <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#rayleigh-block-fading">RayleighBlockFading</a>, or <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#tapped-delay-line-tdl">TDL</a> which do not account for antenna array
geometries and antenna radiation patterns. However, other models, such as <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#clustered-delay-line-cdl">CDL</a>, <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#urban-microcell-umi">UMi</a>, <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#urban-macrocell-uma">UMa</a>, and <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#rural-macrocell-rma">RMa</a> from the 3GPP 38.901 specification, require it.
    
We will assume here that UT and BS antenna arrays are composed of dual cross-polarized antenna elements with an antenna pattern defined in the 3GPP 38.901 specification. By default, the antenna elements are spaced half of a wavelength apart in both vertical and horizontal directions. You can define your own antenna geometries an radiation patterns if needed.
    
An <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#antennaarray">AntennaArray</a> is always defined in the y-z plane. It’s final orientation will be determined by the orientation of the UT or BS. This parameter can be configured in the <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#channel-model-interface">ChannelModel</a> that we will create later.

```python
[7]:
```

```python
carrier_frequency = 2.6e9 # Carrier frequency in Hz.
                          # This is needed here to define the antenna element spacing.
ut_array = AntennaArray(num_rows=1,
                        num_cols=int(num_ut_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
ut_array.show()
bs_array = AntennaArray(num_rows=1,
                        num_cols=int(num_bs_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
bs_array.show()
```



    
We can also visualize the radiation pattern of an individual antenna element:

```python
[8]:
```

```python
ut_array.show_element_radiation_pattern()
```




### CDL Channel Model<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#CDL-Channel-Model" title="Permalink to this headline"></a>
    
Now, we will create an instance of the CDL channel model.

```python
[9]:
```

```python
delay_spread = 300e-9 # Nominal delay spread in [s]. Please see the CDL documentation
                      # about how to choose this value.
direction = "uplink"  # The `direction` determines if the UT or BS is transmitting.
                      # In the `uplink`, the UT is transmitting.
cdl_model = "B"       # Suitable values are ["A", "B", "C", "D", "E"]
speed = 10            # UT speed [m/s]. BSs are always assumed to be fixed.
                      # The direction of travel will chosen randomly within the x-y plane.
# Configure a channel impulse reponse (CIR) generator for the CDL model.
# cdl() will generate CIRs that can be converted to discrete time or discrete frequency.
cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=speed)
```
