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
### GPU Configuration and Imports
## Simulations
### Evaluate the Impact of Mobility
  
  

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

### Evaluate the Impact of Mobility<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#Evaluate-the-Impact-of-Mobility" title="Permalink to this headline"></a>
    
Let us now have a look at the impact of the UT speed on the uplink performance. We compare the scenarios of perfect and imperfect CSI and 0 m/s and 20 m/s speed. To amplify the detrimental effects of high mobility, we only configure a single OFDM symbol for pilot transmissions at the beginning of the resource grid. With perfect CSI, mobility plays hardly any role. However, once channel estimation is taken into acount, the BLER saturates.
    
If you do not want to run the simulation your self, you skip the next cell and visualize the result in the next cell.

```python
[35]:
```

```python
MOBILITY_SIMS = {
    "ebno_db" : list(np.arange(0, 32, 2.0)),
    "cdl_model" : "D",
    "delay_spread" : 100e-9,
    "domain" : "freq",
    "direction" : "uplink",
    "perfect_csi" : [True, False],
    "speed" : [0.0, 20.0],
    "cyclic_prefix_length" : 6,
    "pilot_ofdm_symbol_indices" : [0],
    "ber" : [],
    "bler" : [],
    "duration" : None
}
start = time.time()
for perfect_csi in MOBILITY_SIMS["perfect_csi"]:
    for speed in MOBILITY_SIMS["speed"]:
        model = Model(domain=MOBILITY_SIMS["domain"],
                  direction=MOBILITY_SIMS["direction"],
                  cdl_model=MOBILITY_SIMS["cdl_model"],
                  delay_spread=MOBILITY_SIMS["delay_spread"],
                  perfect_csi=perfect_csi,
                  speed=speed,
                  cyclic_prefix_length=MOBILITY_SIMS["cyclic_prefix_length"],
                  pilot_ofdm_symbol_indices=MOBILITY_SIMS["pilot_ofdm_symbol_indices"])
        ber, bler = sim_ber(model,
                        MOBILITY_SIMS["ebno_db"],
                        batch_size=256,
                        max_mc_iter=100,
                        num_target_block_errors=1000)
        MOBILITY_SIMS["ber"].append(list(ber.numpy()))
        MOBILITY_SIMS["bler"].append(list(bler.numpy()))
MOBILITY_SIMS["duration"] = time.time() - start
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6161e-01 | 7.3291e-01 |      258170 |     1597440 |         1501 |        2048 |        12.6 |reached target block errors
      2.0 | 9.3824e-02 | 4.7624e-01 |      224817 |     2396160 |         1463 |        3072 |         0.1 |reached target block errors
      4.0 | 4.4328e-02 | 2.3594e-01 |      177027 |     3993600 |         1208 |        5120 |         0.2 |reached target block errors
      6.0 | 1.5049e-02 | 8.7891e-02 |      144242 |     9584640 |         1080 |       12288 |         0.5 |reached target block errors
      8.0 | 3.4631e-03 | 2.1315e-02 |      127238 |    36741120 |         1004 |       47104 |         2.1 |reached target block errors
     10.0 | 6.0279e-04 | 4.0527e-03 |       48146 |    79872000 |          415 |      102400 |         4.6 |reached max iter
     12.0 | 5.9683e-05 | 3.8086e-04 |        4767 |    79872000 |           39 |      102400 |         4.6 |reached max iter
     14.0 | 6.9611e-06 | 6.8359e-05 |         556 |    79872000 |            7 |      102400 |         4.6 |reached max iter
     16.0 | 1.7278e-06 | 9.7656e-06 |         138 |    79872000 |            1 |      102400 |         4.5 |reached max iter
     18.0 | 0.0000e+00 | 0.0000e+00 |           0 |    79872000 |            0 |      102400 |         4.6 |reached max iter
Simulation stopped as no error occurred @ EbNo = 18.0 dB.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6147e-01 | 7.3779e-01 |      257937 |     1597440 |         1511 |        2048 |        12.7 |reached target block errors
      2.0 | 9.9016e-02 | 4.8796e-01 |      237258 |     2396160 |         1499 |        3072 |         0.1 |reached target block errors
      4.0 | 4.2508e-02 | 2.3145e-01 |      169761 |     3993600 |         1185 |        5120 |         0.2 |reached target block errors
      6.0 | 1.3701e-02 | 8.1806e-02 |      142266 |    10383360 |         1089 |       13312 |         0.6 |reached target block errors
      8.0 | 3.5065e-03 | 2.2070e-02 |      126033 |    35942400 |         1017 |       46080 |         2.1 |reached target block errors
     10.0 | 6.4011e-04 | 4.0039e-03 |       51127 |    79872000 |          410 |      102400 |         4.6 |reached max iter
     12.0 | 7.5120e-05 | 4.9805e-04 |        6000 |    79872000 |           51 |      102400 |         4.6 |reached max iter
     14.0 | 2.7168e-06 | 1.9531e-05 |         217 |    79872000 |            2 |      102400 |         4.6 |reached max iter
     16.0 | 0.0000e+00 | 0.0000e+00 |           0 |    79872000 |            0 |      102400 |         4.6 |reached max iter
Simulation stopped as no error occurred @ EbNo = 16.0 dB.
WARNING:tensorflow:From /usr/local/lib/python3.8/dist-packages/tensorflow/python/util/dispatch.py:1082: calling gather (from tensorflow.python.ops.array_ops) with validate_indices is deprecated and will be removed in a future version.
Instructions for updating:
The `validate_indices` argument has no effect. Indices are always validated on CPU and never validated on GPU.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 2.7695e-01 | 1.0000e+00 |      221203 |      798720 |         1024 |        1024 |        12.8 |reached target block errors
      2.0 | 2.6110e-01 | 9.8145e-01 |      208544 |      798720 |         1005 |        1024 |         0.0 |reached target block errors
      4.0 | 2.2669e-01 | 9.1016e-01 |      362117 |     1597440 |         1864 |        2048 |         0.1 |reached target block errors
      6.0 | 1.7395e-01 | 7.6221e-01 |      277872 |     1597440 |         1561 |        2048 |         0.1 |reached target block errors
      8.0 | 1.0000e-01 | 4.6777e-01 |      239622 |     2396160 |         1437 |        3072 |         0.1 |reached target block errors
     10.0 | 5.1870e-02 | 2.6343e-01 |      165718 |     3194880 |         1079 |        4096 |         0.2 |reached target block errors
     12.0 | 1.6063e-02 | 8.9933e-02 |      141124 |     8785920 |         1013 |       11264 |         0.5 |reached target block errors
     14.0 | 3.7753e-03 | 2.2179e-02 |      135694 |    35942400 |         1022 |       46080 |         2.1 |reached target block errors
     16.0 | 6.5346e-04 | 4.0527e-03 |       52193 |    79872000 |          415 |      102400 |         4.6 |reached max iter
     18.0 | 5.6778e-05 | 4.1016e-04 |        4535 |    79872000 |           42 |      102400 |         4.6 |reached max iter
     20.0 | 1.7904e-05 | 9.7656e-05 |        1430 |    79872000 |           10 |      102400 |         4.6 |reached max iter
     22.0 | 1.6777e-06 | 9.7656e-06 |         134 |    79872000 |            1 |      102400 |         4.6 |reached max iter
     24.0 | 0.0000e+00 | 0.0000e+00 |           0 |    79872000 |            0 |      102400 |         4.6 |reached max iter
Simulation stopped as no error occurred @ EbNo = 24.0 dB.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 2.8459e-01 | 9.9902e-01 |      227307 |      798720 |         1023 |        1024 |        12.6 |reached target block errors
      2.0 | 2.6964e-01 | 9.9121e-01 |      215368 |      798720 |         1015 |        1024 |         0.0 |reached target block errors
      4.0 | 2.5126e-01 | 9.7510e-01 |      401371 |     1597440 |         1997 |        2048 |         0.1 |reached target block errors
      6.0 | 2.0439e-01 | 8.5352e-01 |      326507 |     1597440 |         1748 |        2048 |         0.1 |reached target block errors
      8.0 | 1.6192e-01 | 7.2461e-01 |      258655 |     1597440 |         1484 |        2048 |         0.1 |reached target block errors
     10.0 | 9.7994e-02 | 4.6973e-01 |      234809 |     2396160 |         1443 |        3072 |         0.1 |reached target block errors
     12.0 | 6.3871e-02 | 3.3854e-01 |      153046 |     2396160 |         1040 |        3072 |         0.1 |reached target block errors
     14.0 | 3.0211e-02 | 1.7334e-01 |      144779 |     4792320 |         1065 |        6144 |         0.3 |reached target block errors
     16.0 | 1.2865e-02 | 8.5449e-02 |      123302 |     9584640 |         1050 |       12288 |         0.6 |reached target block errors
     18.0 | 5.1862e-03 | 3.9688e-02 |      103558 |    19968000 |         1016 |       25600 |         1.2 |reached target block errors
     20.0 | 1.5217e-03 | 1.8175e-02 |       65631 |    43130880 |         1005 |       55296 |         2.5 |reached target block errors
     22.0 | 4.3807e-04 | 1.1489e-02 |       29741 |    67891200 |         1000 |       87040 |         3.9 |reached target block errors
     24.0 | 1.3831e-04 | 1.0574e-02 |       10274 |    74280960 |         1007 |       95232 |         4.3 |reached target block errors
     26.0 | 3.8288e-05 | 1.0015e-02 |        2997 |    78274560 |         1005 |      100352 |         4.5 |reached target block errors
     28.0 | 2.4988e-05 | 1.0234e-02 |        1916 |    76677120 |         1006 |       98304 |         4.4 |reached target block errors
     30.0 | 2.1836e-05 | 1.0595e-02 |        1622 |    74280960 |         1009 |       95232 |         4.3 |reached target block errors
```
```python
[36]:
```

```python
# Load results (uncomment to show saved results from the cell above)
#MOBILITY_SIMS = eval(" {'ebno_db': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0], 'cdl_model': 'D', 'delay_spread': 1e-07, 'domain': 'freq', 'direction': 'uplink', 'perfect_csi': [True, False], 'speed': [0.0, 20.0], 'cyclic_prefix_length': 6, 'pilot_ofdm_symbol_indices': [0], 'ber': [[0.15169959435096153, 0.10006385216346154, 0.04578732221554487, 0.014152748564369658, 0.003497385589670746, 0.0006175756209935898, 4.672475961538461e-05, 3.342848557692308e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.15599834735576923, 0.09424036792200854, 0.043501602564102564, 0.015064206897702992, 0.0034728119338768115, 0.000610752203525641, 6.844701522435897e-05, 1.4072516025641026e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.27565479767628204, 0.25848482572115383, 0.22961112780448717, 0.16972468449519232, 0.1010475093482906, 0.04783954326923077, 0.016199555652680653, 0.004016548428705441, 0.0006721880008012821, 7.454427083333334e-05, 8.526141826923077e-06, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2835011017628205, 0.2701497395833333, 0.2480888171073718, 0.21193033854166668, 0.16006485376602564, 0.1103377904647436, 0.06517052283653846, 0.0286998781383547, 0.012703033186431624, 0.005103715945512821, 0.0016146116805757136, 0.000522487214110478, 0.00014040694277510683, 4.0696557791435366e-05, 2.5298629981884056e-05, 2.4334147401800328e-05]], 'bler': [[0.708984375, 0.4964192708333333, 0.248046875, 0.08406575520833333, 0.022349964488636364, 0.0041796875, 0.000322265625, 2.9296875e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.71630859375, 0.4788411458333333, 0.237890625, 0.08780924479166667, 0.021739130434782608, 0.004091796875, 0.000556640625, 0.000107421875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9970703125, 0.9765625, 0.92138671875, 0.74755859375, 0.478515625, 0.2455078125, 0.09064275568181818, 0.023890053353658538, 0.00423828125, 0.00052734375, 6.8359375e-05, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9990234375, 0.9931640625, 0.962890625, 0.88232421875, 0.71533203125, 0.5322265625, 0.3470052083333333, 0.16276041666666666, 0.082763671875, 0.0407421875, 0.01862839033018868, 0.012176890432098766, 0.010203043619791666, 0.01010792525773196, 0.010646654211956522, 0.010409740691489361]], 'duration': 705.361558675766} ")
print("Simulation duration: {:1.2f} [h]".format(MOBILITY_SIMS["duration"]/3600))
plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.title("CDL-D MIMO Uplink - Impact of UT mobility")
i = 0
for perfect_csi in MOBILITY_SIMS["perfect_csi"]:
    for speed in MOBILITY_SIMS["speed"]:
        style = "{}".format("-" if perfect_csi else "--")
        s = "{} CSI {}[m/s]".format("Perf." if perfect_csi else "Imperf.", speed)
        plt.semilogy(MOBILITY_SIMS["ebno_db"],
                     MOBILITY_SIMS["bler"][i],
                      style, label=s,)
        i += 1
plt.legend();
plt.ylim([1e-3, 1]);
```


```python
Simulation duration: 0.04 [h]
```


