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
### Evaluate the Impact of Insufficient Cyclic Prefix Length
  
  

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

### Evaluate the Impact of Insufficient Cyclic Prefix Length<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html#Evaluate-the-Impact-of-Insufficient-Cyclic-Prefix-Length" title="Permalink to this headline"></a>
    
As a final example, let us have a look at how to simulate OFDM with an insufficiently long cyclic prefix.
    
It is important to notice, that ISI cannot be simulated in the frequency domain as the <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#channel-with-ofdm-waveform">OFDMChannel</a> implicitly assumes perfectly synchronized and ISI-free transmissions. Having no cyclic prefix translates simply into an improved Eb/No as no energy for its transmission is used.
    
Simulating a channel in the time domain requires significantly more memory and compute which might limit the scenarios for which it can be used.
    
If you do not want to run the simulation your self, you skip the next cell and visualize the result in the next cell.

```python
[37]:
```

```python
CP_SIMS = {
    "ebno_db" : list(np.arange(0, 16, 1.0)),
    "cdl_model" : "C",
    "delay_spread" : 100e-9,
    "subcarrier_spacing" : 15e3,
    "domain" : ["freq", "time"],
    "direction" : "uplink",
    "perfect_csi" : False,
    "speed" : 3.0,
    "cyclic_prefix_length" : [20, 4],
    "pilot_ofdm_symbol_indices" : [2, 11],
    "ber" : [],
    "bler" : [],
    "duration": None
}
start = time.time()
for cyclic_prefix_length in CP_SIMS["cyclic_prefix_length"]:
    for domain in CP_SIMS["domain"]:
        model = Model(domain=domain,
                  direction=CP_SIMS["direction"],
                  cdl_model=CP_SIMS["cdl_model"],
                  delay_spread=CP_SIMS["delay_spread"],
                  perfect_csi=CP_SIMS["perfect_csi"],
                  speed=CP_SIMS["speed"],
                  cyclic_prefix_length=cyclic_prefix_length,
                  pilot_ofdm_symbol_indices=CP_SIMS["pilot_ofdm_symbol_indices"],
                  subcarrier_spacing=CP_SIMS["subcarrier_spacing"])
        ber, bler = sim_ber(model,
                        CP_SIMS["ebno_db"],
                        batch_size=256,
                        max_mc_iter=100,
                        num_target_block_errors=1000)
        CP_SIMS["ber"].append(list(ber.numpy()))
        CP_SIMS["bler"].append(list(bler.numpy()))
CP_SIMS["duration"] = time.time() - start
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 7.1097e-02 | 3.2300e-01 |      209674 |     2949120 |         1323 |        4096 |        12.5 |reached target block errors
      1.0 | 4.9494e-02 | 2.3516e-01 |      182453 |     3686400 |         1204 |        5120 |         0.2 |reached target block errors
      2.0 | 3.7691e-02 | 1.7497e-01 |      166733 |     4423680 |         1075 |        6144 |         0.3 |reached target block errors
      3.0 | 2.4792e-02 | 1.2012e-01 |      164506 |     6635520 |         1107 |        9216 |         0.4 |reached target block errors
      4.0 | 1.7193e-02 | 8.5449e-02 |      152113 |     8847360 |         1050 |       12288 |         0.5 |reached target block errors
      5.0 | 1.0395e-02 | 5.2375e-02 |      145610 |    14008320 |         1019 |       19456 |         0.9 |reached target block errors
      6.0 | 6.8724e-03 | 3.4853e-02 |      146939 |    21381120 |         1035 |       29696 |         1.3 |reached target block errors
      7.0 | 4.3218e-03 | 2.1897e-02 |      143387 |    33177600 |         1009 |       46080 |         2.0 |reached target block errors
      8.0 | 2.3735e-03 | 1.2280e-02 |      139996 |    58982400 |         1006 |       81920 |         3.6 |reached target block errors
      9.0 | 1.3166e-03 | 6.6992e-03 |       97071 |    73728000 |          686 |      102400 |         4.5 |reached max iter
     10.0 | 7.2675e-04 | 3.9648e-03 |       53582 |    73728000 |          406 |      102400 |         4.6 |reached max iter
     11.0 | 3.7358e-04 | 2.0508e-03 |       27543 |    73728000 |          210 |      102400 |         4.6 |reached max iter
     12.0 | 2.2316e-04 | 1.2500e-03 |       16453 |    73728000 |          128 |      102400 |         4.6 |reached max iter
     13.0 | 1.1392e-04 | 6.2500e-04 |        8399 |    73728000 |           64 |      102400 |         4.6 |reached max iter
     14.0 | 5.2070e-05 | 2.8320e-04 |        3839 |    73728000 |           29 |      102400 |         4.6 |reached max iter
     15.0 | 2.2176e-05 | 1.4648e-04 |        1635 |    73728000 |           15 |      102400 |         4.6 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 7.2379e-02 | 3.2812e-01 |      160091 |     2211840 |         1008 |        3072 |        13.3 |reached target block errors
      1.0 | 5.1294e-02 | 2.3750e-01 |      189090 |     3686400 |         1216 |        5120 |         0.8 |reached target block errors
      2.0 | 3.7019e-02 | 1.7594e-01 |      163760 |     4423680 |         1081 |        6144 |         0.9 |reached target block errors
      3.0 | 2.4789e-02 | 1.1784e-01 |      164486 |     6635520 |         1086 |        9216 |         1.4 |reached target block errors
      4.0 | 1.7723e-02 | 8.4066e-02 |      156805 |     8847360 |         1033 |       12288 |         1.8 |reached target block errors
      5.0 | 1.0601e-02 | 5.2375e-02 |      148504 |    14008320 |         1019 |       19456 |         2.9 |reached target block errors
      6.0 | 6.7902e-03 | 3.4982e-02 |      140175 |    20643840 |         1003 |       28672 |         4.3 |reached target block errors
      7.0 | 4.1687e-03 | 2.1569e-02 |      141380 |    33914880 |         1016 |       47104 |         7.0 |reached target block errors
      8.0 | 2.3207e-03 | 1.2231e-02 |      136878 |    58982400 |         1002 |       81920 |        12.2 |reached target block errors
      9.0 | 1.4165e-03 | 7.4805e-03 |      104439 |    73728000 |          766 |      102400 |        15.3 |reached max iter
     10.0 | 7.3530e-04 | 3.9453e-03 |       54212 |    73728000 |          404 |      102400 |        15.3 |reached max iter
     11.0 | 4.2832e-04 | 2.3633e-03 |       31579 |    73728000 |          242 |      102400 |        15.3 |reached max iter
     12.0 | 1.6987e-04 | 1.0156e-03 |       12524 |    73728000 |          104 |      102400 |        15.3 |reached max iter
     13.0 | 1.1386e-04 | 5.6641e-04 |        8395 |    73728000 |           58 |      102400 |        15.3 |reached max iter
     14.0 | 7.5345e-05 | 3.8086e-04 |        5555 |    73728000 |           39 |      102400 |        15.2 |reached max iter
     15.0 | 2.2108e-05 | 1.4648e-04 |        1630 |    73728000 |           15 |      102400 |        15.2 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 5.1978e-02 | 2.4141e-01 |      191610 |     3686400 |         1236 |        5120 |        12.8 |reached target block errors
      1.0 | 4.0247e-02 | 1.8587e-01 |      178038 |     4423680 |         1142 |        6144 |         0.3 |reached target block errors
      2.0 | 2.6527e-02 | 1.2805e-01 |      156465 |     5898240 |         1049 |        8192 |         0.4 |reached target block errors
      3.0 | 1.8807e-02 | 9.1797e-02 |      152526 |     8110080 |         1034 |       11264 |         0.5 |reached target block errors
      4.0 | 1.1853e-02 | 5.9053e-02 |      148558 |    12533760 |         1028 |       17408 |         0.8 |reached target block errors
      5.0 | 7.4701e-03 | 3.6856e-02 |      148704 |    19906560 |         1019 |       27648 |         1.2 |reached target block errors
      6.0 | 4.4296e-03 | 2.2705e-02 |      143698 |    32440320 |         1023 |       45056 |         2.0 |reached target block errors
      7.0 | 2.6306e-03 | 1.3498e-02 |      141584 |    53821440 |         1009 |       74752 |         3.3 |reached target block errors
      8.0 | 1.4671e-03 | 7.6660e-03 |      108163 |    73728000 |          785 |      102400 |         4.6 |reached max iter
      9.0 | 8.5364e-04 | 4.5996e-03 |       62937 |    73728000 |          471 |      102400 |         4.6 |reached max iter
     10.0 | 4.6239e-04 | 2.4512e-03 |       34091 |    73728000 |          251 |      102400 |         4.6 |reached max iter
     11.0 | 2.3062e-04 | 1.3184e-03 |       17003 |    73728000 |          135 |      102400 |         4.5 |reached max iter
     12.0 | 1.2603e-04 | 6.8359e-04 |        9292 |    73728000 |           70 |      102400 |         4.6 |reached max iter
     13.0 | 5.4796e-05 | 3.3203e-04 |        4040 |    73728000 |           34 |      102400 |         4.6 |reached max iter
     14.0 | 2.2366e-05 | 1.4648e-04 |        1649 |    73728000 |           15 |      102400 |         4.6 |reached max iter
     15.0 | 7.2293e-06 | 4.8828e-05 |         533 |    73728000 |            5 |      102400 |         4.6 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 8.7329e-02 | 3.9030e-01 |      193157 |     2211840 |         1199 |        3072 |        13.0 |reached target block errors
      1.0 | 6.4305e-02 | 2.9126e-01 |      189644 |     2949120 |         1193 |        4096 |         0.5 |reached target block errors
      2.0 | 4.9408e-02 | 2.3145e-01 |      182138 |     3686400 |         1185 |        5120 |         0.7 |reached target block errors
      3.0 | 3.5916e-02 | 1.7188e-01 |      158880 |     4423680 |         1056 |        6144 |         0.8 |reached target block errors
      4.0 | 2.5862e-02 | 1.2294e-01 |      171605 |     6635520 |         1133 |        9216 |         1.2 |reached target block errors
      5.0 | 1.6486e-02 | 8.1868e-02 |      145861 |     8847360 |         1006 |       12288 |         1.6 |reached target block errors
      6.0 | 1.1784e-02 | 5.9513e-02 |      147692 |    12533760 |         1036 |       17408 |         2.2 |reached target block errors
      7.0 | 7.1154e-03 | 3.7507e-02 |      141643 |    19906560 |         1037 |       27648 |         3.5 |reached target block errors
      8.0 | 4.3708e-03 | 2.5566e-02 |      125677 |    28753920 |         1021 |       39936 |         5.1 |reached target block errors
      9.0 | 2.5687e-03 | 1.7822e-02 |      106057 |    41287680 |         1022 |       57344 |         7.4 |reached target block errors
     10.0 | 1.6852e-03 | 1.6341e-02 |       74550 |    44236800 |         1004 |       61440 |         7.9 |reached target block errors
     11.0 | 8.9164e-04 | 1.6817e-02 |       38786 |    43499520 |         1016 |       60416 |         7.8 |reached target block errors
     12.0 | 4.7879e-04 | 2.1235e-02 |       16591 |    34652160 |         1022 |       48128 |         6.2 |reached target block errors
     13.0 | 3.3972e-04 | 3.1250e-02 |        8015 |    23592960 |         1024 |       32768 |         4.2 |reached target block errors
     14.0 | 1.8072e-04 | 3.9492e-02 |        3331 |    18432000 |         1011 |       25600 |         3.3 |reached target block errors
     15.0 | 1.4855e-04 | 5.2066e-02 |        2081 |    14008320 |         1013 |       19456 |         2.5 |reached target block errors
```
```python
[38]:
```

```python
# Load results (uncomment to show saved results from the cell above)
#CP_SIMS = eval(" {'ebno_db': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], 'cdl_model': 'C', 'delay_spread': 1e-07, 'subcarrier_spacing': 15000.0, 'domain': ['freq', 'time'], 'direction': 'uplink', 'perfect_csi': False, 'speed': 3.0, 'cyclic_prefix_length': [20, 4], 'pilot_ofdm_symbol_indices': [2, 11], 'ber': [[0.06557798032407407, 0.05356038411458333, 0.036498360339506174, 0.02517761752136752, 0.017377266589506172, 0.011118797019675926, 0.006675488753019323, 0.004243488387664524, 0.002491552599862259, 0.0012659071180555555, 0.0007226888020833333, 0.0004105794270833333, 0.00021171875, 0.00011207682291666667, 4.0809461805555556e-05, 2.279730902777778e-05], [0.06876168536324787, 0.051859085648148145, 0.03822603202160494, 0.02548694577991453, 0.016959404550827423, 0.011013454861111112, 0.006892088627304434, 0.004137017877252252, 0.002463582466684675, 0.0014250651041666667, 0.0007145616319444444, 0.00037141927083333336, 0.00019383680555555555, 0.00010481770833333334, 4.1644965277777777e-05, 1.9932725694444446e-05], [0.05217441340488215, 0.03722222222222222, 0.027770973104990583, 0.018300805910669193, 0.012147801143483709, 0.0072280695408950615, 0.004458076408844189, 0.0026281823645104897, 0.0016104890352813088, 0.0008862955729166666, 0.00046706814236111113, 0.0002002495659722222, 0.00011341145833333334, 4.968532986111111e-05, 2.0464409722222224e-05, 1.4811197916666667e-05], [0.0816061000631313, 0.06929166666666667, 0.052485826280381946, 0.035026493778935186, 0.023676058021336554, 0.017314453125, 0.011337403130032207, 0.007286241319444445, 0.004440556877759382, 0.0027163238447260626, 0.0013986585930973266, 0.0009150437801932367, 0.0006104324281090033, 0.00034796381644684255, 0.00018975482723577235, 0.00016822318007662836]], 'bler': [[0.2960069444444444, 0.248046875, 0.17447916666666666, 0.12139423076923077, 0.08697916666666666, 0.05490451388888889, 0.03400135869565218, 0.021543560606060608, 0.012939049586776859, 0.0068515625, 0.0037109375, 0.002203125, 0.001171875, 0.0006171875, 0.000234375, 0.0001171875], [0.3097956730769231, 0.23910984848484848, 0.17604166666666668, 0.12163461538461538, 0.08352726063829788, 0.055182658450704226, 0.03506866591928251, 0.021114864864864864, 0.012662074554294975, 0.0074375, 0.0037265625, 0.0019609375, 0.0010234375, 0.0005703125, 0.0002265625, 0.000109375], [0.24076704545454544, 0.1751736111111111, 0.1340042372881356, 0.08939985795454546, 0.05891682330827068, 0.036205150462962965, 0.022494612068965518, 0.013671875, 0.008275953389830509, 0.0046953125, 0.0024296875, 0.001171875, 0.000609375, 0.00025, 0.000125, 8.59375e-05], [0.37073863636363635, 0.315, 0.246337890625, 0.16438802083333334, 0.1146965579710145, 0.0875, 0.05672554347826087, 0.03807645631067961, 0.025895074503311258, 0.018019153225806453, 0.01474389097744361, 0.016983695652173912, 0.021193258807588076, 0.028158723021582732, 0.038147865853658536, 0.05387931034482758]], 'duration': 5074.19091796875} ")
print("Simulation duration: {:1.2f} [h]".format(CP_SIMS["duration"]/3600))
plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.title("CDL-B MIMO Uplink - Impact of Cyclic Prefix Length")
i = 0
for cyclic_prefix_length in CP_SIMS["cyclic_prefix_length"]:
    for domain in CP_SIMS["domain"]:
        s = "{} Domain, CP length: {}".format("Freq" if domain=="freq" else "Time",
                                               cyclic_prefix_length)
        plt.semilogy(CP_SIMS["ebno_db"],
                     CP_SIMS["bler"][i],
                     label=s)
        i += 1
plt.legend();
plt.ylim([1e-3, 1]);
```


```python
Simulation duration: 0.09 [h]
```


    
One can make a few important observations from the figure above:
<ol class="arabic simple">
- The length of the cyclic prefix has no impact on the performance if the system is simulated in the frequency domain.
The reason why the two curves for both frequency-domain simulations do not overlap is that the cyclic prefix length affects the way the Eb/No is computed.
- With a sufficiently large cyclic prefix (in our case `cyclic_prefix_length` `=` `20` `>=` `l_tot` `=` `17` ), the performance of time and frequency-domain simulations are identical.
- With a too small cyclic prefix length, the performance degrades. At high SNR, inter-symbol interference (from multiple streams) becomes the dominating source of interference.
</ol>
<script type="application/vnd.jupyter.widget-state+json">
{"state": {}, "version_major": 2, "version_minor": 0}
</script>