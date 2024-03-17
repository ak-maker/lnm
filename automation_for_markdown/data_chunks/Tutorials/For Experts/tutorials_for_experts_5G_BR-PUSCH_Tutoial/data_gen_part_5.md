# 5G NR PUSCH Tutorial<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html#5G-NR-PUSCH-Tutorial" title="Permalink to this headline"></a>
    
This notebook provides an introduction to Sionna’s <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html">5G New Radio (NR) module</a> and, in particular, the <a class="reference external" href="https://nvlabs.github.io/sionna/api/nr.html#pusch">physical uplink shared channel (PUSCH)</a>. This module provides implementations of a small subset of the physical layer functionalities as described in the 3GPP specifications <a class="reference external" href="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3213">38.211</a>,
<a class="reference external" href="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3214">38.212</a> and <a class="reference external" href="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3216">38.214</a>.
    
You will
 
- Get an understanding of the different components of a PUSCH configuration, such as the carrier, DMRS, and transport block,
- Learn how to rapidly simulate PUSCH transmissions for multiple transmitters,
- Modify the PUSCHReceiver to use a custom MIMO Detector.
# Table of Content
## GPU Configuration and Imports
## End-to-end PUSCH Simulations
  
  

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

## End-to-end PUSCH Simulations<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html#End-to-end-PUSCH-Simulations" title="Permalink to this headline"></a>
    
We will now implement a end-to-end Keras model that is capable of running PUSCH simulations for many different configurations. You can use it as a boilerplate template for your own experiments.

```python
[36]:
```

```python
# We need to enable sionna.config.xla_compat before we can use
# tf.function with jit_compile=True.
# See https://nvlabs.github.io/sionna/api/config.html#sionna.Config.xla_compat
sionna.config.xla_compat=True
class Model(tf.keras.Model):
    """Simulate PUSCH transmissions over a 3GPP 38.901 model.
    This model runs BER simulations for a multiuser MIMO uplink channel
    compliant with the 5G NR PUSCH specifications.
    You can pick different scenarios, i.e., channel models, perfect or
    estimated CSI, as well as different MIMO detectors (LMMSE or KBest).
    You can chosse to run simulations in either time ("time") or frequency ("freq")
    domains and configure different user speeds.
    Parameters
    ----------
    scenario : str, one of ["umi", "uma", "rma"]
        3GPP 38.901 channel model to be used
    perfect_csi : bool
        Determines if perfect CSI is assumed or if the CSI is estimated
    domain :  str, one of ["freq", "time"]
        Domain in which the simulations are carried out.
        Time domain modelling is typically more complex but allows modelling
        of realistic effects such as inter-symbol interference of subcarrier
        interference due to very high speeds.
    detector : str, one of ["lmmse", "kbest"]
        MIMO detector to be used. Note that each detector has additional
        parameters that can be configured in the source code of the _init_ call.
    speed: float
        User speed (m/s)
    Input
    -----
    batch_size : int
        Number of simultaneously simulated slots
    ebno_db : float
        Signal-to-noise-ratio
    Output
    ------
    b : [batch_size, num_tx, tb_size], tf.float
        Transmitted information bits
    b_hat : [batch_size, num_tx, tb_size], tf.float
        Decoded information bits
    """
    def __init__(self,
                 scenario,    # "umi", "uma", "rma"
                 perfect_csi, # bool
                 domain,      # "freq", "time"
                 detector,    # "lmmse", "kbest"
                 speed        # float
                ):
        super().__init__()
        self._scenario = scenario
        self._perfect_csi = perfect_csi
        self._domain = domain
        self._speed = speed
        self._carrier_frequency = 3.5e9
        self._subcarrier_spacing = 30e3
        self._num_tx = 4
        self._num_tx_ant = 4
        self._num_layers = 2
        self._num_rx_ant = 16
        self._mcs_index = 14
        self._mcs_table = 1
        self._num_prb = 16
        # Create PUSCHConfigs
        # PUSCHConfig for the first transmitter
        pusch_config = PUSCHConfig()
        pusch_config.carrier.subcarrier_spacing = self._subcarrier_spacing/1000
        pusch_config.carrier.n_size_grid = self._num_prb
        pusch_config.num_antenna_ports = self._num_tx_ant
        pusch_config.num_layers = self._num_layers
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 1
        pusch_config.dmrs.dmrs_port_set = list(range(self._num_layers))
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 3
        pusch_config.tb.mcs_index = self._mcs_index
        pusch_config.tb.mcs_table = self._mcs_table
        # Create PUSCHConfigs for the other transmitters by cloning of the first PUSCHConfig
        # and modifying the used DMRS ports.
        pusch_configs = [pusch_config]
        for i in range(1, self._num_tx):
            pc = pusch_config.clone()
            pc.dmrs.dmrs_port_set = list(range(i*self._num_layers, (i+1)*self._num_layers))
            pusch_configs.append(pc)
        # Create PUSCHTransmitter
        self._pusch_transmitter = PUSCHTransmitter(pusch_configs, output_domain=self._domain)
        # Create PUSCHReceiver
        self._l_min, self._l_max = time_lag_discrete_time_channel(self._pusch_transmitter.resource_grid.bandwidth)

        rx_tx_association = np.ones([1, self._num_tx], bool)
        stream_management = StreamManagement(rx_tx_association,
                                             self._num_layers)
        assert detector in["lmmse", "kbest"], "Unsupported MIMO detector"
        if detector=="lmmse":
            detector = LinearDetector(equalizer="lmmse",
                                      output="bit",
                                      demapping_method="maxlog",
                                      resource_grid=self._pusch_transmitter.resource_grid,
                                      stream_management=stream_management,
                                      constellation_type="qam",
                                      num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol)
        elif detector=="kbest":
            detector = KBestDetector(output="bit",
                                     num_streams=self._num_tx*self._num_layers,
                                     k=64,
                                     resource_grid=self._pusch_transmitter.resource_grid,
                                     stream_management=stream_management,
                                     constellation_type="qam",
                                     num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol)
        if self._perfect_csi:
            self._pusch_receiver = PUSCHReceiver(self._pusch_transmitter,
                                                 mimo_detector=detector,
                                                 input_domain=self._domain,
                                                 channel_estimator="perfect",
                                                 l_min = self._l_min)
        else:
            self._pusch_receiver = PUSCHReceiver(self._pusch_transmitter,
                                                 mimo_detector=detector,
                                                 input_domain=self._domain,
                                                 l_min = self._l_min)
        # Configure antenna arrays
        self._ut_array = AntennaArray(
                                 num_rows=1,
                                 num_cols=int(self._num_tx_ant/2),
                                 polarization="dual",
                                 polarization_type="cross",
                                 antenna_pattern="omni",
                                 carrier_frequency=self._carrier_frequency)
        self._bs_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_rx_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)
        # Configure the channel model
        if self._scenario == "umi":
            self._channel_model = UMi(carrier_frequency=self._carrier_frequency,
                                      o2i_model="low",
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "uma":
            self._channel_model = UMa(carrier_frequency=self._carrier_frequency,
                                      o2i_model="low",
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "rma":
            self._channel_model = RMa(carrier_frequency=self._carrier_frequency,
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        # Configure the actual channel
        if domain=="freq":
            self._channel = OFDMChannel(
                                self._channel_model,
                                self._pusch_transmitter.resource_grid,
                                normalize_channel=True,
                                return_channel=True)
        else:
            self._channel = TimeChannel(
                                self._channel_model,
                                self._pusch_transmitter.resource_grid.bandwidth,
                                self._pusch_transmitter.resource_grid.num_time_samples,
                                l_min=self._l_min,
                                l_max=self._l_max,
                                normalize_channel=True,
                                return_channel=True)
    def new_topology(self, batch_size):
        """Set new topology"""
        topology = gen_topology(batch_size,
                                self._num_tx,
                                self._scenario,
                                min_ut_velocity=self._speed,
                                max_ut_velocity=self._speed)
        self._channel_model.set_topology(*topology)
    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        x, b = self._pusch_transmitter(batch_size)
        no = ebnodb2no(ebno_db,
                       self._pusch_transmitter._num_bits_per_symbol,
                       self._pusch_transmitter._target_coderate,
                       self._pusch_transmitter.resource_grid)
        y, h = self._channel([x, no])
        if self._perfect_csi:
            b_hat = self._pusch_receiver([y, h, no])
        else:
            b_hat = self._pusch_receiver([y, no])
        return b, b_hat
```

    
We will now compare the PUSCH BLER performance over the 3GPP 38.901 UMi channel model with different detectors and either perfect or imperfect CSI. Note that these simulations might take some time depending or you available hardware. You can reduce the `batch_size` if the model does not fit into the memory of your GPU. Running the simulations in the time domain will significantly increase the complexity and you might need to decrease the `batch_size` further. The code will also run on CPU if
not GPU is available.
    
Note that the XLA compilation step can take several minutes (but the simulations will be much quicker compared to eager or graph mode.
    
If you do not want to run the simulation yourself, you can skip the next cell and visualize the results in the next cell.

```python
[37]:
```

```python
PUSCH_SIMS = {
    "scenario" : ["umi"],
    "domain" : ["freq"],
    "perfect_csi" : [True, False],
    "detector" : ["kbest", "lmmse"],
    "ebno_db" : list(range(-2,11)),
    "speed" : 3.0,
    "batch_size_freq" : 128,
    "batch_size_time" : 28, # Reduced batch size from time-domain modeling
    "bler" : [],
    "ber" : []
    }
start = time.time()
for scenario in PUSCH_SIMS["scenario"]:
    for domain in PUSCH_SIMS["domain"]:
        for perfect_csi in PUSCH_SIMS["perfect_csi"]:
            batch_size = PUSCH_SIMS["batch_size_freq"] if domain=="freq" else PUSCH_SIMS["batch_size_time"]
            for detector in PUSCH_SIMS["detector"]:
                model = Model(scenario, perfect_csi, domain, detector, PUSCH_SIMS["speed"])
                ber, bler = sim_ber(model,
                            PUSCH_SIMS["ebno_db"],
                            batch_size=batch_size,
                            max_mc_iter=1000,
                            num_target_block_errors=200)
                PUSCH_SIMS["ber"].append(list(ber.numpy()))
                PUSCH_SIMS["bler"].append(list(bler.numpy()))
PUSCH_SIMS["duration"] = time.time() - start
```


```python
WARNING:tensorflow:From /usr/local/lib/python3.8/dist-packages/tensorflow/python/autograph/pyct/static_analysis/liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
Instructions for updating:
Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -2.0 | 8.4141e-02 | 8.6523e-01 |      352915 |     4194304 |          443 |         512 |       174.7 |reached target block errors
     -1.0 | 3.8089e-02 | 5.2539e-01 |      159757 |     4194304 |          269 |         512 |         0.6 |reached target block errors
      0.0 | 1.3623e-02 | 2.3633e-01 |      114277 |     8388608 |          242 |        1024 |         1.2 |reached target block errors
      1.0 | 5.1620e-03 | 7.0312e-02 |      129906 |    25165824 |          216 |        3072 |         3.5 |reached target block errors
      2.0 | 1.8941e-03 | 2.2895e-02 |      142999 |    75497472 |          211 |        9216 |        10.6 |reached target block errors
      3.0 | 6.8813e-04 | 8.1787e-03 |      138539 |   201326592 |          201 |       24576 |        28.4 |reached target block errors
      4.0 | 2.8596e-04 | 3.1917e-03 |      147528 |   515899392 |          201 |       62976 |        73.4 |reached target block errors
      5.0 | 1.2890e-04 | 1.1800e-03 |      181657 |  1409286144 |          203 |      172032 |       201.4 |reached target block errors
      6.0 | 9.3746e-05 | 7.2742e-04 |      211149 |  2252341248 |          200 |      274944 |       321.9 |reached target block errors
      7.0 | 3.3643e-05 | 2.9883e-04 |      141111 |  4194304000 |          153 |      512000 |       599.8 |reached max iter
      8.0 | 2.6884e-05 | 1.8945e-04 |      112758 |  4194304000 |           97 |      512000 |       599.8 |reached max iter
      9.0 | 1.2900e-05 | 1.0742e-04 |       54107 |  4194304000 |           55 |      512000 |       599.8 |reached max iter
     10.0 | 6.6764e-06 | 4.6875e-05 |       28003 |  4194304000 |           24 |      512000 |       599.8 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -2.0 | 3.2366e-02 | 5.0195e-01 |      135754 |     4194304 |          257 |         512 |       171.7 |reached target block errors
     -1.0 | 1.5961e-02 | 2.4121e-01 |      133888 |     8388608 |          247 |        1024 |         1.0 |reached target block errors
      0.0 | 9.8741e-03 | 1.3411e-01 |      124245 |    12582912 |          206 |        1536 |         1.4 |reached target block errors
      1.0 | 4.3543e-03 | 6.5569e-02 |      127843 |    29360128 |          235 |        3584 |         3.4 |reached target block errors
      2.0 | 2.7020e-03 | 3.7109e-02 |      124664 |    46137344 |          209 |        5632 |         5.2 |reached target block errors
      3.0 | 1.5692e-03 | 2.1073e-02 |      125056 |    79691776 |          205 |        9728 |         9.3 |reached target block errors
      4.0 | 8.9329e-04 | 1.2251e-02 |      123642 |   138412032 |          207 |       16896 |        16.1 |reached target block errors
      5.0 | 5.4429e-04 | 7.2443e-03 |      125561 |   230686720 |          204 |       28160 |        26.8 |reached target block errors
      6.0 | 2.9038e-04 | 3.8869e-03 |      123013 |   423624704 |          201 |       51712 |        48.7 |reached target block errors
      7.0 | 2.1879e-04 | 2.7646e-03 |      130307 |   595591168 |          201 |       72704 |        69.2 |reached target block errors
      8.0 | 1.3987e-04 | 1.5751e-03 |      145486 |  1040187392 |          200 |      126976 |       121.0 |reached target block errors
      9.0 | 7.8782e-05 | 9.8387e-04 |      132505 |  1681915904 |          202 |      205312 |       195.2 |reached target block errors
     10.0 | 6.4391e-05 | 7.2391e-04 |      147192 |  2285895680 |          202 |      279040 |       266.3 |reached target block errors
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -2.0 | 1.5518e-01 | 9.9414e-01 |      650858 |     4194304 |          509 |         512 |        43.2 |reached target block errors
     -1.0 | 1.2702e-01 | 9.7266e-01 |      532773 |     4194304 |          498 |         512 |         0.6 |reached target block errors
      0.0 | 8.9072e-02 | 8.4961e-01 |      373596 |     4194304 |          435 |         512 |         0.6 |reached target block errors
      1.0 | 3.6323e-02 | 5.1367e-01 |      152349 |     4194304 |          263 |         512 |         0.6 |reached target block errors
      2.0 | 1.5565e-02 | 2.3438e-01 |      130566 |     8388608 |          240 |        1024 |         1.2 |reached target block errors
      3.0 | 8.8474e-03 | 1.1523e-01 |      148435 |    16777216 |          236 |        2048 |         2.3 |reached target block errors
      4.0 | 5.3303e-03 | 5.1270e-02 |      178854 |    33554432 |          210 |        4096 |         4.7 |reached target block errors
      5.0 | 3.5277e-03 | 2.8878e-02 |      207146 |    58720256 |          207 |        7168 |         8.2 |reached target block errors
      6.0 | 2.9088e-03 | 2.3093e-02 |      207410 |    71303168 |          201 |        8704 |        10.0 |reached target block errors
      7.0 | 2.5939e-03 | 1.8694e-02 |      228468 |    88080384 |          201 |       10752 |        12.4 |reached target block errors
      8.0 | 2.0388e-03 | 1.3672e-02 |      247983 |   121634816 |          203 |       14848 |        17.2 |reached target block errors
      9.0 | 1.7823e-03 | 1.2451e-02 |      239211 |   134217728 |          204 |       16384 |        19.0 |reached target block errors
     10.0 | 1.9271e-03 | 1.3739e-02 |      234399 |   121634816 |          204 |       14848 |        17.2 |reached target block errors
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -2.0 | 1.0344e-01 | 9.1992e-01 |      433850 |     4194304 |          471 |         512 |        43.9 |reached target block errors
     -1.0 | 6.6115e-02 | 7.2461e-01 |      277305 |     4194304 |          371 |         512 |         0.5 |reached target block errors
      0.0 | 4.3680e-02 | 5.3320e-01 |      183209 |     4194304 |          273 |         512 |         0.5 |reached target block errors
      1.0 | 2.1767e-02 | 2.8027e-01 |      182593 |     8388608 |          287 |        1024 |         1.0 |reached target block errors
      2.0 | 1.3199e-02 | 1.6536e-01 |      166083 |    12582912 |          254 |        1536 |         1.4 |reached target block errors
      3.0 | 7.3069e-03 | 8.9844e-02 |      153236 |    20971520 |          230 |        2560 |         2.4 |reached target block errors
      4.0 | 5.2081e-03 | 5.6920e-02 |      152911 |    29360128 |          204 |        3584 |         3.4 |reached target block errors
      5.0 | 4.0943e-03 | 4.3620e-02 |      154555 |    37748736 |          201 |        4608 |         4.3 |reached target block errors
      6.0 | 3.9949e-03 | 3.5006e-02 |      217828 |    54525952 |          233 |        6656 |         6.2 |reached target block errors
      7.0 | 2.3833e-03 | 2.0559e-02 |      189928 |    79691776 |          200 |        9728 |         9.1 |reached target block errors
      8.0 | 2.3061e-03 | 1.7578e-02 |      222467 |    96468992 |          207 |       11776 |        11.1 |reached target block errors
      9.0 | 2.3562e-03 | 1.8466e-02 |      217420 |    92274688 |          208 |       11264 |        10.6 |reached target block errors
     10.0 | 2.1590e-03 | 1.5325e-02 |      235439 |   109051904 |          204 |       13312 |        12.6 |reached target block errors
```
```python
[41]:
```

```python
# Load results (un-comment to show saved results from the cell above)
PUSCH_SIMS = eval("{'scenario': ['umi'], 'domain': ['freq'], 'perfect_csi': [True, False], 'detector': ['kbest', 'lmmse'], 'ebno_db': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'speed': 3.0, 'batch_size_freq': 128, 'batch_size_time': 28, 'bler': [[0.865234375, 0.525390625, 0.236328125, 0.0703125, 0.022894965277777776, 0.0081787109375, 0.0031916920731707315, 0.0011800130208333333, 0.0007274208566108007, 0.000298828125, 0.000189453125, 0.000107421875, 4.6875e-05], [0.501953125, 0.2412109375, 0.13411458333333334, 0.06556919642857142, 0.037109375, 0.021073190789473683, 0.012251420454545454, 0.007244318181818182, 0.0038869121287128713, 0.0027646346830985913, 0.0015751008064516128, 0.0009838684538653367, 0.0007239105504587156], [0.994140625, 0.97265625, 0.849609375, 0.513671875, 0.234375, 0.115234375, 0.05126953125, 0.028878348214285716, 0.023092830882352942, 0.018694196428571428, 0.013671875, 0.012451171875, 0.013739224137931034], [0.919921875, 0.724609375, 0.533203125, 0.2802734375, 0.16536458333333334, 0.08984375, 0.056919642857142856, 0.043619791666666664, 0.035006009615384616, 0.02055921052631579, 0.017578125, 0.018465909090909092, 0.01532451923076923]], 'ber': [[0.08414149284362793, 0.03808903694152832, 0.013622879981994629, 0.00516200065612793, 0.0018940899107191297, 0.0006881306568781534, 0.0002859627328267912, 0.00012890001138051352, 9.374645169220823e-05, 3.3643484115600584e-05, 2.6883602142333983e-05, 1.2900114059448242e-05, 6.676435470581055e-06], [0.032366275787353516, 0.015960693359375, 0.009874105453491211, 0.004354306629725865, 0.00270201943137429, 0.0015692459909539473, 0.0008932893926447088, 0.0005442922765558416, 0.0002903820264457476, 0.00021878598441540356, 0.00013986518306116904, 7.878217911185171e-05, 6.43913898992976e-05], [0.15517663955688477, 0.12702298164367676, 0.08907222747802734, 0.036322832107543945, 0.015564680099487305, 0.008847415447235107, 0.005330264568328857, 0.003527675356183733, 0.0029088469112620633, 0.0025938579014369418, 0.002038750155218716, 0.0017822608351707458, 0.001927071604235419], [0.10343790054321289, 0.06611466407775879, 0.043680429458618164, 0.0217667818069458, 0.013199090957641602, 0.007306861877441406, 0.005208117621285575, 0.004094309277004666, 0.003994941711425781, 0.002383282310084293, 0.0023060985233472743, 0.002356225794011896, 0.002158962763272799]], 'duration': 4399.180883407593}")
print("Simulation duration: {:1.2f} [h]".format(PUSCH_SIMS["duration"]/3600))
plt.figure()
plt.title("5G NR PUSCH over UMi Channel Model (8x16)")
plt.xlabel("SNR (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.xlim([PUSCH_SIMS["ebno_db"][0], PUSCH_SIMS["ebno_db"][-1]])
plt.ylim([1e-5, 1.0])
i = 0
legend = []
for scenario in PUSCH_SIMS["scenario"]:
    for domain in PUSCH_SIMS["domain"]:
        for perfect_csi in PUSCH_SIMS["perfect_csi"]:
            for detector in PUSCH_SIMS["detector"]:
                plt.semilogy(PUSCH_SIMS["ebno_db"], PUSCH_SIMS["bler"][i])
                i += 1
                csi = "Perf. CSI" if perfect_csi else "Imperf. CSI"
                det = "K-Best" if detector=="kbest" else "LMMSE"
                legend.append(det + " " + csi)
plt.legend(legend);
```


```python
Simulation duration: 1.22 [h]
```


    
Hopefully you have enjoyed this tutorial on Sionna’s 5G NR PUSCH module!
    
Please have a look at the <a class="reference external" href="https://nvlabs.github.io/sionna/api/sionna.html">API documentation</a> of the various components or the other available <a class="reference external" href="https://nvlabs.github.io/sionna/tutorials.html">tutorials</a> to learn more.