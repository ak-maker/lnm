# Introduction to Iterative Detection and Decoding<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Introduction_to_Iterative_Detection_and_Decoding.html#Introduction-to-Iterative-Detection-and-Decoding" title="Permalink to this headline"></a>
    
In this notebook, you will learn how to set-up an iterative detection and decoding (IDD) scheme (first presented in [1]) by combining multiple available components in Sionna.
    
For a gentle introduction to MIMO simulations, we refer to the notebooks <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Simple_MIMO_Simulation.html">“Simple MIMO Simulations”</a> and <a class="reference external" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html">“MIMO OFDM Transmissions over CDL”</a>.
    
You will evaluate the performance of IDD with OFDM MIMO detection and soft-input soft-output (SISO) LDPC decoding and compare it againts several non-iterative detectors, such as soft-output LMMSE, K-Best, and expectation propagation (EP), as well as iterative SISO MMSE-PIC detection [2].
    
For the non-IDD models, the signal processing pipeline looks as follows:
    

## Iterative Detection and Decoding<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Introduction_to_Iterative_Detection_and_Decoding.html#Iterative-Detection-and-Decoding" title="Permalink to this headline"></a>
    
The IDD MIMO receiver iteratively exchanges soft-information between the data detector and the channel decoder, which works as follows:
    
    
We denote by $\mathrm{L}^{D}$ the <em>a posteriori</em> information (represented by log-likelihood ratios, LLRs) and by $\mathrm{L}^{E} = \mathrm{L}^{D} - \mathrm{L}^{A}$ the extrinsic information, which corresponds to the information gain in $\mathrm{L}^{D}$ relative to the <em>a priori</em> information $\mathrm{L}^{A}$. The <em>a priori</em> LLRs represent soft information, provided to either the input of the detector (i.e., $\mathrm{L}^{A}_{Det}$) or the decoder (i.e.,
$\mathrm{L}^{A}_{Dec}$). While exchanging extrinsic information is standard for classical IDD, the SISO MMSE-PIC detector [2] turned out to work better when provided with the full <em>a posteriori</em> information from the decoder.
    
Originally, IDD was proposed with a resetting (Turbo) decoder [1]. However, state-of-the-art IDD with LDPC message passing decoding showed better performance with a non-resetting decoder [3], particularly for a low number of decoding iterations. Therefore, we will forward the decoder state (i.e., the check node to variable node messages) from each IDD iteration to the next.

# Table of Content
## GPU Configuration and Imports
## Non-IDD versus IDD Benchmarks
## Discussion-Optimizing IDD with Machine Learning
## Comments
## List of References
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Introduction_to_Iterative_Detection_and_Decoding.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

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
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sionna.mimo import StreamManagement
from sionna.utils import QAMSource, BinarySource, sim_ber, ebnodb2no, QAMSource, expand_to_rank
from sionna.mapping import Mapper, Constellation
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LinearDetector, KBestDetector, EPDetector, \
    RemoveNulledSubcarriers, MMSEPICDetector
from sionna.channel import GenerateOFDMChannel, OFDMChannel, RayleighBlockFading, gen_single_sector_topology
from sionna.channel.tr38901 import UMa, Antenna, PanelArray
from sionna.fec.ldpc import LDPC5GEncoder
from sionna.fec.ldpc import LDPC5GDecoder
```
```python
[2]:
```

```python
import tensorflow as tf
from tensorflow.keras import Model
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
```

## Non-IDD versus IDD Benchmarks<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Introduction_to_Iterative_Detection_and_Decoding.html#Non-IDD-versus-IDD-Benchmarks" title="Permalink to this headline"></a>

```python
[6]:
```

```python
# Range of SNR (dB)
snr_range_cest = np.linspace(ebno_db_min_cest, ebno_db_max_cest, num_steps)
snr_range_perf_csi = np.linspace(ebno_db_min_perf_csi, ebno_db_max_perf_csi, num_steps)
def run_idd_sim(snr_range, perfect_csi_rayleigh):
    lmmse = NonIddModel(detector="lmmse", perfect_csi_rayleigh=perfect_csi_rayleigh)
    k_best = NonIddModel(detector="k-best", perfect_csi_rayleigh=perfect_csi_rayleigh)
    ep = NonIddModel(detector="ep", perfect_csi_rayleigh=perfect_csi_rayleigh)
    idd2 = IddModel(num_idd_iter=2, perfect_csi_rayleigh=perfect_csi_rayleigh)
    idd3 = IddModel(num_idd_iter=3, perfect_csi_rayleigh=perfect_csi_rayleigh)
    ber_lmmse, bler_lmmse = sim_ber(lmmse,
                                    snr_range,
                                    batch_size=batch_size,
                                    max_mc_iter=num_iter,
                                    num_target_block_errors=int(batch_size * num_iter * 0.1))
    ber_ep, bler_ep = sim_ber(ep,
                              snr_range,
                              batch_size=batch_size,
                              max_mc_iter=num_iter,
                              num_target_block_errors=int(batch_size * num_iter * 0.1))
    ber_kbest, bler_kbest = sim_ber(k_best,
                                    snr_range,
                                    batch_size=batch_size,
                                    max_mc_iter=num_iter,
                                    num_target_block_errors=int(batch_size * num_iter * 0.1))
    ber_idd2, bler_idd2 = sim_ber(idd2,
                                  snr_range,
                                  batch_size=batch_size,
                                  max_mc_iter=num_iter,
                                  num_target_block_errors=int(batch_size * num_iter * 0.1))
    ber_idd3, bler_idd3 = sim_ber(idd3,
                                  snr_range,
                                  batch_size=batch_size,
                                  max_mc_iter=num_iter,
                                  num_target_block_errors=int(batch_size * num_iter * 0.1))
    return bler_lmmse, bler_ep, bler_kbest, bler_idd2, bler_idd3

BLER = {}
# Perfect CSI
bler_lmmse, bler_ep, bler_kbest, bler_idd2, bler_idd3 = run_idd_sim(snr_range_perf_csi, perfect_csi_rayleigh=True)
BLER['Perf. CSI / LMMSE'] = bler_lmmse
BLER['Perf. CSI / EP'] = bler_ep
BLER['Perf. CSI / K-Best'] = bler_kbest
BLER['Perf. CSI / IDD2'] = bler_idd2
BLER['Perf. CSI / IDD3'] = bler_idd3
# Estimated CSI
bler_lmmse, bler_ep, bler_kbest, bler_idd2, bler_idd3 = run_idd_sim(snr_range_cest, perfect_csi_rayleigh=False)
BLER['Ch. Est. / LMMSE'] = bler_lmmse
BLER['Ch. Est. / EP'] = bler_ep
BLER['Ch. Est. / K-Best'] = bler_kbest
BLER['Ch. Est. / IDD2'] = bler_idd2
BLER['Ch. Est. / IDD3'] = bler_idd3
```


```python
WARNING:tensorflow:From /usr/local/lib/python3.8/dist-packages/tensorflow/python/autograph/pyct/static_analysis/liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
Instructions for updating:
Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 2.1174e-01 | 1.0000e+00 |      249776 |     1179648 |         1024 |        1024 |         5.7 |reached target block errors
     -9.0 | 1.8563e-01 | 1.0000e+00 |      218982 |     1179648 |         1024 |        1024 |         0.3 |reached target block errors
     -8.0 | 1.0665e-01 | 9.5898e-01 |      125808 |     1179648 |          982 |        1024 |         0.3 |reached target block errors
     -7.0 | 1.6909e-02 | 3.3828e-01 |       49867 |     2949120 |          866 |        2560 |         0.9 |reached target block errors
     -6.0 | 1.1546e-03 | 3.3143e-02 |       33030 |    28606464 |          823 |       24832 |         8.3 |reached target block errors
     -5.0 | 6.4903e-05 | 2.3804e-03 |        2450 |    37748736 |           78 |       32768 |        11.0 |reached max iter
     -4.0 | 2.6491e-07 | 1.2207e-04 |          10 |    37748736 |            4 |       32768 |        10.9 |reached max iter
     -3.0 | 0.0000e+00 | 0.0000e+00 |           0 |    37748736 |            0 |       32768 |        10.9 |reached max iter
Simulation stopped as no error occurred @ EbNo = -3.0 dB.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 2.1091e-01 | 1.0000e+00 |      248794 |     1179648 |         1024 |        1024 |         3.8 |reached target block errors
     -9.0 | 1.8493e-01 | 1.0000e+00 |      218155 |     1179648 |         1024 |        1024 |         0.4 |reached target block errors
     -8.0 | 9.8120e-02 | 9.5508e-01 |      115747 |     1179648 |          978 |        1024 |         0.4 |reached target block errors
     -7.0 | 1.1955e-02 | 2.8027e-01 |       42309 |     3538944 |          861 |        3072 |         1.2 |reached target block errors
     -6.0 | 4.5117e-04 | 1.6541e-02 |       17031 |    37748736 |          542 |       32768 |        12.7 |reached max iter
     -5.0 | 1.5471e-05 | 7.6294e-04 |         584 |    37748736 |           25 |       32768 |        12.5 |reached max iter
     -4.0 | 0.0000e+00 | 0.0000e+00 |           0 |    37748736 |            0 |       32768 |        12.7 |reached max iter
Simulation stopped as no error occurred @ EbNo = -4.0 dB.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 2.1164e-01 | 1.0000e+00 |      249660 |     1179648 |         1024 |        1024 |         6.3 |reached target block errors
     -9.0 | 1.8471e-01 | 1.0000e+00 |      217898 |     1179648 |         1024 |        1024 |         2.1 |reached target block errors
     -8.0 | 1.1558e-01 | 9.9512e-01 |      136338 |     1179648 |         1019 |        1024 |         2.1 |reached target block errors
     -7.0 | 1.8488e-02 | 5.6315e-01 |       32714 |     1769472 |          865 |        1536 |         3.2 |reached target block errors
     -6.0 | 9.5951e-04 | 4.6762e-02 |       19525 |    20348928 |          826 |       17664 |        36.3 |reached target block errors
     -5.0 | 1.9338e-05 | 1.4343e-03 |         730 |    37748736 |           47 |       32768 |        67.2 |reached max iter
     -4.0 | 1.5895e-07 | 6.1035e-05 |           6 |    37748736 |            2 |       32768 |        67.2 |reached max iter
     -3.0 | 0.0000e+00 | 0.0000e+00 |           0 |    37748736 |            0 |       32768 |        67.0 |reached max iter
Simulation stopped as no error occurred @ EbNo = -3.0 dB.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 2.1164e-01 | 1.0000e+00 |      249662 |     1179648 |         1024 |        1024 |         8.2 |reached target block errors
     -9.0 | 1.8639e-01 | 1.0000e+00 |      219878 |     1179648 |         1024 |        1024 |         0.6 |reached target block errors
     -8.0 | 8.6596e-02 | 6.9844e-01 |      127691 |     1474560 |          894 |        1280 |         0.7 |reached target block errors
     -7.0 | 3.4296e-03 | 4.6535e-02 |       69788 |    20348928 |          822 |       17664 |        10.3 |reached target block errors
     -6.0 | 6.1750e-05 | 8.5449e-04 |        2331 |    37748736 |           28 |       32768 |        18.9 |reached max iter
     -5.0 | 1.0596e-06 | 3.0518e-05 |          40 |    37748736 |            1 |       32768 |        18.9 |reached max iter
     -4.0 | 0.0000e+00 | 0.0000e+00 |           0 |    37748736 |            0 |       32768 |        18.9 |reached max iter
Simulation stopped as no error occurred @ EbNo = -4.0 dB.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 2.1188e-01 | 1.0000e+00 |      249945 |     1179648 |         1024 |        1024 |         7.8 |reached target block errors
     -9.0 | 1.8657e-01 | 9.9805e-01 |      220084 |     1179648 |         1022 |        1024 |         0.8 |reached target block errors
     -8.0 | 7.6331e-02 | 5.7943e-01 |      135065 |     1769472 |          890 |        1536 |         1.2 |reached target block errors
     -7.0 | 1.6263e-03 | 1.7822e-02 |       61390 |    37748736 |          584 |       32768 |        25.7 |reached max iter
     -6.0 | 1.7325e-05 | 2.4414e-04 |         654 |    37748736 |            8 |       32768 |        25.5 |reached max iter
     -5.0 | 0.0000e+00 | 0.0000e+00 |           0 |    37748736 |            0 |       32768 |        25.7 |reached max iter
Simulation stopped as no error occurred @ EbNo = -5.0 dB.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 2.9984e-01 | 1.0000e+00 |      353711 |     1179648 |         1024 |        1024 |        12.3 |reached target block errors
     -8.0 | 2.6258e-01 | 1.0000e+00 |      309751 |     1179648 |         1024 |        1024 |         0.4 |reached target block errors
     -6.0 | 2.2838e-01 | 1.0000e+00 |      269403 |     1179648 |         1024 |        1024 |         0.4 |reached target block errors
     -4.0 | 1.4912e-01 | 9.0723e-01 |      175910 |     1179648 |          929 |        1024 |         0.4 |reached target block errors
     -2.0 | 4.0182e-02 | 3.2930e-01 |      118503 |     2949120 |          843 |        2560 |         1.1 |reached target block errors
      0.0 | 9.7169e-03 | 8.6571e-02 |      106028 |    10911744 |          820 |        9472 |         4.1 |reached target block errors
      2.0 | 2.8845e-03 | 2.3529e-02 |      108885 |    37748736 |          771 |       32768 |        14.3 |reached max iter
      4.0 | 1.2734e-03 | 9.0942e-03 |       48069 |    37748736 |          298 |       32768 |        14.1 |reached max iter
      6.0 | 7.8371e-04 | 5.0354e-03 |       29584 |    37748736 |          165 |       32768 |        14.3 |reached max iter
      8.0 | 9.9370e-04 | 6.1340e-03 |       37511 |    37748736 |          201 |       32768 |        14.2 |reached max iter
     10.0 | 7.6546e-04 | 4.9744e-03 |       28895 |    37748736 |          163 |       32768 |        14.4 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 3.0051e-01 | 1.0000e+00 |      354491 |     1179648 |         1024 |        1024 |         6.8 |reached target block errors
     -8.0 | 2.6452e-01 | 1.0000e+00 |      312037 |     1179648 |         1024 |        1024 |         0.5 |reached target block errors
     -6.0 | 2.2600e-01 | 1.0000e+00 |      266602 |     1179648 |         1024 |        1024 |         0.5 |reached target block errors
     -4.0 | 1.4565e-01 | 9.0625e-01 |      171821 |     1179648 |          928 |        1024 |         0.5 |reached target block errors
     -2.0 | 4.0697e-02 | 3.4258e-01 |      120021 |     2949120 |          877 |        2560 |         1.3 |reached target block errors
      0.0 | 7.6236e-03 | 6.9633e-02 |      103421 |    13565952 |          820 |       11776 |         5.8 |reached target block errors
      2.0 | 1.9709e-03 | 1.5198e-02 |       74400 |    37748736 |          498 |       32768 |        16.0 |reached max iter
      4.0 | 9.2236e-04 | 6.6528e-03 |       34818 |    37748736 |          218 |       32768 |        16.1 |reached max iter
      6.0 | 7.6371e-04 | 5.6152e-03 |       28829 |    37748736 |          184 |       32768 |        16.1 |reached max iter
      8.0 | 1.0540e-03 | 6.3477e-03 |       39788 |    37748736 |          208 |       32768 |        15.9 |reached max iter
     10.0 | 9.5132e-04 | 6.2866e-03 |       35911 |    37748736 |          206 |       32768 |        16.1 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 3.0523e-01 | 1.0000e+00 |      360064 |     1179648 |         1024 |        1024 |         8.7 |reached target block errors
     -8.0 | 2.6928e-01 | 1.0000e+00 |      317651 |     1179648 |         1024 |        1024 |         2.2 |reached target block errors
     -6.0 | 2.3226e-01 | 1.0000e+00 |      273984 |     1179648 |         1024 |        1024 |         2.2 |reached target block errors
     -4.0 | 1.6056e-01 | 9.7266e-01 |      189406 |     1179648 |          996 |        1024 |         2.2 |reached target block errors
     -2.0 | 3.5856e-02 | 3.6936e-01 |       95168 |     2654208 |          851 |        2304 |         5.0 |reached target block errors
      0.0 | 6.1384e-03 | 5.8168e-02 |       99566 |    16220160 |          819 |       14080 |        30.4 |reached target block errors
      2.0 | 1.2737e-03 | 1.0498e-02 |       48079 |    37748736 |          344 |       32768 |        70.6 |reached max iter
      4.0 | 9.2064e-04 | 5.4932e-03 |       34753 |    37748736 |          180 |       32768 |        70.7 |reached max iter
      6.0 | 9.2936e-04 | 5.0049e-03 |       35082 |    37748736 |          164 |       32768 |        70.9 |reached max iter
      8.0 | 7.4246e-04 | 4.6082e-03 |       28027 |    37748736 |          151 |       32768 |        70.6 |reached max iter
     10.0 | 7.8665e-04 | 5.1270e-03 |       29695 |    37748736 |          168 |       32768 |        70.6 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 3.0156e-01 | 1.0000e+00 |      355731 |     1179648 |         1024 |        1024 |        11.2 |reached target block errors
     -8.0 | 2.6622e-01 | 1.0000e+00 |      314046 |     1179648 |         1024 |        1024 |         0.7 |reached target block errors
     -6.0 | 2.3190e-01 | 1.0000e+00 |      273565 |     1179648 |         1024 |        1024 |         0.7 |reached target block errors
     -4.0 | 1.5164e-01 | 8.2324e-01 |      178884 |     1179648 |          843 |        1024 |         0.7 |reached target block errors
     -2.0 | 3.0409e-02 | 2.0581e-01 |      143489 |     4718592 |          843 |        4096 |         2.8 |reached target block errors
      0.0 | 4.0669e-03 | 3.0469e-02 |      125935 |    30965760 |          819 |       26880 |        18.0 |reached target block errors
      2.0 | 1.4712e-03 | 9.0942e-03 |       55536 |    37748736 |          298 |       32768 |        22.0 |reached max iter
      4.0 | 6.7668e-04 | 3.9368e-03 |       25544 |    37748736 |          129 |       32768 |        21.9 |reached max iter
      6.0 | 7.9192e-04 | 4.2725e-03 |       29894 |    37748736 |          140 |       32768 |        21.7 |reached max iter
      8.0 | 8.3711e-04 | 5.1575e-03 |       31600 |    37748736 |          169 |       32768 |        22.3 |reached max iter
     10.0 | 7.4954e-04 | 4.7913e-03 |       28294 |    37748736 |          157 |       32768 |        22.0 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
    -10.0 | 3.0150e-01 | 1.0000e+00 |      355662 |     1179648 |         1024 |        1024 |        11.6 |reached target block errors
     -8.0 | 2.6787e-01 | 1.0000e+00 |      315993 |     1179648 |         1024 |        1024 |         0.9 |reached target block errors
     -6.0 | 2.3101e-01 | 1.0000e+00 |      272511 |     1179648 |         1024 |        1024 |         0.9 |reached target block errors
     -4.0 | 1.4615e-01 | 7.9844e-01 |      215510 |     1474560 |         1022 |        1280 |         1.2 |reached target block errors
     -2.0 | 2.3105e-02 | 1.4453e-01 |      156722 |     6782976 |          851 |        5888 |         5.3 |reached target block errors
      0.0 | 3.5593e-03 | 2.1973e-02 |      134358 |    37748736 |          720 |       32768 |        29.4 |reached max iter
      2.0 | 1.3395e-03 | 7.1411e-03 |       50564 |    37748736 |          234 |       32768 |        29.2 |reached max iter
      4.0 | 7.4816e-04 | 3.9062e-03 |       28242 |    37748736 |          128 |       32768 |        29.4 |reached max iter
      6.0 | 7.6657e-04 | 3.9368e-03 |       28937 |    37748736 |          129 |       32768 |        29.4 |reached max iter
      8.0 | 8.2713e-04 | 4.1504e-03 |       31223 |    37748736 |          136 |       32768 |        29.3 |reached max iter
     10.0 | 7.2932e-04 | 4.2114e-03 |       27531 |    37748736 |          138 |       32768 |        29.4 |reached max iter
```

    
Finally, we plot the simulation results and observe that IDD outperforms the non-iterative methods by about 1 dB in the scenario with iid Rayleigh fading channels and perfect CSI. In the scenario with 3GPP UMa channels and estimated CSI, IDD performs slightly better than K-best, at considerably lower runtime.

```python
[7]:
```

```python
fig, ax = plt.subplots(1,2, figsize=(16,7))
fig.suptitle(f"{n_ue}x{NUM_RX_ANT} MU-MIMO UL | {2**num_bits_per_symbol}-QAM")
## Perfect CSI Rayleigh
ax[0].set_title("Perfect CSI iid. Rayleigh")
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / LMMSE'], 'x-', label='LMMSE', c='C0')
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / EP'], 'o--', label='EP', c='C0')
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / K-Best'], 's-.', label='K-Best', c='C0')
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / IDD2'], 'd:', label=r'IDD $I=2$', c='C1')
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / IDD3'], 'd:', label=r'IDD $I=3$', c='C2')
ax[0].set_xlabel(r"$E_b/N0$")
ax[0].set_ylabel("BLER")
ax[0].set_ylim((1e-4, 1.0))
ax[0].legend()
ax[0].grid(True)
## Estimated CSI Rayleigh
ax[1].set_title("Estimated CSI 3GPP UMa")
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / LMMSE'], 'x-', label='LMMSE', c='C0')
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / EP'], 'o--', label='EP', c='C0')
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / K-Best'], 's-.', label='K-Best', c='C0')
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / IDD2'], 'd:', label=r'IDD $I=2$', c='C1')
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / IDD3'], 'd:', label=r'IDD $I=3$', c='C2')
ax[1].set_xlabel(r"$E_b/N0$")
ax[1].set_ylabel("BLER")
ax[1].set_ylim((1e-3, 1.0))
ax[1].legend()
ax[1].grid(True)
plt.show()
```

## Discussion-Optimizing IDD with Machine Learning<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Introduction_to_Iterative_Detection_and_Decoding.html#Discussion-Optimizing-IDD-with-Machine-Learning" title="Permalink to this headline"></a>
    
Recent work [4] showed that IDD can be significantly improved by deep-unfolding, which applies machine learning to automatically tune hyperparameters of classical algorithms. The proposed <em>Deep-Unfolded Interleaved Detection and Decoding</em> method showed performance gains of up to 1.4 dB at the same computational complexity. A link to the simulation code is available in the <a class="reference external" href="https://nvlabs.github.io/sionna/made_with_sionna.html#duidd-deep-unfolded-interleaved-detection-and-decoding-for-mimo-wireless-systems">“Made with
Sionna”</a> section.

## Comments<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Introduction_to_Iterative_Detection_and_Decoding.html#Comments" title="Permalink to this headline"></a>
 
- As discussed in [3], IDD receivers with a non-resetting decoder converge faster than with resetting decoders. However, a resetting decoder (which does not forward `msg_vn`) might perform slightly better for a large number of message passing decoding iterations. Among other quantities, a scaling of the forwarded decoder state is optimized in the DUIDD receiver [4].
- With estimated channels, we observed that the MMSE-PIC output LLRs become large, much larger as with non-iterative receive processing.
## List of References<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Introduction_to_Iterative_Detection_and_Decoding.html#List-of-References" title="Permalink to this headline"></a>
    
[1] B. Hochwald and S. Ten Brink, <a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/1194444">“Achieving near-capacity on a multiple-antenna channel,”</a> IEEE Trans. Commun., vol. 51, no. 3, pp. 389–399, Mar. 2003.
    
[2] C. Studer, S. Fateh, and D. Seethaler, <a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/5779722">“ASIC implementation of soft-input soft-output MIMO detection using MMSE parallel interference cancellation,”</a> IEEE Journal of Solid-State Circuits, vol. 46, no. 7, pp. 1754–1765, Jul. 2011.
    
[3] W.-C. Sun, W.-H. Wu, C.-H. Yang, and Y.-L. Ueng, <a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/7272776">“An iterative detection and decoding receiver for LDPC-coded MIMO systems,”</a> IEEE Trans. Circuits Syst. I, vol. 62, no. 10, pp. 2512–2522, Oct. 2015.
    
[4] R. Wiesmayr, C. Dick, J. Hoydis, and C. Studer, <a class="reference external" href="https://arxiv.org/abs/2212.07816">“DUIDD: Deep-unfolded interleaved detection and decoding for MIMO wireless systems,”</a> in Asilomar Conf. Signals, Syst., Comput., Oct. 2022.