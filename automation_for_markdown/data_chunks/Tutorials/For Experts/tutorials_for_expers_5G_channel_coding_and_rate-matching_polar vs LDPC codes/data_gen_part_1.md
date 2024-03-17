# 5G Channel Coding and Rate-Matching: Polar vs. LDPC Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html#5G-Channel-Coding-and-Rate-Matching:-Polar-vs. LDPC-Codes" title="Permalink to this headline"></a>
    
<em>“For block lengths of about 500, an IBM 7090 computer requires about 0.1 seconds per iteration to decode a block by probabilistic decoding scheme. Consequently, many hours of computation time are necessary to evaluate even a</em> $P(e)$ <em>in the order of</em> ${10^{-4}}$ <em>.”</em> Robert G. Gallager, 1963 [7]
    
In this notebook, you will learn about the different coding schemes in 5G NR and how rate-matching works (cf. 3GPP TS 38.212 [3]). The coding schemes are compared under different length/rate settings and for different decoders.
    
You will learn about the following components:
 
- 5G low-density parity-checks (LDPC) codes [7]. These codes support - without further segmentation - up to <em>k=8448</em> information bits per codeword [3] for a wide range of coderates.
- Polar codes [1] including CRC concatenation and rate-matching for 5G compliant en-/decoding is implemented for the Polar uplink control channel (UCI) [3]. Besides Polar codes, Reed-Muller (RM) codes and several decoders are available:
 
- Successive cancellation (SC) decoding [1]
- Successive cancellation list (SCL) decoding [2]
- Hybrid SC / SCL decoding for enhanced throughput
- Iterative belief propagation (BP) decoding [6]



    
Further, we will demonstrate the basic functionality of the Sionna forward error correction (FEC) module which also includes support for:
 
- Convolutional codes with non-recursive encoding and Viterbi/BCJR decoding
- Turbo codes and iterative BCJR decoding
- Ordered statistics decoding (OSD) for any binary, linear code
- Interleaving and scrambling

    
For additional technical background we refer the interested reader to [4,5,8].
    
Please note that block segmentation is not implemented as it only concatenates multiple code blocks without increasing the effective codewords length (from decoder’s perspective).
    
Some simulations in this notebook require severe simulation time, in particular if parameter sweeps are involved (e.g., different length comparisons). Please keep in mind that each cell in this notebook already contains the pre-computed outputs and no new execution is required to understand the examples.

# Table of Content
## GPU Configuration and Imports
## BER Performance of 5G Coding Schemes
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

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
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.polar import PolarEncoder, Polar5GEncoder, PolarSCLDecoder, Polar5GDecoder, PolarSCDecoder
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.fec.polar.utils import generate_5g_ranking, generate_rm_code
from sionna.fec.conv import ConvEncoder, ViterbiDecoder, BCJRDecoder
from sionna.fec.turbo import TurboEncoder, TurboDecoder
from sionna.fec.linear import OSDecoder
from sionna.utils import BinarySource, ebnodb2no
from sionna.utils.metrics import  count_block_errors
from sionna.channel import AWGN
from sionna.utils.plotting import PlotBER
```
```python
[2]:
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time # for throughput measurements
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

## BER Performance of 5G Coding Schemes<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html#BER-Performance-of-5G-Coding-Schemes" title="Permalink to this headline"></a>
    
Let us first focus on short length coding, e.g., for internet of things (IoT) and ultra-reliable low-latency communications (URLLC). We aim to reproduce similar results as in [9] for the coding schemes supported by Sionna.
    
For a detailed explanation of the `PlotBER` class, we refer to the example notebook on <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html">Bit-Interleaved Coded Modulation</a>.
    
The Sionna API allows to pass an encoder object/layer to the decoder initialization for the 5G decoders. This means that the decoder is directly <em>associated</em> to a specific encoder and <em>knows</em> all relevant code parameters. Please note that - of course - no data or information bits are exchanged between these two associated components. It just simplifies handling of the code parameters, in particular, if rate-matching is used.
    
Let us define the system model first. We use encoder and decoder as input parameter such that the model remains flexible w.r.t. the coding scheme.

```python
[4]:
```

```python
class System_Model(tf.keras.Model):
    """System model for channel coding BER simulations.
    This model allows to simulate BERs over an AWGN channel with
    QAM modulation. Arbitrary FEC encoder/decoder layers can be used to
    initialize the model.
    Parameters
    ----------
        k: int
            number of information bits per codeword.
        n: int
            codeword length.
        num_bits_per_symbol: int
            number of bits per QAM symbol.
        encoder: Keras layer
            A Keras layer that encodes information bit tensors.
        decoder: Keras layer
            A Keras layer that decodes llr tensors.
        demapping_method: str
            A string denoting the demapping method. Can be either "app" or "maxlog".
        sim_esno: bool
            A boolean defaults to False. If true, no rate-adjustment is done for the SNR calculation.
         cw_estiamtes: bool
            A boolean defaults to False. If true, codewords instead of information estimates are returned.
    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.
        ebno_db: float or tf.float
            A float defining the simulation SNR.
    Output
    ------
        (u, u_hat):
            Tuple:
        u: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the transmitted information bits.
        u_hat: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the estimated information bits.
    """
    def __init__(self,
                 k,
                 n,
                 num_bits_per_symbol,
                 encoder,
                 decoder,
                 demapping_method="app",
                 sim_esno=False,
                 cw_estimates=False):
        super().__init__()
        # store values internally
        self.k = k
        self.n = n
        self.sim_esno = sim_esno # disable rate-adjustment for SNR calc
        self.cw_estimates=cw_estimates # if true codewords instead of info bits are returned
        # number of bit per QAM symbol
        self.num_bits_per_symbol = num_bits_per_symbol
        # init components
        self.source = BinarySource()
        # initialize mapper and demapper for constellation object
        self.constellation = Constellation("qam",
                                num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(demapping_method,
                                 constellation=self.constellation)
        # the channel can be replaced by more sophisticated models
        self.channel = AWGN()
        # FEC encoder / decoder
        self.encoder = encoder
        self.decoder = decoder
    @tf.function() # enable graph mode for increased throughputs
    def call(self, batch_size, ebno_db):
        # calculate noise variance
        if self.sim_esno:
                no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=1,
                       coderate=1)
        else:
            no = ebnodb2no(ebno_db,
                           num_bits_per_symbol=self.num_bits_per_symbol,
                           coderate=self.k/self.n)
        u = self.source([batch_size, self.k]) # generate random data
        c = self.encoder(u) # explicitly encode
        x = self.mapper(c) # map c to symbols x
        y = self.channel([x, no]) # transmit over AWGN channel
        llr_ch = self.demapper([y, no]) # demap y to LLRs
        u_hat = self.decoder(llr_ch) # run FEC decoder (incl. rate-recovery)
        if self.cw_estimates:
            return c, u_hat
        return u, u_hat
```

    
And let us define the codes to be simulated.

```python
[5]:
```

```python
# code parameters
k = 64 # number of information bits per codeword
n = 128 # desired codeword length
# Create list of encoder/decoder pairs to be analyzed.
# This allows automated evaluation of the whole list later.
codes_under_test = []
# 5G LDPC codes with 20 BP iterations
enc = LDPC5GEncoder(k=k, n=n)
dec = LDPC5GDecoder(enc, num_iter=20)
name = "5G LDPC BP-20"
codes_under_test.append([enc, dec, name])
# Polar Codes (SC decoding)
enc = Polar5GEncoder(k=k, n=n)
dec = Polar5GDecoder(enc, dec_type="SC")
name = "5G Polar+CRC SC"
codes_under_test.append([enc, dec, name])
# Polar Codes (SCL decoding) with list size 8.
# The CRC is automatically added by the layer.
enc = Polar5GEncoder(k=k, n=n)
dec = Polar5GDecoder(enc, dec_type="SCL", list_size=8)
name = "5G Polar+CRC SCL-8"
codes_under_test.append([enc, dec, name])
### non-5G coding schemes
# RM codes with SCL decoding
f,_,_,_,_ = generate_rm_code(3,7) # equals k=64 and n=128 code
enc = PolarEncoder(f, n)
dec = PolarSCLDecoder(f, n, list_size=8)
name = "Reed Muller (RM) SCL-8"
codes_under_test.append([enc, dec, name])
# Conv. code with Viterbi decoding
enc = ConvEncoder(rate=1/2, constraint_length=8)
dec = ViterbiDecoder(gen_poly=enc.gen_poly, method="soft_llr")
name = "Conv. Code Viterbi (constraint length 8)"
codes_under_test.append([enc, dec, name])
# Turbo. codes
enc = TurboEncoder(rate=1/2, constraint_length=4, terminate=False) # no termination used due to the rate loss
dec = TurboDecoder(enc, num_iter=8)
name = "Turbo Code (constraint length 4)"
codes_under_test.append([enc, dec, name])
```


```python
Warning: 5G Polar codes use an integrated CRC that cannot be materialized with SC decoding and, thus, causes a degraded performance. Please consider SCL decoding instead.
```

    
<em>Remark</em>: some of the coding schemes are not 5G relevant, but are included in this comparison for the sake of completeness.
    
Generate a new BER plot figure to save and plot simulation results efficiently.

```python
[6]:
```

```python
ber_plot128 = PlotBER(f"Performance of Short Length Codes (k={k}, n={n})")
```

    
And run the BER simulation for each code.

```python
[7]:
```

```python
num_bits_per_symbol = 2 # QPSK
ebno_db = np.arange(0, 5, 0.5) # sim SNR range
# run ber simulations for each code we have added to the list
for code in codes_under_test:
    print("\nRunning: " + code[2])
    # generate a new model with the given encoder/decoder
    model = System_Model(k=k,
                         n=n,
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=code[0],
                         decoder=code[1])
    # the first argument must be a callable (function) that yields u and u_hat for batch_size and ebno
    ber_plot128.simulate(model, # the function have defined previously
                         ebno_dbs=ebno_db, # SNR to simulate
                         legend=code[2], # legend string for plotting
                         max_mc_iter=100, # run 100 Monte Carlo runs per SNR point
                         num_target_block_errors=1000, # continue with next SNR point after 1000 bit errors
                         batch_size=10000, # batch-size per Monte Carlo run
                         soft_estimates=False, # the model returns hard-estimates
                         early_stop=True, # stop simulation if no error has been detected at current SNR point
                         show_fig=False, # we show the figure after all results are simulated
                         add_bler=True, # in case BLER is also interesting
                         forward_keyboard_interrupt=True); # should be True in a loop
# and show the figure
ber_plot128(ylim=(1e-5, 1), show_bler=False) # we set the ylim to 1e-5 as otherwise more extensive simulations would be required for accurate curves.

```


```python

Running: 5G LDPC BP-20
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6724e-01 | 8.5960e-01 |      107031 |      640000 |         8596 |       10000 |         2.5 |reached target block errors
      0.5 | 1.2503e-01 | 6.9560e-01 |       80018 |      640000 |         6956 |       10000 |         0.1 |reached target block errors
      1.0 | 8.8070e-02 | 5.1250e-01 |       56365 |      640000 |         5125 |       10000 |         0.1 |reached target block errors
      1.5 | 5.2178e-02 | 3.1040e-01 |       33394 |      640000 |         3104 |       10000 |         0.1 |reached target block errors
      2.0 | 2.5391e-02 | 1.5390e-01 |       16250 |      640000 |         1539 |       10000 |         0.1 |reached target block errors
      2.5 | 1.0280e-02 | 6.4150e-02 |       13159 |     1280000 |         1283 |       20000 |         0.2 |reached target block errors
      3.0 | 3.3266e-03 | 2.0760e-02 |       10645 |     3200000 |         1038 |       50000 |         0.5 |reached target block errors
      3.5 | 9.5947e-04 | 6.0882e-03 |       10439 |    10880000 |         1035 |      170000 |         1.6 |reached target block errors
      4.0 | 2.0158e-04 | 1.3400e-03 |        9676 |    48000000 |         1005 |      750000 |         7.1 |reached target block errors
      4.5 | 4.0484e-05 | 2.5700e-04 |        2591 |    64000000 |          257 |     1000000 |         9.5 |reached max iter
Running: 5G Polar+CRC SC
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 4.0980e-01 | 9.5260e-01 |      262275 |      640000 |         9526 |       10000 |         4.8 |reached target block errors
      0.5 | 3.6786e-01 | 8.9330e-01 |      235431 |      640000 |         8933 |       10000 |         0.0 |reached target block errors
      1.0 | 3.0912e-01 | 7.9180e-01 |      197837 |      640000 |         7918 |       10000 |         0.0 |reached target block errors
      1.5 | 2.4575e-01 | 6.5500e-01 |      157277 |      640000 |         6550 |       10000 |         0.0 |reached target block errors
      2.0 | 1.7330e-01 | 4.7950e-01 |      110914 |      640000 |         4795 |       10000 |         0.0 |reached target block errors
      2.5 | 1.0759e-01 | 3.1080e-01 |       68859 |      640000 |         3108 |       10000 |         0.0 |reached target block errors
      3.0 | 6.0220e-02 | 1.7530e-01 |       38541 |      640000 |         1753 |       10000 |         0.0 |reached target block errors
      3.5 | 2.8487e-02 | 8.3300e-02 |       36463 |     1280000 |         1666 |       20000 |         0.1 |reached target block errors
      4.0 | 1.0125e-02 | 3.1375e-02 |       25920 |     2560000 |         1255 |       40000 |         0.1 |reached target block errors
      4.5 | 3.1420e-03 | 9.7091e-03 |       22120 |     7040000 |         1068 |      110000 |         0.4 |reached target block errors
Running: 5G Polar+CRC SCL-8
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 3.3954e-01 | 7.9370e-01 |      217305 |      640000 |         7937 |       10000 |        16.3 |reached target block errors
      0.5 | 2.5614e-01 | 6.2320e-01 |      163931 |      640000 |         6232 |       10000 |         2.3 |reached target block errors
      1.0 | 1.7195e-01 | 4.2970e-01 |      110045 |      640000 |         4297 |       10000 |         2.3 |reached target block errors
      1.5 | 9.5338e-02 | 2.4580e-01 |       61016 |      640000 |         2458 |       10000 |         2.3 |reached target block errors
      2.0 | 3.8995e-02 | 1.0390e-01 |       24957 |      640000 |         1039 |       10000 |         2.3 |reached target block errors
      2.5 | 1.2763e-02 | 3.4967e-02 |       24505 |     1920000 |         1049 |       30000 |         6.9 |reached target block errors
      3.0 | 2.6419e-03 | 7.5214e-03 |       23671 |     8960000 |         1053 |      140000 |        32.1 |reached target block errors
      3.5 | 4.2701e-04 | 1.2613e-03 |       21863 |    51200000 |         1009 |      800000 |       183.4 |reached target block errors
      4.0 | 5.9375e-05 | 1.7100e-04 |        3800 |    64000000 |          171 |     1000000 |       229.1 |reached max iter
      4.5 | 3.3125e-06 | 9.0000e-06 |         212 |    64000000 |            9 |     1000000 |       229.2 |reached max iter
Running: Reed Muller (RM) SCL-8
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 2.7000e-01 | 6.4760e-01 |      172801 |      640000 |         6476 |       10000 |        12.8 |reached target block errors
      0.5 | 1.9087e-01 | 4.7100e-01 |      122160 |      640000 |         4710 |       10000 |         2.0 |reached target block errors
      1.0 | 1.1507e-01 | 2.9300e-01 |       73643 |      640000 |         2930 |       10000 |         2.0 |reached target block errors
      1.5 | 5.9103e-02 | 1.5370e-01 |       37826 |      640000 |         1537 |       10000 |         2.0 |reached target block errors
      2.0 | 2.3795e-02 | 6.3450e-02 |       30458 |     1280000 |         1269 |       20000 |         4.0 |reached target block errors
      2.5 | 7.2339e-03 | 1.9750e-02 |       27778 |     3840000 |         1185 |       60000 |        12.0 |reached target block errors
      3.0 | 1.6989e-03 | 4.7667e-03 |       22833 |    13440000 |         1001 |      210000 |        41.9 |reached target block errors
      3.5 | 2.5781e-04 | 7.3300e-04 |       16500 |    64000000 |          733 |     1000000 |       199.6 |reached max iter
      4.0 | 3.1578e-05 | 8.3000e-05 |        2021 |    64000000 |           83 |     1000000 |       199.6 |reached max iter
      4.5 | 2.3437e-06 | 6.0000e-06 |         150 |    64000000 |            6 |     1000000 |       199.6 |reached max iter
Running: Conv. Code Viterbi (constraint length 8)
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6208e-01 | 6.8980e-01 |      103733 |      640000 |         6898 |       10000 |         1.8 |reached target block errors
      0.5 | 1.0615e-01 | 5.4740e-01 |       67936 |      640000 |         5474 |       10000 |         0.5 |reached target block errors
      1.0 | 6.0327e-02 | 4.0450e-01 |       38609 |      640000 |         4045 |       10000 |         0.5 |reached target block errors
      1.5 | 3.2498e-02 | 2.7790e-01 |       20799 |      640000 |         2779 |       10000 |         0.5 |reached target block errors
      2.0 | 1.6691e-02 | 1.8970e-01 |       10682 |      640000 |         1897 |       10000 |         0.5 |reached target block errors
      2.5 | 7.9234e-03 | 1.1960e-01 |        5071 |      640000 |         1196 |       10000 |         0.5 |reached target block errors
      3.0 | 4.0820e-03 | 8.0250e-02 |        5225 |     1280000 |         1605 |       20000 |         1.0 |reached target block errors
      3.5 | 1.9516e-03 | 4.7400e-02 |        3747 |     1920000 |         1422 |       30000 |         1.5 |reached target block errors
      4.0 | 1.1066e-03 | 3.1350e-02 |        2833 |     2560000 |         1254 |       40000 |         1.9 |reached target block errors
      4.5 | 6.0313e-04 | 1.9083e-02 |        2316 |     3840000 |         1145 |       60000 |         2.9 |reached target block errors
Running: Turbo Code (constraint length 4)
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.0916e-01 | 7.8380e-01 |       69865 |      640000 |         7838 |       10000 |         3.8 |reached target block errors
      0.5 | 7.6463e-02 | 6.0200e-01 |       48936 |      640000 |         6020 |       10000 |         1.3 |reached target block errors
      1.0 | 4.6916e-02 | 4.0020e-01 |       30026 |      640000 |         4002 |       10000 |         1.2 |reached target block errors
      1.5 | 2.4842e-02 | 2.2510e-01 |       15899 |      640000 |         2251 |       10000 |         1.2 |reached target block errors
      2.0 | 9.7844e-03 | 9.2300e-02 |       12524 |     1280000 |         1846 |       20000 |         2.5 |reached target block errors
      2.5 | 2.9223e-03 | 3.0625e-02 |        7481 |     2560000 |         1225 |       40000 |         5.1 |reached target block errors
      3.0 | 8.1080e-04 | 9.6545e-03 |        5708 |     7040000 |         1062 |      110000 |        13.9 |reached target block errors
      3.5 | 1.7529e-04 | 2.6605e-03 |        4263 |    24320000 |         1011 |      380000 |        47.7 |reached target block errors
      4.0 | 3.2750e-05 | 6.6900e-04 |        2096 |    64000000 |          669 |     1000000 |       125.4 |reached max iter
      4.5 | 8.3281e-06 | 2.3100e-04 |         533 |    64000000 |          231 |     1000000 |       125.3 |reached max iter
```


    
And let’s also look at the block-error-rate.

```python
[8]:
```

```python
ber_plot128(ylim=(1e-5, 1), show_ber=False)
```


    
Please keep in mind that the decoding complexity differs significantly and should be also included in a fair comparison as shown in Section <a class="reference external" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html#Throughput-and-Decoding-Complexity">Throughput and Decoding Complexity</a>.

### Performance under Optimal Decoding<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html#Performance-under-Optimal-Decoding" title="Permalink to this headline"></a>
    
The achievable error-rate performance of a coding scheme depends on the strength of the code construction and the performance of the actual decoding algorithm. We now approximate the maximum-likelihood performance of all previous coding schemes by using the ordered statistics decoder (OSD) [12].

```python
[9]:
```

```python
# overwrite existing legend entries for OSD simulations
legends = ["5G LDPC", "5G Polar+CRC", "5G Polar+CRC", "RM", "Conv. Code", "Turbo Code"]
# run ber simulations for each code we have added to the list
for idx, code in enumerate(codes_under_test):
    if idx==2: # skip second polar code (same code only different decoder)
        continue
    print("\nRunning: " + code[2])
    # initialize encoder
    encoder = code[0]
    # encode dummy bits to init conv encoders (otherwise k is not defined)
    encoder(tf.zeros((1, k)))
    # OSD can be directly associated to an encoder
    decoder = OSDecoder(encoder=encoder, t=4)
    # generate a new model with the given encoder/decoder
    model = System_Model(k=k,
                         n=n,
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=encoder,
                         decoder=decoder,
                         cw_estimates=True) # OSD returns codeword estimates and not info bit estimates
    # the first argument must be a callable (function) that yields u and u_hat for batch_size and ebno
    ber_plot128.simulate(tf.function(model, jit_compile=True),
                         ebno_dbs=ebno_db, # SNR to simulate
                         legend=legends[idx]+f" OSD-{decoder.t} ", # legend string for plotting
                         max_mc_iter=1000, # run 100 Monte Carlo runs per SNR point
                         num_target_block_errors=1000, # continue with next SNR point after 1000 bit errors
                         batch_size=1000, # batch-size per Monte Carlo run
                         soft_estimates=False, # the model returns hard-estimates
                         early_stop=True, # stop simulation if no error has been detected at current SNR point
                         show_fig=False, # we show the figure after all results are simulated
                         add_bler=True, # in case BLER is also interesting
                         forward_keyboard_interrupt=True); # should be True in a loop

```


```python

Running: 5G LDPC BP-20
Note: Required memory complexity is large for the given code parameters and t=4. Please consider small batch-sizes to keep the inference complexity small and activate XLA mode if possible.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.0525e-01 | 4.6233e-01 |       40416 |      384000 |         1387 |        3000 |         4.2 |reached target block errors
      0.5 | 5.5930e-02 | 2.5625e-01 |       28636 |      512000 |         1025 |        4000 |         2.0 |reached target block errors
      1.0 | 2.4980e-02 | 1.1889e-01 |       28777 |     1152000 |         1070 |        9000 |         4.6 |reached target block errors
      1.5 | 8.3019e-03 | 4.1040e-02 |       26566 |     3200000 |         1026 |       25000 |        12.7 |reached target block errors
      2.0 | 2.1109e-03 | 1.1055e-02 |       24588 |    11648000 |         1006 |       91000 |        46.7 |reached target block errors
      2.5 | 3.8392e-04 | 2.1874e-03 |       22556 |    58752000 |         1004 |      459000 |       236.2 |reached target block errors
      3.0 | 4.9438e-05 | 3.2400e-04 |        6328 |   128000000 |          324 |     1000000 |       512.5 |reached max iter
      3.5 | 5.0078e-06 | 3.9000e-05 |         641 |   128000000 |           39 |     1000000 |       511.6 |reached max iter
      4.0 | 3.6719e-07 | 4.0000e-06 |          47 |   128000000 |            4 |     1000000 |       512.2 |reached max iter
      4.5 | 0.0000e+00 | 0.0000e+00 |           0 |   128000000 |            0 |     1000000 |       512.5 |reached max iter
Simulation stopped as no error occurred @ EbNo = 4.5 dB.

Running: 5G Polar+CRC SC
Note: Required memory complexity is large for the given code parameters and t=4. Please consider small batch-sizes to keep the inference complexity small and activate XLA mode if possible.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.0706e-01 | 4.4833e-01 |       41110 |      384000 |         1345 |        3000 |         4.4 |reached target block errors
      0.5 | 5.7684e-02 | 2.4560e-01 |       36918 |      640000 |         1228 |        5000 |         2.5 |reached target block errors
      1.0 | 2.5269e-02 | 1.0990e-01 |       32344 |     1280000 |         1099 |       10000 |         5.1 |reached target block errors
      1.5 | 7.8858e-03 | 3.5276e-02 |       29272 |     3712000 |         1023 |       29000 |        14.8 |reached target block errors
      2.0 | 1.7343e-03 | 7.8976e-03 |       28192 |    16256000 |         1003 |      127000 |        64.9 |reached target block errors
      2.5 | 2.6134e-04 | 1.2516e-03 |       26728 |   102272000 |         1000 |      799000 |       408.2 |reached target block errors
      3.0 | 2.6187e-05 | 1.3300e-04 |        3352 |   128000000 |          133 |     1000000 |       510.5 |reached max iter
      3.5 | 1.7031e-06 | 8.0000e-06 |         218 |   128000000 |            8 |     1000000 |       510.0 |reached max iter
      4.0 | 0.0000e+00 | 0.0000e+00 |           0 |   128000000 |            0 |     1000000 |       510.5 |reached max iter
Simulation stopped as no error occurred @ EbNo = 4.0 dB.

Running: Reed Muller (RM) SCL-8
Note: Required memory complexity is large for the given code parameters and t=4. Please consider small batch-sizes to keep the inference complexity small and activate XLA mode if possible.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 9.9979e-02 | 4.8533e-01 |       38392 |      384000 |         1456 |        3000 |         4.3 |reached target block errors
      0.5 | 5.8141e-02 | 3.0425e-01 |       29768 |      512000 |         1217 |        4000 |         2.0 |reached target block errors
      1.0 | 2.5547e-02 | 1.4088e-01 |       26160 |     1024000 |         1127 |        8000 |         4.1 |reached target block errors
      1.5 | 9.7431e-03 | 5.8222e-02 |       22448 |     2304000 |         1048 |       18000 |         9.2 |reached target block errors
      2.0 | 2.8170e-03 | 1.8182e-02 |       19832 |     7040000 |         1000 |       55000 |        28.1 |reached target block errors
      2.5 | 5.9362e-04 | 4.0732e-03 |       18692 |    31488000 |         1002 |      246000 |       125.7 |reached target block errors
      3.0 | 1.0056e-04 | 7.4500e-04 |       12872 |   128000000 |          745 |     1000000 |       510.3 |reached max iter
      3.5 | 1.3063e-05 | 9.8000e-05 |        1672 |   128000000 |           98 |     1000000 |       510.3 |reached max iter
      4.0 | 6.2500e-07 | 5.0000e-06 |          80 |   128000000 |            5 |     1000000 |       510.2 |reached max iter
      4.5 | 1.2500e-07 | 1.0000e-06 |          16 |   128000000 |            1 |     1000000 |       510.1 |reached max iter
Running: Conv. Code Viterbi (constraint length 8)
Note: Required memory complexity is large for the given code parameters and t=4. Please consider small batch-sizes to keep the inference complexity small and activate XLA mode if possible.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 9.7660e-02 | 7.0150e-01 |       25001 |      256000 |         1403 |        2000 |         2.9 |reached target block errors
      0.5 | 6.5164e-02 | 5.5900e-01 |       16682 |      256000 |         1118 |        2000 |         1.0 |reached target block errors
      1.0 | 3.6641e-02 | 4.1567e-01 |       14070 |      384000 |         1247 |        3000 |         1.5 |reached target block errors
      1.5 | 1.9215e-02 | 2.7100e-01 |        9838 |      512000 |         1084 |        4000 |         2.0 |reached target block errors
      2.0 | 1.0513e-02 | 1.8833e-01 |        8074 |      768000 |         1130 |        6000 |         3.1 |reached target block errors
      2.5 | 5.0686e-03 | 1.1822e-01 |        5839 |     1152000 |         1064 |        9000 |         4.6 |reached target block errors
      3.0 | 2.7242e-03 | 7.8538e-02 |        4533 |     1664000 |         1021 |       13000 |         6.6 |reached target block errors
      3.5 | 1.4941e-03 | 5.1800e-02 |        3825 |     2560000 |         1036 |       20000 |        10.2 |reached target block errors
      4.0 | 7.7959e-04 | 3.0545e-02 |        3293 |     4224000 |         1008 |       33000 |        16.8 |reached target block errors
      4.5 | 4.3529e-04 | 1.8887e-02 |        2953 |     6784000 |         1001 |       53000 |        27.1 |reached target block errors
Running: Turbo Code (constraint length 4)
Note: Required memory complexity is large for the given code parameters and t=4. Please consider small batch-sizes to keep the inference complexity small and activate XLA mode if possible.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.0087e-01 | 5.0400e-01 |       25823 |      256000 |         1008 |        2000 |         3.1 |reached target block errors
      0.5 | 6.4128e-02 | 3.4400e-01 |       24625 |      384000 |         1032 |        3000 |         1.5 |reached target block errors
      1.0 | 3.0613e-02 | 1.7683e-01 |       23511 |      768000 |         1061 |        6000 |         3.1 |reached target block errors
      1.5 | 1.2736e-02 | 8.1692e-02 |       21193 |     1664000 |         1062 |       13000 |         6.7 |reached target block errors
      2.0 | 3.9779e-03 | 2.9500e-02 |       17312 |     4352000 |         1003 |       34000 |        17.4 |reached target block errors
      2.5 | 1.0436e-03 | 1.0192e-02 |       13225 |    12672000 |         1009 |       99000 |        50.7 |reached target block errors
      3.0 | 2.3167e-04 | 3.0895e-03 |        9608 |    41472000 |         1001 |      324000 |       165.9 |reached target block errors
      3.5 | 7.3588e-05 | 1.2706e-03 |        7413 |   100736000 |         1000 |      787000 |       402.9 |reached target block errors
      4.0 | 2.3914e-05 | 4.7400e-04 |        3061 |   128000000 |          474 |     1000000 |       511.9 |reached max iter
      4.5 | 7.0391e-06 | 1.5300e-04 |         901 |   128000000 |          153 |     1000000 |       512.1 |reached max iter
```

    
And let’s plot the results.
    
<em>Remark</em>: we define a custom plotting function to enable a nicer visualization of OSD vs. non-OSD results.

```python
[15]:
```

```python
# for simplicity, we only plot a subset of the simulated curves
# focus on BLER
plots_to_show = ['5G LDPC BP-20 (BLER)', '5G LDPC OSD-4  (BLER)', '5G Polar+CRC SCL-8 (BLER)', '5G Polar+CRC OSD-4  (BLER)', 'Reed Muller (RM) SCL-8 (BLER)', 'RM OSD-4  (BLER)', 'Conv. Code Viterbi (constraint length 8) (BLER)', 'Conv. Code OSD-4  (BLER)', 'Turbo Code (constraint length 4) (BLER)', 'Turbo Code OSD-4  (BLER)']
# find indices of relevant curves
idx = []
for p in plots_to_show:
    for i,l in enumerate(ber_plot128._legends):
        if p==l:
            idx.append(i)
# generate new figure
fig, ax = plt.subplots(figsize=(16,12))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title(f"Performance under Ordered Statistic Decoding (k={k},n={n})", fontsize=25)
plt.grid(which="both")
plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
plt.ylabel(r"BLER", fontsize=25)
# plot pairs of BLER curves (non-osd vs. osd)
for i in range(int(len(idx)/2)):
    # non-OSD
    plt.semilogy(ebno_db,
                 ber_plot128._bers[idx[2*i]],
                 c='C%d'%(i),
                 label=ber_plot128._legends[idx[2*i]].replace(" (BLER)", ""), #remove "(BLER)" from label
                 linewidth=2)
    # OSD
    plt.semilogy(ebno_db,
                 ber_plot128._bers[idx[2*i+1]],
                 c='C%d'%(i),
                 label= ber_plot128._legends[idx[2*i+1]].replace(" (BLER)", ""), #remove "(BLER)" from label
                 linestyle = "--",
                 linewidth=2)
plt.legend(fontsize=20)
plt.xlim([0, 4.5])
plt.ylim([1e-4, 1]);

```


    
As can be seen, the performance of Polar and Convolutional codes is in practice close to their ML performance. For other codes such as LDPC codes, there is a practical performance gap under BP decoding which tends to be smaller for longer codes.

### Performance of Longer LDPC Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html#Performance-of-Longer-LDPC-Codes" title="Permalink to this headline"></a>
    
Now, let us have a look at the performance gains due to longer codewords. For this, we scale the length of the LDPC code and compare the results (same rate, same decoder, same channel).

```python
[8]:
```

```python
# init new figure
ber_plot_ldpc = PlotBER(f"BER/BLER Performance of LDPC Codes @ Fixed Rate=0.5")
```
```python
[9]:
```

```python
# code parameters to simulate
ns = [128, 256, 512, 1000, 2000, 4000, 8000, 16000]  # number of codeword bits per codeword
rate = 0.5 # fixed coderate
# create list of encoder/decoder pairs to be analyzed
codes_under_test = []
# 5G LDPC codes
for n in ns:
    k = int(rate*n) # calculate k for given n and rate
    enc = LDPC5GEncoder(k=k, n=n)
    dec = LDPC5GDecoder(enc, num_iter=20)
    name = f"5G LDPC BP-20 (n={n})"
    codes_under_test.append([enc, dec, name, k, n])

```
```python
[10]:
```

```python
# and simulate the results
num_bits_per_symbol = 2 # QPSK
ebno_db = np.arange(0, 5, 0.25) # sim SNR range
# note that the waterfall for long codes can be steep and requires a fine
# SNR quantization
# run ber simulations for each case
for code in codes_under_test:
    print("Running: " + code[2])
    model = System_Model(k=code[3],
                         n=code[4],
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=code[0],
                         decoder=code[1])
    # the first argument must be a callable (function) that yields u and u_hat
    # for given batch_size and ebno
    # we fix the target number of BLOCK errors instead of the BER to
    # ensure that same accurate results for each block lengths is simulated
    ber_plot_ldpc.simulate(model, # the function have defined previously
                           ebno_dbs=ebno_db,
                           legend=code[2],
                           max_mc_iter=100,
                           num_target_block_errors=500, # we fix the target block errors
                           batch_size=1000,
                           soft_estimates=False,
                           early_stop=True,
                           show_fig=False,
                           forward_keyboard_interrupt=True); # should be True in a loop
# and show figure
ber_plot_ldpc(ylim=(1e-5, 1))
```


```python
Running: 5G LDPC BP-20 (n=128)
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6914e-01 | 8.6600e-01 |       10825 |       64000 |          866 |        1000 |         1.0 |reached target block errors
     0.25 | 1.4652e-01 | 7.8400e-01 |        9377 |       64000 |          784 |        1000 |         0.1 |reached target block errors
      0.5 | 1.2748e-01 | 7.2800e-01 |        8159 |       64000 |          728 |        1000 |         0.1 |reached target block errors
     0.75 | 1.0242e-01 | 6.0000e-01 |        6555 |       64000 |          600 |        1000 |         0.1 |reached target block errors
      1.0 | 8.1711e-02 | 4.8800e-01 |       10459 |      128000 |          976 |        2000 |         0.1 |reached target block errors
     1.25 | 6.5227e-02 | 3.8400e-01 |        8349 |      128000 |          768 |        2000 |         0.1 |reached target block errors
      1.5 | 5.1398e-02 | 3.0700e-01 |        6579 |      128000 |          614 |        2000 |         0.1 |reached target block errors
     1.75 | 3.6177e-02 | 2.1933e-01 |        6946 |      192000 |          658 |        3000 |         0.2 |reached target block errors
      2.0 | 2.5227e-02 | 1.4900e-01 |        6458 |      256000 |          596 |        4000 |         0.2 |reached target block errors
     2.25 | 1.6531e-02 | 1.0200e-01 |        5290 |      320000 |          510 |        5000 |         0.3 |reached target block errors
      2.5 | 1.0494e-02 | 6.6250e-02 |        5373 |      512000 |          530 |        8000 |         0.4 |reached target block errors
     2.75 | 6.5373e-03 | 4.0385e-02 |        5439 |      832000 |          525 |       13000 |         0.7 |reached target block errors
      3.0 | 3.5675e-03 | 2.2773e-02 |        5023 |     1408000 |          501 |       22000 |         1.2 |reached target block errors
     3.25 | 1.8422e-03 | 1.2195e-02 |        4834 |     2624000 |          500 |       41000 |         2.3 |reached target block errors
      3.5 | 9.0968e-04 | 6.1341e-03 |        4774 |     5248000 |          503 |       82000 |         4.8 |reached target block errors
     3.75 | 4.7891e-04 | 2.9900e-03 |        3065 |     6400000 |          299 |      100000 |         5.6 |reached max iter
      4.0 | 1.8422e-04 | 1.1300e-03 |        1179 |     6400000 |          113 |      100000 |         5.6 |reached max iter
     4.25 | 1.1438e-04 | 7.4000e-04 |         732 |     6400000 |           74 |      100000 |         5.6 |reached max iter
      4.5 | 5.5313e-05 | 3.5000e-04 |         354 |     6400000 |           35 |      100000 |         5.7 |reached max iter
     4.75 | 1.1094e-05 | 9.0000e-05 |          71 |     6400000 |            9 |      100000 |         5.6 |reached max iter
Running: 5G LDPC BP-20 (n=256)
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6655e-01 | 9.4200e-01 |       21318 |      128000 |          942 |        1000 |         1.0 |reached target block errors
     0.25 | 1.4567e-01 | 8.8100e-01 |       18646 |      128000 |          881 |        1000 |         0.1 |reached target block errors
      0.5 | 1.2033e-01 | 7.7200e-01 |       15402 |      128000 |          772 |        1000 |         0.1 |reached target block errors
     0.75 | 9.4398e-02 | 6.3800e-01 |       12083 |      128000 |          638 |        1000 |         0.1 |reached target block errors
      1.0 | 6.7824e-02 | 4.9150e-01 |       17363 |      256000 |          983 |        2000 |         0.1 |reached target block errors
     1.25 | 4.6043e-02 | 3.5300e-01 |       11787 |      256000 |          706 |        2000 |         0.1 |reached target block errors
      1.5 | 3.1776e-02 | 2.4000e-01 |       12202 |      384000 |          720 |        3000 |         0.2 |reached target block errors
     1.75 | 1.8992e-02 | 1.5250e-01 |        9724 |      512000 |          610 |        4000 |         0.2 |reached target block errors
      2.0 | 9.2221e-03 | 8.0857e-02 |        8263 |      896000 |          566 |        7000 |         0.4 |reached target block errors
     2.25 | 4.7396e-03 | 4.2083e-02 |        7280 |     1536000 |          505 |       12000 |         0.7 |reached target block errors
      2.5 | 2.2689e-03 | 1.9808e-02 |        7551 |     3328000 |          515 |       26000 |         1.5 |reached target block errors
     2.75 | 9.2346e-04 | 8.5424e-03 |        6974 |     7552000 |          504 |       59000 |         3.5 |reached target block errors
      3.0 | 3.2531e-04 | 3.0000e-03 |        4164 |    12800000 |          300 |      100000 |         5.8 |reached max iter
     3.25 | 1.2242e-04 | 1.1500e-03 |        1567 |    12800000 |          115 |      100000 |         5.9 |reached max iter
      3.5 | 3.8672e-05 | 4.0000e-04 |         495 |    12800000 |           40 |      100000 |         5.9 |reached max iter
     3.75 | 8.5938e-06 | 1.2000e-04 |         110 |    12800000 |           12 |      100000 |         5.8 |reached max iter
      4.0 | 1.5625e-06 | 2.0000e-05 |          20 |    12800000 |            2 |      100000 |         5.8 |reached max iter
     4.25 | 3.1250e-07 | 1.0000e-05 |           4 |    12800000 |            1 |      100000 |         5.9 |reached max iter
      4.5 | 0.0000e+00 | 0.0000e+00 |           0 |    12800000 |            0 |      100000 |         5.9 |reached max iter
Simulation stopped as no error occurred @ EbNo = 4.5 dB.
Running: 5G LDPC BP-20 (n=512)
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6162e-01 | 9.7100e-01 |       41376 |      256000 |          971 |        1000 |         1.0 |reached target block errors
     0.25 | 1.3645e-01 | 9.3300e-01 |       34931 |      256000 |          933 |        1000 |         0.1 |reached target block errors
      0.5 | 1.1016e-01 | 8.3000e-01 |       28202 |      256000 |          830 |        1000 |         0.1 |reached target block errors
     0.75 | 7.9887e-02 | 6.6900e-01 |       20451 |      256000 |          669 |        1000 |         0.1 |reached target block errors
      1.0 | 5.1861e-02 | 4.6150e-01 |       26553 |      512000 |          923 |        2000 |         0.1 |reached target block errors
     1.25 | 2.9461e-02 | 2.8550e-01 |       15084 |      512000 |          571 |        2000 |         0.1 |reached target block errors
      1.5 | 1.4026e-02 | 1.4900e-01 |       14363 |     1024000 |          596 |        4000 |         0.3 |reached target block errors
     1.75 | 5.2413e-03 | 5.8667e-02 |       12076 |     2304000 |          528 |        9000 |         0.6 |reached target block errors
      2.0 | 1.9423e-03 | 2.3810e-02 |       10442 |     5376000 |          500 |       21000 |         1.3 |reached target block errors
     2.25 | 6.3080e-04 | 7.9063e-03 |       10335 |    16384000 |          506 |       64000 |         4.1 |reached target block errors
      2.5 | 1.5441e-04 | 2.0400e-03 |        3953 |    25600000 |          204 |      100000 |         6.4 |reached max iter
     2.75 | 3.3320e-05 | 5.1000e-04 |         853 |    25600000 |           51 |      100000 |         6.3 |reached max iter
      3.0 | 4.7266e-06 | 1.3000e-04 |         121 |    25600000 |           13 |      100000 |         6.4 |reached max iter
     3.25 | 2.3438e-07 | 2.0000e-05 |           6 |    25600000 |            2 |      100000 |         6.4 |reached max iter
      3.5 | 0.0000e+00 | 0.0000e+00 |           0 |    25600000 |            0 |      100000 |         6.4 |reached max iter
Simulation stopped as no error occurred @ EbNo = 3.5 dB.
Running: 5G LDPC BP-20 (n=1000)
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6359e-01 | 9.9800e-01 |       81793 |      500000 |          998 |        1000 |         1.0 |reached target block errors
     0.25 | 1.3874e-01 | 9.8100e-01 |       69368 |      500000 |          981 |        1000 |         0.1 |reached target block errors
      0.5 | 9.9932e-02 | 9.0700e-01 |       49966 |      500000 |          907 |        1000 |         0.1 |reached target block errors
     0.75 | 6.5646e-02 | 7.0900e-01 |       32823 |      500000 |          709 |        1000 |         0.1 |reached target block errors
      1.0 | 3.3873e-02 | 4.6600e-01 |       33873 |     1000000 |          932 |        2000 |         0.2 |reached target block errors
     1.25 | 1.3356e-02 | 2.2533e-01 |       20034 |     1500000 |          676 |        3000 |         0.2 |reached target block errors
      1.5 | 4.1151e-03 | 7.8000e-02 |       14403 |     3500000 |          546 |        7000 |         0.6 |reached target block errors
     1.75 | 7.8215e-04 | 1.9308e-02 |       10168 |    13000000 |          502 |       26000 |         2.1 |reached target block errors
      2.0 | 1.1394e-04 | 3.3300e-03 |        5697 |    50000000 |          333 |      100000 |         7.9 |reached max iter
     2.25 | 1.1760e-05 | 4.9000e-04 |         588 |    50000000 |           49 |      100000 |         7.9 |reached max iter
      2.5 | 1.1600e-06 | 5.0000e-05 |          58 |    50000000 |            5 |      100000 |         7.9 |reached max iter
     2.75 | 8.2000e-07 | 2.0000e-05 |          41 |    50000000 |            2 |      100000 |         7.9 |reached max iter
      3.0 | 0.0000e+00 | 0.0000e+00 |           0 |    50000000 |            0 |      100000 |         7.9 |reached max iter
Simulation stopped as no error occurred @ EbNo = 3.0 dB.
Running: 5G LDPC BP-20 (n=2000)
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.5922e-01 | 1.0000e+00 |      159218 |     1000000 |         1000 |        1000 |         1.3 |reached target block errors
     0.25 | 1.3586e-01 | 1.0000e+00 |      135862 |     1000000 |         1000 |        1000 |         0.1 |reached target block errors
      0.5 | 9.8168e-02 | 9.7100e-01 |       98168 |     1000000 |          971 |        1000 |         0.1 |reached target block errors
     0.75 | 5.4171e-02 | 8.0800e-01 |       54171 |     1000000 |          808 |        1000 |         0.1 |reached target block errors
      1.0 | 1.9121e-02 | 4.5550e-01 |       38243 |     2000000 |          911 |        2000 |         0.2 |reached target block errors
     1.25 | 4.1725e-03 | 1.5675e-01 |       16690 |     4000000 |          627 |        4000 |         0.4 |reached target block errors
      1.5 | 4.1236e-04 | 2.3000e-02 |        9072 |    22000000 |          506 |       22000 |         2.4 |reached target block errors
     1.75 | 2.9270e-05 | 2.3000e-03 |        2927 |   100000000 |          230 |      100000 |        11.0 |reached max iter
      2.0 | 8.2000e-07 | 1.7000e-04 |          82 |   100000000 |           17 |      100000 |        11.0 |reached max iter
     2.25 | 1.0000e-08 | 1.0000e-05 |           1 |   100000000 |            1 |      100000 |        10.9 |reached max iter
      2.5 | 0.0000e+00 | 0.0000e+00 |           0 |   100000000 |            0 |      100000 |        10.9 |reached max iter
Simulation stopped as no error occurred @ EbNo = 2.5 dB.
Running: 5G LDPC BP-20 (n=4000)
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6200e-01 | 1.0000e+00 |      323995 |     2000000 |         1000 |        1000 |         1.3 |reached target block errors
     0.25 | 1.3757e-01 | 1.0000e+00 |      275132 |     2000000 |         1000 |        1000 |         0.2 |reached target block errors
      0.5 | 9.8322e-02 | 9.9800e-01 |      196644 |     2000000 |          998 |        1000 |         0.2 |reached target block errors
     0.75 | 4.9637e-02 | 9.1400e-01 |       99274 |     2000000 |          914 |        1000 |         0.2 |reached target block errors
      1.0 | 1.0812e-02 | 5.2700e-01 |       21624 |     2000000 |          527 |        1000 |         0.2 |reached target block errors
     1.25 | 9.5500e-04 | 1.0020e-01 |        9550 |    10000000 |          501 |        5000 |         0.9 |reached target block errors
      1.5 | 2.3473e-05 | 5.5385e-03 |        4272 |   182000000 |          504 |       91000 |        16.1 |reached target block errors
     1.75 | 2.3500e-07 | 7.0000e-05 |          47 |   200000000 |            7 |      100000 |        17.6 |reached max iter
      2.0 | 0.0000e+00 | 0.0000e+00 |           0 |   200000000 |            0 |      100000 |        17.6 |reached max iter
Simulation stopped as no error occurred @ EbNo = 2.0 dB.
Running: 5G LDPC BP-20 (n=8000)
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.3612e-01 | 1.0000e+00 |      544473 |     4000000 |         1000 |        1000 |         1.8 |reached target block errors
     0.25 | 1.1098e-01 | 1.0000e+00 |      443911 |     4000000 |         1000 |        1000 |         0.4 |reached target block errors
      0.5 | 7.1998e-02 | 1.0000e+00 |      287993 |     4000000 |         1000 |        1000 |         0.4 |reached target block errors
     0.75 | 2.5447e-02 | 9.4400e-01 |      101788 |     4000000 |          944 |        1000 |         0.4 |reached target block errors
      1.0 | 2.5708e-03 | 4.1150e-01 |       20566 |     8000000 |          823 |        2000 |         0.8 |reached target block errors
     1.25 | 4.0510e-05 | 1.9346e-02 |        4213 |   104000000 |          503 |       26000 |        10.0 |reached target block errors
      1.5 | 9.7500e-08 | 1.4000e-04 |          39 |   400000000 |           14 |      100000 |        38.6 |reached max iter
     1.75 | 1.0000e-08 | 1.0000e-05 |           4 |   400000000 |            1 |      100000 |        38.6 |reached max iter
      2.0 | 0.0000e+00 | 0.0000e+00 |           0 |   400000000 |            0 |      100000 |        38.6 |reached max iter
Simulation stopped as no error occurred @ EbNo = 2.0 dB.
Running: 5G LDPC BP-20 (n=16000)
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.3777e-01 | 1.0000e+00 |     1102186 |     8000000 |         1000 |        1000 |         2.8 |reached target block errors
     0.25 | 1.1079e-01 | 1.0000e+00 |      886359 |     8000000 |         1000 |        1000 |         0.8 |reached target block errors
      0.5 | 7.1611e-02 | 1.0000e+00 |      572892 |     8000000 |         1000 |        1000 |         0.8 |reached target block errors
     0.75 | 2.4264e-02 | 9.9300e-01 |      194114 |     8000000 |          993 |        1000 |         0.8 |reached target block errors
      1.0 | 1.1114e-03 | 4.4550e-01 |       17783 |    16000000 |          891 |        2000 |         1.6 |reached target block errors
     1.25 | 1.4387e-06 | 3.6800e-03 |        1151 |   800000000 |          368 |      100000 |        79.6 |reached max iter
      1.5 | 0.0000e+00 | 0.0000e+00 |           0 |   800000000 |            0 |      100000 |        79.6 |reached max iter
Simulation stopped as no error occurred @ EbNo = 1.5 dB.

```


