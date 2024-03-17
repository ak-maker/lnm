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
## A Deeper Look into the Polar Code Module
## Rate-Matching and Rate-Recovery
  
  

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

## A Deeper Look into the Polar Code Module<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html#A-Deeper-Look-into-the-Polar-Code-Module" title="Permalink to this headline"></a>
    
A Polar code can be defined by a set of `frozen` `bit` and `information` `bit` positions [1]. The package `sionna.fec.polar.utils` supports 5G-compliant Polar code design, but also Reed-Muller (RM) codes are available and can be used within the same encoder/decoder layer. If required, rate-matching and CRC concatenation are handled by the class `sionna.fec.polar.Polar5GEncoder` and `sionna.fec.polar.Polar5GDecoder`, respectively.
    
Further, the following decoders are available:
 
- Successive cancellation (SC) decoding [1]
 
- Fast and low-complexity
- Sub-optimal error-rate performance


- Successive cancellation list (SCL) decoding [2]
 
- Excellent error-rate performance
- High-complexity
- CRC-aided decoding possible


- Hybrid SCL decoder (combined SC and SCL decoder)
 
- Pre-decode with SC and only apply SCL iff CRC fails
- Excellent error-rate performance
- Needs outer CRC (e.g., as done in 5G)
- CPU-based implementation and, thus, no XLA support (+ increased decoding latency)


- Iterative belief propagation (BP) decoding [6]
 
- Produces soft-output estimates
- Sub-optimal error-rate performance



    
Let us now generate a new Polar code.

```python
[11]:
```

```python
code_type = "5G" # try also "RM"
# Load the 5G compliant polar code
if code_type=="5G":
    k = 32
    n = 64
    # load 5G compliant channel ranking [3]
    frozen_pos, info_pos = generate_5g_ranking(k,n)
    print("Generated Polar code of length n = {} and k = {}".format(n, k))
    print("Frozen codeword positions: ", frozen_pos)
# Alternatively Reed-Muller code design is also available
elif code_type=="RM":
    r = 3
    m = 7
    frozen_pos, info_pos, n, k, d_min = generate_rm_code(r, m)
    print("Generated ({},{}) Reed-Muller code of length n = {} and k = {} with minimum distance d_min = {}".format(r, m, n, k, d_min))
    print("Frozen codeword positions: ", frozen_pos)
else:
    print("Code not found")
```


```python
Generated Polar code of length n = 64 and k = 32
Frozen codeword positions:  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 16 17 18 19 20 21 24 25 26
 32 33 34 35 36 37 40 48]
```

    
Now, we can initialize the encoder and a `BinarySource` to generate random Polar codewords.

```python
[12]:
```

```python
# init polar encoder
encoder_polar = PolarEncoder(frozen_pos, n)
# init binary source to generate information bits
source = BinarySource()
# define a batch_size
batch_size = 1
# generate random info bits
u = source([batch_size, k])
# and encode
c = encoder_polar(u)
print("Information bits: ", u.numpy())
print("Polar encoded bits: ", c.numpy())
```


```python
Information bits:  [[1. 0. 1. 1. 1. 1. 0. 1. 0. 1. 0. 1. 1. 0. 1. 1. 0. 1. 1. 1. 1. 0. 1. 1.
  0. 0. 0. 0. 0. 0. 0. 1.]]
Polar encoded bits:  [[0. 0. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 0. 1. 0. 0. 0. 0. 0. 1. 1. 0.
  1. 0. 1. 1. 0. 1. 0. 0. 0. 1. 1. 1. 1. 0. 1. 1. 1. 1. 0. 0. 1. 0. 0. 1.
  1. 0. 1. 1. 0. 0. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1.]]
```

    
As can be seen, the length of the resulting code must be a power of 2. This brings us to the problem of rate-matching and we will now have a closer look how we can adapt the length of the code.

## Rate-Matching and Rate-Recovery<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html#Rate-Matching-and-Rate-Recovery" title="Permalink to this headline"></a>
    
The general task of rate-matching is to enable flexibility of the code w.r.t. the codeword length $n$ and information bit input size $k$ and, thereby, the rate $r = \frac{k}{n}$. In modern communication standards such as 5G NR, these parameters can be adjusted on a bit-level granularity without - in a wider sense - redefining the (mother) code itself. This is enabled by a powerful rate-matching and the corresponding rate-recovery block which will be explained in the following.
    
The principle idea is to select a mother code as close as possible to the desired properties from a set of possible mother codes. For example for Polar codes, the codeword length must be a power of 2, i.e., $n = 32, 64, ..., 512, 1024$. For LDPC codes the codeword length is more flexible (due to the different <em>lifting</em> factors), however, does not allow bit-wise granularity neither. Afterwards, the bit-level granularity is provided by shortening, puncturing and repetitions.
    
To summarize, the rate-matching procedure consists of:
<ol class="arabic simple">
- ) 5G NR defines multiple <em>mother</em> codes with similar properties (e.g., via base-graph lifting of LDPC code or sub-codes for Polar codes)
- ) Puncturing, shortening and repetitions of bits to allow bit-level rate adjustments
</ol>
    
The following figure summarizes the principle for the 5G NR Polar code uplink control channel (UCI). The Fig. is inspired by Fig. 6 in [9].
    
    
For bit-wise length adjustments, the following techniques are commonly used:
<ol class="arabic simple">
- ) <em>Puncturing:</em> A ($k,n$) mother code is punctured by <em>not</em> transmitting $p$ punctured codeword bits. Thus, the rate increases to $r_{\text{pun}} = \frac{k}{n-p} > \frac{k}{n} \quad \forall p > 0$. At the decoder these codeword bits are treated as erasure ($\ell_{\text{ch}} = 0$).
- ) <em>Shortening:</em> A ($k,n$) mother code is shortened by setting $s$ information bits to a fixed (=known) value. Assuming systematic encoding, these $s$ positions are not transmitted leading to a new code of rate $r_{\text{short}} = \frac{k-s}{n-s}<\frac{k}{n}$. At the decoder these codeword bits are treated as known values ($\ell_{\text{ch}} = \infty$).
- ) <em>Repetitions</em> can be used to lower the effective rate. For details we refer the interested reader to [11].
</ol>
    
We will now simulate the performance of rate-matched 5G Polar codes for different lengths and rates. For this, we are interested in the required SNR to achieve a target BLER at $10^{-3}$. Please note that this is a reproduction of the results from [Fig.13a, 4].
    
**Note**: This needs a bisection search as we usually simulate the BLER at fixed SNR and, thus, this is simulation takes some time. Please only execute the cell below if you have enough simulation capabilities.

```python
[13]:
```

```python
# find the EsNo in dB to achieve target_bler
def find_threshold(model, # model to be tested
                   batch_size=1000,
                   max_batch_iter=10, # simulate cws up to batch_size * max_batch_iter
                   max_block_errors=100,  # number of errors before stop
                   target_bler=1e-3): # target error rate to simulate (same as in[4])
        """Bisection search to find required SNR to reach target SNR."""
        # bisection parameters
        esno_db_min = -15 # smallest possible search SNR
        esno_db_max = 15 # largest possible search SNR
        esno_interval = (esno_db_max-esno_db_min)/4 # initial search interval size
        esno_db = 2*esno_interval + esno_db_min # current test SNR
        max_iters = 12 # number of iterations for bisection search
        # run bisection
        for i in range(max_iters):
            num_block_error = 0
            num_cws = 0
            for j in range(max_batch_iter):
                # run model and evaluate BLER
                u, u_hat = model(tf.constant(batch_size, tf.int32),
                                 tf.constant(esno_db, tf.float32))
                num_block_error += count_block_errors(u, u_hat)
                num_cws += batch_size
                # early stop if target number of block errors is reached
                if num_block_error>max_block_errors:
                    break
            bler = num_block_error/num_cws
            # increase SNR if BLER was great than target
            # (larger SNR leads to decreases BLER)
            if bler>target_bler:
                esno_db += esno_interval
            else: # and decrease SNR otherwise
                esno_db -= esno_interval
            esno_interval = esno_interval/2
        # return final SNR after max_iters
        return esno_db

```
```python
[ ]:
```

```python
# run simulations for multiple code parameters
num_bits_per_symbol = 2 # QPSK
# we sweep over multiple values for k and n
ks = np.array([12, 16, 32, 64, 128, 140, 210, 220, 256, 300, 400, 450, 460, 512, 800, 880, 940])
ns = np.array([160, 240, 480, 960])
# we use EsNo instead of EbNo to have the same results as in [4]
esno = np.zeros([len(ns), len(ks)])
for j,n in enumerate(ns):
    for i,k in enumerate(ks):
        if k<n: # only simulate if code parameters are feasible (i.e., r < 1)
            print(f"Finding threshold of k = {k}, n = {n}")
            # initialize new encoder / decoder pair
            enc = Polar5GEncoder(k=k, n=n)
            dec = Polar5GDecoder(enc, dec_type="SCL", list_size=8)
            #build model
            model = System_Model(k=k,
                                 n=n,
                                 num_bits_per_symbol=num_bits_per_symbol,
                                 encoder=enc,
                                 decoder=dec,
                                 sim_esno=True) # no rate adjustment
            # and find threshold via bisection search
            esno[j, i] = find_threshold(model)
            print("Found threshold at: ", esno[j, i])
```
```python
[ ]:
```

```python
# plot the results
leg_str = []
for j,n in enumerate(ns):
    plt.plot(np.log2(ks[ks<n]), esno[j, ks<n])
    leg_str.append("n = {}".format(n))

# define labels manually
x_tick_labels = np.power(2, np.arange(3,11))
plt.xticks(ticks=np.arange(3,11),labels=x_tick_labels, fontsize=18)
# adjusted layout of figure
plt.grid("both")
plt.ylim([-10, 15])
plt.xlabel("Number of information bits $k$", fontsize=20)
plt.yticks(fontsize=18)
plt.ylabel("$E_s/N_0^*$ (dB)", fontsize=20)
plt.legend(leg_str, fontsize=18);
fig = plt.gcf() # get handle to current figure
fig.set_size_inches(15,10)
```


    
This figure equals [Fig. 13a, 4] with a few small exception for extreme low-rate codes. This can be explained by the fact that the 3 explicit parity-bits bits are not implemented, however, these bits are only relevant for for $12\leq k \leq20$. It also explains the degraded performance of the n=960, k=16 code.

