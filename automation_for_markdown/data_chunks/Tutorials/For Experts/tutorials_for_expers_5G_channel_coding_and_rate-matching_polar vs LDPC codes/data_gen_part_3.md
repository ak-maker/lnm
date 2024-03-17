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
## Throughput and Decoding Complexity
  
  

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

## Throughput and Decoding Complexity<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html#Throughput-and-Decoding-Complexity" title="Permalink to this headline"></a>
    
In the last part of this notebook, you will compare the different computational complexity of the different codes and decoders. In theory the complexity is given as:
 
- Successive cancellation list (SCL) decoding of Polar codes scales with $\mathcal{O}(L \cdot n \cdot \operatorname{log} n)$ (with $L=1$ for SC decoding)
- Iterative belief propagation (BP) decoding of LDPC codes scales with $\mathcal{O}(n)$. However, in particular for short codes a complexity comparison should be supported by empirical results.

    
We want to emphasize that the results strongly depend on the exact implementation and may differ for different implementations/optimizations. Implementing the SCL decoder in Tensorflow is a delicate task and requires several design trade-offs to enable a graph implementation which can lead to degraded throughput mainly caused by the missing <em>lazy copy-mechanism</em>. However, - inspired by [10] - the SCL decoder layer supports `hybrid` `SC` decoding meaning that SC decoding is done first and a
second stage SCL decoder operates as afterburner iff the outer CRC check fails. Please note that this modus uses <em>‘tf.py_function’</em> (due to the control flow and the dynamic shape of the decoding graph) and, thus, does not support XLA compilation.

```python
[ ]:
```

```python
def get_throughput(batch_size, ebno_dbs, model, repetitions=1):
    """ Simulate throughput in bit/s per ebno_dbs point.
    The results are average over `repetition` trials.
    Input
    -----
    batch_size: tf.int32
        Batch-size for evaluation.
    ebno_dbs: tf.float32
        A tensor containing SNR points to be evaluated.
    model:
        Function or model that yields the transmitted bits `u` and the
        receiver's estimate `u_hat` for a given ``batch_size`` and
        ``ebno_db``.
    repetitions: int
        An integer defining how many trails of the throughput
        simulation are averaged.
    """
    throughput = np.zeros_like(ebno_dbs)
    # call model once to be sure it is compile properly
    # otherwise time to build graph is measured as well.
    u, u_hat = model(tf.constant(batch_size, tf.int32),
                     tf.constant(0., tf.float32))
    for idx, ebno_db in enumerate(ebno_dbs):
        t_start = time.perf_counter()
        # average over multiple runs
        for _ in range(repetitions):
            u, u_hat = model(tf.constant(batch_size, tf.int32),
                             tf.constant(ebno_db, tf. float32))
        t_stop = time.perf_counter()
        # throughput in bit/s
        throughput[idx] = np.size(u.numpy())*repetitions / (t_stop - t_start)
    return throughput

```
```python
[ ]:
```

```python
# plot throughput and ber together for ldpc codes
# and simulate the results
num_bits_per_symbol = 2 # QPSK
ebno_db = [5] # SNR to simulate
num_bits_per_batch = 5e6 # must be reduced in case of out-of-memory errors
num_repetitions = 20 # average throughput over multiple runs
# run throughput simulations for each code
throughput = np.zeros(len(codes_under_test))
code_length = np.zeros(len(codes_under_test))
for idx, code in enumerate(codes_under_test):
    print("Running: " + code[2])
    # save codeword length for plotting
    code_length[idx] = code[4]
    # init new model for given encoder/decoder
    model = System_Model(k=code[3],
                         n=code[4],
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=code[0],
                         decoder=code[1])
    # scale batch_size such that same number of bits is simulated for all codes
    batch_size = int(num_bits_per_batch / code[4])
    # and measure throughput of the model
    throughput[idx] = get_throughput(batch_size,
                                     ebno_db,
                                     model,
                                     repetitions=num_repetitions)
```


```python
Running: 5G LDPC BP-20 (n=128)
Running: 5G LDPC BP-20 (n=256)
Running: 5G LDPC BP-20 (n=512)
Running: 5G LDPC BP-20 (n=1000)
Running: 5G LDPC BP-20 (n=2000)
Running: 5G LDPC BP-20 (n=4000)
Running: 5G LDPC BP-20 (n=8000)
Running: 5G LDPC BP-20 (n=16000)
```
```python
[ ]:
```

```python
# plot results
plt.figure(figsize=(16,10))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title("Throughput LDPC BP Decoding @ rate=0.5", fontsize=25)
plt.xlabel("Codeword length", fontsize=25)
plt.ylabel("Throughput (Mbit/s)", fontsize=25)
plt.grid(which="both")
# and plot results (logarithmic scale in x-dim)
x_tick_labels = code_length.astype(int)
plt.xticks(ticks=np.log2(code_length),labels=x_tick_labels, fontsize=18)
plt.plot(np.log2(code_length), throughput/1e6)

```


```python
[<matplotlib.lines.Line2D at 0x7fee100c3e20>]
```


    
As expected the throughput of BP decoding is (relatively) constant as the complexity scales linearly with $\mathcal{O}(n)$ and, thus, the complexity <em>per</em> decoded bit remains constant. It is instructive to realize that the above plot is in the log-domain for the x-axis.
    
Let us have a look at what happens for different SNR values.

```python
[ ]:
```

```python
# --- LDPC ---
n = 1000
k = 500
encoder = LDPC5GEncoder(k, n)
decoder = LDPC5GDecoder(encoder)
# init a new model
model = System_Model(k=k,
                     n=n,
                     num_bits_per_symbol=num_bits_per_symbol,
                     encoder=encoder,
                     decoder=decoder)
# run throughput tests at 2 dB and 5 dB
ebno_db = [2, 5]
batch_size = 10000
throughput = get_throughput(batch_size,
                            ebno_db, # snr point
                            model,
                            repetitions=num_repetitions)
# and print the results
for idx, snr_db in enumerate(ebno_db):
    print(f"Throughput @ {snr_db:.1f} dB: {throughput[idx]/1e6:.2f} Mbit/s")
```


```python
Throughput @ 2.0 dB: 10.91 Mbit/s
Throughput @ 5.0 dB: 10.90 Mbit/s
```

    
For most Sionna decoders the throughput is not SNR dependent as early stopping of individual samples within a batch is difficult to realize.
    
However, the `hybrid` `SCL` decoder uses an internal NumPy SCL decoder only if the SC decoder failed similar to [10]. We will now benchmark this decoder for different SNR values.

```python
[ ]:
```

```python
# --- Polar ---
n = 256
k = 128
encoder = Polar5GEncoder(k, n)
decoder = Polar5GDecoder(encoder, "hybSCL")
# init a new model
model = System_Model(k=k,
                     n=n,
                     num_bits_per_symbol=num_bits_per_symbol,
                     encoder=encoder,
                     decoder=decoder)
ebno_db = np.arange(0, 5, 0.5) # EbNo to evaluate
batch_size = 1000
throughput = get_throughput(batch_size,
                            ebno_db, # snr point
                            model,
                            repetitions=num_repetitions)
# and print the results
for idx, snr_db in enumerate(ebno_db):
    print(f"Throughput @ {snr_db:.1f} dB: {throughput[idx]/1e6:.3f} Mbit/s")
```


```python
Throughput @ 0.0 dB: 0.016 Mbit/s
Throughput @ 0.5 dB: 0.017 Mbit/s
Throughput @ 1.0 dB: 0.020 Mbit/s
Throughput @ 1.5 dB: 0.029 Mbit/s
Throughput @ 2.0 dB: 0.047 Mbit/s
Throughput @ 2.5 dB: 0.100 Mbit/s
Throughput @ 3.0 dB: 0.236 Mbit/s
Throughput @ 3.5 dB: 0.893 Mbit/s
Throughput @ 4.0 dB: 1.294 Mbit/s
Throughput @ 4.5 dB: 1.469 Mbit/s
```

    
We can overlay the throughput with the BLER of the SC decoder. This can be intuitively explained by the fact that he `hybrid` `SCL` decoder consists of two decoding stages:
 
- SC decoding for all received codewords.
- SCL decoding <em>iff</em> the CRC does not hold, i.e., SC decoding did not yield the correct codeword.

    
Thus, the throughput directly depends on the BLER of the internal SC decoder.

```python
[ ]:
```

```python
ber_plot_polar = PlotBER("Polar SC/SCL Decoding")
ber_plot_polar.simulate(model, # the function have defined previously
                        ebno_dbs=ebno_db,
                        legend="hybrid SCL decoding",
                        max_mc_iter=100,
                        num_target_block_errors=100, # we fix the target bler
                        batch_size=1000,
                        soft_estimates=False,
                        early_stop=True,
                        add_ber=False,
                        add_bler=True,
                        show_fig=False,
                        forward_keyboard_interrupt=False);
# and add SC decoding
decoder2 = Polar5GDecoder(encoder, "SC")
model = System_Model(k=k,
                     n=n,
                     num_bits_per_symbol=num_bits_per_symbol,
                     encoder=encoder,
                     decoder=decoder2)
ber_plot_polar.simulate(model, # the function have defined previously
                        ebno_dbs=ebno_db,
                        legend="SC decoding",
                        max_mc_iter=100,
                        num_target_block_errors=100, # we fix the target bler
                        batch_size=1000,
                        soft_estimates=False,
                        early_stop=True,
                        add_ber=False, # we only focus on BLER
                        add_bler=True,
                        show_fig=False,
                        forward_keyboard_interrupt=False);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 3.3807e-01 | 8.3300e-01 |       43273 |      128000 |          833 |        1000 |         7.8 |reached target block errors
      0.5 | 2.2667e-01 | 6.0800e-01 |       29014 |      128000 |          608 |        1000 |         7.4 |reached target block errors
      1.0 | 1.1982e-01 | 3.4100e-01 |       15337 |      128000 |          341 |        1000 |         6.3 |reached target block errors
      1.5 | 4.4477e-02 | 1.3400e-01 |        5693 |      128000 |          134 |        1000 |         4.6 |reached target block errors
      2.0 | 9.6211e-03 | 3.2000e-02 |        4926 |      512000 |          128 |        4000 |        10.4 |reached target block errors
      2.5 | 1.2563e-03 | 4.7619e-03 |        3377 |     2688000 |          100 |       21000 |        27.1 |reached target block errors
      3.0 | 1.2359e-04 | 5.0000e-04 |        1582 |    12800000 |           50 |      100000 |        53.0 |reached max iter
      3.5 | 1.6406e-06 | 1.0000e-05 |          21 |    12800000 |            1 |      100000 |        15.8 |reached max iter
      4.0 | 0.0000e+00 | 0.0000e+00 |           0 |    12800000 |            0 |      100000 |        10.4 |reached max iter
Simulation stopped as no error occurred @ EbNo = 4.0 dB.
Warning: 5G Polar codes use an integrated CRC that cannot be materialized with SC decoding and, thus, causes a degraded performance. Please consider SCL decoding instead.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 4.2356e-01 | 9.7900e-01 |       54216 |      128000 |          979 |        1000 |         7.9 |reached target block errors
      0.5 | 3.5630e-01 | 8.9800e-01 |       45607 |      128000 |          898 |        1000 |         0.0 |reached target block errors
      1.0 | 2.8463e-01 | 7.7300e-01 |       36433 |      128000 |          773 |        1000 |         0.1 |reached target block errors
      1.5 | 1.9066e-01 | 5.4700e-01 |       24405 |      128000 |          547 |        1000 |         0.0 |reached target block errors
      2.0 | 1.0170e-01 | 3.2100e-01 |       13017 |      128000 |          321 |        1000 |         0.0 |reached target block errors
      2.5 | 4.2672e-02 | 1.5200e-01 |        5462 |      128000 |          152 |        1000 |         0.0 |reached target block errors
      3.0 | 1.5059e-02 | 5.3000e-02 |        3855 |      256000 |          106 |        2000 |         0.1 |reached target block errors
      3.5 | 4.9531e-03 | 1.8667e-02 |        3804 |      768000 |          112 |        6000 |         0.3 |reached target block errors
      4.0 | 6.6205e-04 | 3.2258e-03 |        2627 |     3968000 |          100 |       31000 |         1.5 |reached target block errors
      4.5 | 1.0281e-04 | 6.0000e-04 |        1316 |    12800000 |           60 |      100000 |         4.6 |reached max iter
```

    
Let us visualize the results.

```python
[ ]:
```

```python
ber_plot_polar()
ax2 = plt.gca().twinx()  # new axis
ax2.plot(ebno_db, throughput, 'g', label="Throughput hybSCL-8")
ax2.legend(fontsize=20)
ax2.set_ylabel("Throughput (bit/s)", fontsize=25);
ax2.tick_params(labelsize=25)
```


    
You can also try:
 
- Analyze different rates
- What happens for different batch-sizes? Can you explain what happens?
- What happens for higher order modulation. Why is the complexity increased?
