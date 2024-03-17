# Part 1: Getting Started with Sionna<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Part-1:-Getting-Started-with-Sionna" title="Permalink to this headline"></a>
    
This tutorial will guide you through Sionna, from its basic principles to the implementation of a point-to-point link with a 5G NR compliant code and a 3GPP channel model. You will also learn how to write custom trainable layers by implementing a state of the art neural receiver, and how to train and evaluate end-to-end communication systems.
    
The tutorial is structured in four notebooks:
 
- **Part I: Getting started with Sionna**
- Part II: Differentiable Communication Systems
- Part III: Advanced Link-level Simulations
- Part IV: Toward Learned Receivers

    
The <a class="reference external" href="https://nvlabs.github.io/sionna">official documentation</a> provides key material on how to use Sionna and how its components are implemented.

# Table of Content
## Imports & Basics
## Forward Error Correction (FEC)
  
  

## Imports & Basics<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Imports-&-Basics" title="Permalink to this headline"></a>

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
    import sionna as sn
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna as sn
# Import TensorFlow and NumPy
import tensorflow as tf
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
import numpy as np
# For plotting
%matplotlib inline
# also try %matplotlib widget
import matplotlib.pyplot as plt
# for performance measurements
import time
# For the implementation of the Keras models
from tensorflow.keras import Model
```

    
We can now access Sionna functions within the `sn` namespace.
    
**Hint**: In Jupyter notebooks, you can run bash commands with `!`.

```python
[2]:
```

```python
!nvidia-smi
```


```python
Tue Mar 15 14:47:45 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   51C    P8    23W / 350W |     53MiB / 24265MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:4C:00.0 Off |                  N/A |
|  0%   33C    P8    24W / 350W |      8MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```
## Forward Error Correction (FEC)<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Forward-Error-Correction-(FEC)" title="Permalink to this headline"></a>
    
We now add channel coding to our transceiver to make it more robust against transmission errors. For this, we will use <a class="reference external" href="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3214">5G compliant low-density parity-check (LDPC) codes and Polar codes</a>. You can find more detailed information in the notebooks <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html">Bit-Interleaved Coded Modulation (BICM)</a> and <a class="reference external" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html">5G Channel Coding
and Rate-Matching: Polar vs. LDPC Codes</a>.

```python
[15]:
```

```python
k = 12
n = 20
encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)
decoder = sn.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)
```

    
Let us encode some random input bits.

```python
[16]:
```

```python
BATCH_SIZE = 1 # one codeword in parallel
u = binary_source([BATCH_SIZE, k])
print("Input bits are: \n", u.numpy())
c = encoder(u)
print("Encoded bits are: \n", c.numpy())
```


```python
Input bits are:
 [[1. 1. 0. 0. 1. 1. 0. 0. 0. 0. 1. 1.]]
Encoded bits are:
 [[1. 1. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 1. 1.]]
```

    
One of the fundamental paradigms of Sionna is batch-processing. Thus, the example above could be executed with for arbitrary batch-sizes to simulate `batch_size` codewords in parallel.
    
However, Sionna can do more - it supports <em>N</em>-dimensional input tensors and, thereby, allows the processing of multiple samples of multiple users and several antennas in a single command line. Let’s say we want to encoded `batch_size` codewords of length `n` for each of the `num_users` connected to each of the `num_basestations`. This means in total we transmit `batch_size` * `n` * `num_users` * `num_basestations` bits.

```python
[17]:
```

```python
BATCH_SIZE = 10 # samples per scenario
num_basestations = 4
num_users = 5 # users per basestation
n = 1000 # codeword length per transmitted codeword
coderate = 0.5 # coderate
k = int(coderate * n) # number of info bits per codeword
# instantiate a new encoder for codewords of length n
encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)
# the decoder must be linked to the encoder (to know the exact code parameters used for encoding)
decoder = sn.fec.ldpc.LDPC5GDecoder(encoder,
                                    hard_out=True, # binary output or provide soft-estimates
                                    return_infobits=True, # or also return (decoded) parity bits
                                    num_iter=20, # number of decoding iterations
                                    cn_type="boxplus-phi") # also try "minsum" decoding
# draw random bits to encode
u = binary_source([BATCH_SIZE, num_basestations, num_users, k])
print("Shape of u: ", u.shape)
# We can immediately encode u for all users, basetation and samples
# This all happens with a single line of code
c = encoder(u)
print("Shape of c: ", c.shape)
print("Total number of processed bits: ", np.prod(c.shape))
```


```python
Shape of u:  (10, 4, 5, 500)
Shape of c:  (10, 4, 5, 1000)
Total number of processed bits:  200000
```

    
This works for arbitrary dimensions and allows a simple extension of the designed system to multi-user or multi-antenna scenarios.
    
Let us now replace the LDPC code by a Polar code. The API remains similar.

```python
[18]:
```

```python
k = 64
n = 128
encoder = sn.fec.polar.Polar5GEncoder(k, n)
decoder = sn.fec.polar.Polar5GDecoder(encoder,
                                      dec_type="SCL") # you can also use "SCL"
```

    
<em>Advanced Remark:</em> The 5G Polar encoder/decoder class directly applies rate-matching and the additional CRC concatenation. This is all done internally and transparent to the user.
    
In case you want to access low-level features of the Polar codes, please use `sionna.fec.polar.PolarEncoder` and the desired decoder (`sionna.fec.polar.PolarSCDecoder`, `sionna.fec.polar.PolarSCLDecoder` or `sionna.fec.polar.PolarBPDecoder`).
    
Further details can be found in the tutorial notebook on <a class="reference external" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html">5G Channel Coding and Rate-Matching: Polar vs. LDPC Codes</a>.
    
<img alt="QAM FEC AWGN" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmkAAACtCAYAAADrl+hZAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAaGVYSWZNTQAqAAAACAAEAQYAAwAAAAEAAgAAARIAAwAAAAEAAQAAASgAAwAAAAEAAgAAh2kABAAAAAEAAAA+AAAAAAADoAEAAwAAAAEAAQAAoAIABAAAAAEAAAJpoAMABAAAAAEAAACtAAAAAF4rGUEAAALkaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA2LjAuMCI+CiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOnRpZmY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vdGlmZi8xLjAvIgogICAgICAgICAgICB4bWxuczpleGlmPSJodHRwOi8vbnMuYWRvYmUuY29tL2V4aWYvMS4wLyI+CiAgICAgICAgIDx0aWZmOkNvbXByZXNzaW9uPjE8L3RpZmY6Q29tcHJlc3Npb24+CiAgICAgICAgIDx0aWZmOlJlc29sdXRpb25Vbml0PjI8L3RpZmY6UmVzb2x1dGlvblVuaXQ+CiAgICAgICAgIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6T3JpZW50YXRpb24+CiAgICAgICAgIDx0aWZmOlBob3RvbWV0cmljSW50ZXJwcmV0YXRpb24+MjwvdGlmZjpQaG90b21ldHJpY0ludGVycHJldGF0aW9uPgogICAgICAgICA8ZXhpZjpQaXhlbFhEaW1lbnNpb24+NjE3PC9leGlmOlBpeGVsWERpbWVuc2lvbj4KICAgICAgICAgPGV4aWY6Q29sb3JTcGFjZT4xPC9leGlmOkNvbG9yU3BhY2U+CiAgICAgICAgIDxleGlmOlBpeGVsWURpbWVuc2lvbj4xNzM8L2V4aWY6UGl4ZWxZRGltZW5z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" />

```python
[19]:
```

```python
class CodedSystemAWGN(Model): # Inherits from Keras Model
    def __init__(self, num_bits_per_symbol, n, coderate):
        super().__init__() # Must call the Keras model initializer
        self.num_bits_per_symbol = num_bits_per_symbol
        self.n = n
        self.k = int(n*coderate)
        self.coderate = coderate
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        self.encoder = sn.fec.ldpc.LDPC5GEncoder(self.k, self.n)
        self.decoder = sn.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)
    #@tf.function # activate graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        no = sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.coderate)
        bits = self.binary_source([batch_size, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        y = self.awgn_channel([x, no])
        llr = self.demapper([y,no])
        bits_hat = self.decoder(llr)
        return bits, bits_hat
```
```python
[20]:
```

```python
CODERATE = 0.5
BATCH_SIZE = 2000
model_coded_awgn = CodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                   n=2048,
                                   coderate=CODERATE)
ber_plots.simulate(model_coded_awgn,
                   ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 15),
                   batch_size=BATCH_SIZE,
                   num_target_block_errors=500,
                   legend="Coded",
                   soft_estimates=False,
                   max_mc_iter=15,
                   show_fig=True,
                   forward_keyboard_interrupt=False);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -3.0 | 2.7896e-01 | 1.0000e+00 |      571312 |     2048000 |         2000 |        2000 |         1.8 |reached target block errors
   -2.429 | 2.6327e-01 | 1.0000e+00 |      539174 |     2048000 |         2000 |        2000 |         1.7 |reached target block errors
   -1.857 | 2.4783e-01 | 1.0000e+00 |      507546 |     2048000 |         2000 |        2000 |         1.7 |reached target block errors
   -1.286 | 2.2687e-01 | 1.0000e+00 |      464636 |     2048000 |         2000 |        2000 |         1.7 |reached target block errors
   -0.714 | 2.0312e-01 | 1.0000e+00 |      415988 |     2048000 |         2000 |        2000 |         1.7 |reached target block errors
   -0.143 | 1.7154e-01 | 1.0000e+00 |      351316 |     2048000 |         2000 |        2000 |         1.7 |reached target block errors
    0.429 | 1.1296e-01 | 9.9050e-01 |      231337 |     2048000 |         1981 |        2000 |         1.7 |reached target block errors
      1.0 | 2.0114e-02 | 4.7150e-01 |       41193 |     2048000 |          943 |        2000 |         1.7 |reached target block errors
    1.571 | 1.7617e-04 | 1.1633e-02 |        5412 |    30720000 |          349 |       30000 |        25.2 |reached max iter
    2.143 | 0.0000e+00 | 0.0000e+00 |           0 |    30720000 |            0 |       30000 |        25.3 |reached max iter
Simulation stopped as no error occurred @ EbNo = 2.1 dB.

```

<img alt="../_images/examples_Sionna_tutorial_part1_52_1.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part1_52_1.png" />

    
As can be seen, the `BerPlot` class uses multiple stopping conditions and stops the simulation after no error occured at a specifc SNR point.
    
**Task**: Replace the coding scheme by a Polar encoder/decoder or a convolutional code with Viterbi decoding.

