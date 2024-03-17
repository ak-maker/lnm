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
## Communication Systems as Keras Models
  
  

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
## Communication Systems as Keras Models<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Communication-Systems-as-Keras-Models" title="Permalink to this headline"></a>
    
It is typically more convenient to wrap a Sionna-based communication system into a <a class="reference external" href="https://keras.io/api/models/model/">Keras models</a>.
    
These models can be simply built by using the <a class="reference external" href="https://keras.io/guides/functional_api/">Keras functional API</a> to stack layers.
    
The following cell implements the previous system as a Keras model.
    
The key functions that need to be defined are `__init__()`, which instantiates the required components, and `__call()__`, which performs forward pass through the end-to-end system.

```python
[12]:
```

```python
class UncodedSystemAWGN(Model): # Inherits from Keras Model
    def __init__(self, num_bits_per_symbol, block_length):
        """
        A keras model of an uncoded transmission over the AWGN channel.
        Parameters
        ----------
        num_bits_per_symbol: int
            The number of bits per constellation symbol, e.g., 4 for QAM16.
        block_length: int
            The number of bits per transmitted message block (will be the codeword length later).
        Input
        -----
        batch_size: int
            The batch_size of the Monte-Carlo simulation.
        ebno_db: float
            The `Eb/No` value (=rate-adjusted SNR) in dB.
        Output
        ------
        (bits, llr):
            Tuple:
        bits: tf.float32
            A tensor of shape `[batch_size, block_length] of 0s and 1s
            containing the transmitted information bits.
        llr: tf.float32
            A tensor of shape `[batch_size, block_length] containing the
            received log-likelihood-ratio (LLR) values.
        """
        super().__init__() # Must call the Keras model initializer
        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = block_length
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
    # @tf.function # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        # no channel coding used; we set coderate=1.0
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        bits = self.binary_source([batch_size, self.block_length]) # Blocklength set to 1024 bits
        x = self.mapper(bits)
        y = self.awgn_channel([x, no])
        llr = self.demapper([y,no])
        return bits, llr
```

    
We need first to instantiate the model.

```python
[13]:
```

```python
model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=1024)
```

    
Sionna provides a utility to easily compute and plot the bit error rate (BER).

```python
[14]:
```

```python
EBN0_DB_MIN = -3.0 # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = 5.0 # Maximum value of Eb/N0 [dB] for simulations
BATCH_SIZE = 2000 # How many examples are processed by Sionna in parallel
ber_plots = sn.utils.PlotBER("AWGN")
ber_plots.simulate(model_uncoded_awgn,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=True);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -3.0 | 1.5825e-01 | 1.0000e+00 |      324099 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
   -2.579 | 1.4687e-01 | 1.0000e+00 |      300799 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
   -2.158 | 1.3528e-01 | 1.0000e+00 |      277061 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
   -1.737 | 1.2323e-01 | 1.0000e+00 |      252373 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
   -1.316 | 1.1246e-01 | 1.0000e+00 |      230320 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
   -0.895 | 1.0107e-01 | 1.0000e+00 |      206992 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
   -0.474 | 9.0021e-02 | 1.0000e+00 |      184362 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
   -0.053 | 8.0165e-02 | 1.0000e+00 |      164177 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    0.368 | 6.9933e-02 | 1.0000e+00 |      143222 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    0.789 | 6.0897e-02 | 1.0000e+00 |      124717 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    1.211 | 5.2020e-02 | 1.0000e+00 |      106537 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    1.632 | 4.3859e-02 | 1.0000e+00 |       89823 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    2.053 | 3.6686e-02 | 1.0000e+00 |       75132 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    2.474 | 3.0071e-02 | 1.0000e+00 |       61586 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    2.895 | 2.4304e-02 | 1.0000e+00 |       49775 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    3.316 | 1.9330e-02 | 1.0000e+00 |       39588 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    3.737 | 1.4924e-02 | 1.0000e+00 |       30565 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    4.158 | 1.1227e-02 | 1.0000e+00 |       22992 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
    4.579 | 8.2632e-03 | 1.0000e+00 |       16923 |     2048000 |         2000 |        2000 |         0.0 |reached target block errors
      5.0 | 5.9722e-03 | 9.9850e-01 |       12231 |     2048000 |         1997 |        2000 |         0.0 |reached target block errors
```

<img alt="../_images/examples_Sionna_tutorial_part1_39_1.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part1_39_1.png" />

    
The `sn.utils.PlotBER` object stores the results and allows to add additional simulations to the previous curves.
    
<em>Remark</em>: In Sionna, a block error is defined to happen if for two tensors at least one position in the last dimension differs (i.e., at least one bit wrongly received per codeword). The bit error rate the total number of erroneous positions divided by the total number of transmitted bits.

