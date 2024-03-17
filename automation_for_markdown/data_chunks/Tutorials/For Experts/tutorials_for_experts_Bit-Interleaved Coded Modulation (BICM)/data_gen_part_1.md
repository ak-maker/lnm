# Bit-Interleaved Coded Modulation (BICM)<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#Bit-Interleaved-Coded-Modulation-(BICM)" title="Permalink to this headline"></a>
    
In this notebook you will learn about the principles of bit interleaved coded modulation (BICM) and focus on the interface between LDPC decoding and demapping for higher order modulation. Further, we will discuss the idea of <em>all-zero codeword</em> simulations that enable bit-error rate simulations without having an explicit LDPC encoder available. In the last part, we analyze what happens for mismatched demapping, e.g., if the SNR is unknown and show how min-sum decoding can have practical
advantages in such cases.
    
<em>“From the coding viewpoint, the modulator, waveform channel, and demodulator together constitute a discrete channel with</em> $q$ <em>input letters and</em> $q'$ <em>output letters. […] the real goal of the modulation system is to create the “best” discrete memoryless channel (DMC) as seen by the coding system.”</em> James L. Massey, 1974 [4, cf. preface in 5].
    
The fact that we usually separate modulation and coding into two individual tasks is strongly connected to the concept of bit-interleaved coded modulation (BICM) [1,2,5]. However, the joint optimization of coding and modulation has a long history, for example by Gottfried Ungerböck’s <em>Trellis coded modulation</em> (TCM) [3] and we refer the interested reader to [1,2,5,6] for these <em>principles of coded modulation</em> [5]. Nonetheless, BICM has become the <em>de facto</em> standard in virtually any modern
communication system due to its engineering simplicity.
    
In this notebook, you will use the following components:
 
- Mapper / demapper and the constellation class
- LDPC5GEncoder / LDPC5GDecoder
- AWGN channel
- BinarySource and GaussianPriorSource
- Interleaver / deinterleaver
- Scrambler / descrambler
# Table of Content
## GPU Configuration and Imports
## System Block Diagram
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

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
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder, LDPCBPDecoder
from sionna.fec.interleaving import RandomInterleaver, Deinterleaver
from sionna.fec.scrambling import Scrambler, Descrambler
from sionna.fec.utils import GaussianPriorSource, load_parity_check_examples, get_exit_analytic, plot_exit_chart, plot_trajectory
from sionna.utils import BinarySource, ebnodb2no, hard_decisions
from sionna.utils.plotting import PlotBER
from sionna.channel import AWGN
```
```python
[2]:
```

```python
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```

## System Block Diagram<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#System-Block-Diagram" title="Permalink to this headline"></a>
    
We introduce the following terminology:
 
- `u` denotes the `k` uncoded information bits
- `c` denotes the `n` codewords bits
- `x` denotes the complex-valued symbols after mapping `m` bits to one symbol
- `y` denotes the (noisy) channel observations
- `l_ch` denotes the demappers llr estimate on each bit `c`
- `u_hat` denotes the estimated information bits at the decoder output

    

