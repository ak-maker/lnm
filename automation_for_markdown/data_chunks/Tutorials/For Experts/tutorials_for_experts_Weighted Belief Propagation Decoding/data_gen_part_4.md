# Weighted Belief Propagation Decoding<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html#Weighted-Belief-Propagation-Decoding" title="Permalink to this headline"></a>
    
This notebooks implements the <em>Weighted Belief Propagation</em> (BP) algorithm as proposed by Nachmani <em>et al.</em> in [1]. The main idea is to leverage BP decoding by additional trainable weights that scale each outgoing variable node (VN) and check node (CN) message. These weights provide additional degrees of freedom and can be trained by stochastic gradient descent (SGD) to improve the BP performance for the given code. If all weights are initialized with <em>1</em>, the algorithm equals the <em>classical</em> BP
algorithm and, thus, the concept can be seen as a generalized BP decoder.
    
Our main focus is to show how Sionna can lower the barrier-to-entry for state-of-the-art research. For this, you will investigate:
 
- How to implement the multi-loss BP decoding with Sionna
- How a single scaling factor can lead to similar results
- What happens for training of the 5G LDPC code

    
The setup includes the following components:
 
- LDPC BP Decoder
- Gaussian LLR source

    
Please note that we implement a simplified version of the original algorithm consisting of two major simplifications:
<ol class="arabic simple">
- ) Only outgoing variable node (VN) messages are weighted. This is possible as the VN operation is linear and it would only increase the memory complexity without increasing the <em>expressive</em> power of the neural network.
- ) We use the same shared weights for all iterations. This can potentially influence the final performance, however, simplifies the implementation and allows to run the decoder with different number of iterations.
</ol>
    
**Note**: If you are not familiar with all-zero codeword-based simulations please have a look into the <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html">Bit-Interleaved Coded Modulation</a> example notebook first.

# Table of Content
## GPU Configuration and Imports
## References
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

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
# Import required Sionna components
from sionna.fec.ldpc import LDPCBPDecoder, LDPC5GEncoder, LDPC5GDecoder
from sionna.utils.metrics import BitwiseMutualInformation
from sionna.fec.utils import GaussianPriorSource, load_parity_check_examples
from sionna.utils import ebnodb2no, hard_decisions
from sionna.utils.metrics import compute_ber
from sionna.utils.plotting import PlotBER
from tensorflow.keras.losses import BinaryCrossentropy
```
```python
[2]:
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
```python
[3]:
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```

## References<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html#References" title="Permalink to this headline"></a>
    
[1] E. Nachmani, Y. Be’ery and D. Burshtein, “Learning to Decode Linear Codes Using Deep Learning,” IEEE Annual Allerton Conference on Communication, Control, and Computing (Allerton), pp. 341-346., 2016. <a class="reference external" href="https://arxiv.org/pdf/1607.04793.pdf">https://arxiv.org/pdf/1607.04793.pdf</a>
    
[2] M. Lian, C. Häger, and H. Pfister, “What can machine learning teach us about communications?” IEEE Information Theory Workshop (ITW), pp. 1-5. 2018.
    
[3] ] M. Pretti, “A message passing algorithm with damping,” J. Statist. Mech.: Theory Practice, p. 11008, Nov. 2005.
    
[4] J.S. Yedidia, W.T. Freeman and Y. Weiss, “Constructing free energy approximations and Generalized Belief Propagation algorithms,” IEEE Transactions on Information Theory, 2005.
    
[5] E. Nachmani, E. Marciano, L. Lugosch, W. Gross, D. Burshtein and Y. Be’ery, “Deep learning methods for improved decoding of linear codes,” IEEE Journal of Selected Topics in Signal Processing, vol. 12, no. 1, pp.119-131, 2018.
<script type="application/vnd.jupyter.widget-state+json">
{"state": {}, "version_major": 2, "version_minor": 0}
</script>