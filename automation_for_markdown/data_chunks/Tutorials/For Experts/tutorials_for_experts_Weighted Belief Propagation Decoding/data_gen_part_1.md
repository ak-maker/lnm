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
## Weighted BP for BCH Codes
### Weights before Training and Simulation of BER
  
  

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

## Weighted BP for BCH Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html#Weighted-BP-for-BCH-Codes" title="Permalink to this headline"></a>
    
First, we define the trainable model consisting of:
 
- LDPC BP decoder
- Gaussian LLR source

    
The idea of the multi-loss function in [1] is to average the loss overall iterations, i.e., not just the final estimate is evaluated. This requires to call the BP decoder <em>iteration-wise</em> by setting `num_iter=1` and `stateful=True` such that the decoder will perform a single iteration and returns its current estimate while also providing the internal messages for the next iteration.
    
A few comments:
 
- We assume the transmission of the all-zero codeword. This allows to train and analyze the decoder without the need of an encoder. Remark: The final decoder can be used for arbitrary codewords.
- We directly generate the channel LLRs with `GaussianPriorSource`. The equivalent LLR distribution could be achieved by transmitting the all-zero codeword over an AWGN channel with BPSK modulation.
- For the proposed <em>multi-loss</em> [1] (i.e., the loss is averaged over all iterations), we need to access the decoders intermediate output after each iteration. This is done by calling the decoding function multiple times while setting `stateful` to True, i.e., the decoder continuous the decoding process at the last message state.
```python
[4]:
```

```python
class WeightedBP(tf.keras.Model):
    """System model for BER simulations of weighted BP decoding.
    This model uses `GaussianPriorSource` to mimic the LLRs after demapping of
    QPSK symbols transmitted over an AWGN channel.
    Parameters
    ----------
        pcm: ndarray
            The parity-check matrix of the code under investigation.
        num_iter: int
            Number of BP decoding iterations.

    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.
        ebno_db: float or tf.float
            A float defining the simulation SNR.
    Output
    ------
        (u, u_hat, loss):
            Tuple:
        u: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the transmitted information bits.
        u_hat: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the estimated information bits.
        loss: tf.float32
            Binary cross-entropy loss between `u` and `u_hat`.
    """
    def __init__(self, pcm, num_iter=5):
        super().__init__()
        # init components
        self.decoder = LDPCBPDecoder(pcm,
                                     num_iter=1, # iterations are done via outer loop (to access intermediate results for multi-loss)
                                     stateful=True, # decoder stores internal messages after call
                                     hard_out=False, # we need to access soft-information
                                     cn_type="boxplus",
                                     trainable=True) # the decoder must be trainable, otherwise no weights are generated
        # used to generate llrs during training (see example notebook on all-zero codeword trick)
        self.llr_source = GaussianPriorSource()
        self._num_iter = num_iter
        self._bce = BinaryCrossentropy(from_logits=True)
    def call(self, batch_size, ebno_db):
        noise_var = ebnodb2no(ebno_db,
                              num_bits_per_symbol=2, # QPSK
                              coderate=coderate)
        # all-zero CW to calculate loss / BER
        c = tf.zeros([batch_size, n])
        # Gaussian LLR source
        llr = self.llr_source([[batch_size, n], noise_var])
        # --- implement multi-loss as proposed by Nachmani et al. [1]---
        loss = 0
        msg_vn = None # internal state of decoder
        for i in range(self._num_iter):
            c_hat, msg_vn = self.decoder((llr, msg_vn)) # perform one decoding iteration; decoder returns soft-values
            loss += self._bce(c, c_hat)  # add loss after each iteration
        loss /= self._num_iter # scale loss by number of iterations
        return c, c_hat, loss
```

    
Load a parity-check matrix used for the experiment. We use the same BCH(63,45) code as in [1]. The code can be replaced by any parity-check matrix of your choice.

```python
[5]:
```

```python
pcm_id = 1 # (63,45) BCH code parity check matrix
pcm, k , n, coderate = load_parity_check_examples(pcm_id=pcm_id, verbose=True)
num_iter = 10 # set number of decoding iterations
# and initialize the model
model = WeightedBP(pcm=pcm, num_iter=num_iter)
```


```python

n: 63, k: 45, coderate: 0.714
```

    
**Note**: weighted BP tends to work better for small number of iterations. The effective gains (compared to the baseline with same number of iterations) vanish with more iterations.

### Weights before Training and Simulation of BER<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html#Weights-before-Training-and-Simulation-of-BER" title="Permalink to this headline"></a>
    
Let us plot the weights after initialization of the decoder to verify that everything is properly initialized. This is equivalent the <em>classical</em> BP decoder.

```python
[6]:
```

```python
# count number of weights/edges
print("Total number of weights: ", np.size(model.decoder.get_weights()))
# and show the weight distribution
model.decoder.show_weights()
```


```python
Total number of weights:  432
```


    
We first simulate (and store) the BER performance <em>before</em> training. For this, we use the `PlotBER` class, which provides a convenient way to store the results for later comparison.

```python
[7]:
```

```python
# SNR to simulate the results
ebno_dbs = np.array(np.arange(1, 7, 0.5))
mc_iters = 100 # number of Monte Carlo iterations
# we generate a new PlotBER() object to simulate, store and plot the BER results
ber_plot = PlotBER("Weighted BP")
# simulate and plot the BER curve of the untrained decoder
ber_plot.simulate(model,
                  ebno_dbs=ebno_dbs,
                  batch_size=1000,
                  num_target_bit_errors=2000, # stop sim after 2000 bit errors
                  legend="Untrained",
                  soft_estimates=True,
                  max_mc_iter=mc_iters,
                  forward_keyboard_interrupt=False);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      1.0 | 8.9492e-02 | 9.7600e-01 |        5638 |       63000 |          976 |        1000 |         0.2 |reached target bit errors
      1.5 | 7.4079e-02 | 9.0800e-01 |        4667 |       63000 |          908 |        1000 |         0.2 |reached target bit errors
      2.0 | 5.9444e-02 | 8.1300e-01 |        3745 |       63000 |          813 |        1000 |         0.2 |reached target bit errors
      2.5 | 4.4667e-02 | 6.6400e-01 |        2814 |       63000 |          664 |        1000 |         0.2 |reached target bit errors
      3.0 | 3.4365e-02 | 5.1700e-01 |        2165 |       63000 |          517 |        1000 |         0.2 |reached target bit errors
      3.5 | 2.1563e-02 | 3.4950e-01 |        2717 |      126000 |          699 |        2000 |         0.3 |reached target bit errors
      4.0 | 1.3460e-02 | 2.3200e-01 |        2544 |      189000 |          696 |        3000 |         0.5 |reached target bit errors
      4.5 | 7.1778e-03 | 1.2880e-01 |        2261 |      315000 |          644 |        5000 |         0.8 |reached target bit errors
      5.0 | 3.9877e-03 | 7.5889e-02 |        2261 |      567000 |          683 |        9000 |         1.4 |reached target bit errors
      5.5 | 2.1240e-03 | 3.9188e-02 |        2141 |     1008000 |          627 |       16000 |         2.5 |reached target bit errors
      6.0 | 1.0169e-03 | 2.0406e-02 |        2050 |     2016000 |          653 |       32000 |         4.9 |reached target bit errors
      6.5 | 4.4312e-04 | 8.5417e-03 |        2010 |     4536000 |          615 |       72000 |        11.1 |reached target bit errors
```


