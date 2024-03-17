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
## Further Experiments
### Learning the 5G LDPC Code
  
  

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

### Learning the 5G LDPC Code<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html#Learning-the-5G-LDPC-Code" title="Permalink to this headline"></a>
    
In this Section, you will experience what happens if we apply the same concept to the 5G LDPC code (including rate matching).
    
For this, we need to define a new model.

```python
[8]:
```

```python
class WeightedBP5G(tf.keras.Model):
    """System model for BER simulations of weighted BP decoding for 5G LDPC codes.
    This model uses `GaussianPriorSource` to mimic the LLRs after demapping of
    QPSK symbols transmitted over an AWGN channel.
    Parameters
    ----------
        k: int
            Number of information bits per codeword.
        n: int
            Codeword length.
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
    def __init__(self, k, n, num_iter=20):
        super().__init__()
        # we need to initialize an encoder for the 5G parameters
        self.encoder = LDPC5GEncoder(k, n)
        self.decoder = LDPC5GDecoder(self.encoder,
                                     num_iter=1, # iterations are done via outer loop (to access intermediate results for multi-loss)
                                     stateful=True,
                                     hard_out=False,
                                     cn_type="boxplus",
                                     trainable=True)
        self.llr_source = GaussianPriorSource()
        self._num_iter = num_iter
        self._coderate = k/n
        self._bce = BinaryCrossentropy(from_logits=True)
    def call(self, batch_size, ebno_db):
        noise_var = ebnodb2no(ebno_db,
                              num_bits_per_symbol=2, # QPSK
                              coderate=self._coderate)
        # BPSK modulated all-zero CW
        c = tf.zeros([batch_size, k]) # decoder only returns info bits
        # use fake llrs from GA
        # works as BP is symmetric
        llr = self.llr_source([[batch_size, n], noise_var])
        # --- implement multi-loss is proposed by Nachmani et al. ---
        loss = 0
        msg_vn = None
        for i in range(self._num_iter):
            c_hat, msg_vn = self.decoder((llr, msg_vn)) # perform one decoding iteration; decoder returns soft-values
            loss += self._bce(c, c_hat)  # add loss after each iteration
        return c, c_hat, loss
```
```python
[9]:
```

```python
# generate model
num_iter = 10
k = 400
n = 800
model5G = WeightedBP5G(k, n, num_iter=num_iter)
# generate baseline BER
ebno_dbs = np.array(np.arange(0, 4, 0.25))
mc_iters = 100 # number of monte carlo iterations
ber_plot_5G = PlotBER("Weighted BP for 5G LDPC")
# simulate the untrained performance
ber_plot_5G.simulate(model5G,
                     ebno_dbs=ebno_dbs,
                     batch_size=1000,
                     num_target_bit_errors=2000, # stop sim after 2000 bit errors
                     legend="Untrained",
                     soft_estimates=True,
                     max_mc_iter=mc_iters);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6660e-01 | 1.0000e+00 |       66640 |      400000 |         1000 |        1000 |         0.2 |reached target bit errors
     0.25 | 1.4864e-01 | 1.0000e+00 |       59455 |      400000 |         1000 |        1000 |         0.2 |reached target bit errors
      0.5 | 1.2470e-01 | 9.9700e-01 |       49880 |      400000 |          997 |        1000 |         0.2 |reached target bit errors
     0.75 | 9.4408e-02 | 9.8000e-01 |       37763 |      400000 |          980 |        1000 |         0.2 |reached target bit errors
      1.0 | 6.6635e-02 | 9.3900e-01 |       26654 |      400000 |          939 |        1000 |         0.2 |reached target bit errors
     1.25 | 4.1078e-02 | 8.1100e-01 |       16431 |      400000 |          811 |        1000 |         0.2 |reached target bit errors
      1.5 | 2.1237e-02 | 6.1200e-01 |        8495 |      400000 |          612 |        1000 |         0.2 |reached target bit errors
     1.75 | 9.2050e-03 | 3.7600e-01 |        3682 |      400000 |          376 |        1000 |         0.2 |reached target bit errors
      2.0 | 2.7175e-03 | 1.7050e-01 |        2174 |      800000 |          341 |        2000 |         0.5 |reached target bit errors
     2.25 | 8.8167e-04 | 6.3833e-02 |        2116 |     2400000 |          383 |        6000 |         1.5 |reached target bit errors
      2.5 | 2.1781e-04 | 2.1875e-02 |        2091 |     9600000 |          525 |       24000 |         5.9 |reached target bit errors
     2.75 | 4.2950e-05 | 4.9600e-03 |        1718 |    40000000 |          496 |      100000 |        24.5 |reached max iter
      3.0 | 6.8000e-06 | 9.1000e-04 |         272 |    40000000 |           91 |      100000 |        24.5 |reached max iter
     3.25 | 9.2500e-07 | 1.8000e-04 |          37 |    40000000 |           18 |      100000 |        24.5 |reached max iter
      3.5 | 2.5000e-08 | 1.0000e-05 |           1 |    40000000 |            1 |      100000 |        24.5 |reached max iter
     3.75 | 0.0000e+00 | 0.0000e+00 |           0 |    40000000 |            0 |      100000 |        24.1 |reached max iter
Simulation stopped as no error occurred @ EbNo = 3.8 dB.

```


    
And let’s train this new model.

```python
[10]:
```

```python
# training parameters
batch_size = 1000
train_iter = 200
clip_value_grad = 10 # gradient clipping seems to be important
# smaller training SNR as the new code is longer (=stronger) than before
ebno_db = 1.5 # rule of thumb: train at ber = 1e-2
# only used as metric
bmi = BitwiseMutualInformation()
# try also different optimizers or different hyperparameters
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
# and let's go
for it in range(0, train_iter):
    with tf.GradientTape() as tape:
        b, llr, loss = model5G(batch_size, ebno_db)
    grads = tape.gradient(loss, model5G.trainable_variables)
    grads = tf.clip_by_value(grads, -clip_value_grad, clip_value_grad, name=None)
    optimizer.apply_gradients(zip(grads, model5G.trainable_weights))
    # calculate and print intermediate metrics
    if it%10==0:
        # calculate ber
        b_hat = hard_decisions(llr)
        ber = compute_ber(b, b_hat)
        # and print results
        mi = bmi(b, llr).numpy()
        l = loss.numpy()
        print(f"Current loss: {l:3f} ber: {ber:.4f} bmi: {mi:.3f}".format())
        bmi.reset_states()
```


```python
Current loss: 1.708751 ber: 0.0204 bmi: 0.925
Current loss: 1.745474 ber: 0.0219 bmi: 0.918
Current loss: 1.741312 ber: 0.0224 bmi: 0.917
Current loss: 1.707712 ber: 0.0208 bmi: 0.923
Current loss: 1.705274 ber: 0.0209 bmi: 0.923
Current loss: 1.706761 ber: 0.0211 bmi: 0.922
Current loss: 1.711995 ber: 0.0212 bmi: 0.921
Current loss: 1.729707 ber: 0.0223 bmi: 0.917
Current loss: 1.692947 ber: 0.0205 bmi: 0.924
Current loss: 1.703924 ber: 0.0203 bmi: 0.924
Current loss: 1.743640 ber: 0.0220 bmi: 0.919
Current loss: 1.719159 ber: 0.0220 bmi: 0.919
Current loss: 1.728399 ber: 0.0221 bmi: 0.920
Current loss: 1.717423 ber: 0.0211 bmi: 0.922
Current loss: 1.743661 ber: 0.0225 bmi: 0.918
Current loss: 1.704675 ber: 0.0212 bmi: 0.923
Current loss: 1.690425 ber: 0.0206 bmi: 0.924
Current loss: 1.728023 ber: 0.0212 bmi: 0.922
Current loss: 1.724549 ber: 0.0212 bmi: 0.922
Current loss: 1.739966 ber: 0.0224 bmi: 0.917
```

    
We now simulate the new results and compare it to the untrained results.

```python
[11]:
```

```python
ebno_dbs = np.array(np.arange(0, 4, 0.25))
batch_size = 1000
mc_iters = 100
ber_plot_5G.simulate(model5G,
                     ebno_dbs=ebno_dbs,
                     batch_size=batch_size,
                     num_target_bit_errors=2000, # stop sim after 2000 bit errors
                     legend="Trained",
                     max_mc_iter=mc_iters,
                     soft_estimates=True);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6568e-01 | 1.0000e+00 |       66273 |      400000 |         1000 |        1000 |         0.2 |reached target bit errors
     0.25 | 1.4965e-01 | 9.9900e-01 |       59858 |      400000 |          999 |        1000 |         0.2 |reached target bit errors
      0.5 | 1.2336e-01 | 9.9900e-01 |       49342 |      400000 |          999 |        1000 |         0.2 |reached target bit errors
     0.75 | 9.6135e-02 | 9.9100e-01 |       38454 |      400000 |          991 |        1000 |         0.3 |reached target bit errors
      1.0 | 6.8543e-02 | 9.4500e-01 |       27417 |      400000 |          945 |        1000 |         0.2 |reached target bit errors
     1.25 | 3.9152e-02 | 8.3300e-01 |       15661 |      400000 |          833 |        1000 |         0.2 |reached target bit errors
      1.5 | 2.2040e-02 | 6.2400e-01 |        8816 |      400000 |          624 |        1000 |         0.2 |reached target bit errors
     1.75 | 9.1300e-03 | 3.8400e-01 |        3652 |      400000 |          384 |        1000 |         0.2 |reached target bit errors
      2.0 | 2.8075e-03 | 1.6600e-01 |        2246 |      800000 |          332 |        2000 |         0.5 |reached target bit errors
     2.25 | 8.5500e-04 | 6.2000e-02 |        2052 |     2400000 |          372 |        6000 |         1.4 |reached target bit errors
      2.5 | 1.9837e-04 | 2.1115e-02 |        2063 |    10400000 |          549 |       26000 |         6.3 |reached target bit errors
     2.75 | 2.9600e-05 | 4.1000e-03 |        1184 |    40000000 |          410 |      100000 |        24.3 |reached max iter
      3.0 | 6.5750e-06 | 9.1000e-04 |         263 |    40000000 |           91 |      100000 |        24.3 |reached max iter
     3.25 | 5.5000e-07 | 1.4000e-04 |          22 |    40000000 |           14 |      100000 |        24.3 |reached max iter
      3.5 | 7.5000e-08 | 3.0000e-05 |           3 |    40000000 |            3 |      100000 |        24.5 |reached max iter
     3.75 | 2.5000e-08 | 1.0000e-05 |           1 |    40000000 |            1 |      100000 |        24.3 |reached max iter
```


    
Unfortunately, we observe only very minor gains for the 5G LDPC code. We empirically observed that gain vanishes for more iterations and longer codewords, i.e., for most practical use-cases of the 5G LDPC code the gains are only minor.
    
However, there may be other `codes` `on` `graphs` that benefit from the principle idea of weighted BP - or other channel setups? Feel free to adjust this notebook and train for your favorite code / channel.
    
Other ideas for own experiments:
 
- Implement weighted BP with unique weights per iteration.
- Apply the concept to (scaled) min-sum decoding as in [5].
- Can you replace the complete CN update by a neural network?
- Verify the results from all-zero simulations for a <em>real</em> system simulation with explicit encoder and random data
- What happens in combination with higher order modulation?

