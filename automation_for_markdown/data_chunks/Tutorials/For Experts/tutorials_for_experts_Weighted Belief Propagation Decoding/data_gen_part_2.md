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
### Training
### Results
## Further Experiments
### Damped BP
  
  

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

### Training<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html#Training" title="Permalink to this headline"></a>
    
We now train the model for a fixed number of SGD training iterations.
    
**Note**: this is a very basic implementation of the training loop. You can also try more sophisticated training loops with early stopping, different hyper-parameters or optimizers etc.

```python
[8]:
```

```python
# training parameters
batch_size = 1000
train_iter = 200
ebno_db = 4.0
clip_value_grad = 10 # gradient clipping for stable training convergence
# bmi is used as metric to evaluate the intermediate results
bmi = BitwiseMutualInformation()
# try also different optimizers or different hyperparameters
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
for it in range(0, train_iter):
    with tf.GradientTape() as tape:
        b, llr, loss = model(batch_size, ebno_db)
    grads = tape.gradient(loss, model.trainable_variables)
    grads = tf.clip_by_value(grads, -clip_value_grad, clip_value_grad, name=None)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # calculate and print intermediate metrics
    # only for information
    # this has no impact on the training
    if it%10==0: # evaluate every 10 iterations
        # calculate ber from received LLRs
        b_hat = hard_decisions(llr) # hard decided LLRs first
        ber = compute_ber(b, b_hat)
        # and print results
        mi = bmi(b, llr).numpy() # calculate bit-wise mutual information
        l = loss.numpy() # copy loss to numpy for printing
        print(f"Current loss: {l:3f} ber: {ber:.4f} bmi: {mi:.3f}".format())
        bmi.reset_states() # reset the BMI metric
```


```python
Current loss: 0.048708 ber: 0.0120 bmi: 0.934
Current loss: 0.058506 ber: 0.0139 bmi: 0.923
Current loss: 0.052293 ber: 0.0125 bmi: 0.934
Current loss: 0.054314 ber: 0.0134 bmi: 0.928
Current loss: 0.051650 ber: 0.0125 bmi: 0.924
Current loss: 0.047477 ber: 0.0133 bmi: 0.931
Current loss: 0.045135 ber: 0.0122 bmi: 0.935
Current loss: 0.050638 ber: 0.0125 bmi: 0.938
Current loss: 0.045256 ber: 0.0119 bmi: 0.949
Current loss: 0.041335 ber: 0.0124 bmi: 0.952
Current loss: 0.040905 ber: 0.0107 bmi: 0.937
Current loss: 0.043627 ber: 0.0125 bmi: 0.949
Current loss: 0.044397 ber: 0.0126 bmi: 0.942
Current loss: 0.043392 ber: 0.0126 bmi: 0.938
Current loss: 0.043059 ber: 0.0133 bmi: 0.947
Current loss: 0.047521 ber: 0.0130 bmi: 0.937
Current loss: 0.040529 ber: 0.0116 bmi: 0.944
Current loss: 0.041838 ber: 0.0128 bmi: 0.942
Current loss: 0.041801 ber: 0.0130 bmi: 0.940
Current loss: 0.042754 ber: 0.0142 bmi: 0.946
```
### Results<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html#Results" title="Permalink to this headline"></a>
    
After training, the weights of the decoder have changed. In average, the weights are smaller after training.

```python
[9]:
```

```python
model.decoder.show_weights() # show weights AFTER training
```


    
And let us compare the new BER performance. For this, we can simply call the ber_plot.simulate() function again as it internally stores all previous results (if `add_results` is True).

```python
[10]:
```

```python
ebno_dbs = np.array(np.arange(1, 7, 0.5))
batch_size = 10000
mc_ites = 100
ber_plot.simulate(model,
                  ebno_dbs=ebno_dbs,
                  batch_size=1000,
                  num_target_bit_errors=2000, # stop sim after 2000 bit errors
                  legend="Trained",
                  max_mc_iter=mc_iters,
                  soft_estimates=True);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      1.0 | 9.0730e-02 | 9.9600e-01 |        5716 |       63000 |          996 |        1000 |         0.2 |reached target bit errors
      1.5 | 7.8889e-02 | 9.8400e-01 |        4970 |       63000 |          984 |        1000 |         0.1 |reached target bit errors
      2.0 | 6.6365e-02 | 9.2500e-01 |        4181 |       63000 |          925 |        1000 |         0.1 |reached target bit errors
      2.5 | 4.9825e-02 | 8.2000e-01 |        3139 |       63000 |          820 |        1000 |         0.1 |reached target bit errors
      3.0 | 3.6603e-02 | 6.4400e-01 |        2306 |       63000 |          644 |        1000 |         0.1 |reached target bit errors
      3.5 | 2.2302e-02 | 4.2000e-01 |        2810 |      126000 |          840 |        2000 |         0.3 |reached target bit errors
      4.0 | 1.2577e-02 | 2.4400e-01 |        2377 |      189000 |          732 |        3000 |         0.5 |reached target bit errors
      4.5 | 6.5778e-03 | 1.3460e-01 |        2072 |      315000 |          673 |        5000 |         0.7 |reached target bit errors
      5.0 | 2.9769e-03 | 6.2818e-02 |        2063 |      693000 |          691 |       11000 |         1.7 |reached target bit errors
      5.5 | 1.3287e-03 | 2.9667e-02 |        2009 |     1512000 |          712 |       24000 |         3.6 |reached target bit errors
      6.0 | 5.2511e-04 | 1.2967e-02 |        2018 |     3843000 |          791 |       61000 |         9.2 |reached target bit errors
      6.5 | 2.0333e-04 | 5.6000e-03 |        1281 |     6300000 |          560 |      100000 |        15.0 |reached max iter
```


## Further Experiments<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html#Further-Experiments" title="Permalink to this headline"></a>
    
You will now see that the memory footprint can be drastically reduced by using the same weight for all messages. In the second part we will apply the concept to the 5G LDPC codes.

### Damped BP<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html#Damped-BP" title="Permalink to this headline"></a>
    
It is well-known that scaling of LLRs / messages can help to improve the performance of BP decoding in some scenarios [3,4]. In particular, this works well for very short codes such as the code we are currently analyzing.
    
We now follow the basic idea of [2] and scale all weights with the same scalar.

```python
[11]:
```

```python
# get weights of trained model
weights_bp = model.decoder.get_weights()
# calc mean value of weights
damping_factor = tf.reduce_mean(weights_bp)
# set all weights to the SAME constant scaling
weights_damped = tf.ones_like(weights_bp) * damping_factor
# and apply the new weights
model.decoder.set_weights(weights_damped)
# let us have look at the new weights again
model.decoder.show_weights()
# and simulate the BER again
leg_str = f"Damped BP (scaling factor {damping_factor.numpy():.3f})"
ber_plot.simulate(model,
                  ebno_dbs=ebno_dbs,
                  batch_size=1000,
                  num_target_bit_errors=2000, # stop sim after 2000 bit errors
                  legend=leg_str,
                  max_mc_iter=mc_iters,
                  soft_estimates=True);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      1.0 | 9.0333e-02 | 9.9500e-01 |        5691 |       63000 |          995 |        1000 |         0.2 |reached target bit errors
      1.5 | 7.6413e-02 | 9.7800e-01 |        4814 |       63000 |          978 |        1000 |         0.2 |reached target bit errors
      2.0 | 6.1556e-02 | 8.9800e-01 |        3878 |       63000 |          898 |        1000 |         0.1 |reached target bit errors
      2.5 | 4.8746e-02 | 7.9700e-01 |        3071 |       63000 |          797 |        1000 |         0.2 |reached target bit errors
      3.0 | 3.5746e-02 | 6.0800e-01 |        2252 |       63000 |          608 |        1000 |         0.1 |reached target bit errors
      3.5 | 2.0857e-02 | 3.7950e-01 |        2628 |      126000 |          759 |        2000 |         0.3 |reached target bit errors
      4.0 | 1.2222e-02 | 2.3433e-01 |        2310 |      189000 |          703 |        3000 |         0.5 |reached target bit errors
      4.5 | 6.4524e-03 | 1.2967e-01 |        2439 |      378000 |          778 |        6000 |         0.9 |reached target bit errors
      5.0 | 2.7712e-03 | 5.8667e-02 |        2095 |      756000 |          704 |       12000 |         1.8 |reached target bit errors
      5.5 | 1.2844e-03 | 2.8960e-02 |        2023 |     1575000 |          724 |       25000 |         3.7 |reached target bit errors
      6.0 | 5.0743e-04 | 1.2032e-02 |        2014 |     3969000 |          758 |       63000 |         9.5 |reached target bit errors
      6.5 | 2.1730e-04 | 5.5200e-03 |        1369 |     6300000 |          552 |      100000 |        15.2 |reached max iter
```



    
When looking at the results, we observe almost the same performance although we only scale by a single scalar. This implies that the number of weights of our model is by far too large and the memory footprint could be reduced significantly. However, isn’t it fascinating to see that this simple concept of weighted BP leads to the same results as the concept of <em>damped BP</em>?
    
**Note**: for more iterations it could be beneficial to implement an individual damping per iteration.

