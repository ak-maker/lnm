# Part 2: Differentiable Communication Systems<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part2.html#Part-2:-Differentiable-Communication-Systems" title="Permalink to this headline"></a>
    
This tutorial will guide you through Sionna, from its basic principles to the implementation of a point-to-point link with a 5G NR compliant code and a 3GPP channel model. You will also learn how to write custom trainable layers by implementing a state of the art neural receiver, and how to train and evaluate end-to-end communication systems.
    
The tutorial is structured in four notebooks:
 
- Part I: Getting started with Sionna
- **Part II: Differentiable Communication Systems**
- Part III: Advanced Link-level Simulations
- Part IV: Toward Learned Receivers

    
The <a class="reference external" href="https://nvlabs.github.io/sionna">official documentation</a> provides key material on how to use Sionna and how its components are implemented.
 
# Table of Content
## Imports
## Creating Custom Layers
  
  

## Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part2.html#Imports" title="Permalink to this headline"></a>

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
import matplotlib.pyplot as plt
# For saving complex Python data structures efficiently
import pickle
# For the implementation of the neural receiver
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer
```

## Creating Custom Layers<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part2.html#Creating-Custom-Layers" title="Permalink to this headline"></a>
    
Custom trainable (or not trainable) algorithms should be implemented as <a class="reference external" href="https://keras.io/api/layers/">Keras layers</a>. All Sionna components, such as the mapper, demapper, channel… are implemented as Keras layers.
    
To illustrate how this can be done, the next cell implements a simple neural network-based demapper which consists of three dense layers.

```python
[12]:
```

```python
class NeuralDemapper(Layer): # Inherits from Keras Layer
    def __init__(self):
        super().__init__()
        # The three dense layers that form the custom trainable neural network-based demapper
        self.dense_1 = Dense(64, 'relu')
        self.dense_2 = Dense(64, 'relu')
        self.dense_3 = Dense(NUM_BITS_PER_SYMBOL, None) # The last layer has no activation and therefore outputs logits, i.e., LLRs
    def call(self, y):
        # y : complex-valued with shape [batch size, block length]
        # y is first mapped to a real-valued tensor with shape
        #  [batch size, block length, 2]
        # where the last dimension consists of the real and imaginary components
        # The dense layers operate on the last dimension, and treat the inner dimensions as batch dimensions, i.e.,
        # all the received symbols are independently processed.
        nn_input = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1)
        z = self.dense_1(nn_input)
        z = self.dense_2(z)
        z = self.dense_3(z) # [batch size, number of symbols per block, number of bits per symbol]
        llr = tf.reshape(z, [tf.shape(y)[0], -1]) # [batch size, number of bits per block]
        return llr
```

    
A custom Keras layer is used as any other Sionna layer, and therefore integration to a Sionna-based communication is straightforward.
    
The following model uses the neural demapper instead of the conventional demapper. It takes at initialization a parameter that indicates if the model is intantiated to be trained or evaluated. When instantiated to be trained, the loss function is returned. Otherwise, the transmitted bits and LLRs are returned.

```python
[13]:
```

```python
class End2EndSystem(Model): # Inherits from Keras Model
    def __init__(self, training):
        super().__init__() # Must call the Keras model initializer
        self.constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=True) # Constellation is trainable
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = NeuralDemapper() # Intantiate the NeuralDemapper custom layer as any other
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) # Loss function
        self.training = training
    @tf.function(jit_compile=True) # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        # no channel coding used; we set coderate=1.0
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                coderate=1.0)
        bits = self.binary_source([batch_size, 1200]) # Blocklength set to 1200 bits
        x = self.mapper(bits)
        y = self.awgn_channel([x, no])
        llr = self.demapper(y)  # Call the NeuralDemapper custom layer as any other
        if self.training:
            loss = self.bce(bits, llr)
            return loss
        else:
            return bits, llr
```

    
When a model that includes a neural network is created, the neural network weights are randomly initialized typically leading to very poor performance.
    
To see this, the following cell benchmarks the previously defined untrained model against a conventional baseline.

```python
[14]:
```

```python
EBN0_DB_MIN = 10.0
EBN0_DB_MAX = 20.0

###############################
# Baseline
###############################
class Baseline(Model): # Inherits from Keras Model
    def __init__(self):
        super().__init__() # Must call the Keras model initializer
        self.constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
    @tf.function # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        # no channel coding used; we set coderate=1.0
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                coderate=1.0)
        bits = self.binary_source([batch_size, 1200]) # Blocklength set to 1200 bits
        x = self.mapper(bits)
        y = self.awgn_channel([x, no])
        llr = self.demapper([y,no])
        return bits, llr
###############################
# Benchmarking
###############################
baseline = Baseline()
model = End2EndSystem(False)
ber_plots = sn.utils.PlotBER("Neural Demapper")
ber_plots.simulate(baseline,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Baseline",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);
ber_plots.simulate(model,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Untrained model",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=True);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     10.0 | 2.6927e-02 | 1.0000e+00 |        4136 |      153600 |          128 |         128 |         0.7 |reached target block errors
   10.526 | 2.1426e-02 | 1.0000e+00 |        3291 |      153600 |          128 |         128 |         0.0 |reached target block errors
   11.053 | 1.6100e-02 | 1.0000e+00 |        2473 |      153600 |          128 |         128 |         0.0 |reached target block errors
   11.579 | 1.2051e-02 | 1.0000e+00 |        1851 |      153600 |          128 |         128 |         0.0 |reached target block errors
   12.105 | 9.1927e-03 | 1.0000e+00 |        1412 |      153600 |          128 |         128 |         0.0 |reached target block errors
   12.632 | 6.5234e-03 | 1.0000e+00 |        1002 |      153600 |          128 |         128 |         0.0 |reached target block errors
   13.158 | 4.4792e-03 | 9.8438e-01 |         688 |      153600 |          126 |         128 |         0.0 |reached target block errors
   13.684 | 2.7474e-03 | 9.6875e-01 |         422 |      153600 |          124 |         128 |         0.0 |reached target block errors
   14.211 | 1.6146e-03 | 8.8281e-01 |         248 |      153600 |          113 |         128 |         0.0 |reached target block errors
   14.737 | 9.9609e-04 | 7.0312e-01 |         306 |      307200 |          180 |         256 |         0.0 |reached target block errors
   15.263 | 5.2083e-04 | 4.7266e-01 |         160 |      307200 |          121 |         256 |         0.0 |reached target block errors
   15.789 | 3.4071e-04 | 3.3333e-01 |         157 |      460800 |          128 |         384 |         0.0 |reached target block errors
   16.316 | 1.4193e-04 | 1.5781e-01 |         109 |      768000 |          101 |         640 |         0.0 |reached target block errors
   16.842 | 6.0961e-05 | 7.1023e-02 |         103 |     1689600 |          100 |        1408 |         0.1 |reached target block errors
   17.368 | 2.4113e-05 | 2.8935e-02 |         100 |     4147200 |          100 |        3456 |         0.2 |reached target block errors
   17.895 | 7.6593e-06 | 9.1912e-03 |         100 |    13056000 |          100 |       10880 |         0.5 |reached target block errors
   18.421 | 2.7995e-06 | 3.3594e-03 |          43 |    15360000 |           43 |       12800 |         0.6 |reached max iter
   18.947 | 6.5104e-07 | 7.8125e-04 |          10 |    15360000 |           10 |       12800 |         0.6 |reached max iter
   19.474 | 6.5104e-08 | 7.8125e-05 |           1 |    15360000 |            1 |       12800 |         0.5 |reached max iter
     20.0 | 0.0000e+00 | 0.0000e+00 |           0 |    15360000 |            0 |       12800 |         0.5 |reached max iter
Simulation stopped as no error occurred @ EbNo = 20.0 dB.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     10.0 | 4.7460e-01 | 1.0000e+00 |       72899 |      153600 |          128 |         128 |         1.3 |reached target block errors
   10.526 | 4.7907e-01 | 1.0000e+00 |       73585 |      153600 |          128 |         128 |         0.0 |reached target block errors
   11.053 | 4.7525e-01 | 1.0000e+00 |       72999 |      153600 |          128 |         128 |         0.0 |reached target block errors
   11.579 | 4.7865e-01 | 1.0000e+00 |       73521 |      153600 |          128 |         128 |         0.0 |reached target block errors
   12.105 | 4.7684e-01 | 1.0000e+00 |       73242 |      153600 |          128 |         128 |         0.0 |reached target block errors
   12.632 | 4.7469e-01 | 1.0000e+00 |       72913 |      153600 |          128 |         128 |         0.0 |reached target block errors
   13.158 | 4.7614e-01 | 1.0000e+00 |       73135 |      153600 |          128 |         128 |         0.0 |reached target block errors
   13.684 | 4.7701e-01 | 1.0000e+00 |       73268 |      153600 |          128 |         128 |         0.0 |reached target block errors
   14.211 | 4.7544e-01 | 1.0000e+00 |       73027 |      153600 |          128 |         128 |         0.0 |reached target block errors
   14.737 | 4.7319e-01 | 1.0000e+00 |       72682 |      153600 |          128 |         128 |         0.0 |reached target block errors
   15.263 | 4.7740e-01 | 1.0000e+00 |       73329 |      153600 |          128 |         128 |         0.0 |reached target block errors
   15.789 | 4.7385e-01 | 1.0000e+00 |       72783 |      153600 |          128 |         128 |         0.0 |reached target block errors
   16.316 | 4.7344e-01 | 1.0000e+00 |       72721 |      153600 |          128 |         128 |         0.0 |reached target block errors
   16.842 | 4.7303e-01 | 1.0000e+00 |       72658 |      153600 |          128 |         128 |         0.0 |reached target block errors
   17.368 | 4.7378e-01 | 1.0000e+00 |       72773 |      153600 |          128 |         128 |         0.0 |reached target block errors
   17.895 | 4.7257e-01 | 1.0000e+00 |       72586 |      153600 |          128 |         128 |         0.0 |reached target block errors
   18.421 | 4.7377e-01 | 1.0000e+00 |       72771 |      153600 |          128 |         128 |         0.0 |reached target block errors
   18.947 | 4.7315e-01 | 1.0000e+00 |       72676 |      153600 |          128 |         128 |         0.0 |reached target block errors
   19.474 | 4.7217e-01 | 1.0000e+00 |       72525 |      153600 |          128 |         128 |         0.0 |reached target block errors
     20.0 | 4.7120e-01 | 1.0000e+00 |       72376 |      153600 |          128 |         128 |         0.0 |reached target block errors
```

<img alt="../_images/examples_Sionna_tutorial_part2_34_1.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part2_34_1.png" />
