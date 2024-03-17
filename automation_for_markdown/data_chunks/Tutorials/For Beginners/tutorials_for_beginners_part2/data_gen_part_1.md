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
## Gradient Computation Through End-to-end Systems
  
  

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

## Gradient Computation Through End-to-end Systems<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part2.html#Gradient-Computation-Through-End-to-end-Systems" title="Permalink to this headline"></a>
    
Let’s start by setting up a simple communication system that transmit bits modulated as QAM symbols over an AWGN channel.
    
However, compared to what we have previously done, we now make the constellation <em>trainable</em>. With Sionna, achieving this requires only setting a boolean parameter to `True` when instantiating the `Constellation` object.

```python
[2]:
```

```python
# Binary source to generate uniform i.i.d. bits
binary_source = sn.utils.BinarySource()
# 256-QAM constellation
NUM_BITS_PER_SYMBOL = 6
constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=True) # The constellation is set to be trainable
# Mapper and demapper
mapper = sn.mapping.Mapper(constellation=constellation)
demapper = sn.mapping.Demapper("app", constellation=constellation)
# AWGN channel
awgn_channel = sn.channel.AWGN()
```

    
As we have already seen, we can now easily simulate forward passes through the system we have just setup

```python
[3]:
```

```python
BATCH_SIZE = 128 # How many examples are processed by Sionna in parallel
EBN0_DB = 17.0 # Eb/N0 in dB
no = sn.utils.ebnodb2no(ebno_db=EBN0_DB,
                        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                        coderate=1.0) # Coderate set to 1 as we do uncoded transmission here
bits = binary_source([BATCH_SIZE,
                        1200]) # Blocklength
x = mapper(bits)
y = awgn_channel([x, no])
llr = demapper([y,no])
```

    
Just for fun, let’s visualize the channel inputs and outputs

```python
[4]:
```

```python
plt.figure(figsize=(8,8))
plt.axes().set_aspect(1.0)
plt.grid(True)
plt.scatter(tf.math.real(y), tf.math.imag(y), label='Output')
plt.scatter(tf.math.real(x), tf.math.imag(x), label='Input')
plt.legend(fontsize=20);
```

<img alt="../_images/examples_Sionna_tutorial_part2_12_0.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part2_12_0.png" />

    
Let’s now <em>optimize</em> the constellation through <em>stochastic gradient descent</em> (SGD). As we will see, this is made very easy by Sionna.
    
We need to define a <em>loss function</em> that we will aim to minimize.
    
We can see the task of the receiver as jointly solving, for each received symbol, `NUM_BITS_PER_SYMBOL` binary classification problems in order to reconstruct the transmitted bits. Therefore, a natural choice for the loss function is the <em>binary cross-entropy</em> (BCE) applied to each bit and to each received symbol.
    
<em>Remark:</em> The LLRs computed by the demapper are <em>logits</em> on the transmitted bits, and can therefore be used as-is to compute the BCE without any additional processing. <em>Remark 2:</em> The BCE is closely related to an achieveable information rate for bit-interleaved coded modulation systems [1,2]
    
[1] Georg Böcherer, “Principles of Coded Modulation”, <a class="reference external" href="http://www.georg-boecherer.de/bocherer2018principles.pdf">available online</a>
    
[2] F. Ait Aoudia and J. Hoydis, “End-to-End Learning for OFDM: From Neural Receivers to Pilotless Communication,” in IEEE Transactions on Wireless Communications, vol. 21, no. 2, pp. 1049-1063, Feb. 2022, doi: 10.1109/TWC.2021.3101364.

```python
[5]:
```

```python
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
print(f"BCE: {bce(bits, llr)}")
```


```python
BCE: 0.0001015052548609674
```

    
One iteration of SGD consists in three steps: 1. Perform a forward pass through the end-to-end system and compute the loss function 2. Compute the gradient of the loss function with respect to the trainable weights 3. Apply the gradient to the weights
    
To enable gradient computation, we need to perform the forward pass (step 1) within a `GradientTape`

```python
[6]:
```

```python
with tf.GradientTape() as tape:
    bits = binary_source([BATCH_SIZE,
                            1200]) # Blocklength
    x = mapper(bits)
    y = awgn_channel([x, no])
    llr = demapper([y,no])
    loss = bce(bits, llr)
```

    
Using the `GradientTape`, computing the gradient is done as follows

```python
[7]:
```

```python
gradient = tape.gradient(loss, tape.watched_variables())
```

    
`gradient` is a list of tensor, each tensor corresponding to a trainable variable of our model.
    
For this model, we only have a single trainable tensor: The constellation of shape [`2`, `2^NUM_BITS_PER_SYMBOL`], the first dimension corresponding to the real and imaginary components of the constellation points.
    
<em>Remark:</em> It is important to notice that the gradient computation was performed <em>through the demapper and channel</em>, which are conventional non-trainable algorithms implemented as <em>differentiable</em> Keras layers. This key feature of Sionna enables the training of end-to-end communication systems that combine both trainable and conventional and/or non-trainable signal processing algorithms.

```python
[8]:
```

```python
for g in gradient:
    print(g.shape)
```


```python
(2, 64)
```

    
Applying the gradient (third step) is performed using an <em>optimizer</em>. <a class="reference external" href="https://www.tensorflow.org/api_docs/python/tf/keras/optimizers">Many optimizers are available as part of TensorFlow</a>, and we use in this notebook `Adam`.

```python
[9]:
```

```python
optimizer = tf.keras.optimizers.Adam(1e-2)
```

    
Using the optimizer, the gradients can be applied to the trainable weights to update them

```python
[10]:
```

```python
optimizer.apply_gradients(zip(gradient, tape.watched_variables()));
```

    
Let compare the constellation before and after the gradient application

```python
[11]:
```

```python
fig = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL).show()
fig.axes[0].scatter(tf.math.real(constellation.points), tf.math.imag(constellation.points), label='After SGD')
fig.axes[0].legend();
```

<img alt="../_images/examples_Sionna_tutorial_part2_26_0.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part2_26_0.png" />

    
The SGD step has led to slight change in the position of the constellation points. Training of a communication system using SGD consists in looping over such SGD steps until a stop criterion is met.

