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
## Setting up Training Loops
  
  

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

## Setting up Training Loops<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part2.html#Setting-up-Training-Loops" title="Permalink to this headline"></a>
    
Training of end-to-end communication systems consists in iterating over SGD steps.
    
The next cell implements a training loop of `NUM_TRAINING_ITERATIONS` iterations. The training SNR is set to $E_b/N_0 = 15$ dB.
    
At each iteration: - A forward pass through the end-to-end system is performed within a gradient tape - The gradients are computed using the gradient tape, and applied using the Adam optimizer - The estimated loss is periodically printed to follow the progress of training

```python
[15]:
```

```python
# Number of iterations used for training
NUM_TRAINING_ITERATIONS = 30000
# Set a seed for reproducibility
tf.random.set_seed(1)
# Instantiating the end-to-end model for training
model_train = End2EndSystem(training=True)
# Adam optimizer (SGD variant)
optimizer = tf.keras.optimizers.Adam()
# Training loop
for i in range(NUM_TRAINING_ITERATIONS):
    # Forward pass
    with tf.GradientTape() as tape:
        loss = model_train(BATCH_SIZE, 15.0) # The model is assumed to return the BMD rate
    # Computing and applying gradients
    grads = tape.gradient(loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    # Print progress
    if i % 100 == 0:
        print(f"{i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss:.2E}", end="\r")
```


```python
29900/30000  Loss: 2.02E-03
```

    
The weights of the trained model are saved using <a class="reference external" href="https://docs.python.org/3/library/pickle.html">pickle</a>.

```python
[16]:
```

```python
# Save the weightsin a file
weights = model_train.get_weights()
with open('weights-neural-demapper', 'wb') as f:
    pickle.dump(weights, f)
```

    
Finally, we evaluate the trained model and benchmark it against the previously introduced baseline.
    
We first instantiate the model for evaluation and load the saved weights.

```python
[17]:
```

```python
# Instantiating the end-to-end model for evaluation
model = End2EndSystem(training=False)
# Run one inference to build the layers and loading the weights
model(tf.constant(1, tf.int32), tf.constant(10.0, tf.float32))
with open('weights-neural-demapper', 'rb') as f:
    weights = pickle.load(f)
    model.set_weights(weights)
```

    
The trained model is then evaluated.

```python
[18]:
```

```python
# Computing and plotting BER
ber_plots.simulate(model,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100,
                  legend="Trained model",
                  soft_estimates=True,
                  max_mc_iter=100,
                  show_fig=True);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     10.0 | 2.6094e-02 | 1.0000e+00 |        4008 |      153600 |          128 |         128 |         0.4 |reached target block errors
   10.526 | 2.0768e-02 | 1.0000e+00 |        3190 |      153600 |          128 |         128 |         0.0 |reached target block errors
   11.053 | 1.5729e-02 | 1.0000e+00 |        2416 |      153600 |          128 |         128 |         0.0 |reached target block errors
   11.579 | 1.1667e-02 | 1.0000e+00 |        1792 |      153600 |          128 |         128 |         0.0 |reached target block errors
   12.105 | 8.3789e-03 | 1.0000e+00 |        1287 |      153600 |          128 |         128 |         0.0 |reached target block errors
   12.632 | 6.1458e-03 | 1.0000e+00 |         944 |      153600 |          128 |         128 |         0.0 |reached target block errors
   13.158 | 3.8411e-03 | 9.7656e-01 |         590 |      153600 |          125 |         128 |         0.0 |reached target block errors
   13.684 | 2.8971e-03 | 9.7656e-01 |         445 |      153600 |          125 |         128 |         0.0 |reached target block errors
   14.211 | 1.6602e-03 | 8.4375e-01 |         255 |      153600 |          108 |         128 |         0.0 |reached target block errors
   14.737 | 9.8958e-04 | 6.7578e-01 |         304 |      307200 |          173 |         256 |         0.0 |reached target block errors
   15.263 | 5.0130e-04 | 4.7656e-01 |         154 |      307200 |          122 |         256 |         0.0 |reached target block errors
   15.789 | 2.5228e-04 | 2.6367e-01 |         155 |      614400 |          135 |         512 |         0.0 |reached target block errors
   16.316 | 1.4453e-04 | 1.6250e-01 |         111 |      768000 |          104 |         640 |         0.0 |reached target block errors
   16.842 | 5.2548e-05 | 5.8594e-02 |         113 |     2150400 |          105 |        1792 |         0.1 |reached target block errors
   17.368 | 2.7083e-05 | 3.1875e-02 |         104 |     3840000 |          102 |        3200 |         0.1 |reached target block errors
   17.895 | 8.6520e-06 | 1.0382e-02 |         101 |    11673600 |          101 |        9728 |         0.3 |reached target block errors
   18.421 | 2.7344e-06 | 3.2812e-03 |          42 |    15360000 |           42 |       12800 |         0.4 |reached max iter
   18.947 | 8.4635e-07 | 1.0156e-03 |          13 |    15360000 |           13 |       12800 |         0.4 |reached max iter
   19.474 | 1.3021e-07 | 1.5625e-04 |           2 |    15360000 |            2 |       12800 |         0.4 |reached max iter
     20.0 | 0.0000e+00 | 0.0000e+00 |           0 |    15360000 |            0 |       12800 |         0.4 |reached max iter
Simulation stopped as no error occurred @ EbNo = 20.0 dB.

```

<img alt="../_images/examples_Sionna_tutorial_part2_43_1.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part2_43_1.png" />

<script type="application/vnd.jupyter.widget-state+json">
{"state": {}, "version_major": 2, "version_minor": 0}
</script>