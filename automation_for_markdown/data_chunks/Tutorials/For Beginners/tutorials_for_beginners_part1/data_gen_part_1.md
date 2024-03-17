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
## Sionna Data-flow and Design Paradigms
  
  

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
## Sionna Data-flow and Design Paradigms<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Sionna-Data-flow-and-Design-Paradigms" title="Permalink to this headline"></a>
    
Sionna inherently parallelizes simulations via <em>batching</em>, i.e., each element in the batch dimension is simulated independently.
    
This means the first tensor dimension is always used for <em>inter-frame</em> parallelization similar to an outer <em>for-loop</em> in Matlab/NumPy simulations, but operations can be operated in parallel.
    
To keep the dataflow efficient, Sionna follows a few simple design principles:
 
- Signal-processing components are implemented as an individual <a class="reference external" href="https://keras.io/api/layers/">Keras layer</a>.
- `tf.float32` is used as preferred datatype and `tf.complex64` for complex-valued datatypes, respectively.
This allows simpler re-use of components (e.g., the same scrambling layer can be used for binary inputs and LLR-values).
- `tf.float64`/`tf.complex128` are available when high precision is needed.
- Models can be developed in <em>eager mode</em> allowing simple (and fast) modification of system parameters.
- Number crunching simulations can be executed in the faster <em>graph mode</em> or even <em>XLA</em> acceleration (experimental) is available for most components.
- Whenever possible, components are automatically differentiable via <a class="reference external" href="https://www.tensorflow.org/guide/autodiff">auto-grad</a> to simplify the deep learning design-flow.
- Code is structured into sub-packages for different tasks such as channel coding, mapping,… (see <a class="reference external" href="http://nvlabs.github.io/sionna/api/sionna.html">API documentation</a> for details).

    
These paradigms simplify the re-useability and reliability of our components for a wide range of communications related applications.

