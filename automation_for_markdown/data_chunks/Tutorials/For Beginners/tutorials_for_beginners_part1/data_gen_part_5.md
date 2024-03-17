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
## Eager vs Graph Mode
## Exercise
  
  

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
## Eager vs Graph Mode<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Eager-vs-Graph-Mode" title="Permalink to this headline"></a>
    
So far, we have executed the example in <em>eager</em> mode. This allows to run TensorFlow ops as if it was written NumPy and simplifies development and debugging.
    
However, to unleash Sionna’s full performance, we need to activate <em>graph</em> mode which can be enabled with the function decorator <em>@tf.function()</em>.
    
We refer to <a class="reference external" href="https://www.tensorflow.org/guide/function">TensorFlow Functions</a> for further details.

```python
[21]:
```

```python
@tf.function() # enables graph-mode of the following function
def run_graph(batch_size, ebno_db):
    # all code inside this function will be executed in graph mode, also calls of other functions
    print(f"Tracing run_graph for values batch_size={batch_size} and ebno_db={ebno_db}.") # print whenever this function is traced
    return model_coded_awgn(batch_size, ebno_db)
```
```python
[22]:
```

```python
batch_size = 10 # try also different batch sizes
ebno_db = 1.5
# run twice - how does the output change?
run_graph(batch_size, ebno_db)
```


```python
Tracing run_graph for values batch_size=10 and ebno_db=1.5.
```
```python
[22]:
```
```python
(<tf.Tensor: shape=(10, 1024), dtype=float32, numpy=
 array([[1., 1., 0., ..., 0., 1., 0.],
        [1., 1., 1., ..., 0., 1., 1.],
        [1., 1., 0., ..., 0., 1., 1.],
        ...,
        [0., 1., 0., ..., 0., 0., 0.],
        [0., 1., 0., ..., 0., 1., 0.],
        [0., 1., 1., ..., 0., 1., 1.]], dtype=float32)>,
 <tf.Tensor: shape=(10, 1024), dtype=float32, numpy=
 array([[1., 1., 0., ..., 0., 1., 0.],
        [1., 1., 1., ..., 0., 1., 1.],
        [1., 1., 0., ..., 0., 1., 1.],
        ...,
        [0., 1., 0., ..., 0., 0., 0.],
        [0., 1., 0., ..., 0., 1., 0.],
        [0., 1., 1., ..., 0., 1., 1.]], dtype=float32)>)
```

    
In graph mode, Python code (i.e., <em>non-TensorFlow code</em>) is only executed whenever the function is <em>traced</em>. This happens whenever the input signature changes.
    
As can be seen above, the print statement was executed, i.e., the graph was traced again.
    
To avoid this re-tracing for different inputs, we now input tensors. You can see that the function is now traced once for input tensors of same dtype.
    
See <a class="reference external" href="https://www.tensorflow.org/guide/function#rules_of_tracing">TensorFlow Rules of Tracing</a> for details.
    
**Task:** change the code above such that tensors are used as input and execute the code with different input values. Understand when re-tracing happens.
    
<em>Remark</em>: if the input to a function is a tensor its signature must change and not <em>just</em> its value. For example the input could have a different size or datatype. For efficient code execution, we usually want to avoid re-tracing of the code if not required.

```python
[23]:
```

```python
# You can print the cached signatures with
print(run_graph.pretty_printed_concrete_signatures())
```


```python
run_graph(batch_size=10, ebno_db=1.5)
  Returns:
    (<1>, <2>)
      <1>: float32 Tensor, shape=(10, 1024)
      <2>: float32 Tensor, shape=(10, 1024)
```

    
We now compare the throughput of the different modes.

```python
[24]:
```

```python
repetitions = 4 # average over multiple runs
batch_size = BATCH_SIZE # try also different batch sizes
ebno_db = 1.5
# --- eager mode ---
t_start = time.perf_counter()
for _ in range(repetitions):
    bits, bits_hat = model_coded_awgn(tf.constant(batch_size, tf.int32),
                                tf.constant(ebno_db, tf. float32))
t_stop = time.perf_counter()
# throughput in bit/s
throughput_eager = np.size(bits.numpy())*repetitions / (t_stop - t_start) / 1e6
print(f"Throughput in Eager mode: {throughput_eager :.3f} Mbit/s")
# --- graph mode ---
# run once to trace graph (ignored for throughput)
run_graph(tf.constant(batch_size, tf.int32),
          tf.constant(ebno_db, tf. float32))
t_start = time.perf_counter()
for _ in range(repetitions):
    bits, bits_hat = run_graph(tf.constant(batch_size, tf.int32),
                                tf.constant(ebno_db, tf. float32))
t_stop = time.perf_counter()
# throughput in bit/s
throughput_graph = np.size(bits.numpy())*repetitions / (t_stop - t_start) / 1e6
print(f"Throughput in graph mode: {throughput_graph :.3f} Mbit/s")

```


```python
Throughput in Eager mode: 1.212 Mbit/s
Tracing run_graph for values batch_size=Tensor(&#34;batch_size:0&#34;, shape=(), dtype=int32) and ebno_db=Tensor(&#34;ebno_db:0&#34;, shape=(), dtype=float32).
Throughput in graph mode: 8.623 Mbit/s
```

    
Let’s run the same simulation as above in graph mode.

```python
[25]:
```

```python
ber_plots.simulate(run_graph,
                   ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 12),
                   batch_size=BATCH_SIZE,
                   num_target_block_errors=500,
                   legend="Coded (Graph mode)",
                   soft_estimates=True,
                   max_mc_iter=100,
                   show_fig=True,
                   forward_keyboard_interrupt=False);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -3.0 | 2.7922e-01 | 1.0000e+00 |      571845 |     2048000 |         2000 |        2000 |         0.2 |reached target block errors
   -2.273 | 2.5948e-01 | 1.0000e+00 |      531422 |     2048000 |         2000 |        2000 |         0.2 |reached target block errors
   -1.545 | 2.3550e-01 | 1.0000e+00 |      482301 |     2048000 |         2000 |        2000 |         0.2 |reached target block errors
   -0.818 | 2.0768e-01 | 1.0000e+00 |      425335 |     2048000 |         2000 |        2000 |         0.2 |reached target block errors
   -0.091 | 1.6918e-01 | 1.0000e+00 |      346477 |     2048000 |         2000 |        2000 |         0.2 |reached target block errors
    0.636 | 7.6115e-02 | 9.1650e-01 |      155883 |     2048000 |         1833 |        2000 |         0.2 |reached target block errors
    1.364 | 1.7544e-03 | 7.2125e-02 |       14372 |     8192000 |          577 |        8000 |         1.0 |reached target block errors
    2.091 | 7.8125e-08 | 2.0000e-05 |          16 |   204800000 |            4 |      200000 |        24.3 |reached max iter
    2.818 | 0.0000e+00 | 0.0000e+00 |           0 |   204800000 |            0 |      200000 |        24.4 |reached max iter
Simulation stopped as no error occurred @ EbNo = 2.8 dB.

```

<img alt="../_images/examples_Sionna_tutorial_part1_63_1.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part1_63_1.png" />

    
**Task:** TensorFlow allows to <em>compile</em> graphs with <a class="reference external" href="https://www.tensorflow.org/xla">XLA</a>. Try to further accelerate the code with XLA (`@tf.function(jit_compile=True)`).
    
<em>Remark</em>: XLA is still an experimental feature and not all TensorFlow (and, thus, Sionna) functions support XLA.
    
**Task 2:** Check the GPU load with `!nvidia-smi`. Find the best tradeoff between batch-size and throughput for your specific GPU architecture.

## Exercise<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Exercise" title="Permalink to this headline"></a>
    
Simulate the coded bit error rate (BER) for a Polar coded and 64-QAM modulation. Assume a codeword length of n = 200 and coderate = 0.5.
    
**Hint**: For Polar codes, successive cancellation list decoding (SCL) gives the best BER performance. However, successive cancellation (SC) decoding (without a list) is less complex.

```python
[26]:
```

```python
n = 200
coderate = 0.5
# *You can implement your code here*

```

<script type="application/vnd.jupyter.widget-state+json">
{"state": {}, "version_major": 2, "version_minor": 0}
</script>