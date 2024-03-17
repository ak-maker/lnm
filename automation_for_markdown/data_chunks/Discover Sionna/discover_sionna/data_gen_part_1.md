# Discover Sionna<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Discover-Sionna" title="Permalink to this headline"></a>
    
This example notebook will guide you through the basic principles and illustrates the key features of <a class="reference external" href="https://nvlabs.github.io/sionna">Sionna</a>. With only a few commands, you can simulate the PHY-layer link-level performance for many 5G-compliant components, including easy visualization of the results.

# Table of Content
## Load Required Packages<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Load-Required-Packages" title="Permalink to this headline"></a>
## Sionna Data-flow and Design Paradigms<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Sionna-Data-flow-and-Design-Paradigms" title="Permalink to this headline"></a>
## Let’s Get Started - The First Layers (<em>Eager Mode</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Let’s-Get-Started---The-First-Layers-(Eager-Mode)" title="Permalink to this headline"></a>
## Batches and Multi-dimensional Tensors<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Batches-and-Multi-dimensional-Tensors" title="Permalink to this headline"></a>
  
  

## Load Required Packages<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Load-Required-Packages" title="Permalink to this headline"></a>
    
The Sionna python package must be <a class="reference external" href="https://nvlabs.github.io/sionna/installation.html">installed</a>.

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
import numpy as np
import tensorflow as tf
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
# IPython "magic function" for inline plots
%matplotlib inline
import matplotlib.pyplot as plt
```

    
**Tip**: you can run bash commands in Jupyter via the `!` operator.

```python
[2]:
```

```python
!nvidia-smi
```


```python
Wed Mar 16 14:05:36 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| 51%   65C    P2   208W / 350W |   5207MiB / 24267MiB |     39%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:4C:00.0 Off |                  N/A |
|  0%   28C    P8    13W / 350W |  17371MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

    
In case multiple GPUs are available, we restrict this notebook to single-GPU usage. You can ignore this command if only one GPU is available.
    
Further, we want to avoid that this notebook instantiates the whole GPU memory when initialized and set `memory_growth` as active.
    
<em>Remark</em>: Sionna does not require a GPU. Everything can also run on your CPU - but you may need to wait a little longer.

```python
[3]:
```

```python
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Index of the GPU to be used
    try:
        #tf.config.set_visible_devices([], 'GPU')
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)
```


```python
Number of GPUs available : 2
Only GPU number 0 used.
```
## Sionna Data-flow and Design Paradigms<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Sionna-Data-flow-and-Design-Paradigms" title="Permalink to this headline"></a>
    
Sionna inherently parallelizes simulations via <em>batching</em>, i.e., each element in the batch dimension is simulated independently.
    
This means the first tensor dimension is always used for <em>inter-frame</em> parallelization similar to an outer <em>for-loop</em> in Matlab/NumPy simulations.
    
To keep the dataflow efficient, Sionna follows a few simple design principles:
 
- Signal-processing components are implemented as an individual <a class="reference external" href="https://keras.io/api/layers/">Keras layer</a>.
- `tf.float32` is used as preferred datatype and `tf.complex64` for complex-valued datatypes, respectively.
This allows simpler re-use of components (e.g., the same scrambling layer can be used for binary inputs and LLR-values).
- Models can be developed in <em>eager mode</em> allowing simple (and fast) modification of system parameters.
- Number crunching simulations can be executed in the faster <em>graph mode</em> or even <em>XLA</em> acceleration is available for most components.
- Whenever possible, components are automatically differentiable via <a class="reference external" href="https://www.tensorflow.org/guide/autodiff">auto-grad</a> to simplify the deep learning design-flow.
- Code is structured into sub-packages for different tasks such as channel coding, mapping,… (see <a class="reference external" href="https://nvlabs.github.io/sionna/api/sionna.html">API documentation</a> for details).

    
The division into individual blocks simplifies deployment and all layers and functions comes with unittests to ensure their correct behavior.
    
These paradigms simplify the re-useability and reliability of our components for a wide range of communications related applications.

## Let’s Get Started - The First Layers (<em>Eager Mode</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Let’s-Get-Started---The-First-Layers-(Eager-Mode)" title="Permalink to this headline"></a>
    
Every layer needs to be initialized once before it can be used.
    
**Tip**: use the <a class="reference external" href="https://nvlabs.github.io/sionna/api/sionna.html">API documentation</a> to find an overview of all existing components.
    
We now want to transmit some symbols over an AWGN channel. First, we need to initialize the corresponding layer.

```python
[4]:
```

```python
channel = sionna.channel.AWGN() # init AWGN channel layer
```

    
In this first example, we want to add Gaussian noise to some given values of `x`.
    
Remember - the first dimension is the <em>batch-dimension</em>.
    
We simulate 2 message frames each containing 4 symbols.
    
<em>Remark</em>: the <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.html#awgn">AWGN channel</a> is defined to be complex-valued.

```python
[5]:
```

```python
# define a (complex-valued) tensor to be transmitted
x = tf.constant([[0., 1.5, 1., 0.],[-1., 0., -2, 3 ]], dtype=tf.complex64)
# let's have look at the shape
print("Shape of x: ", x.shape)
print("Values of x: ", x)
```


```python
Shape of x:  (2, 4)
Values of x:  tf.Tensor(
[[ 0. +0.j  1.5+0.j  1. +0.j  0. +0.j]
 [-1. +0.j  0. +0.j -2. +0.j  3. +0.j]], shape=(2, 4), dtype=complex64)
```

    
We want to simulate the channel at an SNR of 5 dB. For this, we can simply <em>call</em> the previously defined layer `channel`.
    
If you have never used <a class="reference external" href="https://keras.io">Keras</a> you can think of a layer as of a function: it has an input and returns the processed output.
    
<em>Remark</em>: Each time this cell is executed a new noise realization is drawn.

```python
[6]:
```

```python
ebno_db = 5
# calculate noise variance from given EbNo
no = sionna.utils.ebnodb2no(ebno_db = ebno_db,
                            num_bits_per_symbol=2, # QPSK
                            coderate=1)
y = channel([x, no])
print("Noisy symbols are: ", y)
```


```python
Noisy symbols are:  tf.Tensor(
[[ 0.17642795-0.21076633j  1.540727  +0.2577709j   0.676615  -0.14763176j
  -0.14807788-0.01961605j]
 [-0.9018068 -0.04732923j -0.55583185+0.41312575j -1.8852113 -0.23232108j
   3.3803759 +0.2269492j ]], shape=(2, 4), dtype=complex64)
```
## Batches and Multi-dimensional Tensors<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Batches-and-Multi-dimensional-Tensors" title="Permalink to this headline"></a>
    
Sionna natively supports multi-dimensional tensors.
    
Most layers operate at the last dimension and can have arbitrary input shapes (preserved at output).
    
Let us assume we want to add a CRC-24 check to 64 codewords of length 500 (e.g., different CRC per sub-carrier). Further, we want to parallelize the simulation over a batch of 100 samples.

```python
[7]:
```

```python
batch_size = 100 # outer level of parallelism
num_codewords = 64 # codewords per batch sample
info_bit_length = 500 # info bits PER codeword
source = sionna.utils.BinarySource() # yields random bits
u = source([batch_size, num_codewords, info_bit_length]) # call the source layer
print("Shape of u: ", u.shape)
# initialize an CRC encoder with the standard compliant "CRC24A" polynomial
encoder_crc = sionna.fec.crc.CRCEncoder("CRC24A")
decoder_crc = sionna.fec.crc.CRCDecoder(encoder_crc) # connect to encoder
# add the CRC to the information bits u
c = encoder_crc(u) # returns a list [c, crc_valid]
print("Shape of c: ", c.shape)
print("Processed bits: ", np.size(c.numpy()))
# we can also verify the results
# returns list of [info bits without CRC bits, indicator if CRC holds]
u_hat, crc_valid = decoder_crc(c)
print("Shape of u_hat: ", u_hat.shape)
print("Shape of crc_valid: ", crc_valid.shape)
print("Valid CRC check of first codeword: ", crc_valid.numpy()[0,0,0])
```


```python
Shape of u:  (100, 64, 500)
Shape of c:  (100, 64, 524)
Processed bits:  3353600
Shape of u_hat:  (100, 64, 500)
Shape of crc_valid:  (100, 64, 1)
Valid CRC check of first codeword:  True
```

    
We want to do another simulation but for 5 independent users.
    
Instead of defining 5 different tensors, we can simply add another dimension.

```python
[8]:
```

```python
num_users = 5
u = source([batch_size, num_users, num_codewords, info_bit_length])
print("New shape of u: ", u.shape)
# We can re-use the same encoder as before
c = encoder_crc(u)
print("New shape of c: ", c.shape)
print("Processed bits: ", np.size(c.numpy()))
```


```python
New shape of u:  (100, 5, 64, 500)
New shape of c:  (100, 5, 64, 524)
Processed bits:  16768000
```

    
Often a good visualization of results helps to get new research ideas. Thus, Sionna has built-in plotting functions.
    
Let’s have look at a 16-QAM constellation.

```python
[9]:
```

```python
constellation = sionna.mapping.Constellation("qam", num_bits_per_symbol=4)
constellation.show();
```

