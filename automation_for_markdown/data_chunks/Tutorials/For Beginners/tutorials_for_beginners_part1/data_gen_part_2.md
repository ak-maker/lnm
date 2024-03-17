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
## Hello, Sionna!
  
  

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
## Hello, Sionna!<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html#Hello,-Sionna!" title="Permalink to this headline"></a>
    
Let’s start with a very simple simulation: Transmitting QAM symbols over an AWGN channel. We will implement the system shown in the figure below.
    
<img alt="QAM AWGN" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAa8AAAC/CAYAAABJylMuAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAaGVYSWZNTQAqAAAACAAEAQYAAwAAAAEAAgAAARIAAwAAAAEAAQAAASgAAwAAAAEAAgAAh2kABAAAAAEAAAA+AAAAAAADoAEAAwAAAAEAAQAAoAIABAAAAAEAAAGvoAMABAAAAAEAAAC/AAAAAO5zGdUAAALkaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA2LjAuMCI+CiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOnRpZmY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vdGlmZi8xLjAvIgogICAgICAgICAgICB4bWxuczpleGlmPSJodHRwOi8vbnMuYWRvYmUuY29tL2V4aWYvMS4wLyI+CiAgICAgICAgIDx0aWZmOkNvbXByZXNzaW9uPjE8L3RpZmY6Q29tcHJlc3Npb24+CiAgICAgICAgIDx0aWZmOlJlc29sdXRpb25Vbml0PjI8L3RpZmY6UmVzb2x1dGlvblVuaXQ+CiAgICAgICAgIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6T3JpZW50YXRpb24+CiAgICAgICAgIDx0aWZmOlBob3RvbWV0cmljSW50ZXJwcmV0YXRpb24+MjwvdGlmZjpQaG90b21ldHJpY0ludGVycHJldGF0aW9uPgogICAgICAgICA8ZXhpZjpQaXhlbFhEaW1lbnNpb24+NDMxPC9leGlmOlBpeGVsWERpbWVuc2lvbj4KICAgICAgICAgPGV4aWY6Q29sb3JTcGFjZT4xPC9leGlmOkNvbG9yU3BhY2U+CiAgICAgICAgIDxleGlmOlBpeGVsWURpbWVuc2lvbj4xOTE8L2V4aWY6UGl4ZWxZRGltZW5zaW9uPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4K3524XQAANrxJREFUeAHtnQe4VNX19tct9N6LghQrWLCjYkDFGn2woYio2DUxsURiHo0t1vjPJxJrYk80GjV2E6MRQVExNuwNRZAiSu/lcu93fpvs6zDM3Dsz90w5M+/mGc7cM+fss8+7y7vW2muvXVYTJFMSAkJACAgBIRAhBMojVFYVVQgIASEgBISAQ0DkpYYgBISAEBACkUNA5BW5KlOBhYAQEAJCQOSlNiAEhIAQEAKRQ0DkFbkqU4GFgBAQAkJA5KU2IASEgBAQApFDQOQVuSpTgYWAEBACQkDkpTYgBISAEBACkUNA5BW5KlOBhYAQEAJCQOSlNiAEhIAQEAKRQ0DkFbkqU4GFgBAQAkJA5KU2IASEgBAQApFDQOQVuSpTgYWAEBACQkDkpTYgBISAEBACkUNA5BW5KlOBhYAQEAJCQOSlNiAEhIAQEAKRQ0DkFbkqU4GFgBAQAkJA5KU2IASEgBAQApFDQOQVuSpTgYWAEBACQkDkpTYgBISAEBACkUNA5BW5KlOBhYAQEAJCQOSlNiAEhIAQEAKRQ0DkFbkqU4GFgBAQAkJA5KU2IASEgBAQApFDQOQVuSpTgYWAEBACQkDkpTYgBISAEBACkUNA5BW5KlOBhYAQEAJCQOSlNiAEhIAQEAKRQ0DkFbkqU4GFgBAQAkJA5KU2IASEgBAQApFDQOQVuSpTgYWAEBACQkDkpTYgBISAEBACkUNA5BW5KlOBhYAQEAJCQOSlNiAEhIAQEAKRQ0DkFbkqU4GFgBAQAkKgUhAIASGQHgKTJ082/+FOvislR+CUU06x1q1b24477mgDBw60Jk2aJL9YvwiBFBEoqwlSitfqMiFQ8giMGDHCVq1aZQMGDHCDMYDwXSk5Ao899pgtXbrUHnzwQRs1apSdf/75jsyS36FfhED9CIi86sdIVwgBp11BXMcdd5ydd955Tnto2rSpQ8YfBVNiBJYsWWLV1dXG8dFHH7V7773XrrvuOjvggAOkhSWGTGdTQEDklQJIukQIdO3a1caNG2cHHnigtW3bVoBkiAAE9vLLL9vFF19szz//vPXo0SPDnHRbqSMg8ir1FqD3rxeBm266yaZPn+60BWlZ9cJV7wWrV6+266+/3r788ksbO3asderUqd57dIEQiEdA3obxiOhvIRCDAMSFhnD55ZebiCsGmAZ8xWGDea81a9bYO++8Y5CZkhBIFwGRV7qI6fqSQmDRokXOMUOmwnCrHe/DnXfe2d577z3nABNu7sqtFBAQeZVCLesdM0YAN/jBgwdnfL9uTI4AbvMir+T46Je6ERB51Y2Pfi1xBN588023NqnEYcjK67Pu65NPPnHmw6w8QJkWNQIir6KuXr1cQxHAbCiTYUNRTHw/psNly5Y5N/rEV+isEEiOgMgrOTb6RQgIASEgBAoUAZFXgVaMiiUEhIAQEALJERB5JcdGvwgBISAEhECBIiDyKtCKUbGEgBAQAkIgOQIir+TY6BchUHAIsKB38eLFCcu1bt06LfhNiIxOFiMCIq9irFW9U9EhQDSKG2+80Q499FD3Of744110CgLe+vTaa6+5oLdc69Ptt99ukyZNMoiNdM8999iECRNq//bX6SgEooaAyCtqNabyliQCDz30kLG1CKR12mmnORfzCy+80ObPn1+LB1uO3HnnnbZy5cracy+99JI9+eSTtnbtWnfukUcesVdffdWqqqpqr9EXIRBFBEReUaw1lbnkELjjjjtsp512smHDhtnw4cNdVPYPPvjAXnnlFUdEs2bNcqQ0d+5ce/fdd51mxVZ9H330kSMv9iDjmvfff9/efvttaV4l14KK74VFXsVXp3qjIkOAOa4vvvjC9t57b2vVqpU1b97cEdkmm2ziIlRgEnzqqadsyy23tH333deRFabDOXPm2IIFC5zWNWXKFBs/frxbcA2BSfMqskZSgq8j8irBStcrRwsByAmyadasWW3BGzVqZJWVlY6YMAliMjzssMPsyCOPdFHwFy5caJ9++ql16dLFDj74YHvxxRftmWeesZEjR7pdjSE2baJeC6e+RBABkVcEK01FLi0ECKNEiCpMgN7xYt68ec4M2K1bN3vjjTecBvb55587s+DMmTOdUwbmw6222sqZGZ977jl33RFHHGF9+/at1dhKC0m9bTEhIPIqptrUuxQlAmhYaFVPP/20TZ061diN+K677jJ2dx46dKjhhEGQW3Yl5hzmRZw7IK+tt97amRjZQ2vzzTe3zTbbzLbbbjtjvswTYVGCppcqegQqi/4N9YJCoAgQOPfcc+23v/2tnX766VZeXm7Tpk2z3/3ud4b5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" />
    
We will use upper case for naming simulation parameters that are used throughout this notebook
    
Every layer needs to be initialized once before it can be used.
    
**Tip**: Use the <a class="reference external" href="http://nvlabs.github.io/sionna/api/sionna.html">API documentation</a> to find an overview of all existing components. You can directly access the signature and the docstring within jupyter via `Shift+TAB`.
    
<em>Remark</em>: Most layers are defined to be complex-valued.
    
We first need to create a QAM constellation.

```python
[3]:
```

```python
NUM_BITS_PER_SYMBOL = 2 # QPSK
constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
constellation.show();
```

<img alt="../_images/examples_Sionna_tutorial_part1_13_0.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part1_13_0.png" />

    
**Task:** Try to change the modulation order, e.g., to 16-QAM.
    
We then need to setup a mapper to map bits into constellation points. The mapper takes as parameter the constellation.
    
We also need to setup a corresponding demapper to compute log-likelihood ratios (LLRs) from received noisy samples.

```python
[4]:
```

```python
mapper = sn.mapping.Mapper(constellation=constellation)
# The demapper uses the same constellation object as the mapper
demapper = sn.mapping.Demapper("app", constellation=constellation)
```

    
**Tip**: You can access the signature+docstring via `?` command and print the complete class definition via `??` operator.
    
Obviously, you can also access the source code via <a class="reference external" href="https://github.com/nvlabs/sionna/">https://github.com/nvlabs/sionna/</a>.

```python
[5]:
```

```python
# print class definition of the Constellation class
sn.mapping.Mapper??
```


```python
Init signature: sn.mapping.Mapper(*args, **kwargs)
Source:
class Mapper(Layer):
    # pylint: disable=line-too-long
    r&#34;&#34;&#34;
    Mapper(constellation_type=None, num_bits_per_symbol=None, constellation=None, dtype=tf.complex64, **kwargs)
    Maps binary tensors to points of a constellation.
    This class defines a layer that maps a tensor of binary values
    to a tensor of points from a provided constellation.
    Parameters
    ----------
    constellation_type : One of [&#34;qam&#34;, &#34;pam&#34;, &#34;custom&#34;], str
        For &#34;custom&#34;, an instance of :class:`~sionna.mapping.Constellation`
        must be provided.
    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in [&#34;qam&#34;, &#34;pam&#34;].
    constellation :  Constellation
        An instance of :class:`~sionna.mapping.Constellation` or
        `None`. In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.
    dtype : One of [tf.complex64, tf.complex128], tf.DType
        The output dtype. Defaults to tf.complex64.
    Input
    -----
    : [..., n], tf.float or tf.int
        Tensor with with binary entries.
    Output
    ------
    : [...,n/Constellation.num_bits_per_symbol], tf.complex
        The mapped constellation symbols.
    Note
    ----
    The last input dimension must be an integer multiple of the
    number of bits per constellation symbol.
    &#34;&#34;&#34;
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 dtype=tf.complex64,
                 **kwargs
                ):
        super().__init__(dtype=dtype, **kwargs)
        assert dtype in [tf.complex64, tf.complex128],\
            &#34;dtype must be tf.complex64 or tf.complex128&#34;
        #self._dtype = dtype
        #print(dtype, self._dtype)
        if constellation is not None:
            assert constellation_type in [None, &#34;custom&#34;], \
                &#34;&#34;&#34;`constellation_type` must be &#34;custom&#34;.&#34;&#34;&#34;
            assert num_bits_per_symbol in \
                     [None, constellation.num_bits_per_symbol], \
                &#34;&#34;&#34;`Wrong value of `num_bits_per_symbol.`&#34;&#34;&#34;
            self._constellation = constellation
        else:
            assert constellation_type in [&#34;qam&#34;, &#34;pam&#34;], \
                &#34;Wrong constellation type.&#34;
            assert num_bits_per_symbol is not None, \
                &#34;`num_bits_per_symbol` must be provided.&#34;
            self._constellation = Constellation(constellation_type,
                                                num_bits_per_symbol,
                                                dtype=self._dtype)
        self._binary_base = 2**tf.constant(
                        range(self.constellation.num_bits_per_symbol-1,-1,-1))
    @property
    def constellation(self):
        &#34;&#34;&#34;The Constellation used by the Mapper.&#34;&#34;&#34;
        return self._constellation
    def call(self, inputs):
        tf.debugging.assert_greater_equal(tf.rank(inputs), 2,
            message=&#34;The input must have at least rank 2&#34;)
        # Reshape inputs to the desired format
        new_shape = [-1] + inputs.shape[1:-1].as_list() + \
           [int(inputs.shape[-1] / self.constellation.num_bits_per_symbol),
            self.constellation.num_bits_per_symbol]
        inputs_reshaped = tf.cast(tf.reshape(inputs, new_shape), tf.int32)
        # Convert the last dimension to an integer
        int_rep = tf.reduce_sum(inputs_reshaped * self._binary_base, axis=-1)
        # Map integers to constellation symbols
        x = tf.gather(self.constellation.points, int_rep, axis=0)
        return x
File:           ~/.local/lib/python3.8/site-packages/sionna/mapping.py
Type:           type
Subclasses:
```

    
As can be seen, the `Mapper` class inherits from `Layer`, i.e., implements a Keras layer.
    
This allows to simply built complex systems by using the <a class="reference external" href="https://keras.io/guides/functional_api/">Keras functional API</a> to stack layers.
    
Sionna provides as utility a binary source to sample uniform i.i.d. bits.

```python
[6]:
```

```python
binary_source = sn.utils.BinarySource()
```

    
Finally, we need the AWGN channel.

```python
[7]:
```

```python
awgn_channel = sn.channel.AWGN()
```

    
Sionna provides a utility function to compute the noise power spectral density ratio $N_0$ from the energy per bit to noise power spectral density ratio $E_b/N_0$ in dB and a variety of parameters such as the coderate and the nunber of bits per symbol.

```python
[8]:
```

```python
no = sn.utils.ebnodb2no(ebno_db=10.0,
                        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                        coderate=1.0) # Coderate set to 1 as we do uncoded transmission here
```

    
We now have all the components we need to transmit QAM symbols over an AWGN channel.
    
Sionna natively supports multi-dimensional tensors.
    
Most layers operate at the last dimension and can have arbitrary input shapes (preserved at output).

```python
[9]:
```

```python
BATCH_SIZE = 64 # How many examples are processed by Sionna in parallel
bits = binary_source([BATCH_SIZE,
                      1024]) # Blocklength
print("Shape of bits: ", bits.shape)
x = mapper(bits)
print("Shape of x: ", x.shape)
y = awgn_channel([x, no])
print("Shape of y: ", y.shape)
llr = demapper([y, no])
print("Shape of llr: ", llr.shape)
```


```python
Shape of bits:  (64, 1024)
Shape of x:  (64, 512)
Shape of y:  (64, 512)
Shape of llr:  (64, 1024)
```

    
In <em>Eager</em> mode, we can directly access the values of each tensor. This simplify debugging.

```python
[10]:
```

```python
num_samples = 8 # how many samples shall be printed
num_symbols = int(num_samples/NUM_BITS_PER_SYMBOL)
print(f"First {num_samples} transmitted bits: {bits[0,:num_samples]}")
print(f"First {num_symbols} transmitted symbols: {np.round(x[0,:num_symbols], 2)}")
print(f"First {num_symbols} received symbols: {np.round(y[0,:num_symbols], 2)}")
print(f"First {num_samples} demapped llrs: {np.round(llr[0,:num_samples], 2)}")
```


```python
First 8 transmitted bits: [0. 1. 1. 1. 1. 0. 1. 0.]
First 4 transmitted symbols: [ 0.71-0.71j -0.71-0.71j -0.71+0.71j -0.71+0.71j]
First 4 received symbols: [ 0.65-0.61j -0.62-0.69j -0.6 +0.72j -0.86+0.63j]
First 8 demapped llrs: [-36.65  34.36  35.04  38.83  33.96 -40.5   48.47 -35.65]
```

    
Let’s visualize the received noisy samples.

```python
[11]:
```

```python
plt.figure(figsize=(8,8))
plt.axes().set_aspect(1)
plt.grid(True)
plt.title('Channel output')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.scatter(tf.math.real(y), tf.math.imag(y))
plt.tight_layout()
```

<img alt="../_images/examples_Sionna_tutorial_part1_31_0.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part1_31_0.png" />

    
**Task:** One can play with the SNR to visualize the impact on the received samples.
    
**Advanced Task:** Compare the LLR distribution for “app” demapping with “maxlog” demapping. The <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html">Bit-Interleaved Coded Modulation</a> example notebook can be helpful for this task.

