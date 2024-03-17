# Part 4: Toward Learned Receivers<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part4.html#Part-4:-Toward-Learned-Receivers" title="Permalink to this headline"></a>
    
This tutorial will guide you through Sionna, from its basic principles to the implementation of a point-to-point link with a 5G NR compliant code and a 3GPP channel model. You will also learn how to write custom trainable layers by implementing a state of the art neural receiver, and how to train and evaluate end-to-end communication systems.
    
The tutorial is structured in four notebooks:
 
- Part I: Getting started with Sionna
- Part II: Differentiable Communication Systems
- Part III: Advanced Link-level Simulations
- **Part IV: Toward Learned Receivers**

    
The <a class="reference external" href="https://nvlabs.github.io/sionna">official documentation</a> provides key material on how to use Sionna and how its components are implemented.
 
# Table of Content
## Imports
## Benchmarking the Neural Receiver
## Conclusion
  
  

## Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part4.html#Imports" title="Permalink to this headline"></a>

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
# For saving complex Python data structures efficiently
import pickle
# For plotting
%matplotlib inline
import matplotlib.pyplot as plt
# For the implementation of the neural receiver
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu
```

## Benchmarking the Neural Receiver<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part4.html#Benchmarking-the-Neural-Receiver" title="Permalink to this headline"></a>
    
We evaluate the trained model and benchmark it against the previously introduced baselines.
    
We first define and evaluate the baselines.

```python
[5]:
```

```python
class OFDMSystem(Model): # Inherits from Keras Model
    def __init__(self, perfect_csi):
        super().__init__() # Must call the Keras model initializer
        self.perfect_csi = perfect_csi
        n = int(RESOURCE_GRID.num_data_symbols*NUM_BITS_PER_SYMBOL) # Number of coded bits
        k = int(n*CODERATE) # Number of information bits
        self.k = k
        # The binary source will create batches of information bits
        self.binary_source = sn.utils.BinarySource()
        # The encoder maps information bits to coded bits
        self.encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)
        # The mapper maps blocks of information bits to constellation symbols
        self.mapper = sn.mapping.Mapper("qam", NUM_BITS_PER_SYMBOL)
        # The resource grid mapper maps symbols onto an OFDM resource grid
        self.rg_mapper = sn.ofdm.ResourceGridMapper(RESOURCE_GRID)
        # Frequency domain channel
        self.channel = sn.channel.OFDMChannel(CDL, RESOURCE_GRID, add_awgn=True, normalize_channel=True, return_channel=True)
        # The LS channel estimator will provide channel estimates and error variances
        self.ls_est = sn.ofdm.LSChannelEstimator(RESOURCE_GRID, interpolation_type="nn")
        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = sn.ofdm.LMMSEEqualizer(RESOURCE_GRID, STREAM_MANAGEMENT)
        # The demapper produces LLR for all coded bits
        self.demapper = sn.mapping.Demapper("app", "qam", NUM_BITS_PER_SYMBOL)
        # The decoder provides hard-decisions on the information bits
        self.decoder = sn.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)
    @tf.function # Graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        no = sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=NUM_BITS_PER_SYMBOL, coderate=CODERATE, resource_grid=RESOURCE_GRID)
        # Transmitter
        bits = self.binary_source([batch_size, NUM_UT, RESOURCE_GRID.num_streams_per_tx, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        x_rg = self.rg_mapper(x)
        # Channel
        y, h_freq = self.channel([x_rg, no])
        # Receiver
        if self.perfect_csi:
            h_hat, err_var = h_freq, 0.
        else:
            h_hat, err_var = self.ls_est ([y, no])
        x_hat, no_eff = self.lmmse_equ([y, h_hat, err_var, no])
        llr = self.demapper([x_hat, no_eff])
        bits_hat = self.decoder(llr)
        return bits, bits_hat
```
```python
[6]:
```

```python
ber_plots = sn.utils.PlotBER("Advanced neural receiver")
baseline_ls = OFDMSystem(False)
ber_plots.simulate(baseline_ls,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Baseline: LS Estimation",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);
baseline_pcsi = OFDMSystem(True)
ber_plots.simulate(baseline_pcsi,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Baseline: Perfect CSI",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);
```


```python
WARNING:tensorflow:From /usr/local/lib/python3.8/dist-packages/tensorflow/python/util/dispatch.py:1082: calling gather (from tensorflow.python.ops.array_ops) with validate_indices is deprecated and will be removed in a future version.
Instructions for updating:
The `validate_indices` argument has no effect. Indices are always validated on CPU and never validated on GPU.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -3.0 | 3.6894e-01 | 1.0000e+00 |       43069 |      116736 |          128 |         128 |         7.7 |reached target block errors
   -2.579 | 3.5806e-01 | 1.0000e+00 |       41799 |      116736 |          128 |         128 |         0.2 |reached target block errors
   -2.158 | 3.4527e-01 | 1.0000e+00 |       40305 |      116736 |          128 |         128 |         0.2 |reached target block errors
   -1.737 | 3.3213e-01 | 1.0000e+00 |       38771 |      116736 |          128 |         128 |         0.2 |reached target block errors
   -1.316 | 3.2260e-01 | 1.0000e+00 |       37659 |      116736 |          128 |         128 |         0.2 |reached target block errors
   -0.895 | 3.0787e-01 | 1.0000e+00 |       35940 |      116736 |          128 |         128 |         0.2 |reached target block errors
   -0.474 | 2.9344e-01 | 1.0000e+00 |       34255 |      116736 |          128 |         128 |         0.2 |reached target block errors
   -0.053 | 2.7841e-01 | 1.0000e+00 |       32501 |      116736 |          128 |         128 |         0.2 |reached target block errors
    0.368 | 2.6109e-01 | 1.0000e+00 |       30479 |      116736 |          128 |         128 |         0.2 |reached target block errors
    0.789 | 2.4077e-01 | 1.0000e+00 |       28107 |      116736 |          128 |         128 |         0.2 |reached target block errors
    1.211 | 2.2460e-01 | 1.0000e+00 |       26219 |      116736 |          128 |         128 |         0.2 |reached target block errors
    1.632 | 1.9116e-01 | 1.0000e+00 |       22315 |      116736 |          128 |         128 |         0.2 |reached target block errors
    2.053 | 1.5909e-01 | 1.0000e+00 |       18572 |      116736 |          128 |         128 |         0.2 |reached target block errors
    2.474 | 9.3930e-02 | 8.6719e-01 |       10965 |      116736 |          111 |         128 |         0.2 |reached target block errors
    2.895 | 2.1987e-02 | 3.8281e-01 |        7700 |      350208 |          147 |         384 |         0.5 |reached target block errors
    3.316 | 1.5316e-03 | 4.2352e-02 |        3397 |     2217984 |          103 |        2432 |         2.9 |reached target block errors
    3.737 | 1.1607e-04 | 1.6406e-03 |        1355 |    11673600 |           21 |       12800 |        15.5 |reached max iter
    4.158 | 0.0000e+00 | 0.0000e+00 |           0 |    11673600 |            0 |       12800 |        15.5 |reached max iter
Simulation stopped as no error occurred @ EbNo = 4.2 dB.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -3.0 | 2.1695e-01 | 1.0000e+00 |       25326 |      116736 |          128 |         128 |         3.4 |reached target block errors
   -2.579 | 1.9826e-01 | 1.0000e+00 |       23144 |      116736 |          128 |         128 |         0.2 |reached target block errors
   -2.158 | 1.7926e-01 | 1.0000e+00 |       20926 |      116736 |          128 |         128 |         0.2 |reached target block errors
   -1.737 | 1.3810e-01 | 1.0000e+00 |       16121 |      116736 |          128 |         128 |         0.2 |reached target block errors
   -1.316 | 7.1966e-02 | 8.7500e-01 |        8401 |      116736 |          112 |         128 |         0.2 |reached target block errors
   -0.895 | 1.6267e-02 | 3.6719e-01 |        5697 |      350208 |          141 |         384 |         0.5 |reached target block errors
   -0.474 | 6.2963e-04 | 2.8181e-02 |        2058 |     3268608 |          101 |        3584 |         4.3 |reached target block errors
   -0.053 | 4.5916e-05 | 8.5938e-04 |         536 |    11673600 |           11 |       12800 |        15.4 |reached max iter
    0.368 | 2.9126e-05 | 1.5625e-04 |         340 |    11673600 |            2 |       12800 |        15.4 |reached max iter
    0.789 | 1.5676e-05 | 7.8125e-05 |         183 |    11673600 |            1 |       12800 |        15.4 |reached max iter
    1.211 | 0.0000e+00 | 0.0000e+00 |           0 |    11673600 |            0 |       12800 |        15.5 |reached max iter
Simulation stopped as no error occurred @ EbNo = 1.2 dB.

```

    
We then instantiate and evaluate the end-to-end system equipped with the neural receiver.

```python
[7]:
```

```python
# Instantiating the end-to-end model for evaluation
model_neuralrx = OFDMSystemNeuralReceiver(training=False)
# Run one inference to build the layers and loading the weights
model_neuralrx(tf.constant(1, tf.int32), tf.constant(10.0, tf.float32))
with open('weights-ofdm-neuralrx', 'rb') as f:
    weights = pickle.load(f)
    model_neuralrx.set_weights(weights)
```
```python
[8]:
```

```python
# Computing and plotting BER
ber_plots.simulate(model_neuralrx,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100,
                  legend="Neural Receiver",
                  soft_estimates=True,
                  max_mc_iter=100,
                  show_fig=True);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -3.0 | 2.2083e-01 | 1.0000e+00 |       25779 |      116736 |          128 |         128 |         0.3 |reached target block errors
   -2.579 | 2.0480e-01 | 1.0000e+00 |       23907 |      116736 |          128 |         128 |         0.2 |reached target block errors
   -2.158 | 1.8219e-01 | 1.0000e+00 |       21268 |      116736 |          128 |         128 |         0.2 |reached target block errors
   -1.737 | 1.4852e-01 | 1.0000e+00 |       17338 |      116736 |          128 |         128 |         0.2 |reached target block errors
   -1.316 | 9.0503e-02 | 9.4531e-01 |       10565 |      116736 |          121 |         128 |         0.2 |reached target block errors
   -0.895 | 2.2251e-02 | 4.4922e-01 |        5195 |      233472 |          115 |         256 |         0.3 |reached target block errors
   -0.474 | 1.7106e-03 | 6.4303e-02 |        2596 |     1517568 |          107 |        1664 |         2.2 |reached target block errors
   -0.053 | 1.4828e-04 | 3.2812e-03 |        1731 |    11673600 |           42 |       12800 |        16.6 |reached max iter
    0.368 | 6.3305e-05 | 6.2500e-04 |         739 |    11673600 |            8 |       12800 |        16.5 |reached max iter
    0.789 | 8.6520e-06 | 1.5625e-04 |         101 |    11673600 |            2 |       12800 |        16.5 |reached max iter
    1.211 | 4.2832e-07 | 7.8125e-05 |           5 |    11673600 |            1 |       12800 |        16.6 |reached max iter
    1.632 | 0.0000e+00 | 0.0000e+00 |           0 |    11673600 |            0 |       12800 |        16.5 |reached max iter
Simulation stopped as no error occurred @ EbNo = 1.6 dB.

```

<img alt="../_images/examples_Sionna_tutorial_part4_29_1.png" src="https://nvlabs.github.io/sionna/_images/examples_Sionna_tutorial_part4_29_1.png" />
## Conclusion<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part4.html#Conclusion" title="Permalink to this headline"></a>
    
We hope you are excited about Sionna - there is much more to be discovered:
 
- TensorBoard debugging available
- Scaling to multi-GPU simulation is simple
- See the <a class="reference external" href="https://nvlabs.github.io/sionna/tutorials.html">available tutorials</a> for more examples

    
And if something is still missing - the project is open-source: you can modify, add, and extend any component at any time.
    
To get started you can use the `pip` installer:

```python
[ ]:
```

```python
!pip install sionna
```

## References<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part4.html#References" title="Permalink to this headline"></a>
    
[1] <a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/9345504">M. Honkala, D. Korpi and J. M. J. Huttunen, “DeepRx: Fully Convolutional Deep Learning Receiver,” in IEEE Transactions on Wireless Communications, vol. 20, no. 6, pp. 3925-3940, June 2021, doi: 10.1109/TWC.2021.3054520</a>.
    
[2] <a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/9508784">F. Ait Aoudia and J. Hoydis, “End-to-end Learning for OFDM: From Neural Receivers to Pilotless Communication,” in IEEE Transactions on Wireless Communications, doi: 10.1109/TWC.2021.3101364</a>.
    
[3] <a class="reference external" href="https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html">Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, “Deep Residual Learning for Image Recognition”, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778</a>
<script type="application/vnd.jupyter.widget-state+json">
{"state": {}, "version_major": 2, "version_minor": 0}
</script>