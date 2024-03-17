# Neural Receiver for OFDM SIMO Systems<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Neural_Receiver.html#Neural-Receiver-for-OFDM-SIMO-Systems" title="Permalink to this headline"></a>
    
In this notebook, you will learn how to train a neural receiver that implements OFDM detection. The considered setup is shown in the figure below. As one can see, the neural receiver substitutes channel estimation, equalization, and demapping. It takes as input the post-DFT (discrete Fourier transform) received samples, which form the received resource grid, and computes log-likelihood ratios (LLRs) on the transmitted coded bits. These LLRs are then fed to the outer decoder to reconstruct the
transmitted information bits.
    
    
Two baselines are considered for benchmarking, which are shown in the figure above. Both baselines use linear minimum mean square error (LMMSE) equalization and demapping assuming additive white Gaussian noise (AWGN). They differ by how channel estimation is performed:
 
- **Pefect CSI**: Perfect channel state information (CSI) knowledge is assumed.
- **LS estimation**: Uses the transmitted pilots to perform least squares (LS) estimation of the channel with nearest-neighbor interpolation.

    
All the considered end-to-end systems use an LDPC outer code from the 5G NR specification, QPSK modulation, and a 3GPP CDL channel model simulated in the frequency domain.
# Table of Content
## GPU Configuration and Imports
## End-to-end System as a Keras Model
## Evaluation of the Baselines
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Neural_Receiver.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

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
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
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
[2]:
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu
from sionna.channel.tr38901 import Antenna, AntennaArray, CDL
from sionna.channel import OFDMChannel
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber
from sionna.utils import sim_ber
```

## End-to-end System as a Keras Model<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Neural_Receiver.html#End-to-end-System-as-a-Keras-Model" title="Permalink to this headline"></a>
    
The following Keras <em>Model</em> implements the three considered end-to-end systems (perfect CSI baseline, LS estimation baseline, and neural receiver).
    
When instantiating the Keras model, the parameter `system` is used to specify the system to setup, and the parameter `training` is used to specified if the system is instantiated to be trained or to be evaluated. The `training` parameter is only relevant when the neural receiver is used.
    
At each call of this model:
 
- A batch of codewords is randomly sampled, modulated, and mapped to resource grids to form the channel inputs
- A batch of channel realizations is randomly sampled and applied to the channel inputs
- The receiver is executed on the post-DFT received samples to compute LLRs on the coded bits. Which receiver is executed (baseline with perfect CSI knowledge, baseline with LS estimation, or neural receiver) depends on the specified `system` parameter.
- If not training, the outer decoder is applied to reconstruct the information bits
- If training, the BMD rate is estimated over the batch from the LLRs and the transmitted bits
```python
[12]:
```

```python
class E2ESystem(Model):
    r"""
    Keras model that implements the end-to-end systems.
    As the three considered end-to-end systems (perfect CSI baseline, LS estimation baseline, and neural receiver) share most of
    the link components (transmitter, channel model, outer code...), they are implemented using the same Keras model.
    When instantiating the Keras model, the parameter ``system`` is used to specify the system to setup,
    and the parameter ``training`` is used to specified if the system is instantiated to be trained or to be evaluated.
    The ``training`` parameter is only relevant when the neural
    At each call of this model:
    * A batch of codewords is randomly sampled, modulated, and mapped to resource grids to form the channel inputs
    * A batch of channel realizations is randomly sampled and applied to the channel inputs
    * The receiver is executed on the post-DFT received samples to compute LLRs on the coded bits.
      Which receiver is executed (baseline with perfect CSI knowledge, baseline with LS estimation, or neural receiver) depends
      on the specified ``system`` parameter.
    * If not training, the outer decoder is applied to reconstruct the information bits
    * If training, the BMD rate is estimated over the batch from the LLRs and the transmitted bits
    Parameters
    -----------
    system : str
        Specify the receiver to use. Should be one of 'baseline-perfect-csi', 'baseline-ls-estimation' or 'neural-receiver'
    training : bool
        Set to `True` if the system is instantiated to be trained. Set to `False` otherwise. Defaults to `False`.
        If the system is instantiated to be trained, the outer encoder and decoder are not instantiated as they are not required for training.
        This significantly reduces the computational complexity of training.
        If training, the bit-metric decoding (BMD) rate is computed from the transmitted bits and the LLRs. The BMD rate is known to be
        an achievable information rate for BICM systems, and therefore training of the neural receiver aims at maximizing this rate.
    Input
    ------
    batch_size : int
        Batch size
    no : scalar or [batch_size], tf.float
        Noise variance.
        At training, a different noise variance should be sampled for each batch example.
    Output
    -------
    If ``training`` is set to `True`, then the output is a single scalar, which is an estimation of the BMD rate computed over the batch. It
    should be used as objective for training.
    If ``training`` is set to `False`, the transmitted information bits and their reconstruction on the receiver side are returned to
    compute the block/bit error rate.
    """
    def __init__(self, system, training=False):
        super().__init__()
        self._system = system
        self._training = training
        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._encoder = LDPC5GEncoder(k, n)
        self._mapper = Mapper("qam", num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(resource_grid)
        ######################################
        ## Channel
        # A 3GPP CDL channel model is used
        cdl = CDL(cdl_model, delay_spread, carrier_frequency,
                  ut_antenna, bs_array, "uplink", min_speed=speed)
        self._channel = OFDMChannel(cdl, resource_grid, normalize_channel=True, return_channel=True)
        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of `system`
        if "baseline" in system:
            if system == 'baseline-perfect-csi': # Perfect CSI
                self._removed_null_subc = RemoveNulledSubcarriers(resource_grid)
            elif system == 'baseline-ls-estimation': # LS estimation
                self._ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")
            # Components required by both baselines
            self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager, )
            self._demapper = Demapper("app", "qam", num_bits_per_symbol)
        elif system == "neural-receiver": # Neural receiver
            self._neural_receiver = NeuralReceiver()
            self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
    @tf.function
    def call(self, batch_size, ebno_db):
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        # Outer coding is only performed if not training
        if self._training:
            c = self._binary_source([batch_size, 1, 1, n])
        else:
            b = self._binary_source([batch_size, 1, 1, k])
            c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)
        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y,h = self._channel([x_rg, no_])
        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of ``system``
        if "baseline" in self._system:
            if self._system == 'baseline-perfect-csi':
                h_hat = self._removed_null_subc(h) # Extract non-null subcarriers
                err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
            elif self._system == 'baseline-ls-estimation':
                h_hat, err_var = self._ls_est([y, no]) # LS channel estimation with nearest-neighbor
            x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no]) # LMMSE equalization
            no_eff_= expand_to_rank(no_eff, tf.rank(x_hat))
            llr = self._demapper([x_hat, no_eff_]) # Demapping
        elif self._system == "neural-receiver":
            # The neural receover computes LLRs from the frequency domain received symbols and N0
            y = tf.squeeze(y, axis=1)
            llr = self._neural_receiver([y, no])
            llr = insert_dims(llr, 2, 1) # Reshape the input to fit what the resource grid demapper is expected
            llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
            llr = tf.reshape(llr, [batch_size, 1, 1, n]) # Reshape the LLRs to fit what the outer decoder is expected
        # Outer coding is not needed if the information rate is returned
        if self._training:
            # Compute and return BMD rate (in bit), which is known to be an achievable
            # information rate for BICM systems.
            # Training aims at maximizing the BMD rate
            bce = tf.nn.sigmoid_cross_entropy_with_logits(c, llr)
            bce = tf.reduce_mean(bce)
            rate = tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)
            return rate
        else:
            # Outer decoding
            b_hat = self._decoder(llr)
            return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation
```

## Evaluation of the Baselines<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Neural_Receiver.html#Evaluation-of-the-Baselines" title="Permalink to this headline"></a>
    
We evaluate the BERs achieved by the baselines in the next cell.
    
**Note:** Evaluation of the two systems can take a while. Therefore, we provide pre-computed results at the end of this notebook.

```python
[13]:
```

```python
# Range of SNRs over which the systems are evaluated
ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                     ebno_db_max, # Max SNR for evaluation
                     0.5) # Step
```
```python
[14]:
```

```python
# Dictionnary storing the evaluation results
BLER = {}
model = E2ESystem('baseline-perfect-csi')
_,bler = sim_ber(model, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=100)
BLER['baseline-perfect-csi'] = bler.numpy()
model = E2ESystem('baseline-ls-estimation')
_,bler = sim_ber(model, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=100)
BLER['baseline-ls-estimation'] = bler.numpy()
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -5.0 | 2.5345e-01 | 1.0000e+00 |       45158 |      178176 |          128 |         128 |         7.3 |reached target block errors
     -4.5 | 2.3644e-01 | 1.0000e+00 |       42128 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -4.0 | 2.1466e-01 | 1.0000e+00 |       38248 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -3.5 | 1.9506e-01 | 1.0000e+00 |       34755 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -3.0 | 1.6194e-01 | 1.0000e+00 |       28853 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -2.5 | 1.0628e-01 | 9.9219e-01 |       18937 |      178176 |          127 |         128 |         0.1 |reached target block errors
     -2.0 | 1.8395e-02 | 5.6250e-01 |        6555 |      356352 |          144 |         256 |         0.2 |reached target block errors
     -1.5 | 6.6440e-04 | 2.7478e-02 |        3433 |     5167104 |          102 |        3712 |         3.2 |reached target block errors
     -1.0 | 8.7161e-05 | 1.4844e-03 |        1553 |    17817600 |           19 |       12800 |        10.9 |reached max iter
     -0.5 | 2.8904e-05 | 7.8125e-04 |         515 |    17817600 |           10 |       12800 |        10.9 |reached max iter
      0.0 | 1.2347e-05 | 1.5625e-04 |         220 |    17817600 |            2 |       12800 |        10.9 |reached max iter
      0.5 | 1.1337e-05 | 7.8125e-05 |         202 |    17817600 |            1 |       12800 |        10.8 |reached max iter
      1.0 | 8.0819e-06 | 7.8125e-05 |         144 |    17817600 |            1 |       12800 |        10.9 |reached max iter
      1.5 | 1.6837e-07 | 7.8125e-05 |           3 |    17817600 |            1 |       12800 |        10.9 |reached max iter
      2.0 | 0.0000e+00 | 0.0000e+00 |           0 |    17817600 |            0 |       12800 |        10.9 |reached max iter
Simulation stopped as no error occurred @ EbNo = 2.0 dB.
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -5.0 | 3.9096e-01 | 1.0000e+00 |       69659 |      178176 |          128 |         128 |         2.8 |reached target block errors
     -4.5 | 3.8028e-01 | 1.0000e+00 |       67756 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -4.0 | 3.6582e-01 | 1.0000e+00 |       65180 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -3.5 | 3.5540e-01 | 1.0000e+00 |       63324 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -3.0 | 3.4142e-01 | 1.0000e+00 |       60833 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -2.5 | 3.2873e-01 | 1.0000e+00 |       58572 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -2.0 | 3.1137e-01 | 1.0000e+00 |       55478 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -1.5 | 2.9676e-01 | 1.0000e+00 |       52875 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -1.0 | 2.7707e-01 | 1.0000e+00 |       49368 |      178176 |          128 |         128 |         0.1 |reached target block errors
     -0.5 | 2.5655e-01 | 1.0000e+00 |       45711 |      178176 |          128 |         128 |         0.1 |reached target block errors
      0.0 | 2.3697e-01 | 1.0000e+00 |       42223 |      178176 |          128 |         128 |         0.1 |reached target block errors
      0.5 | 2.0973e-01 | 1.0000e+00 |       37369 |      178176 |          128 |         128 |         0.1 |reached target block errors
      1.0 | 1.6844e-01 | 1.0000e+00 |       30012 |      178176 |          128 |         128 |         0.1 |reached target block errors
      1.5 | 8.5578e-02 | 9.2969e-01 |       15248 |      178176 |          119 |         128 |         0.1 |reached target block errors
      2.0 | 1.0147e-02 | 2.5195e-01 |        7232 |      712704 |          129 |         512 |         0.4 |reached target block errors
      2.5 | 7.8271e-04 | 1.2401e-02 |        8786 |    11225088 |          100 |        8064 |         6.9 |reached target block errors
      3.0 | 2.1866e-04 | 2.1094e-03 |        3896 |    17817600 |           27 |       12800 |        11.0 |reached max iter
      3.5 | 9.0528e-05 | 7.0312e-04 |        1613 |    17817600 |            9 |       12800 |        10.9 |reached max iter
      4.0 | 2.9634e-05 | 2.3437e-04 |         528 |    17817600 |            3 |       12800 |        11.0 |reached max iter
      4.5 | 1.9868e-05 | 1.5625e-04 |         354 |    17817600 |            2 |       12800 |        10.9 |reached max iter
      5.0 | 3.8445e-05 | 2.3437e-04 |         685 |    17817600 |            3 |       12800 |        10.9 |reached max iter
      5.5 | 0.0000e+00 | 0.0000e+00 |           0 |    17817600 |            0 |       12800 |        10.9 |reached max iter
Simulation stopped as no error occurred @ EbNo = 5.5 dB.

```
