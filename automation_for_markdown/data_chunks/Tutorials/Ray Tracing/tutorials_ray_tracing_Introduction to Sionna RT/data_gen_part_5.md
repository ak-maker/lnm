# Introduction to Sionna RT<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#Introduction-to-Sionna-RT" title="Permalink to this headline"></a>
    
In this notebook, you will
 
- Discover the basic functionalities of Sionna’s <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html">ray tracing (RT) module</a>
- Learn how to compute coverage maps
- Use ray-traced channels for link-level simulations instead of stochastic channel models
# Table of Content
## GPU Configuration and Imports
## Site-specifc Link-Level Simulations
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

```python
[1]:
```

```python
import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Colab does currently not support the latest version of ipython.
# Thus, the preview does not work in Colab. However, whenever possible we
# strongly recommend to use the scene preview mode.
try: # detect if the notebook runs in Colab
    import google.colab
    colab_compat = True # deactivate preview
except:
    colab_compat = False
resolution = [480,320] # increase for higher quality of renderings
# Allows to exit cell execution in Jupyter
class ExitCell(Exception):
    def _render_traceback_(self):
        pass
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
tf.random.set_seed(1) # Set global random seed for reproducibility

```
```python
[2]:
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time
# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement

```

## Site-specifc Link-Level Simulations<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#Site-specifc-Link-Level-Simulations" title="Permalink to this headline"></a>
    
We will now use Sionna RT for site-specific link-level simulations. For this, we evaluate the BER performance for a MU-MIMO 5G NR system in the uplink direction based on ray traced CIRs for random user positions.
    
We use the 5G NR PUSCH transmitter and receiver from the <a class="reference external" href="https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html">5G NR PUSCH Tutorial</a> notebook. Note that also the systems from the <a class="reference external" href="https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html">MIMO OFDM Transmissions over the CDL Channel Model</a> or the <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Neural_Receiver.html">Neural Receiver for OFDM SIMO Systems</a> tutorials could be used instead.
    
There are different ways to implement uplink scenarios in Sionna RT. In this example, we configure the basestation as transmitter and the user equipments (UEs) as receivers which simplifies the ray tracing. Due to channel reciprocity, one can <em>reverse</em> the direction of the ray traced channels afterwards. For the ray tracer itself, the direction (uplink/downlink) does not change the simulated paths.
    
<em>Note</em>: Running the cells below can take several hours of compute time.

```python
[30]:
```

```python
# System parameters
subcarrier_spacing = 30e3
num_time_steps = 14 # Total number of ofdm symbols per slot
num_tx = 4 # Number of users
num_rx = 1 # Only one receiver considered
num_tx_ant = 4 # Each user has 4 antennas
num_rx_ant = 16 # The receiver is equipped with 16 antennas
# batch_size for CIR generation
batch_size_cir = 1000

```

    
Let us add a new transmitter that acts as basestation. We will later use channel reciprocity to simulate the uplink direction.

```python
[31]:
```

```python
# Remove old tx from scene
scene.remove("tx")
scene.synthetic_array = True # Emulate multiple antennas to reduce ray tracing complexity
# Transmitter (=basestation) has an antenna pattern from 3GPP 38.901
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=int(num_rx_ant/2), # We want to transmitter to be equiped with the 16 rx antennas
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="cross")
# Create transmitter
tx = Transmitter(name="tx",
                 position=[8.5,21,27],
                 look_at=[45,90,1.5]) # optional, defines view direction
scene.add(tx)

```

    
We now need to update the coverage map for the new transmitter.

```python
[32]:
```

```python
max_depth = 5 # Defines max number of ray interactions
# Update coverage_map
cm = scene.coverage_map(max_depth=max_depth,
                        diffraction=True,
                        cm_cell_size=(1., 1.),
                        combining_vec=None,
                        precoding_vec=None,
                        num_samples=int(10e6))

```

    
The function `sample_positions` allows sampling of random user positions from a coverage map. It ensures that only positions are sampled that have a path gain of at least `min_gain_dB` dB and at most `max_gain_dB` dB, i.e., ignores positions without connection to the transmitter. Further, one can set the distance `min_dist` and `max_dist` to sample only points with a certain distance away from the transmitter.

```python
[33]:
```

```python
min_gain_db = -130 # in dB; ignore any position with less than -130 dB path gain
max_gain_db = 0 # in dB; ignore strong paths
# sample points in a 5-400m radius around the receiver
min_dist = 5 # in m
max_dist = 400 # in m
#sample batch_size random user positions from coverage map
ue_pos = cm.sample_positions(batch_size=batch_size_cir,
                             min_gain_db=min_gain_db,
                             max_gain_db=max_gain_db,
                             min_dist=min_dist,
                             max_dist=max_dist)

```

    
We now add new receivers (=UEs) at random position.
    
<em>Remark</em>: This is an example for 5G NR PUSCH (uplink direction), we will reverse the direction of the channel for later BER simulations.

```python
[34]:
```

```python
# Remove old receivers from scene
scene.remove("rx")
for i in range(batch_size_cir):
    scene.remove(f"rx-{i}")
# Configure antenna array for all receivers (=UEs)
scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=int(num_tx_ant/2), # Each receiver is equipped with 4 tx antennas (uplink)
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso", # UE orientation is random
                             polarization="cross")
# Create batch_size receivers
for i in range(batch_size_cir):
    rx = Receiver(name=f"rx-{i}",
                  position=ue_pos[i], # Random position sampled from coverage map
                  )
    scene.add(rx)
# And visualize the scene
if colab_compat:
    scene.render("birds_view", show_devices=True, resolution=resolution);
    raise ExitCell
scene.preview(show_devices=True, coverage_map=cm)

```


    
Each dot represents a random receiver position drawn from the random sampling function of the coverage map. This allows to efficiently sample batches of random channel realizations even in complex scenarios.
    
We can now simulate the CIRs for many different random positions.
    
<em>Remark</em>: Running the cells below can take some time depending on the requested number of CIRs.

```python
[35]:
```

```python
target_num_cirs = 5000 # Defines how many different CIRS are generated.
# Remark: some path are removed if no path was found for this position
max_depth = 5
min_gain_db = -130 # in dB / ignore any position with less than -130 dB path gain
max_gain_db = 0 # in dB / ignore any position with more than 0 dB path gain
# Sample points within a 10-400m radius around the transmitter
min_dist = 10 # in m
max_dist = 400 # in m
# Placeholder to gather channel impulse reponses
a = None
tau = None
# Each simulation returns batch_size_cir results
num_runs = int(np.ceil(target_num_cirs/batch_size_cir))
for idx in range(num_runs):
    print(f"Progress: {idx+1}/{num_runs}", end="\r")
    # Sample random user positions
    ue_pos = cm.sample_positions(
                        batch_size=batch_size_cir,
                        min_gain_db=min_gain_db,
                        max_gain_db=max_gain_db,
                        min_dist=min_dist,
                        max_dist=max_dist)
    # Update all receiver positions
    for idx in range(batch_size_cir):
        scene.receivers[f"rx-{idx}"].position = ue_pos[idx]
    # Simulate CIR
    paths = scene.compute_paths(
                    max_depth=max_depth,
                    diffraction=True,
                    num_samples=1e6) # shared between all tx in a scene
    # Transform paths into channel impulse responses
    paths.reverse_direction = True # Convert to uplink direction
    paths.apply_doppler(sampling_frequency=subcarrier_spacing,
                        num_time_steps=14,
                        tx_velocities=[0.,0.,0],
                        rx_velocities=[3.,3.,0])
    # We fix here the maximum number of paths to 75 which ensures
    # that we can simply concatenate different channel impulse reponses
    a_, tau_ = paths.cir(num_paths=75)
    del paths # Free memory
    if a is None:
        a = a_.numpy()
        tau = tau_.numpy()
    else:
        # Concatenate along the num_tx dimension
        a = np.concatenate([a, a_], axis=3)
        tau = np.concatenate([tau, tau_], axis=2)
del cm # Free memory
# Exchange the num_tx and batchsize dimensions
a = np.transpose(a, [3, 1, 2, 0, 4, 5, 6])
tau = np.transpose(tau, [2, 1, 0, 3])
# Remove CIRs that have no active link (i.e., a is all-zero)
p_link = np.sum(np.abs(a)**2, axis=(1,2,3,4,5,6))
a = a[p_link>0.,...]
tau = tau[p_link>0.,...]
print("Shape of a:", a.shape)
print("Shape of tau: ", tau.shape)

```


```python
Shape of a: (4858, 1, 16, 1, 4, 75, 14)
Shape of tau:  (4858, 1, 1, 75)
```

    
Note that transmitter and receiver have been reversed, i.e., the transmitter now denotes the UE (with 4 antennas each) and the receiver is the basestation (with 16 antennas).
    
<em>Remark</em>: We have removed all positions where the resulting CIR was zero, i.e., no path between transmitter and receiver was found. This comes from the fact that the `sample_position` function samples from the quantized coverage map and randomizes the position within the cell. It can happen that for this random position no connection between transmitter and receiver can be found.
    
Let us now initialize a `data_generator` that samples random UEs from the dataset and <em>yields</em> the previously simulated CIRs.

```python
[36]:
```

```python
class CIRGenerator:
    """Creates a generator from a given dataset of channel impulse responses.
    The generator samples ``num_tx`` different transmitters from the given path
    coefficients `a` and path delays `tau` and stacks the CIRs into a single tensor.
    Note that the generator internally samples ``num_tx`` random transmitters
    from the dataset. For this, the inputs ``a`` and ``tau`` must be given for
    a single transmitter (i.e., ``num_tx`` =1) which will then be stacked
    internally.
    Parameters
    ----------
    a : [batch size, num_rx, num_rx_ant, 1, num_tx_ant, num_paths, num_time_steps], complex
        Path coefficients per transmitter.
    tau : [batch size, num_rx, 1, num_paths], float
        Path delays [s] per transmitter.
    num_tx : int
        Number of transmitters
    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients
    tau : [batch size, num_rx, num_tx, num_paths], tf.float
        Path delays [s]
    """
    def __init__(self,
                 a,
                 tau,
                 num_tx):
        # Copy to tensorflow
        self._a = tf.constant(a, tf.complex64)
        self._tau = tf.constant(tau, tf.float32)
        self._dataset_size = self._a.shape[0]
        self._num_tx = num_tx
    def __call__(self):
        # Generator implements an infinite loop that yields new random samples
        while True:
            # Sample 4 random users and stack them together
            idx,_,_ = tf.random.uniform_candidate_sampler(
                            tf.expand_dims(tf.range(self._dataset_size, dtype=tf.int64), axis=0),
                            num_true=self._dataset_size,
                            num_sampled=self._num_tx,
                            unique=True,
                            range_max=self._dataset_size)
            a = tf.gather(self._a, idx)
            tau = tf.gather(self._tau, idx)
            # Transpose to remove batch dimension
            a = tf.transpose(a, (3,1,2,0,4,5,6))
            tau = tf.transpose(tau, (2,1,0,3))
            # And remove batch-dimension
            a = tf.squeeze(a, axis=0)
            tau = tf.squeeze(tau, axis=0)
            yield a, tau

```

    
We use Sionna’s built-in <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.CIRDataset">CIRDataset</a> to initialize a channel model that can be directly used in Sionna’s <a class="reference external" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.OFDMChannel">OFDMChannel</a> layer.

```python
[38]:
```

```python
batch_size = 20 # Must be the same for the BER simulations as CIRDataset returns fixed batch_size
# Init CIR generator
cir_generator = CIRGenerator(a,
                             tau,
                             num_tx)
# Initialises a channel model that can be directly used by OFDMChannel layer
channel_model = CIRDataset(cir_generator,
                           batch_size,
                           num_rx,
                           num_rx_ant,
                           num_tx,
                           num_tx_ant,
                           75,
                           num_time_steps)
# Delete to free memory
del a, tau

```
```python
[40]:
```

```python
# We need to enable sionna.config.xla_compat before we can use
# tf.function with jit_compile=True.
# See https://nvlabs.github.io/sionna/api/config.html#sionna.Config.xla_compat
sionna.config.xla_compat=False # not supported in CIRDataset
class Model(tf.keras.Model):
    """Simulate PUSCH transmissions over a 3GPP 38.901 model.
    This model runs BER simulations for a multi-user MIMO uplink channel
    compliant with the 5G NR PUSCH specifications.
    You can pick different scenarios, i.e., channel models, perfect or
    estimated CSI, as well as different MIMO detectors (LMMSE or KBest).
    Parameters
    ----------
    channel_model : :class:`~sionna.channel.ChannelModel` object
        An instance of a :class:`~sionna.channel.ChannelModel` object, such as
        :class:`~sionna.channel.RayleighBlockFading` or
        :class:`~sionna.channel.tr38901.UMi` or
        :class:`~sionna.channel.CIRDataset`.
    perfect_csi : bool
        Determines if perfect CSI is assumed or if the CSI is estimated
    detector : str, one of ["lmmse", "kbest"]
        MIMO detector to be used. Note that each detector has additional
        parameters that can be configured in the source code of the _init_ call.
    Input
    -----
    batch_size : int
        Number of simultaneously simulated slots
    ebno_db : float
        Signal-to-noise-ratio
    Output
    ------
    b : [batch_size, num_tx, tb_size], tf.float
        Transmitted information bits
    b_hat : [batch_size, num_tx, tb_size], tf.float
        Decoded information bits
    """
    def __init__(self,
                 channel_model,
                 perfect_csi, # bool
                 detector,    # "lmmse", "kbest"
                ):
        super().__init__()
        self._channel_model = channel_model
        self._perfect_csi = perfect_csi
        # System configuration
        self._num_prb = 16
        self._mcs_index = 14
        self._num_layers = 1
        self._mcs_table = 1
        self._domain = "freq"
        # Below parameters must equal the Path2CIR parameters
        self._num_tx_ant = 4
        self._num_tx = 4
        self._subcarrier_spacing = 30e3 # must be the same as used for Path2CIR
        # PUSCHConfig for the first transmitter
        pusch_config = PUSCHConfig()
        pusch_config.carrier.subcarrier_spacing = self._subcarrier_spacing/1000
        pusch_config.carrier.n_size_grid = self._num_prb
        pusch_config.num_antenna_ports = self._num_tx_ant
        pusch_config.num_layers = self._num_layers
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 1
        pusch_config.dmrs.dmrs_port_set = list(range(self._num_layers))
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.length = 1
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.tb.mcs_index = self._mcs_index
        pusch_config.tb.mcs_table = self._mcs_table
        # Create PUSCHConfigs for the other transmitters by cloning of the first PUSCHConfig
        # and modifying the used DMRS ports.
        pusch_configs = [pusch_config]
        for i in range(1, self._num_tx):
            pc = pusch_config.clone()
            pc.dmrs.dmrs_port_set = list(range(i*self._num_layers, (i+1)*self._num_layers))
            pusch_configs.append(pc)
        # Create PUSCHTransmitter
        self._pusch_transmitter = PUSCHTransmitter(pusch_configs, output_domain=self._domain)
        # Create PUSCHReceiver
        rx_tx_association = np.ones([1, self._num_tx], bool)
        stream_management = StreamManagement(rx_tx_association,
                                             self._num_layers)
        assert detector in["lmmse", "kbest"], "Unsupported MIMO detector"
        if detector=="lmmse":
            detector = LinearDetector(equalizer="lmmse",
                                      output="bit",
                                      demapping_method="maxlog",
                                      resource_grid=self._pusch_transmitter.resource_grid,
                                      stream_management=stream_management,
                                      constellation_type="qam",
                                      num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol)
        elif detector=="kbest":
            detector = KBestDetector(output="bit",
                                     num_streams=self._num_tx*self._num_layers,
                                     k=64,
                                     resource_grid=self._pusch_transmitter.resource_grid,
                                     stream_management=stream_management,
                                     constellation_type="qam",
                                     num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol)
        if self._perfect_csi:
            self._pusch_receiver = PUSCHReceiver(self._pusch_transmitter,
                                                 mimo_detector=detector,
                                                 input_domain=self._domain,
                                                 channel_estimator="perfect")
        else:
            self._pusch_receiver = PUSCHReceiver(self._pusch_transmitter,
                                                 mimo_detector=detector,
                                                 input_domain=self._domain)

        # Configure the actual channel
        self._channel = OFDMChannel(
                            self._channel_model,
                            self._pusch_transmitter.resource_grid,
                            normalize_channel=True,
                            return_channel=True)
    # XLA currently not supported by the CIRDataset function
    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):
        x, b = self._pusch_transmitter(batch_size)
        no = ebnodb2no(ebno_db,
                       self._pusch_transmitter._num_bits_per_symbol,
                       pusch_transmitter._target_coderate,
                       pusch_transmitter.resource_grid)
        y, h = self._channel([x, no])
        if self._perfect_csi:
            b_hat = self._pusch_receiver([y, h, no])
        else:
            b_hat = self._pusch_receiver([y, no])
        return b, b_hat

```

    
We now initialize the end-to-end model with the `CIRDataset`.

```python
[41]:
```

```python
ebno_db = 10.
e2e_model = Model(channel_model,
                  perfect_csi=False, # bool
                  detector="lmmse")  # "lmmse", "kbest"
# We can draw samples from the end-2-end link-level simulations
b, b_hat = e2e_model(batch_size, ebno_db)

```

    
And let’s run the final evaluation for different system configurations.

```python
[42]:
```

```python
ebno_db = np.arange(-3, 18, 2) # sim SNR range
ber_plot = PlotBER(f"Site-Specific MU-MIMO 5G NR PUSCH")
for detector in ["lmmse", "kbest"]:
    for perf_csi in [True, False]:
        e2e_model = Model(channel_model,
                          perfect_csi=perf_csi,
                          detector=detector)
        # define legend
        csi = "Perf. CSI" if perf_csi else "Imperf. CSI"
        det = "K-Best" if detector=="kbest" else "LMMSE"
        l = det + " " + csi
        ber_plot.simulate(
                    e2e_model,
                    ebno_dbs=ebno_db, # SNR to simulate
                    legend=l, # legend string for plotting
                    max_mc_iter=500,
                    num_target_block_errors=2000,
                    batch_size=batch_size, # batch-size per Monte Carlo run
                    soft_estimates=False, # the model returns hard-estimates
                    early_stop=True,
                    show_fig=False,
                    add_bler=True,
                    forward_keyboard_interrupt=True);

```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -3.0 | 1.3426e-01 | 9.7500e-01 |     1394056 |    10383360 |         2028 |        2080 |        14.4 |reached target block errors
     -1.0 | 6.0569e-02 | 5.4375e-01 |     1112682 |    18370560 |         2001 |        3680 |         8.0 |reached target block errors
      1.0 | 3.0802e-02 | 2.6605e-01 |     1168609 |    37939200 |         2022 |        7600 |        16.3 |reached target block errors
      3.0 | 1.8333e-02 | 1.4970e-01 |     1222718 |    66693120 |         2000 |       13360 |        28.5 |reached target block errors
      5.0 | 9.8484e-03 | 8.5119e-02 |     1156321 |   117411840 |         2002 |       23520 |        50.2 |reached target block errors
      7.0 | 5.7053e-03 | 4.9250e-02 |     1139238 |   199680000 |         1970 |       40000 |        85.2 |reached max iter
      9.0 | 3.3868e-03 | 2.9075e-02 |      676280 |   199680000 |         1163 |       40000 |        85.3 |reached max iter
     11.0 | 1.6629e-03 | 1.4825e-02 |      332046 |   199680000 |          593 |       40000 |        85.4 |reached max iter
     13.0 | 1.0874e-03 | 9.6750e-03 |      217127 |   199680000 |          387 |       40000 |        85.2 |reached max iter
     15.0 | 5.7423e-04 | 4.8000e-03 |      114662 |   199680000 |          192 |       40000 |        85.1 |reached max iter
     17.0 | 4.3051e-04 | 3.1000e-03 |       85964 |   199680000 |          124 |       40000 |        85.2 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -3.0 | 2.2992e-01 | 1.0000e+00 |     2295491 |     9984000 |         2000 |        2000 |        14.7 |reached target block errors
     -1.0 | 1.8111e-01 | 9.9904e-01 |     1880529 |    10383360 |         2078 |        2080 |         4.5 |reached target block errors
      1.0 | 1.2673e-01 | 9.6346e-01 |     1315935 |    10383360 |         2004 |        2080 |         4.5 |reached target block errors
      3.0 | 6.2428e-02 | 5.0075e-01 |     1246554 |    19968000 |         2003 |        4000 |         8.7 |reached target block errors
      5.0 | 3.8155e-02 | 2.9390e-01 |     1310429 |    34344960 |         2022 |        6880 |        14.9 |reached target block errors
      7.0 | 2.3349e-02 | 1.7764e-01 |     1324110 |    56709120 |         2018 |       11360 |        24.6 |reached target block errors
      9.0 | 1.4020e-02 | 1.0570e-01 |     1326994 |    94648320 |         2004 |       18960 |        40.9 |reached target block errors
     11.0 | 8.1793e-03 | 6.2562e-02 |     1306587 |   159744000 |         2002 |       32000 |        69.1 |reached target block errors
     13.0 | 5.3070e-03 | 4.2200e-02 |     1059697 |   199680000 |         1688 |       40000 |        86.3 |reached max iter
     15.0 | 3.7709e-03 | 2.9325e-02 |      752975 |   199680000 |         1173 |       40000 |        86.5 |reached max iter
     17.0 | 2.5822e-03 | 2.0975e-02 |      515618 |   199680000 |          839 |       40000 |        86.4 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -3.0 | 1.4514e-01 | 9.9856e-01 |     1507056 |    10383360 |         2077 |        2080 |        32.2 |reached target block errors
     -1.0 | 7.4307e-02 | 8.8664e-01 |      860583 |    11581440 |         2057 |        2320 |        23.0 |reached target block errors
      1.0 | 3.3615e-02 | 3.0241e-01 |     1114221 |    33146880 |         2008 |        6640 |        65.7 |reached target block errors
      3.0 | 1.5859e-02 | 1.3012e-01 |     1222379 |    77076480 |         2009 |       15440 |       153.0 |reached target block errors
      5.0 | 6.7793e-03 | 6.0628e-02 |     1120861 |   165335040 |         2008 |       33120 |       328.7 |reached target block errors
      7.0 | 2.2633e-03 | 2.0250e-02 |      451928 |   199680000 |          810 |       40000 |       398.1 |reached max iter
      9.0 | 6.0958e-04 | 5.2750e-03 |      121721 |   199680000 |          211 |       40000 |       397.8 |reached max iter
     11.0 | 2.9266e-04 | 2.0500e-03 |       58438 |   199680000 |           82 |       40000 |       397.2 |reached max iter
     13.0 | 1.8539e-04 | 1.2000e-03 |       37019 |   199680000 |           48 |       40000 |       396.9 |reached max iter
     15.0 | 9.1491e-05 | 5.5000e-04 |       18269 |   199680000 |           22 |       40000 |       397.1 |reached max iter
     17.0 | 5.7602e-05 | 3.7500e-04 |       11502 |   199680000 |           15 |       40000 |       398.1 |reached max iter
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -3.0 | 2.4055e-01 | 1.0000e+00 |     2401681 |     9984000 |         2000 |        2000 |        31.1 |reached target block errors
     -1.0 | 1.9379e-01 | 1.0000e+00 |     1934840 |     9984000 |         2000 |        2000 |        20.3 |reached target block errors
      1.0 | 1.3773e-01 | 9.8846e-01 |     1430079 |    10383360 |         2056 |        2080 |        21.1 |reached target block errors
      3.0 | 7.0248e-02 | 6.6875e-01 |     1066068 |    15175680 |         2033 |        3040 |        30.9 |reached target block errors
      5.0 | 3.8589e-02 | 3.0196e-01 |     1279094 |    33146880 |         2005 |        6640 |        67.4 |reached target block errors
      7.0 | 1.9754e-02 | 1.5022e-01 |     1325339 |    67092480 |         2019 |       13440 |       136.5 |reached target block errors
      9.0 | 1.0694e-02 | 8.3696e-02 |     1276994 |   119408640 |         2002 |       23920 |       242.5 |reached target block errors
     11.0 | 5.5038e-03 | 4.2000e-02 |     1099001 |   199680000 |         1680 |       40000 |       405.3 |reached max iter
     13.0 | 3.0494e-03 | 2.3975e-02 |      608913 |   199680000 |          959 |       40000 |       405.1 |reached max iter
     15.0 | 2.3187e-03 | 1.8325e-02 |      462991 |   199680000 |          733 |       40000 |       405.3 |reached max iter
     17.0 | 1.8833e-03 | 1.4850e-02 |      376057 |   199680000 |          594 |       40000 |       404.9 |reached max iter
```
```python
[43]:
```

```python
# and show figure
ber_plot(show_bler=True, show_ber=False, ylim=[1e-4,1], xlim=[-3,17])

```


