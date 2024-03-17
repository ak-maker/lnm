# Discover Sionna<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Discover-Sionna" title="Permalink to this headline"></a>
    
This example notebook will guide you through the basic principles and illustrates the key features of <a class="reference external" href="https://nvlabs.github.io/sionna">Sionna</a>. With only a few commands, you can simulate the PHY-layer link-level performance for many 5G-compliant components, including easy visualization of the results.

# Table of Content
## Setting up the End-to-end Model<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Setting-up-the-End-to-end-Model" title="Permalink to this headline"></a>
## Run some Throughput Tests (Graph Mode)<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Run-some-Throughput-Tests-(Graph-Mode)" title="Permalink to this headline"></a>
  
  

## Setting up the End-to-end Model<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Setting-up-the-End-to-end-Model" title="Permalink to this headline"></a>
    
We now define a <em>Keras model</em> that is more convenient for training and Monte-Carlo simulations.
    
We simulate the transmission over a time-varying multi-path channel (the <em>TDL-A</em> model from 3GPP TR38.901). For this, OFDM and a <em>conventional</em> bit-interleaved coded modulation (BICM) scheme with higher order modulation is used. The information bits are protected by a 5G-compliant LDPC code.
    
<em>Remark</em>: Due to the large number of parameters, we define them as dictionary.

```python
[14]:
```

```python
class e2e_model(tf.keras.Model): # inherits from keras.model
    """Example model for end-to-end link-level simulations.
    Parameters
    ----------
    params: dict
        A dictionary defining the system parameters.
    Input
    -----
    batch_size: int or tf.int
        The batch_sizeused for the simulation.
    ebno_db: float or tf.float
        A float defining the simulation SNR.
    Output
    ------
    (b, b_hat):
        Tuple:
    b: tf.float32
        A tensor of shape `[batch_size, k]` containing the transmitted
        information bits.
    b_hat: tf.float32
        A tensor of shape `[batch_size, k]` containing the receiver's
        estimate of the transmitted information bits.
    """
    def __init__(self,
                params):
        super().__init__()

        # Define an OFDM Resource Grid Object
        self.rg = sionna.ofdm.ResourceGrid(
                            num_ofdm_symbols=params["num_ofdm_symbols"],
                            fft_size=params["fft_size"],
                            subcarrier_spacing=params["subcarrier_spacing"],
                            num_tx=1,
                            num_streams_per_tx=1,
                            cyclic_prefix_length=params["cyclic_prefix_length"],
                            pilot_pattern="kronecker",
                            pilot_ofdm_symbol_indices=params["pilot_ofdm_symbol_indices"])
        # Create a Stream Management object
        self.sm = sionna.mimo.StreamManagement(rx_tx_association=np.array([[1]]),
                                               num_streams_per_tx=1)
        self.coderate = params["coderate"]
        self.num_bits_per_symbol = params["num_bits_per_symbol"]
        self.n = int(self.rg.num_data_symbols*self.num_bits_per_symbol)
        self.k = int(self.n*coderate)
        # Init layers
        self.binary_source = sionna.utils.BinarySource()
        self.encoder = sionna.fec.ldpc.encoding.LDPC5GEncoder(self.k, self.n)
        self.interleaver = sionna.fec.interleaving.RowColumnInterleaver(
                                        row_depth=self.num_bits_per_symbol)
        self.deinterleaver = sionna.fec.interleaving.Deinterleaver(self.interleaver)
        self.mapper = sionna.mapping.Mapper("qam", self.num_bits_per_symbol)
        self.rg_mapper = sionna.ofdm.ResourceGridMapper(self.rg)
        self.tdl = sionna.channel.tr38901.TDL(model="A",
                           delay_spread=params["delay_spread"],
                           carrier_frequency=params["carrier_frequency"],
                           min_speed=params["min_speed"],
                           max_speed=params["max_speed"])
        self.channel = sionna.channel.OFDMChannel(self.tdl, self.rg, add_awgn=True, normalize_channel=True)
        self.ls_est = sionna.ofdm.LSChannelEstimator(self.rg, interpolation_type="nn")
        self.lmmse_equ = sionna.ofdm.LMMSEEqualizer(self.rg, self.sm)
        self.demapper = sionna.mapping.Demapper(params["demapping_method"],
                                                "qam", self.num_bits_per_symbol)
        self.decoder = sionna.fec.ldpc.decoding.LDPC5GDecoder(self.encoder,
                                                    hard_out=True,
                                                    cn_type=params["cn_type"],
                                                    num_iter=params["bp_iter"])
        print("Number of pilots: {}".format(self.rg.num_pilot_symbols))
        print("Number of data symbols: {}".format(self.rg.num_data_symbols))
        print("Number of resource elements: {}".format(
                                    self.rg.num_resource_elements))
        print("Pilot overhead: {:.2f}%".format(
                                    self.rg.num_pilot_symbols /
                                    self.rg.num_resource_elements*100))
        print("Cyclic prefix overhead: {:.2f}%".format(
                                    params["cyclic_prefix_length"] /
                                    (params["cyclic_prefix_length"]
                                    +params["fft_size"])*100))
        print("Each frame contains {} information bits".format(self.k))
    def call(self, batch_size, ebno_db):
        # Generate a batch of random bit vectors
        # We need two dummy dimension representing the number of
        # transmitters and streams per transmitter, respectively.
        b = self.binary_source([batch_size, 1, 1, self.k])
        # Encode the bits using the all-zero dummy encoder
        c = self.encoder(b)
        # Interleave the bits before mapping (BICM)
        c_int = self.interleaver(c)
        # Map bits to constellation symbols
        s = self.mapper(c_int)
        # Map symbols onto OFDM ressource grid
        x_rg = self.rg_mapper(s)
        # Transmit over noisy multi-path channel
        no = sionna.utils.ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate, self.rg)
        y = self.channel([x_rg, no])
        # LS Channel estimation with nearest pilot interpolation
        h_hat, err_var = self.ls_est ([y, no])
        # LMMSE Equalization
        x_hat, no_eff = self.lmmse_equ([y, h_hat, err_var, no])
        # Demap to LLRs
        llr = self.demapper([x_hat, no_eff])
        # Deinterleave before decoding
        llr_int = self.deinterleaver(llr)
        # Decode
        b_hat = self.decoder(llr_int)
        # number of simulated bits
        nb_bits = batch_size*self.k
        # transmitted bits and the receiver's estimate after decoding
        return b, b_hat
```

    
Let us define the system parameters for our simulation as dictionary:

```python
[15]:
```

```python
sys_params = {
    # Channel
    "carrier_frequency" : 3.5e9,
    "delay_spread" : 100e-9,
    "min_speed" : 3,
    "max_speed" : 3,
    "tdl_model" : "A",
    # OFDM
    "fft_size" : 256,
    "subcarrier_spacing" : 30e3,
    "num_ofdm_symbols" : 14,
    "cyclic_prefix_length" : 16,
    "pilot_ofdm_symbol_indices" : [2, 11],
    # Code & Modulation
    "coderate" : 0.5,
    "num_bits_per_symbol" : 4,
    "demapping_method" : "app",
    "cn_type" : "boxplus",
    "bp_iter" : 20
}
```

    
…and initialize the model:

```python
[16]:
```

```python
model = e2e_model(sys_params)
```


```python
Number of pilots: 512
Number of data symbols: 3072
Number of resource elements: 3584
Pilot overhead: 14.29%
Cyclic prefix overhead: 5.88%
Each frame contains 6144 information bits
```

    
As before, we can simply <em>call</em> the model to simulate the BER for the given simulation parameters.

```python
[17]:
```

```python
#simulation parameters
ebno_db = 10
batch_size = 200
# and call the model
b, b_hat = model(batch_size, ebno_db)
ber = sionna.utils.metrics.compute_ber(b, b_hat)
nb_bits = np.size(b.numpy())
print("BER: {:.4} at Eb/No of {} dB and {} simulated bits".format(ber.numpy(), ebno_db, nb_bits))
```


```python
BER: 0.006234 at Eb/No of 10 dB and 1228800 simulated bits
```
## Run some Throughput Tests (Graph Mode)<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Run-some-Throughput-Tests-(Graph-Mode)" title="Permalink to this headline"></a>
    
Sionna is not just an easy-to-use library, but also incredibly fast. Let us measure the throughput of the model defined above.
    
We compare <em>eager</em> and <em>graph</em> execution modes (see <a class="reference external" href="https://www.tensorflow.org/guide/intro_to_graphs">Tensorflow Doc</a> for details), as well as <em>eager with XLA</em> (see <a class="reference external" href="https://www.tensorflow.org/xla#enable_xla_for_tensorflow_models">https://www.tensorflow.org/xla#enable_xla_for_tensorflow_models</a>). Note that we need to activate the <a class="reference external" href="https://nvlabs.github.io/sionna/api/config.html#sionna.Config.xla_compat">sionna.config.xla_compat</a> feature for XLA to work.
    
**Tip**: change the `batch_size` to see how the batch parallelism enhances the throughput. Depending on your machine, the `batch_size` may be too large.

```python
[18]:
```

```python
import time # this block requires the timeit library
batch_size = 200
ebno_db = 5 # evalaute SNR point
repetitions = 4 # throughput is averaged over multiple runs
def get_throughput(batch_size, ebno_db, model, repetitions=1):
    """ Simulate throughput in bit/s per ebno_db point.
    The results are average over `repetition` trials.
    Input
    -----
    batch_size: int or tf.int32
        Batch-size for evaluation.
    ebno_db: float or tf.float32
        A tensor containing the SNR points be evaluated
    model:
        Function or model that yields the transmitted bits `u` and the
        receiver's estimate `u_hat` for a given ``batch_size`` and
        ``ebno_db``.
    repetitions: int
        An integer defining how many trails of the throughput
        simulation are averaged.
    """

    # call model once to be sure it is compile properly
    # otherwise time to build graph is measured as well.
    u, u_hat = model(tf.constant(batch_size, tf.int32),
                     tf.constant(ebno_db, tf.float32))
    t_start = time.perf_counter()
    # average over multiple runs
    for _ in range(repetitions):
        u, u_hat = model(tf.constant(batch_size, tf.int32),
                            tf.constant(ebno_db, tf. float32))
    t_stop = time.perf_counter()
    # throughput in bit/s
    throughput = np.size(u.numpy())*repetitions / (t_stop - t_start)
    return throughput
# eager mode - just call the model
def run_eager(batch_size, ebno_db):
    return model(batch_size, ebno_db)
time_eager = get_throughput(batch_size, ebno_db, run_eager, repetitions=4)
# the decorator "@tf.function" enables the graph mode
@tf.function
def run_graph(batch_size, ebno_db):
    return model(batch_size, ebno_db)
time_graph = get_throughput(batch_size, ebno_db, run_graph, repetitions=4)
# the decorator "@tf.function(jit_compile=True)" enables the graph mode with XLA
# we need to activate the sionna.config.xla_compat feature for this to work
sionna.config.xla_compat=True
@tf.function(jit_compile=True)
def run_graph_xla(batch_size, ebno_db):
    return model(batch_size, ebno_db)
time_graph_xla = get_throughput(batch_size, ebno_db, run_graph_xla, repetitions=4)
# we deactivate the sionna.config.xla_compat so that the cell can be run mutiple times
sionna.config.xla_compat=False
print(f"Throughput in eager execution: {time_eager/1e6:.2f} Mb/s")
print(f"Throughput in graph execution: {time_graph/1e6:.2f} Mb/s")
print(f"Throughput in graph execution with XLA: {time_graph_xla/1e6:.2f} Mb/s")
```


```python
WARNING:tensorflow:From /usr/local/lib/python3.8/dist-packages/tensorflow/python/util/dispatch.py:1082: calling gather (from tensorflow.python.ops.array_ops) with validate_indices is deprecated and will be removed in a future version.
Instructions for updating:
The `validate_indices` argument has no effect. Indices are always validated on CPU and never validated on GPU.
Throughput in eager execution: 0.51 Mb/s
Throughput in graph execution: 4.10 Mb/s
Throughput in graph execution with XLA: 43.72 Mb/s
```

    
Obviously, <em>graph</em> execution (with XLA) yields much higher throughputs (at least if a fast GPU is available). Thus, for exhaustive training and Monte-Carlo simulations the <em>graph</em> mode (with XLA and GPU acceleration) is the preferred choice.

