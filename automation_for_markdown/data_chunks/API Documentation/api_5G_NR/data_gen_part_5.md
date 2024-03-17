# 5G NR<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#g-nr" title="Permalink to this headline"></a>
    
This module provides layers and functions to support simulations of
5G NR compliant features, in particular, the physical uplink shared channel (PUSCH). It provides implementations of a subset of the physical layer functionalities as described in the 3GPP specifications <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id1">[3GPP38211]</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38212" id="id2">[3GPP38212]</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id3">[3GPP38214]</a>.
    
The best way to discover this module’s components is by having a look at the <a class="reference external" href="../examples/5G_NR_PUSCH.html">5G NR PUSCH Tutorial</a>.
    
The following code snippet shows how you can make standard-compliant
simulations of the 5G NR PUSCH with a few lines of code:
```python
# Create a PUSCH configuration with default settings
pusch_config = PUSCHConfig()
# Instantiate a PUSCHTransmitter from the PUSCHConfig
pusch_transmitter = PUSCHTransmitter(pusch_config)
# Create a PUSCHReceiver using the PUSCHTransmitter
pusch_receiver = PUSCHReceiver(pusch_transmitter)
# AWGN channel
channel = AWGN()
# Simulate transmissions over the AWGN channel
batch_size = 16
no = 0.1 # Noise variance
x, b = pusch_transmitter(batch_size) # Generate transmit signal and info bits
y = channel([x, no]) # Simulate channel output
b_hat = pusch_receiver([x, no]) # Recover the info bits
# Compute BER
print("BER:", compute_ber(b, b_hat).numpy())
```

    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter" title="sionna.nr.PUSCHTransmitter">`PUSCHTransmitter`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHReceiver" title="sionna.nr.PUSCHReceiver">`PUSCHReceiver`</a> provide high-level abstractions of all required processing blocks. You can easily modify them according to your needs.

# Table of Content
## PUSCH<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#pusch" title="Permalink to this headline"></a>
### PUSCHPilotPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschpilotpattern" title="Permalink to this headline"></a>
### PUSCHPrecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschprecoder" title="Permalink to this headline"></a>
### PUSCHReceiver<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschreceiver" title="Permalink to this headline"></a>
### PUSCHTransmitter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschtransmitter" title="Permalink to this headline"></a>
## Transport Block<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#transport-block" title="Permalink to this headline"></a>
  
  

### PUSCHPilotPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschpilotpattern" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``PUSCHPilotPattern`(<em class="sig-param">`pusch_configs`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_pilot_pattern.html#PUSCHPilotPattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern" title="Permalink to this definition"></a>
    
Class defining a pilot pattern for NR PUSCH.
    
This class defines a <a class="reference internal" href="ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>
that is used to configure an OFDM <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
    
For every transmitter, a separte <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a>
needs to be provided from which the pilot pattern will be created.
Parameters
 
- **pusch_configs** (instance or list of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a>) – PUSCH Configurations according to which the pilot pattern
will created. One configuration is needed for each transmitter.
- **dtype** (<em>tf.Dtype</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`mask`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.mask" title="Permalink to this definition"></a>
    
Mask of the pilot pattern


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.normalize" title="Permalink to this definition"></a>
    
Returns or sets the flag indicating if the pilots
are normalized or not


<em class="property">`property` </em>`num_data_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.num_data_symbols" title="Permalink to this definition"></a>
    
Number of data symbols per transmit stream.


<em class="property">`property` </em>`num_effective_subcarriers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.num_effective_subcarriers" title="Permalink to this definition"></a>
    
Number of effectvie subcarriers


<em class="property">`property` </em>`num_ofdm_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.num_ofdm_symbols" title="Permalink to this definition"></a>
    
Number of OFDM symbols


<em class="property">`property` </em>`num_pilot_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.num_pilot_symbols" title="Permalink to this definition"></a>
    
Number of pilot symbols per transmit stream.


<em class="property">`property` </em>`num_streams_per_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.num_streams_per_tx" title="Permalink to this definition"></a>
    
Number of streams per transmitter


<em class="property">`property` </em>`num_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.num_tx" title="Permalink to this definition"></a>
    
Number of transmitters


<em class="property">`property` </em>`pilots`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.pilots" title="Permalink to this definition"></a>
    
Returns or sets the possibly normalized tensor of pilot symbols.
If pilots are normalized, the normalization will be applied
after new values for pilots have been set. If this is
not the desired behavior, turn normalization off.


`show`(<em class="sig-param">`tx_ind``=``None`</em>, <em class="sig-param">`stream_ind``=``None`</em>, <em class="sig-param">`show_pilot_ind``=``False`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.show" title="Permalink to this definition"></a>
    
Visualizes the pilot patterns for some transmitters and streams.
Input
 
- **tx_ind** (<em>list, int</em>) – Indicates the indices of transmitters to be included.
Defaults to <cite>None</cite>, i.e., all transmitters included.
- **stream_ind** (<em>list, int</em>) – Indicates the indices of streams to be included.
Defaults to <cite>None</cite>, i.e., all streams included.
- **show_pilot_ind** (<em>bool</em>) – Indicates if the indices of the pilot symbols should be shown.


Output
    
**list** (<em>matplotlib.figure.Figure</em>) – List of matplot figure objects showing each the pilot pattern
from a specific transmitter and stream.




<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPilotPattern.trainable" title="Permalink to this definition"></a>
    
Returns if pilots are trainable or not


### PUSCHPrecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschprecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``PUSCHPrecoder`(<em class="sig-param">`precoding_matrices`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_precoder.html#PUSCHPrecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHPrecoder" title="Permalink to this definition"></a>
    
Precodes a batch of modulated symbols mapped onto a resource grid
for PUSCH transmissions. Each transmitter is assumed to have its
own precoding matrix.
Parameters
 
- **precoding_matrices** (<em>list</em><em>, </em><em>[</em><em>num_tx</em><em>, </em><em>num_antenna_ports</em><em>, </em><em>num_layers</em><em>]</em><em> tf.complex</em>) – List of precoding matrices, one for each transmitter.
All precoding matrices must have the same shape.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>]</em>) – Dtype of inputs and outputs. Defaults to tf.complex64.


Input
    
<em>[batch_size, num_tx, num_layers, num_symbols_per_slot, num_subcarriers]</em> – Batch of resource grids to be precoded

Output
    
<em>[batch_size, num_tx, num_antenna_ports, num_symbols_per_slot, num_subcarriers]</em> – Batch of precoded resource grids



### PUSCHReceiver<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschreceiver" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``PUSCHReceiver`(<em class="sig-param">`pusch_transmitter`</em>, <em class="sig-param">`channel_estimator``=``None`</em>, <em class="sig-param">`mimo_detector``=``None`</em>, <em class="sig-param">`tb_decoder``=``None`</em>, <em class="sig-param">`return_tb_crc_status``=``False`</em>, <em class="sig-param">`stream_management``=``None`</em>, <em class="sig-param">`input_domain``=``'freq'`</em>, <em class="sig-param">`l_min``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_receiver.html#PUSCHReceiver">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHReceiver" title="Permalink to this definition"></a>
    
This layer implements a full receiver for batches of 5G NR PUSCH slots sent
by multiple transmitters. Inputs can be in the time or frequency domain.
Perfect channel state information can be optionally provided.
Different channel estimatiors, MIMO detectors, and transport decoders
can be configured.
    
The layer combines multiple processing blocks into a single layer
as shown in the following figure. Blocks with dashed lines are
optional and depend on the configuration.
<a class="reference internal image-reference" href="../_images/pusch_receiver_block_diagram.png"><img alt="../_images/pusch_receiver_block_diagram.png" src="https://nvlabs.github.io/sionna/_images/pusch_receiver_block_diagram.png" style="width: 258.3px; height: 420.59999999999997px;" /></a>
    
If the `input_domain` equals “time”, the inputs $\mathbf{y}$ are first
transformed to resource grids with the <a class="reference internal" href="ofdm.html#sionna.ofdm.OFDMDemodulator" title="sionna.ofdm.OFDMDemodulator">`OFDMDemodulator`</a>.
Then channel estimation is performed, e.g., with the help of the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHLSChannelEstimator" title="sionna.nr.PUSCHLSChannelEstimator">`PUSCHLSChannelEstimator`</a>. If `channel_estimator`
is chosen to be “perfect”, this step is skipped and the input $\mathbf{h}$
is used instead.
Next, MIMO detection is carried out with an arbitrary <a class="reference internal" href="ofdm.html#sionna.ofdm.OFDMDetector" title="sionna.ofdm.OFDMDetector">`OFDMDetector`</a>.
The resulting LLRs for each layer are then combined to transport blocks
with the help of the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerDemapper" title="sionna.nr.LayerDemapper">`LayerDemapper`</a>.
Finally, the transport blocks are decoded with the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder" title="sionna.nr.TBDecoder">`TBDecoder`</a>.
Parameters
 
- **pusch_transmitter** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter" title="sionna.nr.PUSCHTransmitter">`PUSCHTransmitter`</a>) – Transmitter used for the generation of the transmit signals
- **channel_estimator** (<a class="reference internal" href="ofdm.html#sionna.ofdm.BaseChannelEstimator" title="sionna.ofdm.BaseChannelEstimator">`BaseChannelEstimator`</a>, “perfect”, or <cite>None</cite>) – Channel estimator to be used.
If <cite>None</cite>, the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHLSChannelEstimator" title="sionna.nr.PUSCHLSChannelEstimator">`PUSCHLSChannelEstimator`</a> with
linear interpolation is used.
If “perfect”, no channel estimation is performed and the channel state information
`h` must be provided as additional input.
Defaults to <cite>None</cite>.
- **mimo_detector** (<a class="reference internal" href="ofdm.html#sionna.ofdm.OFDMDetector" title="sionna.ofdm.OFDMDetector">`OFDMDetector`</a> or <cite>None</cite>) – MIMO Detector to be used.
If <cite>None</cite>, the <a class="reference internal" href="ofdm.html#sionna.ofdm.LinearDetector" title="sionna.ofdm.LinearDetector">`LinearDetector`</a> with
LMMSE detection is used.
Defaults to <cite>None</cite>.
- **tb_decoder** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder" title="sionna.nr.TBDecoder">`TBDecoder`</a> or <cite>None</cite>) – Transport block decoder to be used.
If <cite>None</cite>, the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder" title="sionna.nr.TBDecoder">`TBDecoder`</a> with its
default settings is used.
Defaults to <cite>None</cite>.
- **return_tb_crc_status** (<em>bool</em>) – If <cite>True</cite>, the status of the transport block CRC is returned
as additional output.
Defaults to <cite>False</cite>.
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> or <cite>None</cite>) – Stream management configuration to be used.
If <cite>None</cite>, it is assumed that there is a single receiver
which decodes all streams of all transmitters.
Defaults to <cite>None</cite>.
- **input_domain** (<em>str</em><em>, </em><em>one of</em><em> [</em><em>"freq"</em><em>, </em><em>"time"</em><em>]</em>) – Domain of the input signal.
Defaults to “freq”.
- **l_min** (int or <cite>None</cite>) – Smallest time-lag for the discrete complex baseband channel.
Only needed if `input_domain` equals “time”.
Defaults to <cite>None</cite>.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(y, h, no)** – Tuple:
- **y** (<em>[batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex or [batch size, num_rx, num_rx_ant, num_time_samples + l_max - l_min], tf.complex</em>) – Frequency- or time-domain input signal
- **h** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_max - l_min, l_max - l_min + 1], tf.complex</em>) – Perfect channel state information in either frequency or time domain
(depending on `input_domain`) to be used for detection.
Only required if `channel_estimator` equals “perfect”.
- **no** (<em>[batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float</em>) – Variance of the AWGN


Output
 
- **b_hat** (<em>[batch_size, num_tx, tb_size], tf.float</em>) – Decoded information bits
- **tb_crc_status** (<em>[batch_size, num_tx], tf.bool</em>) – Transport block CRC status



<p class="rubric">Example
```python
>>> pusch_config = PUSCHConfig()
>>> pusch_transmitter = PUSCHTransmitter(pusch_config)
>>> pusch_receiver = PUSCHReceiver(pusch_transmitter)
>>> channel = AWGN()
>>> x, b = pusch_transmitter(16)
>>> no = 0.1
>>> y = channel([x, no])
>>> b_hat = pusch_receiver([x, no])
>>> compute_ber(b, b_hat)
<tf.Tensor: shape=(), dtype=float64, numpy=0.0>
```
<em class="property">`property` </em>`resource_grid`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHReceiver.resource_grid" title="Permalink to this definition"></a>
    
OFDM resource grid underlying the PUSCH transmissions


### PUSCHTransmitter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschtransmitter" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``PUSCHTransmitter`(<em class="sig-param">`pusch_configs`</em>, <em class="sig-param">`return_bits``=``True`</em>, <em class="sig-param">`output_domain``=``'freq'`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`verbose``=``False`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_transmitter.html#PUSCHTransmitter">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter" title="Permalink to this definition"></a>
    
This layer generates batches of 5G NR PUSCH slots for multiple transmitters
with random or provided payloads. Frequency- or time-domain outputs can be generated.
    
It combines multiple processing blocks into a single layer
as shown in the following figure. Blocks with dashed lines are
optional and depend on the configuration.
<a class="reference internal image-reference" href="../_images/pusch_transmitter_block_diagram.png"><img alt="../_images/pusch_transmitter_block_diagram.png" src="https://nvlabs.github.io/sionna/_images/pusch_transmitter_block_diagram.png" style="width: 364.2px; height: 543.0px;" /></a>
    
Information bits $\mathbf{b}$ that are either randomly generated or
provided as input are encoded into a transport block by the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder" title="sionna.nr.TBEncoder">`TBEncoder`</a>.
The encoded bits are then mapped to QAM constellation symbols by the <a class="reference internal" href="mapping.html#sionna.mapping.Mapper" title="sionna.mapping.Mapper">`Mapper`</a>.
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper" title="sionna.nr.LayerMapper">`LayerMapper`</a> splits the modulated symbols into different layers
which are then mapped onto OFDM resource grids by the <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGridMapper" title="sionna.ofdm.ResourceGridMapper">`ResourceGridMapper`</a>.
If precoding is enabled in the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a>, the resource grids
are further precoded so that there is one for each transmitter and antenna port.
If `output_domain` equals “freq”, these are the outputs $\mathbf{x}$.
If `output_domain` is chosen to be “time”, the resource grids are transformed into
time-domain signals by the <a class="reference internal" href="ofdm.html#sionna.ofdm.OFDMModulator" title="sionna.ofdm.OFDMModulator">`OFDMModulator`</a>.
Parameters
 
- **pusch_configs** (instance or list of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a>) – PUSCH Configurations according to which the resource grid and pilot pattern
will created. One configuration is needed for each transmitter.
- **return_bits** (<em>bool</em>) – If set to <cite>True</cite>, the layer generates random information bits
to be transmitted and returns them together with the transmit signal.
Defaults to <cite>True</cite>.
- **output_domain** (<em>str</em><em>, </em><em>one of</em><em> [</em><em>"freq"</em><em>, </em><em>"time"</em><em>]</em>) – The domain of the output. Defaults to “freq”.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>]</em>) – Dtype of inputs and outputs. Defaults to tf.complex64.
- **verbose** (<em>bool</em>) – If <cite>True</cite>, additional parameters are printed during initialization.
Defaults to <cite>False</cite>.


Input
 
- **One of**
- **batch_size** (<em>int</em>) – Batch size of random transmit signals to be generated,
if `return_bits` is <cite>True</cite>.
- **b** (<em>[batch_size, num_tx, tb_size], tf.float</em>) – Information bits to be transmitted,
if `return_bits` is <cite>False</cite>.


Output
 
- **x** (<em>[batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex or [batch size, num_tx, num_tx_ant, num_time_samples], tf.complex</em>) – Transmit signal in either frequency or time domain, depending on `output_domain`.
- **b** (<em>[batch_size, num_tx, tb_size], tf.float</em>) – Transmitted information bits.
Only returned if `return_bits` is <cite>True</cite>.



<p class="rubric">Example
```python
>>> pusch_config = PUSCHConfig()
>>> pusch_transmitter = PUSCHTransmitter(pusch_config)
>>> x, b = pusch_transmitter(16)
>>> print("Shape of x:", x.shape)
Shape of x: (16, 1, 1, 14, 48)
>>> print("Shape of b:", b.shape)
Shape of b: (16, 1, 1352)
```
<em class="property">`property` </em>`pilot_pattern`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter.pilot_pattern" title="Permalink to this definition"></a>
    
Aggregate pilot pattern of all transmitters


<em class="property">`property` </em>`resource_grid`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter.resource_grid" title="Permalink to this definition"></a>
    
OFDM resource grid underlying the PUSCH transmissions


`show`()<a class="reference internal" href="../_modules/sionna/nr/pusch_transmitter.html#PUSCHTransmitter.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter.show" title="Permalink to this definition"></a>
    
Print all properties of the PUSCHConfig and children


## Transport Block<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#transport-block" title="Permalink to this headline"></a>

