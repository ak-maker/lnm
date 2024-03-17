# Wireless<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#wireless" title="Permalink to this headline"></a>
    
This module provides layers and functions that implement wireless channel models.
Models currently available include <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.AWGN" title="sionna.channel.AWGN">`AWGN`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#flat-fading">flat-fading</a> with (optional) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation">`SpatialCorrelation`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="sionna.channel.RayleighBlockFading">`RayleighBlockFading`</a>, as well as models from the 3rd Generation Partnership Project (3GPP) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id1">[TR38901]</a>: <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tdl">TDL</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#cdl">CDL</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#umi">UMi</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#uma">UMa</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rma">RMa</a>. It is also possible to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#external-datasets">use externally generated CIRs</a>.
    
Apart from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#flat-fading">flat-fading</a>, all of these models generate channel impulse responses (CIRs) that can then be used to
implement a channel transfer function in the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#time-domain">time domain</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#ofdm-waveform">assuming an OFDM waveform</a>.
    
This is achieved using the different functions, classes, and Keras layers which
operate as shown in the figures below.
<p class="caption">Fig. 7 Channel module architecture for time domain simulations.<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id30" title="Permalink to this image"></a>

<p class="caption">Fig. 8 Channel module architecture for simulations assuming OFDM waveform.<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id31" title="Permalink to this image"></a>
    
A channel model generate CIRs from which channel responses in the time domain
or in the frequency domain are computed using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_time_channel" title="sionna.channel.cir_to_time_channel">`cir_to_time_channel()`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_ofdm_channel" title="sionna.channel.cir_to_ofdm_channel">`cir_to_ofdm_channel()`</a> functions, respectively.
If one does not need access to the raw CIRs, the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateTimeChannel" title="sionna.channel.GenerateTimeChannel">`GenerateTimeChannel`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateOFDMChannel" title="sionna.channel.GenerateOFDMChannel">`GenerateOFDMChannel`</a> classes can be used to conveniently
sample CIRs and generate channel responses in the desired domain.
    
Once the channel responses in the time or frequency domain are computed, they
can be applied to the channel input using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyTimeChannel" title="sionna.channel.ApplyTimeChannel">`ApplyTimeChannel`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyOFDMChannel" title="sionna.channel.ApplyOFDMChannel">`ApplyOFDMChannel`</a> Keras layers.
    
The following code snippets show how to setup and run a Rayleigh block fading
model assuming an OFDM waveform, and without accessing the CIRs or
channel responses.
This is the easiest way to setup a channel model.
Setting-up other models is done in a similar way, except for
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.AWGN" title="sionna.channel.AWGN">`AWGN`</a> (see the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.AWGN" title="sionna.channel.AWGN">`AWGN`</a>
class documentation).
```python
rayleigh = RayleighBlockFading(num_rx = 1,
                               num_rx_ant = 32,
                               num_tx = 4,
                               num_tx_ant = 2)
channel  = OFDMChannel(channel_model = rayleigh,
                       resource_grid = rg)
```

    
where `rg` is an instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
    
Running the channel model is done as follows:
```python
# x is the channel input
# no is the noise variance
y = channel([x, no])
```

    
To use the time domain representation of the channel, one can use
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.TimeChannel" title="sionna.channel.TimeChannel">`TimeChannel`</a> instead of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.OFDMChannel" title="sionna.channel.OFDMChannel">`OFDMChannel`</a>.
    
If access to the channel responses is needed, one can separate their
generation from their application to the channel input by setting up the channel
model as follows:
```python
rayleigh = RayleighBlockFading(num_rx = 1,
                               num_rx_ant = 32,
                               num_tx = 4,
                               num_tx_ant = 2)
generate_channel = GenerateOFDMChannel(channel_model = rayleigh,
                                       resource_grid = rg)
apply_channel = ApplyOFDMChannel()
```

    
where `rg` is an instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
Running the channel model is done as follows:
```python
# Generate a batch of channel responses
h = generate_channel(batch_size)
# Apply the channel
# x is the channel input
# no is the noise variance
y = apply_channel([x, h, no])
```

    
Generating and applying the channel in the time domain can be achieved by using
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateTimeChannel" title="sionna.channel.GenerateTimeChannel">`GenerateTimeChannel`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyTimeChannel" title="sionna.channel.ApplyTimeChannel">`ApplyTimeChannel`</a> instead of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateOFDMChannel" title="sionna.channel.GenerateOFDMChannel">`GenerateOFDMChannel`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyOFDMChannel" title="sionna.channel.ApplyOFDMChannel">`ApplyOFDMChannel`</a>, respectively.
    
To access the CIRs, setting up the channel can be done as follows:
```python
rayleigh = RayleighBlockFading(num_rx = 1,
                               num_rx_ant = 32,
                               num_tx = 4,
                               num_tx_ant = 2)
apply_channel = ApplyOFDMChannel()
```

    
and running the channel model as follows:
```python
cir = rayleigh(batch_size)
h = cir_to_ofdm_channel(frequencies, *cir)
y = apply_channel([x, h, no])
```

    
where `frequencies` are the subcarrier frequencies in the baseband, which can
be computed using the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.subcarrier_frequencies" title="sionna.channel.subcarrier_frequencies">`subcarrier_frequencies()`</a> utility
function.
    
Applying the channel in the time domain can be done by using
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_time_channel" title="sionna.channel.cir_to_time_channel">`cir_to_time_channel()`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyTimeChannel" title="sionna.channel.ApplyTimeChannel">`ApplyTimeChannel`</a> instead of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_ofdm_channel" title="sionna.channel.cir_to_ofdm_channel">`cir_to_ofdm_channel()`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyOFDMChannel" title="sionna.channel.ApplyOFDMChannel">`ApplyOFDMChannel`</a>, respectively.
    
For the purpose of the present document, the following symbols apply:
<table class="docutils align-default">
<colgroup>
<col style="width: 24%" />
<col style="width: 76%" />
</colgroup>
<tbody>
<tr class="row-odd"><td>    
$N_T (u)$</td>
<td>    
Number of transmitters (transmitter index)</td>
</tr>
<tr class="row-even"><td>    
$N_R (v)$</td>
<td>    
Number of receivers (receiver index)</td>
</tr>
<tr class="row-odd"><td>    
$N_{TA} (k)$</td>
<td>    
Number of antennas per transmitter (transmit antenna index)</td>
</tr>
<tr class="row-even"><td>    
$N_{RA} (l)$</td>
<td>    
Number of antennas per receiver (receive antenna index)</td>
</tr>
<tr class="row-odd"><td>    
$N_S (s)$</td>
<td>    
Number of OFDM symbols (OFDM symbol index)</td>
</tr>
<tr class="row-even"><td>    
$N_F (n)$</td>
<td>    
Number of subcarriers (subcarrier index)</td>
</tr>
<tr class="row-odd"><td>    
$N_B (b)$</td>
<td>    
Number of time samples forming the channel input (baseband symbol index)</td>
</tr>
<tr class="row-even"><td>    
$L_{\text{min}}$</td>
<td>    
Smallest time-lag for the discrete complex baseband channel</td>
</tr>
<tr class="row-odd"><td>    
$L_{\text{max}}$</td>
<td>    
Largest time-lag for the discrete complex baseband channel</td>
</tr>
<tr class="row-even"><td>    
$M (m)$</td>
<td>    
Number of paths (clusters) forming a power delay profile (path index)</td>
</tr>
<tr class="row-odd"><td>    
$\tau_m(t)$</td>
<td>    
$m^{th}$ path (cluster) delay at time step $t$</td>
</tr>
<tr class="row-even"><td>    
$a_m(t)$</td>
<td>    
$m^{th}$ path (cluster) complex coefficient at time step $t$</td>
</tr>
<tr class="row-odd"><td>    
$\Delta_f$</td>
<td>    
Subcarrier spacing</td>
</tr>
<tr class="row-even"><td>    
$W$</td>
<td>    
Bandwidth</td>
</tr>
<tr class="row-odd"><td>    
$N_0$</td>
<td>    
Noise variance</td>
</tr>
</tbody>
</table>
    
All transmitters are equipped with $N_{TA}$ antennas and all receivers
with $N_{RA}$ antennas.
    
A channel model, such as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="sionna.channel.RayleighBlockFading">`RayleighBlockFading`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="sionna.channel.tr38901.UMi">`UMi`</a>, is used to generate for each link between
antenna $k$ of transmitter $u$ and antenna $l$ of receiver
$v$ a power delay profile
$(a_{u, k, v, l, m}(t), \tau_{u, v, m}), 0 \leq m \leq M-1$.
The delays are assumed not to depend on time $t$, and transmit and receive
antennas $k$ and $l$.
Such a power delay profile corresponds to the channel impulse response

$$
h_{u, k, v, l}(t,\tau) =
\sum_{m=0}^{M-1} a_{u, k, v, l,m}(t) \delta(\tau - \tau_{u, v, m})
$$
    
where $\delta(\cdot)$ is the Dirac delta measure.
For example, in the case of Rayleigh block fading, the power delay profiles are
time-invariant and such that for every link $(u, k, v, l)$

$$
\begin{split}\begin{align}
   M                     &= 1\\
   \tau_{u, v, 0}  &= 0\\
   a_{u, k, v, l, 0}     &\sim \mathcal{CN}(0,1).
\end{align}\end{split}
$$
    
3GPP channel models use the procedure depicted in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id2">[TR38901]</a> to generate power
delay profiles. With these models, the power delay profiles are time-<em>variant</em>
in the event of mobility.

# Table of Content
## 3GPP 38.901 channel models<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#gpp-38-901-channel-models" title="Permalink to this headline"></a>
### Rural macrocell (RMa)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rural-macrocell-rma" title="Permalink to this headline"></a>
## External datasets<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#external-datasets" title="Permalink to this headline"></a>
## Utility functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#utility-functions" title="Permalink to this headline"></a>
### subcarrier_frequencies<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#subcarrier-frequencies" title="Permalink to this headline"></a>
### time_lag_discrete_time_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#time-lag-discrete-time-channel" title="Permalink to this headline"></a>
  
  

### Rural macrocell (RMa)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rural-macrocell-rma" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``RMa`(<em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`ut_array`</em>, <em class="sig-param">`bs_array`</em>, <em class="sig-param">`direction`</em>, <em class="sig-param">`enable_pathloss``=``True`</em>, <em class="sig-param">`enable_shadow_fading``=``True`</em>, <em class="sig-param">`always_generate_lsp``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/rma.html#RMa">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.RMa" title="Permalink to this definition"></a>
    
Rural macrocell (RMa) channel model from 3GPP <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id20">[TR38901]</a> specification.
    
Setting up a RMa model requires configuring the network topology, i.e., the
UTs and BSs locations, UTs velocities, etc. This is achieved using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.RMa.set_topology" title="sionna.channel.tr38901.RMa.set_topology">`set_topology()`</a> method. Setting a different
topology for each batch example is possible. The batch size used when setting up the network topology
is used for the link simulations.
    
The following code snippet shows how to setup an RMa channel model assuming
an OFDM waveform:
```python
>>> # UT and BS panel arrays
>>> bs_array = PanelArray(num_rows_per_panel = 4,
...                       num_cols_per_panel = 4,
...                       polarization = 'dual',
...                       polarization_type = 'cross',
...                       antenna_pattern = '38.901',
...                       carrier_frequency = 3.5e9)
>>> ut_array = PanelArray(num_rows_per_panel = 1,
...                       num_cols_per_panel = 1,
...                       polarization = 'single',
...                       polarization_type = 'V',
...                       antenna_pattern = 'omni',
...                       carrier_frequency = 3.5e9)
>>> # Instantiating RMa channel model
>>> channel_model = RMa(carrier_frequency = 3.5e9,
...                     ut_array = ut_array,
...                     bs_array = bs_array,
...                     direction = 'uplink')
>>> # Setting up network topology
>>> # ut_loc: UTs locations
>>> # bs_loc: BSs locations
>>> # ut_orientations: UTs array orientations
>>> # bs_orientations: BSs array orientations
>>> # in_state: Indoor/outdoor states of UTs
>>> channel_model.set_topology(ut_loc,
...                            bs_loc,
...                            ut_orientations,
...                            bs_orientations,
...                            ut_velocities,
...                            in_state)
>>> # Instanting the OFDM channel
>>> channel = OFDMChannel(channel_model = channel_model,
...                       resource_grid = rg)
```

    
where `rg` is an instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
Parameters
 
- **carrier_frequency** (<em>float</em>) – Carrier frequency [Hz]
- **rx_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the receivers. All receivers share the same
antenna array configuration.
- **tx_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the transmitters. All transmitters share the
same antenna array configuration.
- **direction** (<em>str</em>) – Link direction. Either “uplink” or “downlink”.
- **enable_pathloss** (<em>bool</em>) – If <cite>True</cite>, apply pathloss. Otherwise doesn’t. Defaults to <cite>True</cite>.
- **enable_shadow_fading** (<em>bool</em>) – If <cite>True</cite>, apply shadow fading. Otherwise doesn’t.
Defaults to <cite>True</cite>.
- **average_street_width** (<em>float</em>) – Average street width [m]. Defaults to 5m.
- **average_street_width** – Average building height [m]. Defaults to 20m.
- **always_generate_lsp** (<em>bool</em>) – If <cite>True</cite>, new large scale parameters (LSPs) are generated for every
new generation of channel impulse responses. Otherwise, always reuse
the same LSPs, except if the topology is changed. Defaults to
<cite>False</cite>.
- **dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.


Input
 
- **num_time_steps** (<em>int</em>) – Number of time steps
- **sampling_frequency** (<em>float</em>) – Sampling frequency [Hz]


Output
 
- **a** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx, num_tx, num_paths], tf.float</em>) – Path delays [s]




`set_topology`(<em class="sig-param">`ut_loc``=``None`</em>, <em class="sig-param">`bs_loc``=``None`</em>, <em class="sig-param">`ut_orientations``=``None`</em>, <em class="sig-param">`bs_orientations``=``None`</em>, <em class="sig-param">`ut_velocities``=``None`</em>, <em class="sig-param">`in_state``=``None`</em>, <em class="sig-param">`los``=``None`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.RMa.set_topology" title="Permalink to this definition"></a>
    
Set the network topology.
    
It is possible to set up a different network topology for each batch
example. The batch size used when setting up the network topology
is used for the link simulations.
    
When calling this function, not specifying a parameter leads to the
reuse of the previously given value. Not specifying a value that was not
set at a former call rises an error.
Input
 
- **ut_loc** (<em>[batch size,num_ut, 3], tf.float</em>) – Locations of the UTs
- **bs_loc** (<em>[batch size,num_bs, 3], tf.float</em>) – Locations of BSs
- **ut_orientations** (<em>[batch size,num_ut, 3], tf.float</em>) – Orientations of the UTs arrays [radian]
- **bs_orientations** (<em>[batch size,num_bs, 3], tf.float</em>) – Orientations of the BSs arrays [radian]
- **ut_velocities** (<em>[batch size,num_ut, 3], tf.float</em>) – Velocity vectors of UTs
- **in_state** (<em>[batch size,num_ut], tf.bool</em>) – Indoor/outdoor state of UTs. <cite>True</cite> means indoor and <cite>False</cite>
means outdoor.
- **los** (tf.bool or <cite>None</cite>) – If not <cite>None</cite> (default value), all UTs located outdoor are
forced to be in LoS if `los` is set to <cite>True</cite>, or in NLoS
if it is set to <cite>False</cite>. If set to <cite>None</cite>, the LoS/NLoS states
of UTs is set following 3GPP specification <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id21">[TR38901]</a>.




**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.


`show_topology`(<em class="sig-param">`bs_index``=``0`</em>, <em class="sig-param">`batch_index``=``0`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.RMa.show_topology" title="Permalink to this definition"></a>
    
Shows the network topology of the batch example with index
`batch_index`.
    
The `bs_index` parameter specifies with respect to which BS the
LoS/NLoS state of UTs is indicated.
Input
 
- **bs_index** (<em>int</em>) – BS index with respect to which the LoS/NLoS state of UTs is
indicated. Defaults to 0.
- **batch_index** (<em>int</em>) – Batch example for which the topology is shown. Defaults to 0.





## External datasets<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#external-datasets" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``CIRDataset`(<em class="sig-param">`cir_generator`</em>, <em class="sig-param">`batch_size`</em>, <em class="sig-param">`num_rx`</em>, <em class="sig-param">`num_rx_ant`</em>, <em class="sig-param">`num_tx`</em>, <em class="sig-param">`num_tx_ant`</em>, <em class="sig-param">`num_paths`</em>, <em class="sig-param">`num_time_steps`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/cir_dataset.html#CIRDataset">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.CIRDataset" title="Permalink to this definition"></a>
    
Creates a channel model from a dataset that can be used with classes such as
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.TimeChannel" title="sionna.channel.TimeChannel">`TimeChannel`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.OFDMChannel" title="sionna.channel.OFDMChannel">`OFDMChannel`</a>.
The dataset is defined by a <a class="reference external" href="https://wiki.python.org/moin/Generators">generator</a>.
    
The batch size is configured when instantiating the dataset or through the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.CIRDataset.batch_size" title="sionna.channel.CIRDataset.batch_size">`batch_size`</a> property.
The number of time steps (<cite>num_time_steps</cite>) and sampling frequency (<cite>sampling_frequency</cite>) can only be set when instantiating the dataset.
The specified values must be in accordance with the data.
<p class="rubric">Example
    
The following code snippet shows how to use this class as a channel model.
```python
>>> my_generator = MyGenerator(...)
>>> channel_model = sionna.channel.CIRDataset(my_generator,
...                                           batch_size,
...                                           num_rx,
...                                           num_rx_ant,
...                                           num_tx,
...                                           num_tx_ant,
...                                           num_paths,
...                                           num_time_steps+l_tot-1)
>>> channel = sionna.channel.TimeChannel(channel_model, bandwidth, num_time_steps)
```

    
where `MyGenerator` is a generator
```python
>>> class MyGenerator:
...
...     def __call__(self):
...         ...
...         yield a, tau
```

    
that returns complex-valued path coefficients `a` with shape
<cite>[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]</cite>
and real-valued path delays `tau` (in second)
<cite>[num_rx, num_tx, num_paths]</cite>.
Parameters
 
- **cir_generator** – Generator that returns channel impulse responses `(a,` `tau)` where
`a` is the tensor of channel coefficients of shape
<cite>[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]</cite>
and dtype `dtype`, and `tau` the tensor of path delays
of shape  <cite>[num_rx, num_tx, num_paths]</cite> and dtype `dtype.`
`real_dtype`.
- **batch_size** (<em>int</em>) – Batch size
- **num_rx** (<em>int</em>) – Number of receivers ($N_R$)
- **num_rx_ant** (<em>int</em>) – Number of antennas per receiver ($N_{RA}$)
- **num_tx** (<em>int</em>) – Number of transmitters ($N_T$)
- **num_tx_ant** (<em>int</em>) – Number of antennas per transmitter ($N_{TA}$)
- **num_paths** (<em>int</em>) – Number of paths ($M$)
- **num_time_steps** (<em>int</em>) – Number of time steps
- **dtype** (<em>tf.DType</em>) – Complex datatype to use for internal processing and output.
Defaults to <cite>tf.complex64</cite>.


Output
 
- **a** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx, num_tx, num_paths], tf.float</em>) – Path delays [s]




<em class="property">`property` </em>`batch_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.CIRDataset.batch_size" title="Permalink to this definition"></a>
    
Batch size


## Utility functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#utility-functions" title="Permalink to this headline"></a>

### subcarrier_frequencies<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#subcarrier-frequencies" title="Permalink to this headline"></a>

`sionna.channel.``subcarrier_frequencies`(<em class="sig-param">`num_subcarriers`</em>, <em class="sig-param">`subcarrier_spacing`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#subcarrier_frequencies">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.subcarrier_frequencies" title="Permalink to this definition"></a>
    
Compute the baseband frequencies of `num_subcarrier` subcarriers spaced by
`subcarrier_spacing`, i.e.,
```python
>>> # If num_subcarrier is even:
>>> frequencies = [-num_subcarrier/2, ..., 0, ..., num_subcarrier/2-1] * subcarrier_spacing
>>>
>>> # If num_subcarrier is odd:
>>> frequencies = [-(num_subcarrier-1)/2, ..., 0, ..., (num_subcarrier-1)/2] * subcarrier_spacing
```

Input
 
- **num_subcarriers** (<em>int</em>) – Number of subcarriers
- **subcarrier_spacing** (<em>float</em>) – Subcarrier spacing [Hz]
- **dtype** (<em>tf.DType</em>) – Datatype to use for internal processing and output.
If a complex datatype is provided, the corresponding precision of
real components is used.
Defaults to <cite>tf.complex64</cite> (<cite>tf.float32</cite>).


Output
    
**frequencies** ([`num_subcarrier`], tf.float) – Baseband frequencies of subcarriers



### time_lag_discrete_time_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#time-lag-discrete-time-channel" title="Permalink to this headline"></a>

`sionna.channel.``time_lag_discrete_time_channel`(<em class="sig-param">`bandwidth`</em>, <em class="sig-param">`maximum_delay_spread``=``3e-06`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#time_lag_discrete_time_channel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.time_lag_discrete_time_channel" title="Permalink to this definition"></a>
    
Compute the smallest and largest time-lag for the descrete complex baseband
channel, i.e., $L_{\text{min}}$ and $L_{\text{max}}$.
    
The smallest time-lag ($L_{\text{min}}$) returned is always -6, as this value
was found small enough for all models included in Sionna.
    
The largest time-lag ($L_{\text{max}}$) is computed from the `bandwidth`
and `maximum_delay_spread` as follows:

$$
L_{\text{max}} = \lceil W \tau_{\text{max}} \rceil + 6
$$
    
where $L_{\text{max}}$ is the largest time-lag, $W$ the `bandwidth`,
and $\tau_{\text{max}}$ the `maximum_delay_spread`.
    
The default value for the `maximum_delay_spread` is 3us, which was found
to be large enough to include most significant paths with all channel models
included in Sionna assuming a nominal delay spread of 100ns.

**Note**
    
The values of $L_{\text{min}}$ and $L_{\text{max}}$ computed
by this function are only recommended values.
$L_{\text{min}}$ and $L_{\text{max}}$ should be set according to
the considered channel model. For OFDM systems, one also needs to be careful
that the effective length of the complex baseband channel is not larger than
the cyclic prefix length.

Input
 
- **bandwidth** (<em>float</em>) – Bandwith ($W$) [Hz]
- **maximum_delay_spread** (<em>float</em>) – Maximum delay spread [s]. Defaults to 3us.


Output
 
- **l_min** (<em>int</em>) – Smallest time-lag ($L_{\text{min}}$) for the descrete complex baseband
channel. Set to -6, , as this value was found small enough for all models
included in Sionna.
- **l_max** (<em>int</em>) – Largest time-lag ($L_{\text{max}}$) for the descrete complex baseband
channel




