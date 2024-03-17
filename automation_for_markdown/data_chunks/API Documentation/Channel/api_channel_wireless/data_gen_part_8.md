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
### Urban microcell (UMi)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#urban-microcell-umi" title="Permalink to this headline"></a>
### Urban macrocell (UMa)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#urban-macrocell-uma" title="Permalink to this headline"></a>
  
  

### Urban microcell (UMi)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#urban-microcell-umi" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``UMi`(<em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`o2i_model`</em>, <em class="sig-param">`ut_array`</em>, <em class="sig-param">`bs_array`</em>, <em class="sig-param">`direction`</em>, <em class="sig-param">`enable_pathloss``=``True`</em>, <em class="sig-param">`enable_shadow_fading``=``True`</em>, <em class="sig-param">`always_generate_lsp``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/umi.html#UMi">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="Permalink to this definition"></a>
    
Urban microcell (UMi) channel model from 3GPP <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id14">[TR38901]</a> specification.
    
Setting up a UMi model requires configuring the network topology, i.e., the
UTs and BSs locations, UTs velocities, etc. This is achieved using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi.set_topology" title="sionna.channel.tr38901.UMi.set_topology">`set_topology()`</a> method. Setting a different
topology for each batch example is possible. The batch size used when setting up the network topology
is used for the link simulations.
    
The following code snippet shows how to setup a UMi channel model operating
in the frequency domain:
```python
>>> # UT and BS panel arrays
>>> bs_array = PanelArray(num_rows_per_panel = 4,
...                       num_cols_per_panel = 4,
...                       polarization = 'dual',
...                       polarization_type  = 'cross',
...                       antenna_pattern = '38.901',
...                       carrier_frequency = 3.5e9)
>>> ut_array = PanelArray(num_rows_per_panel = 1,
...                       num_cols_per_panel = 1,
...                       polarization = 'single',
...                       polarization_type = 'V',
...                       antenna_pattern = 'omni',
...                       carrier_frequency = 3.5e9)
>>> # Instantiating UMi channel model
>>> channel_model = UMi(carrier_frequency = 3.5e9,
...                     o2i_model = 'low',
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
>>> # Instanting the frequency domain channel
>>> channel = OFDMChannel(channel_model = channel_model,
...                       resource_grid = rg)
```

    
where `rg` is an instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
Parameters
 
- **carrier_frequency** (<em>float</em>) – Carrier frequency in Hertz
- **o2i_model** (<em>str</em>) – Outdoor-to-indoor loss model for UTs located indoor.
Set this parameter to “low” to use the low-loss model, or to “high”
to use the high-loss model.
See section 7.4.3 of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id15">[TR38901]</a> for details.
- **rx_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the receivers. All receivers share the same
antenna array configuration.
- **tx_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the transmitters. All transmitters share the
same antenna array configuration.
- **direction** (<em>str</em>) – Link direction. Either “uplink” or “downlink”.
- **enable_pathloss** (<em>bool</em>) – If <cite>True</cite>, apply pathloss. Otherwise doesn’t. Defaults to <cite>True</cite>.
- **enable_shadow_fading** (<em>bool</em>) – If <cite>True</cite>, apply shadow fading. Otherwise doesn’t.
Defaults to <cite>True</cite>.
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




`set_topology`(<em class="sig-param">`ut_loc``=``None`</em>, <em class="sig-param">`bs_loc``=``None`</em>, <em class="sig-param">`ut_orientations``=``None`</em>, <em class="sig-param">`bs_orientations``=``None`</em>, <em class="sig-param">`ut_velocities``=``None`</em>, <em class="sig-param">`in_state``=``None`</em>, <em class="sig-param">`los``=``None`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi.set_topology" title="Permalink to this definition"></a>
    
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
of UTs is set following 3GPP specification <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id16">[TR38901]</a>.




**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.


`show_topology`(<em class="sig-param">`bs_index``=``0`</em>, <em class="sig-param">`batch_index``=``0`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi.show_topology" title="Permalink to this definition"></a>
    
Shows the network topology of the batch example with index
`batch_index`.
    
The `bs_index` parameter specifies with respect to which BS the
LoS/NLoS state of UTs is indicated.
Input
 
- **bs_index** (<em>int</em>) – BS index with respect to which the LoS/NLoS state of UTs is
indicated. Defaults to 0.
- **batch_index** (<em>int</em>) – Batch example for which the topology is shown. Defaults to 0.





### Urban macrocell (UMa)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#urban-macrocell-uma" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``UMa`(<em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`o2i_model`</em>, <em class="sig-param">`ut_array`</em>, <em class="sig-param">`bs_array`</em>, <em class="sig-param">`direction`</em>, <em class="sig-param">`enable_pathloss``=``True`</em>, <em class="sig-param">`enable_shadow_fading``=``True`</em>, <em class="sig-param">`always_generate_lsp``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/uma.html#UMa">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMa" title="Permalink to this definition"></a>
    
Urban macrocell (UMa) channel model from 3GPP <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id17">[TR38901]</a> specification.
    
Setting up a UMa model requires configuring the network topology, i.e., the
UTs and BSs locations, UTs velocities, etc. This is achieved using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMa.set_topology" title="sionna.channel.tr38901.UMa.set_topology">`set_topology()`</a> method. Setting a different
topology for each batch example is possible. The batch size used when setting up the network topology
is used for the link simulations.
    
The following code snippet shows how to setup an UMa channel model assuming
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
>>> # Instantiating UMa channel model
>>> channel_model = UMa(carrier_frequency = 3.5e9,
...                     o2i_model = 'low',
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
 
- **carrier_frequency** (<em>float</em>) – Carrier frequency in Hertz
- **o2i_model** (<em>str</em>) – Outdoor-to-indoor loss model for UTs located indoor.
Set this parameter to “low” to use the low-loss model, or to “high”
to use the high-loss model.
See section 7.4.3 of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id18">[TR38901]</a> for details.
- **rx_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the receivers. All receivers share the same
antenna array configuration.
- **tx_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the transmitters. All transmitters share the
same antenna array configuration.
- **direction** (<em>str</em>) – Link direction. Either “uplink” or “downlink”.
- **enable_pathloss** (<em>bool</em>) – If <cite>True</cite>, apply pathloss. Otherwise doesn’t. Defaults to <cite>True</cite>.
- **enable_shadow_fading** (<em>bool</em>) – If <cite>True</cite>, apply shadow fading. Otherwise doesn’t.
Defaults to <cite>True</cite>.
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




`set_topology`(<em class="sig-param">`ut_loc``=``None`</em>, <em class="sig-param">`bs_loc``=``None`</em>, <em class="sig-param">`ut_orientations``=``None`</em>, <em class="sig-param">`bs_orientations``=``None`</em>, <em class="sig-param">`ut_velocities``=``None`</em>, <em class="sig-param">`in_state``=``None`</em>, <em class="sig-param">`los``=``None`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMa.set_topology" title="Permalink to this definition"></a>
    
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
of UTs is set following 3GPP specification <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id19">[TR38901]</a>.




**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.


`show_topology`(<em class="sig-param">`bs_index``=``0`</em>, <em class="sig-param">`batch_index``=``0`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMa.show_topology" title="Permalink to this definition"></a>
    
Shows the network topology of the batch example with index
`batch_index`.
    
The `bs_index` parameter specifies with respect to which BS the
LoS/NLoS state of UTs is indicated.
Input
 
- **bs_index** (<em>int</em>) – BS index with respect to which the LoS/NLoS state of UTs is
indicated. Defaults to 0.
- **batch_index** (<em>int</em>) – Batch example for which the topology is shown. Defaults to 0.





