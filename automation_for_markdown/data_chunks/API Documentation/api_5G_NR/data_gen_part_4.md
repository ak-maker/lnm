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
### PUSCHDMRSConfig<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschdmrsconfig" title="Permalink to this headline"></a>
### PUSCHLSChannelEstimator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschlschannelestimator" title="Permalink to this headline"></a>
  
  

### PUSCHDMRSConfig<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschdmrsconfig" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``PUSCHDMRSConfig`(<em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_dmrs_config.html#PUSCHDMRSConfig">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="Permalink to this definition"></a>
    
The PUSCHDMRSConfig objects sets parameters related to the generation
of demodulation reference signals (DMRS) for a physical uplink shared
channel (PUSCH), as described in Section 6.4.1.1 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id15">[3GPP38211]</a>.
    
All configurable properties can be provided as keyword arguments during the
initialization or changed later.
<p class="rubric">Example
```python
>>> dmrs_config = PUSCHDMRSConfig(config_type=2)
>>> dmrs_config.additional_position = 1
```
<em class="property">`property` </em>`additional_position`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.additional_position" title="Permalink to this definition"></a>
    
Maximum number of additional DMRS positions
    
The actual number of used DMRS positions depends on
the length of the PUSCH symbol allocation.
Type
    
int, 0 (default) | 1 | 2 | 3




<em class="property">`property` </em>`allowed_dmrs_ports`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.allowed_dmrs_ports" title="Permalink to this definition"></a>
    
List of nominal antenna
ports
    
The maximum number of allowed antenna ports <cite>max_num_dmrs_ports</cite>
depends on the DMRS <cite>config_type</cite> and <cite>length</cite>. It can be
equal to 4, 6, 8, or 12.
Type
    
list, [0,…,max_num_dmrs_ports-1], read-only




<em class="property">`property` </em>`beta`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.beta" title="Permalink to this definition"></a>
    
Ratio of PUSCH energy per resource element
(EPRE) to DMRS EPRE $\beta^{\text{DMRS}}_\text{PUSCH}$
Table 6.2.2-1 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id16">[3GPP38214]</a>
Type
    
float, read-only




<em class="property">`property` </em>`cdm_groups`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.cdm_groups" title="Permalink to this definition"></a>
    
List of CDM groups
$\lambda$ for all ports
in the <cite>dmrs_port_set</cite> as defined in
Table 6.4.1.1.3-1 or 6.4.1.1.3-2 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id17">[3GPP38211]</a>
    
Depends on the <cite>config_type</cite>.
Type
    
list, elements in [0,1,2], read-only




<em class="property">`property` </em>`config_type`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.config_type" title="Permalink to this definition"></a>
    
DMRS configuration type
    
The configuration type determines the frequency density of
DMRS signals. With configuration type 1, six subcarriers per PRB are
used for each antenna port, with configuration type 2, four
subcarriers are used.
Type
    
int, 1 (default) | 2




<em class="property">`property` </em>`deltas`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.deltas" title="Permalink to this definition"></a>
    
List of delta (frequency)
shifts $\Delta$ for all ports in the <cite>port_set</cite> as defined in
Table 6.4.1.1.3-1 or 6.4.1.1.3-2 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id18">[3GPP38211]</a>
    
Depends on the <cite>config_type</cite>.
Type
    
list, elements in [0,1,2,4], read-only




<em class="property">`property` </em>`dmrs_port_set`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.dmrs_port_set" title="Permalink to this definition"></a>
    
List of used DMRS antenna ports
    
The elements in this list must all be from the list of
<cite>allowed_dmrs_ports</cite> which depends on the <cite>config_type</cite> as well as
the <cite>length</cite>. If set to <cite>[]</cite>, the port set will be equal to
[0,…,num_layers-1], where
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_layers" title="sionna.nr.PUSCHConfig.num_layers">`num_layers`</a> is a property of the
parent <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a> instance.
Type
    
list, [] (default) | [0,…,11]




<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.length" title="Permalink to this definition"></a>
    
Number of front-loaded DMRS symbols
A value of 1 corresponds to “single-symbol” DMRS, a value
of 2 corresponds to “double-symbol” DMRS.
Type
    
int, 1 (default) | 2




<em class="property">`property` </em>`n_id`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.n_id" title="Permalink to this definition"></a>
    
Scrambling
identities
    
Defines the scrambling identities $N_\text{ID}^0$ and
$N_\text{ID}^1$ as a 2-tuple of integers. If <cite>None</cite>,
the property <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.n_cell_id" title="sionna.nr.CarrierConfig.n_cell_id">`n_cell_id`</a> of the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig" title="sionna.nr.CarrierConfig">`CarrierConfig`</a> is used.
Type
    
2-tuple, None (default), [[0,…,65535], [0,…,65535]]




<em class="property">`property` </em>`n_scid`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.n_scid" title="Permalink to this definition"></a>
    
DMRS scrambling initialization
$n_\text{SCID}$
Type
    
int, 0 (default) | 1




<em class="property">`property` </em>`num_cdm_groups_without_data`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.num_cdm_groups_without_data" title="Permalink to this definition"></a>
    
Number of CDM groups without data
    
This parameter controls how many REs are available for data
transmission in a DMRS symbol. It should be greater or equal to
the maximum configured number of CDM groups. A value of
1 corresponds to CDM group 0, a value of 2 corresponds to
CDM groups 0 and 1, and a value of 3 corresponds to
CDM groups 0, 1, and 2.
Type
    
int, 2 (default) | 1 | 3




<em class="property">`property` </em>`type_a_position`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.type_a_position" title="Permalink to this definition"></a>
    
Position of first DMRS OFDM symbol
    
Defines the position of the first DMRS symbol within a slot.
This parameter only applies if the property
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.mapping_type" title="sionna.nr.PUSCHConfig.mapping_type">`mapping_type`</a> of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a> is equal to “A”.
Type
    
int, 2 (default) | 3




<em class="property">`property` </em>`w_f`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.w_f" title="Permalink to this definition"></a>
    
Frequency weight vectors
$w_f(k')$ for all ports in the port set as defined in
Table 6.4.1.1.3-1 or 6.4.1.1.3-2 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id19">[3GPP38211]</a>
Type
    
matrix, elements in [-1,1], read-only




<em class="property">`property` </em>`w_t`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig.w_t" title="Permalink to this definition"></a>
    
Time weight vectors
$w_t(l')$ for all ports in the port set as defined in
Table 6.4.1.1.3-1 or 6.4.1.1.3-2 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id20">[3GPP38211]</a>
Type
    
matrix, elements in [-1,1], read-only




### PUSCHLSChannelEstimator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschlschannelestimator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``PUSCHLSChannelEstimator`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`dmrs_length`</em>, <em class="sig-param">`dmrs_additional_position`</em>, <em class="sig-param">`num_cdm_groups_without_data`</em>, <em class="sig-param">`interpolation_type``=``'nn'`</em>, <em class="sig-param">`interpolator``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_channel_estimation.html#PUSCHLSChannelEstimator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHLSChannelEstimator" title="Permalink to this definition"></a>
    
Layer implementing least-squares (LS) channel estimation for NR PUSCH Transmissions.
    
After LS channel estimation at the pilot positions, the channel estimates
and error variances are interpolated accross the entire resource grid using
a specified interpolation function.
    
The implementation is similar to that of <a class="reference internal" href="ofdm.html#sionna.ofdm.LSChannelEstimator" title="sionna.ofdm.LSChannelEstimator">`LSChannelEstimator`</a>.
However, it additional takes into account the separation of streams in the same CDM group
as defined in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a>. This is done through
frequency and time averaging of adjacent LS channel estimates.
Parameters
 
- **resource_grid** (<a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **dmrs_length** (<em>int</em><em>, </em><em>[</em><em>1</em><em>,</em><em>2</em><em>]</em>) – Length of DMRS symbols. See <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a>.
- **dmrs_additional_position** (<em>int</em><em>, </em><em>[</em><em>0</em><em>,</em><em>1</em><em>,</em><em>2</em><em>,</em><em>3</em><em>]</em>) – Number of additional DMRS symbols.
See <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a>.
- **num_cdm_groups_without_data** (<em>int</em><em>, </em><em>[</em><em>1</em><em>,</em><em>2</em><em>,</em><em>3</em><em>]</em>) – Number of CDM groups masked for data transmissions.
See <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a>.
- **interpolation_type** (<em>One of</em><em> [</em><em>"nn"</em><em>, </em><em>"lin"</em><em>, </em><em>"lin_time_avg"</em><em>]</em><em>, </em><em>string</em>) – The interpolation method to be used.
It is ignored if `interpolator` is not <cite>None</cite>.
Available options are <a class="reference internal" href="ofdm.html#sionna.ofdm.NearestNeighborInterpolator" title="sionna.ofdm.NearestNeighborInterpolator">`NearestNeighborInterpolator`</a> (<cite>“nn</cite>”)
or <a class="reference internal" href="ofdm.html#sionna.ofdm.LinearInterpolator" title="sionna.ofdm.LinearInterpolator">`LinearInterpolator`</a> without (<cite>“lin”</cite>) or with
averaging across OFDM symbols (<cite>“lin_time_avg”</cite>).
Defaults to “nn”.
- **interpolator** (<a class="reference internal" href="ofdm.html#sionna.ofdm.BaseChannelInterpolator" title="sionna.ofdm.BaseChannelInterpolator"><em>BaseChannelInterpolator</em></a>) – An instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.BaseChannelInterpolator" title="sionna.ofdm.BaseChannelInterpolator">`BaseChannelInterpolator`</a>,
such as <a class="reference internal" href="ofdm.html#sionna.ofdm.LMMSEInterpolator" title="sionna.ofdm.LMMSEInterpolator">`LMMSEInterpolator`</a>,
or <cite>None</cite>. In the latter case, the interpolator specified
by `interpolation_type` is used.
Otherwise, the `interpolator` is used and `interpolation_type`
is ignored.
Defaults to <cite>None</cite>.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(y, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], tf.complex</em>) – Observed resource grid
- **no** (<em>[batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float</em>) – Variance of the AWGN


Output
 
- **h_ls** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex</em>) – Channel estimates across the entire resource grid for all
transmitters and streams
- **err_var** (Same shape as `h_ls`, tf.float) – Channel estimation error variance across the entire resource grid
for all transmitters and streams




