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
### PanelArray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#panelarray" title="Permalink to this headline"></a>
### Antenna<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#antenna" title="Permalink to this headline"></a>
### AntennaArray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#antennaarray" title="Permalink to this headline"></a>
  
  

### PanelArray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#panelarray" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``PanelArray`(<em class="sig-param">`num_rows_per_panel`</em>, <em class="sig-param">`num_cols_per_panel`</em>, <em class="sig-param">`polarization`</em>, <em class="sig-param">`polarization_type`</em>, <em class="sig-param">`antenna_pattern`</em>, <em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`num_rows``=``1`</em>, <em class="sig-param">`num_cols``=``1`</em>, <em class="sig-param">`panel_vertical_spacing``=``None`</em>, <em class="sig-param">`panel_horizontal_spacing``=``None`</em>, <em class="sig-param">`element_vertical_spacing``=``None`</em>, <em class="sig-param">`element_horizontal_spacing``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/antenna.html#PanelArray">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="Permalink to this definition"></a>
    
Antenna panel array following the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id5">[TR38901]</a> specification.
    
This class is used to create models of the panel arrays used by the
transmitters and receivers and that need to be specified when using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#cdl">CDL</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#umi">UMi</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#uma">UMa</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rma">RMa</a>
models.
<p class="rubric">Example
```python
>>> array = PanelArray(num_rows_per_panel = 4,
...                    num_cols_per_panel = 4,
...                    polarization = 'dual',
...                    polarization_type = 'VH',
...                    antenna_pattern = '38.901',
...                    carrier_frequency = 3.5e9,
...                    num_cols = 2,
...                    panel_horizontal_spacing = 3.)
>>> array.show()
```

Parameters
 
- **num_rows_per_panel** (<em>int</em>) – Number of rows of elements per panel
- **num_cols_per_panel** (<em>int</em>) – Number of columns of elements per panel
- **polarization** (<em>str</em>) – Polarization, either “single” or “dual”
- **polarization_type** (<em>str</em>) – Type of polarization. For single polarization, must be “V” or “H”.
For dual polarization, must be “VH” or “cross”.
- **antenna_pattern** (<em>str</em>) – Element radiation pattern, either “omni” or “38.901”
- **carrier_frequency** (<em>float</em>) – Carrier frequency [Hz]
- **num_rows** (<em>int</em>) – Number of rows of panels. Defaults to 1.
- **num_cols** (<em>int</em>) – Number of columns of panels. Defaults to 1.
- **panel_vertical_spacing** (<cite>None</cite> or float) – Vertical spacing of panels [multiples of wavelength].
Must be greater than the panel width.
If set to <cite>None</cite> (default value), it is set to the panel width + 0.5.
- **panel_horizontal_spacing** (<cite>None</cite> or float) – Horizontal spacing of panels [in multiples of wavelength].
Must be greater than the panel height.
If set to <cite>None</cite> (default value), it is set to the panel height + 0.5.
- **element_vertical_spacing** (<cite>None</cite> or float) – Element vertical spacing [multiple of wavelength].
Defaults to 0.5 if set to <cite>None</cite>.
- **element_horizontal_spacing** (<cite>None</cite> or float) – Element horizontal spacing [multiple of wavelength].
Defaults to 0.5 if set to <cite>None</cite>.
- **dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`ant_ind_pol1`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.ant_ind_pol1" title="Permalink to this definition"></a>
    
Indices of antenna elements with the first polarization direction


<em class="property">`property` </em>`ant_ind_pol2`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.ant_ind_pol2" title="Permalink to this definition"></a>
    
Indices of antenna elements with the second polarization direction.
Only defined with dual polarization.


<em class="property">`property` </em>`ant_pol1`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.ant_pol1" title="Permalink to this definition"></a>
    
Field of an antenna element with the first polarization direction


<em class="property">`property` </em>`ant_pol2`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.ant_pol2" title="Permalink to this definition"></a>
    
Field of an antenna element with the second polarization direction.
Only defined with dual polarization.


<em class="property">`property` </em>`ant_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.ant_pos" title="Permalink to this definition"></a>
    
Positions of the antennas


<em class="property">`property` </em>`ant_pos_pol1`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.ant_pos_pol1" title="Permalink to this definition"></a>
    
Positions of the antenna elements with the first polarization
direction


<em class="property">`property` </em>`ant_pos_pol2`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.ant_pos_pol2" title="Permalink to this definition"></a>
    
Positions of antenna elements with the second polarization direction.
Only defined with dual polarization.


<em class="property">`property` </em>`element_horizontal_spacing`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.element_horizontal_spacing" title="Permalink to this definition"></a>
    
Horizontal spacing between the antenna elements within a panel
[multiple of wavelength]


<em class="property">`property` </em>`element_vertical_spacing`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.element_vertical_spacing" title="Permalink to this definition"></a>
    
Vertical spacing between the antenna elements within a panel
[multiple of wavelength]


<em class="property">`property` </em>`num_ant`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.num_ant" title="Permalink to this definition"></a>
    
Total number of antenna elements


<em class="property">`property` </em>`num_cols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.num_cols" title="Permalink to this definition"></a>
    
Number of columns of panels


<em class="property">`property` </em>`num_cols_per_panel`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.num_cols_per_panel" title="Permalink to this definition"></a>
    
Number of columns of elements per panel


<em class="property">`property` </em>`num_panels`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.num_panels" title="Permalink to this definition"></a>
    
Number of panels


<em class="property">`property` </em>`num_panels_ant`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.num_panels_ant" title="Permalink to this definition"></a>
    
Number of antenna elements per panel


<em class="property">`property` </em>`num_rows`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.num_rows" title="Permalink to this definition"></a>
    
Number of rows of panels


<em class="property">`property` </em>`num_rows_per_panel`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.num_rows_per_panel" title="Permalink to this definition"></a>
    
Number of rows of elements per panel


<em class="property">`property` </em>`panel_horizontal_spacing`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.panel_horizontal_spacing" title="Permalink to this definition"></a>
    
Horizontal spacing between the panels [multiple of wavelength]


<em class="property">`property` </em>`panel_vertical_spacing`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.panel_vertical_spacing" title="Permalink to this definition"></a>
    
Vertical spacing between the panels [multiple of wavelength]


<em class="property">`property` </em>`polarization`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.polarization" title="Permalink to this definition"></a>
    
Polarization (“single” or “dual”)


<em class="property">`property` </em>`polarization_type`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.polarization_type" title="Permalink to this definition"></a>
    
Polarization type. “V” or “H” for single polarization.
“VH” or “cross” for dual polarization.


`show`()<a class="reference internal" href="../_modules/sionna/channel/tr38901/antenna.html#PanelArray.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.show" title="Permalink to this definition"></a>
    
Show the panel array geometry


`show_element_radiation_pattern`()<a class="reference internal" href="../_modules/sionna/channel/tr38901/antenna.html#PanelArray.show_element_radiation_pattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray.show_element_radiation_pattern" title="Permalink to this definition"></a>
    
Show the radiation field of antenna elements forming the panel


### Antenna<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#antenna" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``Antenna`(<em class="sig-param">`polarization`</em>, <em class="sig-param">`polarization_type`</em>, <em class="sig-param">`antenna_pattern`</em>, <em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/antenna.html#Antenna">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.Antenna" title="Permalink to this definition"></a>
    
Single antenna following the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id6">[TR38901]</a> specification.
    
This class is a special case of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray">`PanelArray`</a>,
and can be used in lieu of it.
Parameters
 
- **polarization** (<em>str</em>) – Polarization, either “single” or “dual”
- **polarization_type** (<em>str</em>) – Type of polarization. For single polarization, must be “V” or “H”.
For dual polarization, must be “VH” or “cross”.
- **antenna_pattern** (<em>str</em>) – Element radiation pattern, either “omni” or “38.901”
- **carrier_frequency** (<em>float</em>) – Carrier frequency [Hz]
- **dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




### AntennaArray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#antennaarray" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``AntennaArray`(<em class="sig-param">`num_rows`</em>, <em class="sig-param">`num_cols`</em>, <em class="sig-param">`polarization`</em>, <em class="sig-param">`polarization_type`</em>, <em class="sig-param">`antenna_pattern`</em>, <em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`vertical_spacing`</em>, <em class="sig-param">`horizontal_spacing`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/antenna.html#AntennaArray">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.AntennaArray" title="Permalink to this definition"></a>
    
Antenna array following the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id7">[TR38901]</a> specification.
    
This class is a special case of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray">`PanelArray`</a>,
and can used in lieu of it.
Parameters
 
- **num_rows** (<em>int</em>) – Number of rows of elements
- **num_cols** (<em>int</em>) – Number of columns of elements
- **polarization** (<em>str</em>) – Polarization, either “single” or “dual”
- **polarization_type** (<em>str</em>) – Type of polarization. For single polarization, must be “V” or “H”.
For dual polarization, must be “VH” or “cross”.
- **antenna_pattern** (<em>str</em>) – Element radiation pattern, either “omni” or “38.901”
- **carrier_frequency** (<em>float</em>) – Carrier frequency [Hz]
- **vertical_spacing** (<cite>None</cite> or float) – Element vertical spacing [multiple of wavelength].
Defaults to 0.5 if set to <cite>None</cite>.
- **horizontal_spacing** (<cite>None</cite> or float) – Element horizontal spacing [multiple of wavelength].
Defaults to 0.5 if set to <cite>None</cite>.
- **dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




