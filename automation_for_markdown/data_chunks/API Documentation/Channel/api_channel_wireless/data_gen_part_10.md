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
## Utility functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#utility-functions" title="Permalink to this headline"></a>
### deg_2_rad<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#deg-2-rad" title="Permalink to this headline"></a>
### rad_2_deg<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rad-2-deg" title="Permalink to this headline"></a>
### wrap_angle_0_360<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#wrap-angle-0-360" title="Permalink to this headline"></a>
### drop_uts_in_sector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#drop-uts-in-sector" title="Permalink to this headline"></a>
### relocate_uts<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#relocate-uts" title="Permalink to this headline"></a>
### set_3gpp_scenario_parameters<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#set-3gpp-scenario-parameters" title="Permalink to this headline"></a>
### gen_single_sector_topology<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#gen-single-sector-topology" title="Permalink to this headline"></a>
  
  

### deg_2_rad<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#deg-2-rad" title="Permalink to this headline"></a>

`sionna.channel.``deg_2_rad`(<em class="sig-param">`x`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#deg_2_rad">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.deg_2_rad" title="Permalink to this definition"></a>
    
Convert degree to radian
Input
    
**x** (<em>Tensor</em>) – Angles in degree

Output
    
**y** (<em>Tensor</em>) – Angles `x` converted to radian



### rad_2_deg<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rad-2-deg" title="Permalink to this headline"></a>

`sionna.channel.``rad_2_deg`(<em class="sig-param">`x`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#rad_2_deg">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.rad_2_deg" title="Permalink to this definition"></a>
    
Convert radian to degree
Input
    
**x** (<em>Tensor</em>) – Angles in radian

Output
    
**y** (<em>Tensor</em>) – Angles `x` converted to degree



### wrap_angle_0_360<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#wrap-angle-0-360" title="Permalink to this headline"></a>

`sionna.channel.``wrap_angle_0_360`(<em class="sig-param">`angle`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#wrap_angle_0_360">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.wrap_angle_0_360" title="Permalink to this definition"></a>
    
Wrap `angle` to (0,360)
Input
    
**angle** (<em>Tensor</em>) – Input to wrap

Output
    
**y** (<em>Tensor</em>) – `angle` wrapped to (0,360)



### drop_uts_in_sector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#drop-uts-in-sector" title="Permalink to this headline"></a>

`sionna.channel.``drop_uts_in_sector`(<em class="sig-param">`batch_size`</em>, <em class="sig-param">`num_ut`</em>, <em class="sig-param">`min_bs_ut_dist`</em>, <em class="sig-param">`isd`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#drop_uts_in_sector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.drop_uts_in_sector" title="Permalink to this definition"></a>
    
Uniformly sample UT locations from a sector.
    
The sector from which UTs are sampled is shown in the following figure.
The BS is assumed to be located at the origin (0,0) of the coordinate
system.
<a class="reference internal image-reference" href="../_images/drop_uts_in_sector.png"><img alt="../_images/drop_uts_in_sector.png" src="https://nvlabs.github.io/sionna/_images/drop_uts_in_sector.png" style="width: 307.5px; height: 216.9px;" /></a>

Input
 
- **batch_size** (<em>int</em>) – Batch size
- **num_ut** (<em>int</em>) – Number of UTs to sample per batch example
- **min_bs_ut_dist** (<em>tf.float</em>) – Minimum BS-UT distance [m]
- **isd** (<em>tf.float</em>) – Inter-site distance, i.e., the distance between two adjacent BSs [m]
- **dtype** (<em>tf.DType</em>) – Datatype to use for internal processing and output.
If a complex datatype is provided, the corresponding precision of
real components is used.
Defaults to <cite>tf.complex64</cite> (<cite>tf.float32</cite>).


Output
    
**ut_loc** (<em>[batch_size, num_ut, 2], tf.float</em>) – UTs locations in the X-Y plan



### relocate_uts<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#relocate-uts" title="Permalink to this headline"></a>

`sionna.channel.``relocate_uts`(<em class="sig-param">`ut_loc`</em>, <em class="sig-param">`sector_id`</em>, <em class="sig-param">`cell_loc`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#relocate_uts">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.relocate_uts" title="Permalink to this definition"></a>
    
Relocate the UTs by rotating them into the sector with index `sector_id`
and transposing them to the cell centered on `cell_loc`.
    
`sector_id` gives the index of the sector to which the UTs are
rotated to. The picture below shows how the three sectors of a cell are
indexed.
<a class="reference internal image-reference" href="../_images/panel_array_sector_id.png"><img alt="../_images/panel_array_sector_id.png" src="https://nvlabs.github.io/sionna/_images/panel_array_sector_id.png" style="width: 188.1px; height: 162.9px;" /></a>
<p class="caption">Fig. 9 Indexing of sectors<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id32" title="Permalink to this image"></a>
    
If `sector_id` is a scalar, then all UTs are relocated to the same
sector indexed by `sector_id`.
If `sector_id` is a tensor, it should be broadcastable with
[`batch_size`, `num_ut`], and give the sector in which each UT or
batch example is relocated to.
    
When calling the function, `ut_loc` gives the locations of the UTs to
relocate, which are all assumed to be in sector with index 0, and in the
cell centered on the origin (0,0).
Input
 
- **ut_loc** (<em>[batch_size, num_ut, 2], tf.float</em>) – UTs locations in the X-Y plan
- **sector_id** (<em>Tensor broadcastable with [batch_size, num_ut], int</em>) – Indexes of the sector to which to relocate the UTs
- **cell_loc** (<em>Tensor broadcastable with [batch_size, num_ut], tf.float</em>) – Center of the cell to which to transpose the UTs


Output
    
**ut_loc** (<em>[batch_size, num_ut, 2], tf.float</em>) – Relocated UTs locations in the X-Y plan



### set_3gpp_scenario_parameters<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#set-3gpp-scenario-parameters" title="Permalink to this headline"></a>

`sionna.channel.``set_3gpp_scenario_parameters`(<em class="sig-param">`scenario`</em>, <em class="sig-param">`min_bs_ut_dist``=``None`</em>, <em class="sig-param">`isd``=``None`</em>, <em class="sig-param">`bs_height``=``None`</em>, <em class="sig-param">`min_ut_height``=``None`</em>, <em class="sig-param">`max_ut_height``=``None`</em>, <em class="sig-param">`indoor_probability``=``None`</em>, <em class="sig-param">`min_ut_velocity``=``None`</em>, <em class="sig-param">`max_ut_velocity``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#set_3gpp_scenario_parameters">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.set_3gpp_scenario_parameters" title="Permalink to this definition"></a>
    
Set valid parameters for a specified 3GPP system level `scenario`
(RMa, UMi, or UMa).
    
If a parameter is given, then it is returned. If it is set to <cite>None</cite>,
then a parameter valid according to the chosen scenario is returned
(see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id25">[TR38901]</a>).
Input
 
- **scenario** (<em>str</em>) – System level model scenario. Must be one of “rma”, “umi”, or “uma”.
- **min_bs_ut_dist** (<em>None or tf.float</em>) – Minimum BS-UT distance [m]
- **isd** (<em>None or tf.float</em>) – Inter-site distance [m]
- **bs_height** (<em>None or tf.float</em>) – BS elevation [m]
- **min_ut_height** (<em>None or tf.float</em>) – Minimum UT elevation [m]
- **max_ut_height** (<em>None or tf.float</em>) – Maximum UT elevation [m]
- **indoor_probability** (<em>None or tf.float</em>) – Probability of a UT to be indoor
- **min_ut_velocity** (<em>None or tf.float</em>) – Minimum UT velocity [m/s]
- **max_ut_velocity** (<em>None or tf.float</em>) – Maximim UT velocity [m/s]
- **dtype** (<em>tf.DType</em>) – Datatype to use for internal processing and output.
If a complex datatype is provided, the corresponding precision of
real components is used.
Defaults to <cite>tf.complex64</cite> (<cite>tf.float32</cite>).


Output
 
- **min_bs_ut_dist** (<em>tf.float</em>) – Minimum BS-UT distance [m]
- **isd** (<em>tf.float</em>) – Inter-site distance [m]
- **bs_height** (<em>tf.float</em>) – BS elevation [m]
- **min_ut_height** (<em>tf.float</em>) – Minimum UT elevation [m]
- **max_ut_height** (<em>tf.float</em>) – Maximum UT elevation [m]
- **indoor_probability** (<em>tf.float</em>) – Probability of a UT to be indoor
- **min_ut_velocity** (<em>tf.float</em>) – Minimum UT velocity [m/s]
- **max_ut_velocity** (<em>tf.float</em>) – Maximim UT velocity [m/s]




### gen_single_sector_topology<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#gen-single-sector-topology" title="Permalink to this headline"></a>

`sionna.channel.``gen_single_sector_topology`(<em class="sig-param">`batch_size`</em>, <em class="sig-param">`num_ut`</em>, <em class="sig-param">`scenario`</em>, <em class="sig-param">`min_bs_ut_dist``=``None`</em>, <em class="sig-param">`isd``=``None`</em>, <em class="sig-param">`bs_height``=``None`</em>, <em class="sig-param">`min_ut_height``=``None`</em>, <em class="sig-param">`max_ut_height``=``None`</em>, <em class="sig-param">`indoor_probability``=``None`</em>, <em class="sig-param">`min_ut_velocity``=``None`</em>, <em class="sig-param">`max_ut_velocity``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#gen_single_sector_topology">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.gen_single_sector_topology" title="Permalink to this definition"></a>
    
Generate a batch of topologies consisting of a single BS located at the
origin and `num_ut` UTs randomly and uniformly dropped in a cell sector.
    
The following picture shows the sector from which UTs are sampled.
<a class="reference internal image-reference" href="../_images/drop_uts_in_sector.png"><img alt="../_images/drop_uts_in_sector.png" src="https://nvlabs.github.io/sionna/_images/drop_uts_in_sector.png" style="width: 307.5px; height: 216.9px;" /></a>
    
UTs orientations are randomly and uniformly set, whereas the BS orientation
is set such that the it is oriented towards the center of the sector.
    
The drop configuration can be controlled through the optional parameters.
Parameters set to <cite>None</cite> are set to valid values according to the chosen
`scenario` (see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id26">[TR38901]</a>).
    
The returned batch of topologies can be used as-is with the
`set_topology()` method of the system level models, i.e.
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="sionna.channel.tr38901.UMi">`UMi`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMa" title="sionna.channel.tr38901.UMa">`UMa`</a>,
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.RMa" title="sionna.channel.tr38901.RMa">`RMa`</a>.
<p class="rubric">Example
```python
>>> # Create antenna arrays
>>> bs_array = PanelArray(num_rows_per_panel = 4,
...                      num_cols_per_panel = 4,
...                      polarization = 'dual',
...                      polarization_type = 'VH',
...                      antenna_pattern = '38.901',
...                      carrier_frequency = 3.5e9)
>>>
>>> ut_array = PanelArray(num_rows_per_panel = 1,
...                       num_cols_per_panel = 1,
...                       polarization = 'single',
...                       polarization_type = 'V',
...                       antenna_pattern = 'omni',
...                       carrier_frequency = 3.5e9)
>>> # Create channel model
>>> channel_model = UMi(carrier_frequency = 3.5e9,
...                     o2i_model = 'low',
...                     ut_array = ut_array,
...                     bs_array = bs_array,
...                     direction = 'uplink')
>>> # Generate the topology
>>> topology = gen_single_sector_topology(batch_size = 100,
...                                       num_ut = 4,
...                                       scenario = 'umi')
>>> # Set the topology
>>> ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
>>> channel_model.set_topology(ut_loc,
...                            bs_loc,
...                            ut_orientations,
...                            bs_orientations,
...                            ut_velocities,
...                            in_state)
>>> channel_model.show_topology()
```

Input
 
- **batch_size** (<em>int</em>) – Batch size
- **num_ut** (<em>int</em>) – Number of UTs to sample per batch example
- **scenario** (<em>str</em>) – System leven model scenario. Must be one of “rma”, “umi”, or “uma”.
- **min_bs_ut_dist** (<em>None or tf.float</em>) – Minimum BS-UT distance [m]
- **isd** (<em>None or tf.float</em>) – Inter-site distance [m]
- **bs_height** (<em>None or tf.float</em>) – BS elevation [m]
- **min_ut_height** (<em>None or tf.float</em>) – Minimum UT elevation [m]
- **max_ut_height** (<em>None or tf.float</em>) – Maximum UT elevation [m]
- **indoor_probability** (<em>None or tf.float</em>) – Probability of a UT to be indoor
- **min_ut_velocity** (<em>None or tf.float</em>) – Minimum UT velocity [m/s]
- **max_ut_velocity** (<em>None or tf.float</em>) – Maximim UT velocity [m/s]
- **dtype** (<em>tf.DType</em>) – Datatype to use for internal processing and output.
If a complex datatype is provided, the corresponding precision of
real components is used.
Defaults to <cite>tf.complex64</cite> (<cite>tf.float32</cite>).


Output
 
- **ut_loc** (<em>[batch_size, num_ut, 3], tf.float</em>) – UTs locations
- **bs_loc** (<em>[batch_size, 1, 3], tf.float</em>) – BS location. Set to (0,0,0) for all batch examples.
- **ut_orientations** (<em>[batch_size, num_ut, 3], tf.float</em>) – UTs orientations [radian]
- **bs_orientations** (<em>[batch_size, 1, 3], tf.float</em>) – BS orientations [radian]. Oriented towards the center of the sector.
- **ut_velocities** (<em>[batch_size, num_ut, 3], tf.float</em>) – UTs velocities [m/s]
- **in_state** (<em>[batch_size, num_ut], tf.float</em>) – Indoor/outdoor state of UTs. <cite>True</cite> means indoor, <cite>False</cite> means
outdoor.




