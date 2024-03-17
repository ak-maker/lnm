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
### gen_single_sector_topology_interferers<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#gen-single-sector-topology-interferers" title="Permalink to this headline"></a>
### exp_corr_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#exp-corr-mat" title="Permalink to this headline"></a>
### one_ring_corr_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#one-ring-corr-mat" title="Permalink to this headline"></a>
  
  

### gen_single_sector_topology_interferers<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#gen-single-sector-topology-interferers" title="Permalink to this headline"></a>

`sionna.channel.``gen_single_sector_topology_interferers`(<em class="sig-param">`batch_size`</em>, <em class="sig-param">`num_ut`</em>, <em class="sig-param">`num_interferer`</em>, <em class="sig-param">`scenario`</em>, <em class="sig-param">`min_bs_ut_dist``=``None`</em>, <em class="sig-param">`isd``=``None`</em>, <em class="sig-param">`bs_height``=``None`</em>, <em class="sig-param">`min_ut_height``=``None`</em>, <em class="sig-param">`max_ut_height``=``None`</em>, <em class="sig-param">`indoor_probability``=``None`</em>, <em class="sig-param">`min_ut_velocity``=``None`</em>, <em class="sig-param">`max_ut_velocity``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#gen_single_sector_topology_interferers">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.gen_single_sector_topology_interferers" title="Permalink to this definition"></a>
    
Generate a batch of topologies consisting of a single BS located at the
origin, `num_ut` UTs randomly and uniformly dropped in a cell sector, and
`num_interferer` interfering UTs randomly dropped in the adjacent cells.
    
The following picture shows how UTs are sampled
<a class="reference internal image-reference" href="../_images/drop_uts_in_sector_interferers.png"><img alt="../_images/drop_uts_in_sector_interferers.png" src="https://nvlabs.github.io/sionna/_images/drop_uts_in_sector_interferers.png" style="width: 383.4px; height: 383.09999999999997px;" /></a>
    
UTs orientations are randomly and uniformly set, whereas the BS orientation
is set such that it is oriented towards the center of the sector it
serves.
    
The drop configuration can be controlled through the optional parameters.
Parameters set to <cite>None</cite> are set to valid values according to the chosen
`scenario` (see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id27">[TR38901]</a>).
    
The returned batch of topologies can be used as-is with the
`set_topology()` method of the system level models, i.e.
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="sionna.channel.tr38901.UMi">`UMi`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMa" title="sionna.channel.tr38901.UMa">`UMa`</a>,
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.RMa" title="sionna.channel.tr38901.RMa">`RMa`</a>.
    
In the returned `ut_loc`, `ut_orientations`, `ut_velocities`, and
`in_state` tensors, the first `num_ut` items along the axis with index
1 correspond to the served UTs, whereas the remaining `num_interferer`
items correspond to the interfering UTs.
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
>>> topology = gen_single_sector_topology_interferers(batch_size = 100,
...                                                   num_ut = 4,
...                                                   num_interferer = 4,
...                                                   scenario = 'umi')
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
- **num_interferer** (<em>int</em>) – Number of interfeering UTs per batch example
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
 
- **ut_loc** (<em>[batch_size, num_ut, 3], tf.float</em>) – UTs locations. The first `num_ut` items along the axis with index
1 correspond to the served UTs, whereas the remaining
`num_interferer` items correspond to the interfeering UTs.
- **bs_loc** (<em>[batch_size, 1, 3], tf.float</em>) – BS location. Set to (0,0,0) for all batch examples.
- **ut_orientations** (<em>[batch_size, num_ut, 3], tf.float</em>) – UTs orientations [radian]. The first `num_ut` items along the
axis with index 1 correspond to the served UTs, whereas the
remaining `num_interferer` items correspond to the interfeering
UTs.
- **bs_orientations** (<em>[batch_size, 1, 3], tf.float</em>) – BS orientation [radian]. Oriented towards the center of the sector.
- **ut_velocities** (<em>[batch_size, num_ut, 3], tf.float</em>) – UTs velocities [m/s]. The first `num_ut` items along the axis
with index 1 correspond to the served UTs, whereas the remaining
`num_interferer` items correspond to the interfeering UTs.
- **in_state** (<em>[batch_size, num_ut], tf.float</em>) – Indoor/outdoor state of UTs. <cite>True</cite> means indoor, <cite>False</cite> means
outdoor. The first `num_ut` items along the axis with
index 1 correspond to the served UTs, whereas the remaining
`num_interferer` items correspond to the interfeering UTs.




### exp_corr_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#exp-corr-mat" title="Permalink to this headline"></a>

`sionna.channel.``exp_corr_mat`(<em class="sig-param">`a`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#exp_corr_mat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.exp_corr_mat" title="Permalink to this definition"></a>
    
Generate exponential correlation matrices.
    
This function computes for every element $a$ of a complex-valued
tensor $\mathbf{a}$ the corresponding $n\times n$ exponential
correlation matrix $\mathbf{R}(a,n)$, defined as (Eq. 1, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#mal2018" id="id28">[MAL2018]</a>):

$$
\begin{split}\mathbf{R}(a,n)_{i,j} = \begin{cases}
            1 & \text{if } i=j\\
            a^{i-j}  & \text{if } i>j\\
            (a^\star)^{j-i}  & \text{if } j<i, j=1,\dots,n\\
          \end{cases}\end{split}
$$
    
where $|a|<1$ and $\mathbf{R}\in\mathbb{C}^{n\times n}$.
Input
 
- **a** (<em>[n_0, …, n_k], tf.complex</em>) – A tensor of arbitrary rank whose elements
have an absolute value smaller than one.
- **n** (<em>int</em>) – Number of dimensions of the output correlation matrices.
- **dtype** (<em>tf.complex64, tf.complex128</em>) – The dtype of the output.


Output
    
**R** (<em>[n_0, …, n_k, n, n], tf.complex</em>) – A tensor of the same dtype as the input tensor $\mathbf{a}$.



### one_ring_corr_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#one-ring-corr-mat" title="Permalink to this headline"></a>

`sionna.channel.``one_ring_corr_mat`(<em class="sig-param">`phi_deg`</em>, <em class="sig-param">`num_ant`</em>, <em class="sig-param">`d_h``=``0.5`</em>, <em class="sig-param">`sigma_phi_deg``=``15`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#one_ring_corr_mat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.one_ring_corr_mat" title="Permalink to this definition"></a>
    
Generate covariance matrices from the one-ring model.
    
This function generates approximate covariance matrices for the
so-called <cite>one-ring</cite> model (Eq. 2.24) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#bhs2017" id="id29">[BHS2017]</a>. A uniform
linear array (ULA) with uniform antenna spacing is assumed. The elements
of the covariance matrices are computed as:

$$
\mathbf{R}_{\ell,m} =
      \exp\left( j2\pi d_\text{H} (\ell -m)\sin(\varphi) \right)
      \exp\left( -\frac{\sigma_\varphi^2}{2}
      \left( 2\pi d_\text{H}(\ell -m)\cos(\varphi) \right)^2 \right)
$$
    
for $\ell,m = 1,\dots, M$, where $M$ is the number of antennas,
$\varphi$ is the angle of arrival, $d_\text{H}$ is the antenna
spacing in multiples of the wavelength,
and $\sigma^2_\varphi$ is the angular standard deviation.
Input
 
- **phi_deg** (<em>[n_0, …, n_k], tf.float</em>) – A tensor of arbitrary rank containing azimuth angles (deg) of arrival.
- **num_ant** (<em>int</em>) – Number of antennas
- **d_h** (<em>float</em>) – Antenna spacing in multiples of the wavelength. Defaults to 0.5.
- **sigma_phi_deg** (<em>float</em>) – Angular standard deviation (deg). Defaults to 15 (deg). Values greater
than 15 should not be used as the approximation becomes invalid.
- **dtype** (<em>tf.complex64, tf.complex128</em>) – The dtype of the output.


Output
    
**R** ([n_0, …, n_k, num_ant, nun_ant], <cite>dtype</cite>) – Tensor containing the covariance matrices of the desired dtype.




References:
TR38901(<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id2">2</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id4">3</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id5">4</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id6">5</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id7">6</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id8">7</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id11">8</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id12">9</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id13">10</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id14">11</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id15">12</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id16">13</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id17">14</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id18">15</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id19">16</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id20">17</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id21">18</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id25">19</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id26">20</a>,<a href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id27">21</a>)
    
3GPP TR 38.901,
“Study on channel model for frequencies from 0.5 to 100 GHz”, Release 16.1

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id10">TS38141-1</a>
    
3GPP TS 38.141-1
“Base Station (BS) conformance testing Part 1: Conducted conformance testing”,
Release 17

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id3">Tse</a>
    
D. Tse and P. Viswanath, “Fundamentals of wireless communication“,
Cambridge university press, 2005.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id9">SoS</a>
<ol class="upperalpha simple" start="3">
- Xiao, Y. R. Zheng and N. C. Beaulieu, “Novel Sum-of-Sinusoids Simulation Models for Rayleigh and Rician Fading Channels,” in IEEE Transactions on Wireless Communications, vol. 5, no. 12, pp. 3667-3679, December 2006, doi: 10.1109/TWC.2006.256990.
</ol>

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id28">MAL2018</a>
    
Ranjan K. Mallik,
“The exponential correlation matrix: Eigen-analysis and
applications”, IEEE Trans. Wireless Commun., vol. 17, no. 7,
pp. 4690-4705, Jul. 2018.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#id29">BHS2017</a>
    
Emil Björnson, Jakob Hoydis and Luca Sanguinetti (2017),
<a class="reference external" href="https://massivemimobook.com">“Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency”</a>,
Foundations and Trends in Signal Processing:
Vol. 11, No. 3-4, pp 154–655.



