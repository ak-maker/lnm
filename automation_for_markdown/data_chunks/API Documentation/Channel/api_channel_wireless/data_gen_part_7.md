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
### Clustered delay line (CDL)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#clustered-delay-line-cdl" title="Permalink to this headline"></a>
  
  

### Clustered delay line (CDL)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#clustered-delay-line-cdl" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.tr38901.``CDL`(<em class="sig-param">`model`</em>, <em class="sig-param">`delay_spread`</em>, <em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`ut_array`</em>, <em class="sig-param">`bs_array`</em>, <em class="sig-param">`direction`</em>, <em class="sig-param">`min_speed``=``0.`</em>, <em class="sig-param">`max_speed``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/tr38901/cdl.html#CDL">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.CDL" title="Permalink to this definition"></a>
    
Clustered delay line (CDL) channel model from the 3GPP <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id12">[TR38901]</a> specification.
    
The power delay profiles (PDPs) are normalized to have a total energy of one.
    
If a minimum speed and a maximum speed are specified such that the
maximum speed is greater than the minimum speed, then UTs speeds are
randomly and uniformly sampled from the specified interval for each link
and each batch example.
    
The CDL model only works for systems with a single transmitter and a single
receiver. The transmitter and receiver can be equipped with multiple
antennas.
<p class="rubric">Example
    
The following code snippet shows how to setup a CDL channel model assuming
an OFDM waveform:
```python
>>> # Panel array configuration for the transmitter and receiver
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
>>> # CDL channel model
>>> cdl = CDL(model = "A",
>>>           delay_spread = 300e-9,
...           carrier_frequency = 3.5e9,
...           ut_array = ut_array,
...           bs_array = bs_array,
...           direction = 'uplink')
>>> channel = OFDMChannel(channel_model = cdl,
...                       resource_grid = rg)
```

    
where `rg` is an instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
<p class="rubric">Notes
    
The following tables from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id13">[TR38901]</a> provide typical values for the delay
spread.
<table class="docutils align-default">
<colgroup>
<col style="width: 58%" />
<col style="width: 42%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head">    
Model</th>
<th class="head">    
Delay spread [ns]</th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td>    
Very short delay spread</td>
<td>    
$10$</td>
</tr>
<tr class="row-odd"><td>    
Short short delay spread</td>
<td>    
$10$</td>
</tr>
<tr class="row-even"><td>    
Nominal delay spread</td>
<td>    
$100$</td>
</tr>
<tr class="row-odd"><td>    
Long delay spread</td>
<td>    
$300$</td>
</tr>
<tr class="row-even"><td>    
Very long delay spread</td>
<td>    
$1000$</td>
</tr>
</tbody>
</table>
<table class="docutils align-default">
<colgroup>
<col style="width: 30%" />
<col style="width: 27%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 5%" />
<col style="width: 6%" />
<col style="width: 6%" />
<col style="width: 5%" />
<col style="width: 6%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head" colspan="2" rowspan="2">    
Delay spread [ns]</th>
<th class="head" colspan="7">    
Frequency [GHz]</th>
</tr>
<tr class="row-even"><th class="head">    
2</th>
<th class="head">    
6</th>
<th class="head">    
15</th>
<th class="head">    
28</th>
<th class="head">    
39</th>
<th class="head">    
60</th>
<th class="head">    
70</th>
</tr>
</thead>
<tbody>
<tr class="row-odd"><td rowspan="3">    
Indoor office</td>
<td>    
Short delay profile</td>
<td>    
20</td>
<td>    
16</td>
<td>    
16</td>
<td>    
16</td>
<td>    
16</td>
<td>    
16</td>
<td>    
16</td>
</tr>
<tr class="row-even"><td>    
Normal delay profile</td>
<td>    
39</td>
<td>    
30</td>
<td>    
24</td>
<td>    
20</td>
<td>    
18</td>
<td>    
16</td>
<td>    
16</td>
</tr>
<tr class="row-odd"><td>    
Long delay profile</td>
<td>    
59</td>
<td>    
53</td>
<td>    
47</td>
<td>    
43</td>
<td>    
41</td>
<td>    
38</td>
<td>    
37</td>
</tr>
<tr class="row-even"><td rowspan="3">    
UMi Street-canyon</td>
<td>    
Short delay profile</td>
<td>    
65</td>
<td>    
45</td>
<td>    
37</td>
<td>    
32</td>
<td>    
30</td>
<td>    
27</td>
<td>    
26</td>
</tr>
<tr class="row-odd"><td>    
Normal delay profile</td>
<td>    
129</td>
<td>    
93</td>
<td>    
76</td>
<td>    
66</td>
<td>    
61</td>
<td>    
55</td>
<td>    
53</td>
</tr>
<tr class="row-even"><td>    
Long delay profile</td>
<td>    
634</td>
<td>    
316</td>
<td>    
307</td>
<td>    
301</td>
<td>    
297</td>
<td>    
293</td>
<td>    
291</td>
</tr>
<tr class="row-odd"><td rowspan="3">    
UMa</td>
<td>    
Short delay profile</td>
<td>    
93</td>
<td>    
93</td>
<td>    
85</td>
<td>    
80</td>
<td>    
78</td>
<td>    
75</td>
<td>    
74</td>
</tr>
<tr class="row-even"><td>    
Normal delay profile</td>
<td>    
363</td>
<td>    
363</td>
<td>    
302</td>
<td>    
266</td>
<td>    
249</td>
<td>    
228</td>
<td>    
221</td>
</tr>
<tr class="row-odd"><td>    
Long delay profile</td>
<td>    
1148</td>
<td>    
1148</td>
<td>    
955</td>
<td>    
841</td>
<td>    
786</td>
<td>    
720</td>
<td>    
698</td>
</tr>
<tr class="row-even"><td rowspan="3">    
RMa / RMa O2I</td>
<td>    
Short delay profile</td>
<td>    
32</td>
<td>    
32</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
</tr>
<tr class="row-odd"><td>    
Normal delay profile</td>
<td>    
37</td>
<td>    
37</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
</tr>
<tr class="row-even"><td>    
Long delay profile</td>
<td>    
153</td>
<td>    
153</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
<td>    
N/A</td>
</tr>
<tr class="row-odd"><td rowspan="2">    
UMi / UMa O2I</td>
<td>    
Normal delay profile</td>
<td colspan="7">    
242</td>
</tr>
<tr class="row-even"><td>    
Long delay profile</td>
<td colspan="7">    
616</td>
</tr>
</tbody>
</table>
Parameters
 
- **model** (<em>str</em>) – CDL model to use. Must be one of “A”, “B”, “C”, “D” or “E”.
- **delay_spread** (<em>float</em>) – RMS delay spread [s].
- **carrier_frequency** (<em>float</em>) – Carrier frequency [Hz].
- **ut_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the UTs. All UTs share the same antenna array
configuration.
- **bs_array** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray"><em>PanelArray</em></a>) – Panel array used by the Bs. All BSs share the same antenna array
configuration.
- **direction** (<em>str</em>) – Link direction. Must be either “uplink” or “downlink”.
- **ut_orientation** (<cite>None</cite> or Tensor of shape [3], tf.float) – Orientation of the UT. If set to <cite>None</cite>, [$\pi$, 0, 0] is used.
Defaults to <cite>None</cite>.
- **bs_orientation** (<cite>None</cite> or Tensor of shape [3], tf.float) – Orientation of the BS. If set to <cite>None</cite>, [0, 0, 0] is used.
Defaults to <cite>None</cite>.
- **min_speed** (<em>float</em>) – Minimum speed [m/s]. Defaults to 0.
- **max_speed** (<em>None</em><em> or </em><em>float</em>) – Maximum speed [m/s]. If set to <cite>None</cite>,
then `max_speed` takes the same value as `min_speed`.
Defaults to <cite>None</cite>.
- **dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.


Input
 
- **batch_size** (<em>int</em>) – Batch size
- **num_time_steps** (<em>int</em>) – Number of time steps
- **sampling_frequency** (<em>float</em>) – Sampling frequency [Hz]


Output
 
- **a** (<em>[batch size, num_rx = 1, num_rx_ant, num_tx = 1, num_tx_ant, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx = 1, num_tx = 1, num_paths], tf.float</em>) – Path delays [s]




<em class="property">`property` </em>`delay_spread`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.CDL.delay_spread" title="Permalink to this definition"></a>
    
RMS delay spread [s]


<em class="property">`property` </em>`delays`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.CDL.delays" title="Permalink to this definition"></a>
    
Path delays [s]


<em class="property">`property` </em>`k_factor`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.CDL.k_factor" title="Permalink to this definition"></a>
    
K-factor in linear scale. Only available with LoS models.


<em class="property">`property` </em>`los`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.CDL.los" title="Permalink to this definition"></a>
    
<cite>True</cite> is this is a LoS model. <cite>False</cite> otherwise.


<em class="property">`property` </em>`num_clusters`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.CDL.num_clusters" title="Permalink to this definition"></a>
    
Number of paths ($M$)


<em class="property">`property` </em>`powers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.CDL.powers" title="Permalink to this definition"></a>
    
Path powers in linear scale


