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
## Channel with OFDM waveform<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#channel-with-ofdm-waveform" title="Permalink to this headline"></a>
### OFDMChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#ofdmchannel" title="Permalink to this headline"></a>
### GenerateOFDMChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#generateofdmchannel" title="Permalink to this headline"></a>
### ApplyOFDMChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#applyofdmchannel" title="Permalink to this headline"></a>
### cir_to_ofdm_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#cir-to-ofdm-channel" title="Permalink to this headline"></a>
## Rayleigh block fading<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rayleigh-block-fading" title="Permalink to this headline"></a>
## 3GPP 38.901 channel models<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#gpp-38-901-channel-models" title="Permalink to this headline"></a>
  
  

### OFDMChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#ofdmchannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``OFDMChannel`(<em class="sig-param">`channel_model`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`add_awgn``=``True`</em>, <em class="sig-param">`normalize_channel``=``False`</em>, <em class="sig-param">`return_channel``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/ofdm_channel.html#OFDMChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.OFDMChannel" title="Permalink to this definition"></a>
    
Generate channel frequency responses and apply them to channel inputs
assuming an OFDM waveform with no ICI nor ISI.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer
in a Keras model.
    
For each OFDM symbol $s$ and subcarrier $n$, the channel output is computed as follows:

$$
y_{s,n} = \widehat{h}_{s, n} x_{s,n} + w_{s,n}
$$
    
where $y_{s,n}$ is the channel output computed by this layer,
$\widehat{h}_{s, n}$ the frequency channel response,
$x_{s,n}$ the channel input `x`, and $w_{s,n}$ the additive noise.
    
For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna
of each receiver and by summing over all the antennas of all transmitters.
    
The channel frequency response for the $s^{th}$ OFDM symbol and
$n^{th}$ subcarrier is computed from a given channel impulse response
$(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1$ generated by the `channel_model`
as follows:

$$
\widehat{h}_{s, n} = \sum_{m=0}^{M-1} a_{m}(s) e^{-j2\pi n \Delta_f \tau_{m}}
$$
    
where $\Delta_f$ is the subcarrier spacing, and $s$ is used as time
step to indicate that the channel impulse response can change from one OFDM symbol to the
next in the event of mobility, even if it is assumed static over the duration
of an OFDM symbol.
Parameters
 
- **channel_model** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a> object) – An instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a> object, such as
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="sionna.channel.RayleighBlockFading">`RayleighBlockFading`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="sionna.channel.tr38901.UMi">`UMi`</a>.
- **resource_grid** (<a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>) – Resource grid
- **add_awgn** (<em>bool</em>) – If set to <cite>False</cite>, no white Gaussian noise is added.
Defaults to <cite>True</cite>.
- **normalize_channel** (<em>bool</em>) – If set to <cite>True</cite>, the channel is normalized over the resource grid
to ensure unit average energy per resource element. Defaults to <cite>False</cite>.
- **return_channel** (<em>bool</em>) – If set to <cite>True</cite>, the channel response is returned in addition to the
channel output. Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – Complex datatype to use for internal processing and output.
Defaults to tf.complex64.


Input
 
- **(x, no) or x** – Tuple or Tensor:
- **x** (<em>[batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel inputs
- **no** (<em>Scalar or Tensor, tf.float</em>) – Scalar or tensor whose shape can be broadcast to the shape of the
channel outputs:
[batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size].
Only required if `add_awgn` is set to <cite>True</cite>.
The noise power `no` is per complex dimension. If `no` is a scalar,
noise of the same variance will be added to the outputs.
If `no` is a tensor, it must have a shape that can be broadcast to
the shape of the channel outputs. This allows, e.g., adding noise of
different variance to each example in a batch. If `no` has a lower
rank than the channel outputs, then `no` will be broadcast to the
shape of the channel outputs by adding dummy dimensions after the last
axis.


Output
 
- **y** (<em>[batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel outputs
- **h_freq** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – (Optional) Channel frequency responses. Returned only if
`return_channel` is set to <cite>True</cite>.




### GenerateOFDMChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#generateofdmchannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``GenerateOFDMChannel`(<em class="sig-param">`channel_model`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`normalize_channel``=``False`</em>)<a class="reference internal" href="../_modules/sionna/channel/generate_ofdm_channel.html#GenerateOFDMChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateOFDMChannel" title="Permalink to this definition"></a>
    
Generate channel frequency responses.
The channel impulse response is constant over the duration of an OFDM symbol.
    
Given a channel impulse response
$(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1$, generated by the `channel_model`,
the channel frequency response for the $s^{th}$ OFDM symbol and
$n^{th}$ subcarrier is computed as follows:

$$
\widehat{h}_{s, n} = \sum_{m=0}^{M-1} a_{m}(s) e^{-j2\pi n \Delta_f \tau_{m}}
$$
    
where $\Delta_f$ is the subcarrier spacing, and $s$ is used as time
step to indicate that the channel impulse response can change from one OFDM symbol to the
next in the event of mobility, even if it is assumed static over the duration
of an OFDM symbol.
Parameters
 
- **channel_model** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a> object) – An instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a> object, such as
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="sionna.channel.RayleighBlockFading">`RayleighBlockFading`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="sionna.channel.tr38901.UMi">`UMi`</a>.
- **resource_grid** (<a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>) – Resource grid
- **normalize_channel** (<em>bool</em>) – If set to <cite>True</cite>, the channel is normalized over the resource grid
to ensure unit average energy per resource element. Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – Complex datatype to use for internal processing and output.
Defaults to <cite>tf.complex64</cite>.


Input
    
**batch_size** (<em>int</em>) – Batch size. Defaults to <cite>None</cite> for channel models that do not require this paranmeter.

Output
    
**h_freq** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex</em>) – Channel frequency responses



### ApplyOFDMChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#applyofdmchannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``ApplyOFDMChannel`(<em class="sig-param">`add_awgn``=``True`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/apply_ofdm_channel.html#ApplyOFDMChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyOFDMChannel" title="Permalink to this definition"></a>
    
Apply single-tap channel frequency responses to channel inputs.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer
in a Keras model.
    
For each OFDM symbol $s$ and subcarrier $n$, the single-tap channel
is applied as follows:

$$
y_{s,n} = \widehat{h}_{s, n} x_{s,n} + w_{s,n}
$$
    
where $y_{s,n}$ is the channel output computed by this layer,
$\widehat{h}_{s, n}$ the frequency channel response (`h_freq`),
$x_{s,n}$ the channel input `x`, and $w_{s,n}$ the additive noise.
    
For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna
of each receiver and by summing over all the antennas of all transmitters.
Parameters
 
- **add_awgn** (<em>bool</em>) – If set to <cite>False</cite>, no white Gaussian noise is added.
Defaults to <cite>True</cite>.
- **dtype** (<em>tf.DType</em>) – Complex datatype to use for internal processing and output. Defaults to
<cite>tf.complex64</cite>.


Input
 
- **(x, h_freq, no) or (x, h_freq)** – Tuple:
- **x** (<em>[batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel inputs
- **h_freq** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel frequency responses
- **no** (<em>Scalar or Tensor, tf.float</em>) – Scalar or tensor whose shape can be broadcast to the shape of the
channel outputs:
[batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size].
Only required if `add_awgn` is set to <cite>True</cite>.
The noise power `no` is per complex dimension. If `no` is a
scalar, noise of the same variance will be added to the outputs.
If `no` is a tensor, it must have a shape that can be broadcast to
the shape of the channel outputs. This allows, e.g., adding noise of
different variance to each example in a batch. If `no` has a lower
rank than the channel outputs, then `no` will be broadcast to the
shape of the channel outputs by adding dummy dimensions after the
last axis.


Output
    
**y** (<em>[batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel outputs



### cir_to_ofdm_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#cir-to-ofdm-channel" title="Permalink to this headline"></a>

`sionna.channel.``cir_to_ofdm_channel`(<em class="sig-param">`frequencies`</em>, <em class="sig-param">`a`</em>, <em class="sig-param">`tau`</em>, <em class="sig-param">`normalize``=``False`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#cir_to_ofdm_channel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_ofdm_channel" title="Permalink to this definition"></a>
    
Compute the frequency response of the channel at `frequencies`.
    
Given a channel impulse response
$(a_{m}, \tau_{m}), 0 \leq m \leq M-1$ (inputs `a` and `tau`),
the channel frequency response for the frequency $f$
is computed as follows:

$$
\widehat{h}(f) = \sum_{m=0}^{M-1} a_{m} e^{-j2\pi f \tau_{m}}
$$

Input
 
- **frequencies** (<em>[fft_size], tf.float</em>) – Frequencies at which to compute the channel response
- **a** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], tf.float</em>) – Path delays
- **normalize** (<em>bool</em>) – If set to <cite>True</cite>, the channel is normalized over the resource grid
to ensure unit average energy per resource element. Defaults to <cite>False</cite>.


Output
    
**h_f** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], tf.complex</em>) – Channel frequency responses at `frequencies`



## Rayleigh block fading<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rayleigh-block-fading" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``RayleighBlockFading`(<em class="sig-param">`num_rx`</em>, <em class="sig-param">`num_rx_ant`</em>, <em class="sig-param">`num_tx`</em>, <em class="sig-param">`num_tx_ant`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/channel/rayleigh_block_fading.html#RayleighBlockFading">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="Permalink to this definition"></a>
    
Generate channel impulse responses corresponding to a Rayleigh block
fading channel model.
    
The channel impulse responses generated are formed of a single path with
zero delay and a normally distributed fading coefficient.
All time steps of a batch example share the same channel coefficient
(block fading).
    
This class can be used in conjunction with the classes that simulate the
channel response in time or frequency domain, i.e.,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.OFDMChannel" title="sionna.channel.OFDMChannel">`OFDMChannel`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.TimeChannel" title="sionna.channel.TimeChannel">`TimeChannel`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateOFDMChannel" title="sionna.channel.GenerateOFDMChannel">`GenerateOFDMChannel`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyOFDMChannel" title="sionna.channel.ApplyOFDMChannel">`ApplyOFDMChannel`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateTimeChannel" title="sionna.channel.GenerateTimeChannel">`GenerateTimeChannel`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyTimeChannel" title="sionna.channel.ApplyTimeChannel">`ApplyTimeChannel`</a>.
Parameters
 
- **num_rx** (<em>int</em>) – Number of receivers ($N_R$)
- **num_rx_ant** (<em>int</em>) – Number of antennas per receiver ($N_{RA}$)
- **num_tx** (<em>int</em>) – Number of transmitters ($N_T$)
- **num_tx_ant** (<em>int</em>) – Number of antennas per transmitter ($N_{TA}$)
- **dtype** (<em>tf.DType</em>) – Complex datatype to use for internal processing and output.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **batch_size** (<em>int</em>) – Batch size
- **num_time_steps** (<em>int</em>) – Number of time steps


Output
 
- **a** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths = 1, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx, num_tx, num_paths = 1], tf.float</em>) – Path delays [s]




## 3GPP 38.901 channel models<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#gpp-38-901-channel-models" title="Permalink to this headline"></a>
    
The submodule `tr38901` implements 3GPP channel models from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#tr38901" id="id4">[TR38901]</a>.
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#cdl">CDL</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#umi">UMi</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#uma">UMa</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rma">RMa</a>
models require setting-up antenna models for the transmitters and
receivers. This is achieved using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.PanelArray" title="sionna.channel.tr38901.PanelArray">`PanelArray`</a> class.
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#umi">UMi</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#uma">UMa</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#rma">RMa</a> models require
setting-up a network topology, specifying, e.g., the user terminals (UTs) and
base stations (BSs) locations, the UTs velocities, etc.
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#utility-functions">Utility functions</a> are available to help laying out
complex topologies or to quickly setup simple but widely used topologies.

