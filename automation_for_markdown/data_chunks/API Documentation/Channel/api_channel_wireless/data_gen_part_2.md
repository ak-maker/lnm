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
## Flat-fading channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#flat-fading-channel" title="Permalink to this headline"></a>
### PerColumnModel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#percolumnmodel" title="Permalink to this headline"></a>
## Channel model interface<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#channel-model-interface" title="Permalink to this headline"></a>
## Time domain channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#time-domain-channel" title="Permalink to this headline"></a>
### TimeChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#timechannel" title="Permalink to this headline"></a>
### GenerateTimeChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#generatetimechannel" title="Permalink to this headline"></a>
  
  

### PerColumnModel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#percolumnmodel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``PerColumnModel`(<em class="sig-param">`r_rx`</em>)<a class="reference internal" href="../_modules/sionna/channel/spatial_correlation.html#PerColumnModel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.PerColumnModel" title="Permalink to this definition"></a>
    
Per-column model for spatial correlation.
    
Given a batch of matrices $\mathbf{H}\in\mathbb{C}^{M\times K}$
and correlation matrices $\mathbf{R}_k\in\mathbb{C}^{M\times M}, k=1,\dots,K$,
this function will generate the output $\mathbf{H}_\text{corr}\in\mathbb{C}^{M\times K}$,
with columns

$$
\mathbf{h}^\text{corr}_k = \mathbf{R}^{\frac12}_k \mathbf{h}_k,\quad k=1, \dots, K
$$
    
where $\mathbf{h}_k$ is the kth column of $\mathbf{H}$.
Note that all $\mathbf{R}_k\in\mathbb{C}^{M\times M}$ must
be positive semi-definite, such as the ones generated
by <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.one_ring_corr_mat" title="sionna.channel.one_ring_corr_mat">`one_ring_corr_mat()`</a>.
    
This model is typically used to simulate a MIMO channel between multiple
single-antenna users and a base station with multiple antennas.
The resulting SIMO channel for each user has a different spatial correlation.
Parameters
    
**r_rx** (<em>[</em><em>...</em><em>, </em><em>M</em><em>, </em><em>M</em><em>]</em><em>, </em><em>tf.complex</em>) – Tensor containing the receive correlation matrices. If
the rank of `r_rx` is smaller than that of the input `h`,
it will be broadcast. For a typically use of this model, `r_rx`
has shape […, K, M, M], i.e., a different correlation matrix for each
column of `h`.

Input
    
**h** (<em>[…, M, K], tf.complex</em>) – Tensor containing spatially uncorrelated
channel coeffficients.

Output
    
**h_corr** (<em>[…, M, K], tf.complex</em>) – Tensor containing the spatially
correlated channel coefficients.



<em class="property">`property` </em>`r_rx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.PerColumnModel.r_rx" title="Permalink to this definition"></a>
    
Tensor containing the receive correlation matrices.

**Note**
    
If you want to set this property in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.


## Channel model interface<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#channel-model-interface" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``ChannelModel`<a class="reference internal" href="../_modules/sionna/channel/channel_model.html#ChannelModel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="Permalink to this definition"></a>
    
Abstract class that defines an interface for channel models.
    
Any channel model which generates channel impulse responses must implement this interface.
All the channel models available in Sionna, such as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="sionna.channel.RayleighBlockFading">`RayleighBlockFading`</a> or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.TDL" title="sionna.channel.tr38901.TDL">`TDL`</a>, implement this interface.
    
<em>Remark:</em> Some channel models only require a subset of the input parameters.
Input
 
- **batch_size** (<em>int</em>) – Batch size
- **num_time_steps** (<em>int</em>) – Number of time steps
- **sampling_frequency** (<em>float</em>) – Sampling frequency [Hz]


Output
 
- **a** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx, num_tx, num_paths], tf.float</em>) – Path delays [s]




## Time domain channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#time-domain-channel" title="Permalink to this headline"></a>
    
The model of the channel in the time domain assumes pulse shaping and receive
filtering are performed using a conventional sinc filter (see, e.g., <a class="reference internal" href="../em_primer.html#tse" id="id3">[Tse]</a>).
Using sinc for transmit and receive filtering, the discrete-time domain received
signal at time step $b$ is

$$
y_{v, l, b} = \sum_{u=0}^{N_{T}-1}\sum_{k=0}^{N_{TA}-1}
   \sum_{\ell = L_{\text{min}}}^{L_{\text{max}}}
   \bar{h}_{u, k, v, l, b, \ell} x_{u, k, b-\ell}
   + w_{v, l, b}
$$
    
where $x_{u, k, b}$ is the baseband symbol transmitted by transmitter
$u$ on antenna $k$ and at time step $b$,
$w_{v, l, b} \sim \mathcal{CN}\left(0,N_0\right)$ the additive white
Gaussian noise, and $\bar{h}_{u, k, v, l, b, \ell}$ the channel filter tap
at time step $b$ and for time-lag $\ell$, which is given by

$$
\bar{h}_{u, k, v, l, b, \ell}
= \sum_{m=0}^{M-1} a_{u, k, v, l, m}\left(\frac{b}{W}\right)
   \text{sinc}\left( \ell - W\tau_{u, v, m} \right).
$$

**Note**
    
The two parameters $L_{\text{min}}$ and $L_{\text{max}}$ control the smallest
and largest time-lag for the discrete-time channel model, respectively.
They are set when instantiating <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.TimeChannel" title="sionna.channel.TimeChannel">`TimeChannel`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateTimeChannel" title="sionna.channel.GenerateTimeChannel">`GenerateTimeChannel`</a>, and when calling the utility
function <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_time_channel" title="sionna.channel.cir_to_time_channel">`cir_to_time_channel()`</a>.
Because the sinc filter is neither time-limited nor causal, the discrete-time
channel model is not causal. Therefore, ideally, one would set
$L_{\text{min}} = -\infty$ and $L_{\text{max}} = +\infty$.
In practice, however, these two parameters need to be set to reasonable
finite values. Values for these two parameters can be computed using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.time_lag_discrete_time_channel" title="sionna.channel.time_lag_discrete_time_channel">`time_lag_discrete_time_channel()`</a> utility function from
a given bandwidth and maximum delay spread.
This function returns $-6$ for $L_{\text{min}}$. $L_{\text{max}}$ is computed
from the specified bandwidth and maximum delay spread, which default value is
$3 \mu s$. These values for $L_{\text{min}}$ and the maximum delay spread
were found to be valid for all the models available in Sionna when an RMS delay
spread of 100ns is assumed.

### TimeChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#timechannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``TimeChannel`(<em class="sig-param">`channel_model`</em>, <em class="sig-param">`bandwidth`</em>, <em class="sig-param">`num_time_samples`</em>, <em class="sig-param">`maximum_delay_spread``=``3e-6`</em>, <em class="sig-param">`l_min``=``None`</em>, <em class="sig-param">`l_max``=``None`</em>, <em class="sig-param">`normalize_channel``=``False`</em>, <em class="sig-param">`add_awgn``=``True`</em>, <em class="sig-param">`return_channel``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/time_channel.html#TimeChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.TimeChannel" title="Permalink to this definition"></a>
    
Generate channel responses and apply them to channel inputs in the time domain.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer
in a Keras model.
    
The channel output consists of `num_time_samples` + `l_max` - `l_min`
time samples, as it is the result of filtering the channel input of length
`num_time_samples` with the time-variant channel filter  of length
`l_max` - `l_min` + 1. In the case of a single-input single-output link and given a sequence of channel
inputs $x_0,\cdots,x_{N_B}$, where $N_B$ is `num_time_samples`, this
layer outputs

$$
y_b = \sum_{\ell = L_{\text{min}}}^{L_{\text{max}}} x_{b-\ell} \bar{h}_{b,\ell} + w_b
$$
    
where $L_{\text{min}}$ corresponds `l_min`, $L_{\text{max}}$ to `l_max`, $w_b$ to
the additive noise, and $\bar{h}_{b,\ell}$ to the
$\ell^{th}$ tap of the $b^{th}$ channel sample.
This layer outputs $y_b$ for $b$ ranging from $L_{\text{min}}$ to
$N_B + L_{\text{max}} - 1$, and $x_{b}$ is set to 0 for $b < 0$ or $b \geq N_B$.
The channel taps $\bar{h}_{b,\ell}$ are computed assuming a sinc filter
is used for pulse shaping and receive filtering. Therefore, given a channel impulse response
$(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1$, generated by the `channel_model`,
the channel taps are computed as follows:

$$
\bar{h}_{b, \ell}
= \sum_{m=0}^{M-1} a_{m}\left(\frac{b}{W}\right)
    \text{sinc}\left( \ell - W\tau_{m} \right)
$$
    
for $\ell$ ranging from `l_min` to `l_max`, and where $W$ is
the `bandwidth`.
    
For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna of each receiver and by summing over all the antennas of all transmitters.
Parameters
 
- **channel_model** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a> object) – An instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a>, such as
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="sionna.channel.RayleighBlockFading">`RayleighBlockFading`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="sionna.channel.tr38901.UMi">`UMi`</a>.
- **bandwidth** (<em>float</em>) – Bandwidth ($W$) [Hz]
- **num_time_samples** (<em>int</em>) – Number of time samples forming the channel input ($N_B$)
- **maximum_delay_spread** (<em>float</em>) – Maximum delay spread [s].
Used to compute the default value of `l_max` if `l_max` is set to
<cite>None</cite>. If a value is given for `l_max`, this parameter is not used.
It defaults to 3us, which was found
to be large enough to include most significant paths with all channel
models included in Sionna assuming a nominal delay spread of 100ns.
- **l_min** (<em>int</em>) – Smallest time-lag for the discrete complex baseband channel ($L_{\text{min}}$).
If set to <cite>None</cite>, defaults to the value given by <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.time_lag_discrete_time_channel" title="sionna.channel.time_lag_discrete_time_channel">`time_lag_discrete_time_channel()`</a>.
- **l_max** (<em>int</em>) – Largest time-lag for the discrete complex baseband channel ($L_{\text{max}}$).
If set to <cite>None</cite>, it is computed from `bandwidth` and `maximum_delay_spread`
using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.time_lag_discrete_time_channel" title="sionna.channel.time_lag_discrete_time_channel">`time_lag_discrete_time_channel()`</a>. If it is not set to <cite>None</cite>,
then the parameter `maximum_delay_spread` is not used.
- **add_awgn** (<em>bool</em>) – If set to <cite>False</cite>, no white Gaussian noise is added.
Defaults to <cite>True</cite>.
- **normalize_channel** (<em>bool</em>) – If set to <cite>True</cite>, the channel is normalized over the block size
to ensure unit average energy per time step. Defaults to <cite>False</cite>.
- **return_channel** (<em>bool</em>) – If set to <cite>True</cite>, the channel response is returned in addition to the
channel output. Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – Complex datatype to use for internal processing and output.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(x, no) or x** – Tuple or Tensor:
- **x** (<em>[batch size, num_tx, num_tx_ant, num_time_samples], tf.complex</em>) – Channel inputs
- **no** (<em>Scalar or Tensor, tf.float</em>) – Scalar or tensor whose shape can be broadcast to the shape of the
channel outputs: [batch size, num_rx, num_rx_ant, num_time_samples].
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
 
- **y** (<em>[batch size, num_rx, num_rx_ant, num_time_samples + l_max - l_min], tf.complex</em>) – Channel outputs
The channel output consists of `num_time_samples` + `l_max` - `l_min`
time samples, as it is the result of filtering the channel input of length
`num_time_samples` with the time-variant channel filter  of length
`l_max` - `l_min` + 1.
- **h_time** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_max - l_min, l_max - l_min + 1], tf.complex</em>) – (Optional) Channel responses. Returned only if `return_channel`
is set to <cite>True</cite>.
For each batch example, `num_time_samples` + `l_max` - `l_min` time
steps of the channel realizations are generated to filter the channel input.




### GenerateTimeChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#generatetimechannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``GenerateTimeChannel`(<em class="sig-param">`channel_model`</em>, <em class="sig-param">`bandwidth`</em>, <em class="sig-param">`num_time_samples`</em>, <em class="sig-param">`l_min`</em>, <em class="sig-param">`l_max`</em>, <em class="sig-param">`normalize_channel``=``False`</em>)<a class="reference internal" href="../_modules/sionna/channel/generate_time_channel.html#GenerateTimeChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateTimeChannel" title="Permalink to this definition"></a>
    
Generate channel responses in the time domain.
    
For each batch example, `num_time_samples` + `l_max` - `l_min` time steps of a
channel realization are generated by this layer.
These can be used to filter a channel input of length `num_time_samples` using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyTimeChannel" title="sionna.channel.ApplyTimeChannel">`ApplyTimeChannel`</a> layer.
    
The channel taps $\bar{h}_{b,\ell}$ (`h_time`) returned by this layer
are computed assuming a sinc filter is used for pulse shaping and receive filtering.
Therefore, given a channel impulse response
$(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1$, generated by the `channel_model`,
the channel taps are computed as follows:

$$
\bar{h}_{b, \ell}
= \sum_{m=0}^{M-1} a_{m}\left(\frac{b}{W}\right)
    \text{sinc}\left( \ell - W\tau_{m} \right)
$$
    
for $\ell$ ranging from `l_min` to `l_max`, and where $W$ is
the `bandwidth`.
Parameters
 
- **channel_model** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a> object) – An instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ChannelModel" title="sionna.channel.ChannelModel">`ChannelModel`</a>, such as
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.RayleighBlockFading" title="sionna.channel.RayleighBlockFading">`RayleighBlockFading`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.UMi" title="sionna.channel.tr38901.UMi">`UMi`</a>.
- **bandwidth** (<em>float</em>) – Bandwidth ($W$) [Hz]
- **num_time_samples** (<em>int</em>) – Number of time samples forming the channel input ($N_B$)
- **l_min** (<em>int</em>) – Smallest time-lag for the discrete complex baseband channel ($L_{\text{min}}$)
- **l_max** (<em>int</em>) – Largest time-lag for the discrete complex baseband channel ($L_{\text{max}}$)
- **normalize_channel** (<em>bool</em>) – If set to <cite>True</cite>, the channel is normalized over the block size
to ensure unit average energy per time step. Defaults to <cite>False</cite>.


Input
    
**batch_size** (<em>int</em>) – Batch size. Defaults to <cite>None</cite> for channel models that do not require this paranmeter.

Output
    
**h_time** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_max - l_min, l_max - l_min + 1], tf.complex</em>) – Channel responses.
For each batch example, `num_time_samples` + `l_max` - `l_min` time steps of a
channel realization are generated by this layer.
These can be used to filter a channel input of length `num_time_samples` using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyTimeChannel" title="sionna.channel.ApplyTimeChannel">`ApplyTimeChannel`</a> layer.



