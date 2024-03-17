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
## Time domain channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#time-domain-channel" title="Permalink to this headline"></a>
### ApplyTimeChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#applytimechannel" title="Permalink to this headline"></a>
### cir_to_time_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#cir-to-time-channel" title="Permalink to this headline"></a>
### time_to_ofdm_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#time-to-ofdm-channel" title="Permalink to this headline"></a>
## Channel with OFDM waveform<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#channel-with-ofdm-waveform" title="Permalink to this headline"></a>
  
  

### ApplyTimeChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#applytimechannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``ApplyTimeChannel`(<em class="sig-param">`num_time_samples`</em>, <em class="sig-param">`l_tot`</em>, <em class="sig-param">`add_awgn``=``True`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/apply_time_channel.html#ApplyTimeChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyTimeChannel" title="Permalink to this definition"></a>
    
Apply time domain channel responses `h_time` to channel inputs `x`,
by filtering the channel inputs with time-variant channel responses.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer
in a Keras model.
    
For each batch example, `num_time_samples` + `l_tot` - 1 time steps of a
channel realization are required to filter the channel inputs.
    
The channel output consists of `num_time_samples` + `l_tot` - 1
time samples, as it is the result of filtering the channel input of length
`num_time_samples` with the time-variant channel filter  of length
`l_tot`. In the case of a single-input single-output link and given a sequence of channel
inputs $x_0,\cdots,x_{N_B}$, where $N_B$ is `num_time_samples`, this
layer outputs

$$
y_b = \sum_{\ell = 0}^{L_{\text{tot}}} x_{b-\ell} \bar{h}_{b,\ell} + w_b
$$
    
where $L_{\text{tot}}$ corresponds `l_tot`, $w_b$ to the additive noise, and
$\bar{h}_{b,\ell}$ to the $\ell^{th}$ tap of the $b^{th}$ channel sample.
This layer outputs $y_b$ for $b$ ranging from 0 to
$N_B + L_{\text{tot}} - 1$, and $x_{b}$ is set to 0 for $b \geq N_B$.
    
For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna
of each receiver and by summing over all the antennas of all transmitters.
Parameters
 
- **num_time_samples** (<em>int</em>) – Number of time samples forming the channel input ($N_B$)
- **l_tot** (<em>int</em>) – Length of the channel filter ($L_{\text{tot}} = L_{\text{max}} - L_{\text{min}} + 1$)
- **add_awgn** (<em>bool</em>) – If set to <cite>False</cite>, no white Gaussian noise is added.
Defaults to <cite>True</cite>.
- **dtype** (<em>tf.DType</em>) – Complex datatype to use for internal processing and output.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(x, h_time, no) or (x, h_time)** – Tuple:
- **x** (<em>[batch size, num_tx, num_tx_ant, num_time_samples], tf.complex</em>) – Channel inputs
- **h_time** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_tot - 1, l_tot], tf.complex</em>) – Channel responses.
For each batch example, `num_time_samples` + `l_tot` - 1 time steps of a
channel realization are required to filter the channel inputs.
- **no** (<em>Scalar or Tensor, tf.float</em>) – Scalar or tensor whose shape can be broadcast to the shape of the channel outputs: [batch size, num_rx, num_rx_ant, num_time_samples + l_tot - 1].
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
    
**y** (<em>[batch size, num_rx, num_rx_ant, num_time_samples + l_tot - 1], tf.complex</em>) – Channel outputs.
The channel output consists of `num_time_samples` + `l_tot` - 1
time samples, as it is the result of filtering the channel input of length
`num_time_samples` with the time-variant channel filter  of length
`l_tot`.



### cir_to_time_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#cir-to-time-channel" title="Permalink to this headline"></a>

`sionna.channel.``cir_to_time_channel`(<em class="sig-param">`bandwidth`</em>, <em class="sig-param">`a`</em>, <em class="sig-param">`tau`</em>, <em class="sig-param">`l_min`</em>, <em class="sig-param">`l_max`</em>, <em class="sig-param">`normalize``=``False`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#cir_to_time_channel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.cir_to_time_channel" title="Permalink to this definition"></a>
    
Compute the channel taps forming the discrete complex-baseband
representation of the channel from the channel impulse response
(`a`, `tau`).
    
This function assumes that a sinc filter is used for pulse shaping and receive
filtering. Therefore, given a channel impulse response
$(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1$, the channel taps
are computed as follows:

$$
\bar{h}_{b, \ell}
= \sum_{m=0}^{M-1} a_{m}\left(\frac{b}{W}\right)
    \text{sinc}\left( \ell - W\tau_{m} \right)
$$
    
for $\ell$ ranging from `l_min` to `l_max`, and where $W$ is
the `bandwidth`.
Input
 
- **bandwidth** (<em>float</em>) – Bandwidth [Hz]
- **a** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], tf.float</em>) – Path delays [s]
- **l_min** (<em>int</em>) – Smallest time-lag for the discrete complex baseband channel ($L_{\text{min}}$)
- **l_max** (<em>int</em>) – Largest time-lag for the discrete complex baseband channel ($L_{\text{max}}$)
- **normalize** (<em>bool</em>) – If set to <cite>True</cite>, the channel is normalized over the block size
to ensure unit average energy per time step. Defaults to <cite>False</cite>.


Output
    
**hm** (<em>[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1], tf.complex</em>) – Channel taps coefficients



### time_to_ofdm_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#time-to-ofdm-channel" title="Permalink to this headline"></a>

`sionna.channel.``time_to_ofdm_channel`(<em class="sig-param">`h_t`</em>, <em class="sig-param">`rg`</em>, <em class="sig-param">`l_min`</em>)<a class="reference internal" href="../_modules/sionna/channel/utils.html#time_to_ofdm_channel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.time_to_ofdm_channel" title="Permalink to this definition"></a>
    
Compute the channel frequency response from the discrete complex-baseband
channel impulse response.
    
Given a discrete complex-baseband channel impulse response
$\bar{h}_{b,\ell}$, for $\ell$ ranging from $L_\text{min}\le 0$
to $L_\text{max}$, the discrete channel frequency response is computed as

$$
\hat{h}_{b,n} = \sum_{k=0}^{L_\text{max}} \bar{h}_{b,k} e^{-j \frac{2\pi kn}{N}} + \sum_{k=L_\text{min}}^{-1} \bar{h}_{b,k} e^{-j \frac{2\pi n(N+k)}{N}}, \quad n=0,\dots,N-1
$$
    
where $N$ is the FFT size and $b$ is the time step.
    
This function only produces one channel frequency response per OFDM symbol, i.e.,
only values of $b$ corresponding to the start of an OFDM symbol (after
cyclic prefix removal) are considered.
Input
 
- **h_t** (<em>[…num_time_steps,l_max-l_min+1], tf.complex</em>) – Tensor of discrete complex-baseband channel impulse responses
- **resource_grid** (<a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>) – Resource grid
- **l_min** (<em>int</em>) – Smallest time-lag for the discrete complex baseband
channel impulse response ($L_{\text{min}}$)


Output
    
**h_f** (<em>[…,num_ofdm_symbols,fft_size], tf.complex</em>) – Tensor of discrete complex-baseband channel frequency responses



**Note**
    
Note that the result of this function is generally different from the
output of `cir_to_ofdm_channel()` because
the discrete complex-baseband channel impulse response is truncated
(see `cir_to_time_channel()`). This effect
can be observed in the example below.
<p class="rubric">Examples
```python
# Setup resource grid and channel model
tf.random.set_seed(4)
sm = StreamManagement(np.array([[1]]), 1)
rg = ResourceGrid(num_ofdm_symbols=1,
                  fft_size=1024,
                  subcarrier_spacing=15e3)
tdl = TDL("A", 100e-9, 3.5e9)
# Generate CIR
cir = tdl(batch_size=1, num_time_steps=1, sampling_frequency=rg.bandwidth)
# Generate OFDM channel from CIR
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
h_freq = tf.squeeze(cir_to_ofdm_channel(frequencies, *cir, normalize=True))
# Generate time channel from CIR
l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min=l_min, l_max=l_max, normalize=True)
# Generate OFDM channel from time channel
h_freq_hat = tf.squeeze(time_to_ofdm_channel(h_time, rg, l_min))
# Visualize results
plt.figure()
plt.plot(np.real(h_freq), "-")
plt.plot(np.real(h_freq_hat), "--")
plt.plot(np.imag(h_freq), "-")
plt.plot(np.imag(h_freq_hat), "--")
plt.xlabel("Subcarrier index")
plt.ylabel(r"Channel frequency response")
plt.legend(["OFDM Channel (real)", "OFDM Channel from time (real)", "OFDM Channel (imag)", "OFDM Channel from time (imag)"])
```


## Channel with OFDM waveform<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#channel-with-ofdm-waveform" title="Permalink to this headline"></a>
    
To implement the channel response assuming an OFDM waveform, it is assumed that
the power delay profiles are invariant over the duration of an OFDM symbol.
Moreover, it is assumed that the duration of the cyclic prefix (CP) equals at
least the maximum delay spread. These assumptions are common in the literature, as they
enable modeling of the channel transfer function in the frequency domain as a
single-tap channel.
    
For every link $(u, k, v, l)$ and resource element $(s,n)$,
the frequency channel response is obtained by computing the Fourier transform of
the channel response at the subcarrier frequencies, i.e.,

$$
\begin{split}\begin{align}
\widehat{h}_{u, k, v, l, s, n}
   &= \int_{-\infty}^{+\infty} h_{u, k, v, l}(s,\tau) e^{-j2\pi n \Delta_f \tau} d\tau\\
   &= \sum_{m=0}^{M-1} a_{u, k, v, l, m}(s)
   e^{-j2\pi n \Delta_f \tau_{u, k, v, l, m}}
\end{align}\end{split}
$$
    
where $s$ is used as time step to indicate that the channel response can
change from one OFDM symbol to the next in the event of mobility, even if it is
assumed static over the duration of an OFDM symbol.
    
For every receive antenna $l$ of every receiver $v$, the
received signal $y_{v, l, s, n}$ for resource element
$(s, n)$ is computed by

$$
y_{v, l, s, n} = \sum_{u=0}^{N_{T}-1}\sum_{k=0}^{N_{TA}-1}
   \widehat{h}_{u, k, v, l, s, n} x_{u, k, s, n}
   + w_{v, l, s, n}
$$
    
where $x_{u, k, s, n}$ is the baseband symbol transmitted by transmitter
$u$ on antenna $k$ and resource element $(s, n)$, and
$w_{v, l, s, n} \sim \mathcal{CN}\left(0,N_0\right)$ the additive white
Gaussian noise.

**Note**
    
This model does not account for intersymbol interference (ISI) nor
intercarrier interference (ICI). To model the ICI due to channel aging over
the duration of an OFDM symbol or the ISI due to a delay spread exceeding the
CP duration, one would need to simulate the channel in the time domain.
This can be achieved by using the <a class="reference internal" href="ofdm.html#sionna.ofdm.OFDMModulator" title="sionna.ofdm.OFDMModulator">`OFDMModulator`</a> and
<a class="reference internal" href="ofdm.html#sionna.ofdm.OFDMDemodulator" title="sionna.ofdm.OFDMDemodulator">`OFDMDemodulator`</a> layers, and the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#time-domain">time domain channel model</a>.
By doing so, one performs inverse discrete Fourier transform (IDFT) on
the transmitter side and discrete Fourier transform (DFT) on the receiver side
on top of a single-carrier sinc-shaped waveform.
This is equivalent to
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#ofdm-waveform">simulating the channel in the frequency domain</a> if no
ISI nor ICI is assumed, but allows the simulation of these effects in the
event of a non-stationary channel or long delay spreads.
Note that simulating the channel in the time domain is typically significantly
more computationally demanding that simulating the channel in the frequency
domain.

