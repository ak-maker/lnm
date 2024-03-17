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
## AWGN<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#awgn" title="Permalink to this headline"></a>
## Flat-fading channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#flat-fading-channel" title="Permalink to this headline"></a>
### FlatFadingChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#flatfadingchannel" title="Permalink to this headline"></a>
### GenerateFlatFadingChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#generateflatfadingchannel" title="Permalink to this headline"></a>
### ApplyFlatFadingChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#applyflatfadingchannel" title="Permalink to this headline"></a>
### SpatialCorrelation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#spatialcorrelation" title="Permalink to this headline"></a>
### KroneckerModel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#kroneckermodel" title="Permalink to this headline"></a>
  
  

## AWGN<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#awgn" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``AWGN`(<em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/awgn.html#AWGN">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.AWGN" title="Permalink to this definition"></a>
    
Add complex AWGN to the inputs with a certain variance.
    
This class inherits from the Keras <cite>Layer</cite> class and can be used as layer in
a Keras model.
    
This layer adds complex AWGN noise with variance `no` to the input.
The noise has variance `no/2` per real dimension.
It can be either a scalar or a tensor which can be broadcast to the shape
of the input.
<p class="rubric">Example
    
Setting-up:
```python
>>> awgn_channel = AWGN()
```

    
Running:
```python
>>> # x is the channel input
>>> # no is the noise variance
>>> y = awgn_channel((x, no))
```

Parameters
    
**dtype** (<em>Complex tf.DType</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.

Input
 
- **(x, no)** – Tuple:
- **x** (<em>Tensor, tf.complex</em>) – Channel input
- **no** (<em>Scalar or Tensor, tf.float</em>) – Scalar or tensor whose shape can be broadcast to the shape of `x`.
The noise power `no` is per complex dimension. If `no` is a
scalar, noise of the same variance will be added to the input.
If `no` is a tensor, it must have a shape that can be broadcast to
the shape of `x`. This allows, e.g., adding noise of different
variance to each example in a batch. If `no` has a lower rank than
`x`, then `no` will be broadcast to the shape of `x` by adding
dummy dimensions after the last axis.


Output
    
**y** (Tensor with same shape as `x`, tf.complex) – Channel output



## Flat-fading channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#flat-fading-channel" title="Permalink to this headline"></a>

### FlatFadingChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#flatfadingchannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``FlatFadingChannel`(<em class="sig-param">`num_tx_ant`</em>, <em class="sig-param">`num_rx_ant`</em>, <em class="sig-param">`spatial_corr``=``None`</em>, <em class="sig-param">`add_awgn``=``True`</em>, <em class="sig-param">`return_channel``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/flat_fading_channel.html#FlatFadingChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.FlatFadingChannel" title="Permalink to this definition"></a>
    
Applies random channel matrices to a vector input and adds AWGN.
    
This class combines <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateFlatFadingChannel" title="sionna.channel.GenerateFlatFadingChannel">`GenerateFlatFadingChannel`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyFlatFadingChannel" title="sionna.channel.ApplyFlatFadingChannel">`ApplyFlatFadingChannel`</a> and computes the output of
a flat-fading channel with AWGN.
    
For a given batch of input vectors $\mathbf{x}\in\mathbb{C}^{K}$,
the output is

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$
    
where $\mathbf{H}\in\mathbb{C}^{M\times K}$ are randomly generated
flat-fading channel matrices and
$\mathbf{n}\in\mathbb{C}^{M}\sim\mathcal{CN}(0, N_o\mathbf{I})$
is an AWGN vector that is optionally added.
    
A <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation">`SpatialCorrelation`</a> can be configured and the
channel realizations optionally returned. This is useful to simulate
receiver algorithms with perfect channel knowledge.
Parameters
 
- **num_tx_ant** (<em>int</em>) – Number of transmit antennas.
- **num_rx_ant** (<em>int</em>) – Number of receive antennas.
- **spatial_corr** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation"><em>SpatialCorrelation</em></a><em>, </em><em>None</em>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation">`SpatialCorrelation`</a> or <cite>None</cite>.
Defaults to <cite>None</cite>.
- **add_awgn** (<em>bool</em>) – Indicates if AWGN noise should be added to the output.
Defaults to <cite>True</cite>.
- **return_channel** (<em>bool</em>) – Indicates if the channel realizations should be returned.
Defaults  to <cite>False</cite>.
- **dtype** (<em>tf.complex64</em><em>, </em><em>tf.complex128</em>) – The dtype of the output. Defaults to <cite>tf.complex64</cite>.


Input
 
- **(x, no)** – Tuple or Tensor:
- **x** (<em>[batch_size, num_tx_ant], tf.complex</em>) – Tensor of transmit vectors.
- **no** (<em>Scalar of Tensor, tf.float</em>) – The noise power `no` is per complex dimension.
Only required if `add_awgn==True`.
Will be broadcast to the dimensions of the channel output if needed.
For more details, see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.AWGN" title="sionna.channel.AWGN">`AWGN`</a>.


Output
 
- **(y, h)** – Tuple or Tensor:
- **y** ([batch_size, num_rx_ant, num_tx_ant], `dtype`) – Channel output.
- **h** ([batch_size, num_rx_ant, num_tx_ant], `dtype`) – Channel realizations. Will only be returned if
`return_channel==True`.




<em class="property">`property` </em>`apply`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.FlatFadingChannel.apply" title="Permalink to this definition"></a>
    
Calls the internal <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyFlatFadingChannel" title="sionna.channel.ApplyFlatFadingChannel">`ApplyFlatFadingChannel`</a>.


<em class="property">`property` </em>`generate`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.FlatFadingChannel.generate" title="Permalink to this definition"></a>
    
Calls the internal <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateFlatFadingChannel" title="sionna.channel.GenerateFlatFadingChannel">`GenerateFlatFadingChannel`</a>.


<em class="property">`property` </em>`spatial_corr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.FlatFadingChannel.spatial_corr" title="Permalink to this definition"></a>
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation">`SpatialCorrelation`</a> to be used.


### GenerateFlatFadingChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#generateflatfadingchannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``GenerateFlatFadingChannel`(<em class="sig-param">`num_tx_ant`</em>, <em class="sig-param">`num_rx_ant`</em>, <em class="sig-param">`spatial_corr``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/flat_fading_channel.html#GenerateFlatFadingChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateFlatFadingChannel" title="Permalink to this definition"></a>
    
Generates tensors of flat-fading channel realizations.
    
This class generates batches of random flat-fading channel matrices.
A spatial correlation can be applied.
Parameters
 
- **num_tx_ant** (<em>int</em>) – Number of transmit antennas.
- **num_rx_ant** (<em>int</em>) – Number of receive antennas.
- **spatial_corr** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation"><em>SpatialCorrelation</em></a><em>, </em><em>None</em>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation">`SpatialCorrelation`</a> or <cite>None</cite>.
Defaults to <cite>None</cite>.
- **dtype** (<em>tf.complex64</em><em>, </em><em>tf.complex128</em>) – The dtype of the output. Defaults to <cite>tf.complex64</cite>.


Input
    
**batch_size** (<em>int</em>) – The batch size, i.e., the number of channel matrices to generate.

Output
    
**h** ([batch_size, num_rx_ant, num_tx_ant], `dtype`) – Batch of random flat fading channel matrices.



<em class="property">`property` </em>`spatial_corr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.GenerateFlatFadingChannel.spatial_corr" title="Permalink to this definition"></a>
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="sionna.channel.SpatialCorrelation">`SpatialCorrelation`</a> to be used.


### ApplyFlatFadingChannel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#applyflatfadingchannel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``ApplyFlatFadingChannel`(<em class="sig-param">`add_awgn``=``True`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/channel/flat_fading_channel.html#ApplyFlatFadingChannel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.ApplyFlatFadingChannel" title="Permalink to this definition"></a>
    
Applies given channel matrices to a vector input and adds AWGN.
    
This class applies a given tensor of flat-fading channel matrices
to an input tensor. AWGN noise can be optionally added.
Mathematically, for channel matrices
$\mathbf{H}\in\mathbb{C}^{M\times K}$
and input $\mathbf{x}\in\mathbb{C}^{K}$, the output is

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$
    
where $\mathbf{n}\in\mathbb{C}^{M}\sim\mathcal{CN}(0, N_o\mathbf{I})$
is an AWGN vector that is optionally added.
Parameters
 
- **add_awgn** (<em>bool</em>) – Indicates if AWGN noise should be added to the output.
Defaults to <cite>True</cite>.
- **dtype** (<em>tf.complex64</em><em>, </em><em>tf.complex128</em>) – The dtype of the output. Defaults to <cite>tf.complex64</cite>.


Input
 
- **(x, h, no)** – Tuple:
- **x** (<em>[batch_size, num_tx_ant], tf.complex</em>) – Tensor of transmit vectors.
- **h** (<em>[batch_size, num_rx_ant, num_tx_ant], tf.complex</em>) – Tensor of channel realizations. Will be broadcast to the
dimensions of `x` if needed.
- **no** (<em>Scalar or Tensor, tf.float</em>) – The noise power `no` is per complex dimension.
Only required if `add_awgn==True`.
Will be broadcast to the shape of `y`.
For more details, see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.AWGN" title="sionna.channel.AWGN">`AWGN`</a>.


Output
    
**y** ([batch_size, num_rx_ant, num_tx_ant], `dtype`) – Channel output.



### SpatialCorrelation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#spatialcorrelation" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``SpatialCorrelation`<a class="reference internal" href="../_modules/sionna/channel/spatial_correlation.html#SpatialCorrelation">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.SpatialCorrelation" title="Permalink to this definition"></a>
    
Abstract class that defines an interface for spatial correlation functions.
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.FlatFadingChannel" title="sionna.channel.FlatFadingChannel">`FlatFadingChannel`</a> model can be configured with a
spatial correlation model.
Input
    
**h** (<em>tf.complex</em>) – Tensor of arbitrary shape containing spatially uncorrelated
channel coefficients

Output
    
**h_corr** (<em>tf.complex</em>) – Tensor of the same shape and dtype as `h` containing the spatially
correlated channel coefficients.



### KroneckerModel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#kroneckermodel" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.channel.``KroneckerModel`(<em class="sig-param">`r_tx``=``None`</em>, <em class="sig-param">`r_rx``=``None`</em>)<a class="reference internal" href="../_modules/sionna/channel/spatial_correlation.html#KroneckerModel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.KroneckerModel" title="Permalink to this definition"></a>
    
Kronecker model for spatial correlation.
    
Given a batch of matrices $\mathbf{H}\in\mathbb{C}^{M\times K}$,
$\mathbf{R}_\text{tx}\in\mathbb{C}^{K\times K}$, and
$\mathbf{R}_\text{rx}\in\mathbb{C}^{M\times M}$, this function
will generate the following output:

$$
\mathbf{H}_\text{corr} = \mathbf{R}^{\frac12}_\text{rx} \mathbf{H} \mathbf{R}^{\frac12}_\text{tx}
$$
    
Note that $\mathbf{R}_\text{tx}\in\mathbb{C}^{K\times K}$ and $\mathbf{R}_\text{rx}\in\mathbb{C}^{M\times M}$
must be positive semi-definite, such as the ones generated by
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.exp_corr_mat" title="sionna.channel.exp_corr_mat">`exp_corr_mat()`</a>.
Parameters
 
- **r_tx** (<em>[</em><em>...</em><em>, </em><em>K</em><em>, </em><em>K</em><em>]</em><em>, </em><em>tf.complex</em>) – Tensor containing the transmit correlation matrices. If
the rank of `r_tx` is smaller than that of the input `h`,
it will be broadcast.
- **r_rx** (<em>[</em><em>...</em><em>, </em><em>M</em><em>, </em><em>M</em><em>]</em><em>, </em><em>tf.complex</em>) – Tensor containing the receive correlation matrices. If
the rank of `r_rx` is smaller than that of the input `h`,
it will be broadcast.


Input
    
**h** (<em>[…, M, K], tf.complex</em>) – Tensor containing spatially uncorrelated
channel coeffficients.

Output
    
**h_corr** (<em>[…, M, K], tf.complex</em>) – Tensor containing the spatially
correlated channel coefficients.



<em class="property">`property` </em>`r_rx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.KroneckerModel.r_rx" title="Permalink to this definition"></a>
    
Tensor containing the receive correlation matrices.

**Note**
    
If you want to set this property in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.


<em class="property">`property` </em>`r_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.KroneckerModel.r_tx" title="Permalink to this definition"></a>
    
Tensor containing the transmit correlation matrices.

**Note**
    
If you want to set this property in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.


