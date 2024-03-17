# Orthogonal Frequency-Division Multiplexing (OFDM)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#orthogonal-frequency-division-multiplexing-ofdm" title="Permalink to this headline"></a>
    
This module provides layers and functions to support
simulation of OFDM-based systems. The key component is the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> that defines how data and pilot symbols
are mapped onto a sequence of OFDM symbols with a given FFT size. The resource
grid can also define guard and DC carriers which are nulled. In 4G/5G parlance,
a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> would be a slot.
Once a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> is defined, one can use the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGridMapper" title="sionna.ofdm.ResourceGridMapper">`ResourceGridMapper`</a> to map a tensor of complex-valued
data symbols onto the resource grid, prior to OFDM modulation using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMModulator" title="sionna.ofdm.OFDMModulator">`OFDMModulator`</a> or further processing in the
frequency domain.
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a> allows for a fine-grained configuration
of how transmitters send pilots for each of their streams or antennas. As the
management of pilots in multi-cell MIMO setups can quickly become complicated,
the module provides the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.KroneckerPilotPattern" title="sionna.ofdm.KroneckerPilotPattern">`KroneckerPilotPattern`</a> class
that automatically generates orthogonal pilot transmissions for all transmitters
and streams.
    
Additionally, the module contains layers for channel estimation, precoding,
equalization, and detection,
such as the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LSChannelEstimator" title="sionna.ofdm.LSChannelEstimator">`LSChannelEstimator`</a>, the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ZFPrecoder" title="sionna.ofdm.ZFPrecoder">`ZFPrecoder`</a>, and the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEEqualizer" title="sionna.ofdm.LMMSEEqualizer">`LMMSEEqualizer`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LinearDetector" title="sionna.ofdm.LinearDetector">`LinearDetector`</a>.
These are good starting points for the development of more advanced algorithms
and provide robust baselines for benchmarking.

# Table of Content
## Channel Estimation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#channel-estimation" title="Permalink to this headline"></a>
### NearestNeighborInterpolator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#nearestneighborinterpolator" title="Permalink to this headline"></a>
### tdl_time_cov_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#tdl-time-cov-mat" title="Permalink to this headline"></a>
### tdl_freq_cov_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#tdl-freq-cov-mat" title="Permalink to this headline"></a>
## Precoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#precoding" title="Permalink to this headline"></a>
### ZFPrecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#zfprecoder" title="Permalink to this headline"></a>
## Equalization<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#equalization" title="Permalink to this headline"></a>
### OFDMEqualizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmequalizer" title="Permalink to this headline"></a>
### LMMSEEqualizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#lmmseequalizer" title="Permalink to this headline"></a>
  
  

### NearestNeighborInterpolator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#nearestneighborinterpolator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``NearestNeighborInterpolator`(<em class="sig-param">`pilot_pattern`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#NearestNeighborInterpolator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.NearestNeighborInterpolator" title="Permalink to this definition"></a>
    
Nearest-neighbor channel estimate interpolation on a resource grid.
    
This class assigns to each element of an OFDM resource grid one of
`num_pilots` provided channel estimates and error
variances according to the nearest neighbor method. It is assumed
that the measurements were taken at the nonzero positions of a
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>.
    
The figure below shows how four channel estimates are interpolated
accross a resource grid. Grey fields indicate measurement positions
while the colored regions show which resource elements are assigned
to the same measurement value.
Parameters
    
**pilot_pattern** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern"><em>PilotPattern</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>

Input
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimates for the pilot-carrying resource elements
- **err_var** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimation error variances for the pilot-carrying resource elements


Output
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel estimates accross the entire resource grid for all
transmitters and streams
- **err_var** (Same shape as `h_hat`, tf.float) – Channel estimation error variances accross the entire resource grid
for all transmitters and streams




### tdl_time_cov_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#tdl-time-cov-mat" title="Permalink to this headline"></a>

`sionna.ofdm.``tdl_time_cov_mat`(<em class="sig-param">`model`</em>, <em class="sig-param">`speed`</em>, <em class="sig-param">`carrier_frequency`</em>, <em class="sig-param">`ofdm_symbol_duration`</em>, <em class="sig-param">`num_ofdm_symbols`</em>, <em class="sig-param">`los_angle_of_arrival``=``0.7853981633974483`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#tdl_time_cov_mat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.tdl_time_cov_mat" title="Permalink to this definition"></a>
    
Computes the time covariance matrix of a
<a class="reference internal" href="channel.wireless.html#sionna.channel.tr38901.TDL" title="sionna.channel.tr38901.TDL">`TDL`</a> channel model.
    
For non-line-of-sight (NLoS) model, the channel time covariance matrix
$\mathbf{R^{(t)}}$ of a TDL channel model is

$$
\mathbf{R^{(t)}}_{u,v} = J_0 \left( \nu \Delta_t \left( u-v \right) \right)
$$
    
where $J_0$ is the zero-order Bessel function of the first kind,
$\Delta_t$ the duration of an OFDM symbol, and $\nu$ the Doppler
spread defined by

$$
\nu = 2 \pi \frac{v}{c} f_c
$$
    
where $v$ is the movement speed, $c$ the speed of light, and
$f_c$ the carrier frequency.
    
For line-of-sight (LoS) channel models, the channel time covariance matrix
is

$$
\mathbf{R^{(t)}}_{u,v} = P_{\text{NLoS}} J_0 \left( \nu \Delta_t \left( u-v \right) \right) + P_{\text{LoS}}e^{j \nu \Delta_t \left( u-v \right) \cos{\alpha_{\text{LoS}}}}
$$
    
where $\alpha_{\text{LoS}}$ is the angle-of-arrival for the LoS path,
$P_{\text{NLoS}}$ the total power of NLoS paths, and
$P_{\text{LoS}}$ the power of the LoS path. The power delay profile
is assumed to have unit power, i.e., $P_{\text{NLoS}} + P_{\text{LoS}} = 1$.
Input
 
- **model** (<em>str</em>) – TDL model for which to return the covariance matrix.
Should be one of “A”, “B”, “C”, “D”, or “E”.
- **speed** (<em>float</em>) – Speed [m/s]
- **carrier_frequency** (<em>float</em>) – Carrier frequency [Hz]
- **ofdm_symbol_duration** (<em>float</em>) – Duration of an OFDM symbol [s]
- **num_ofdm_symbols** (<em>int</em>) – Number of OFDM symbols
- **los_angle_of_arrival** (<em>float</em>) – Angle-of-arrival for LoS path [radian]. Only used with LoS models.
Defaults to $\pi/4$.
- **dtype** (<em>tf.DType</em>) – Datatype to use for the output.
Should be one of <cite>tf.complex64</cite> or <cite>tf.complex128</cite>.
Defaults to <cite>tf.complex64</cite>.


Output
    
**cov_mat** (<em>[num_ofdm_symbols, num_ofdm_symbols], tf.complex</em>) – Channel time covariance matrix



### tdl_freq_cov_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#tdl-freq-cov-mat" title="Permalink to this headline"></a>

`sionna.ofdm.``tdl_freq_cov_mat`(<em class="sig-param">`model`</em>, <em class="sig-param">`subcarrier_spacing`</em>, <em class="sig-param">`fft_size`</em>, <em class="sig-param">`delay_spread`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#tdl_freq_cov_mat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.tdl_freq_cov_mat" title="Permalink to this definition"></a>
    
Computes the frequency covariance matrix of a
<a class="reference internal" href="channel.wireless.html#sionna.channel.tr38901.TDL" title="sionna.channel.tr38901.TDL">`TDL`</a> channel model.
    
The channel frequency covariance matrix $\mathbf{R}^{(f)}$ of a TDL channel model is

$$
\mathbf{R}^{(f)}_{u,v} = \sum_{\ell=1}^L P_\ell e^{-j 2 \pi \tau_\ell \Delta_f (u-v)}, 1 \leq u,v \leq M
$$
    
where $M$ is the FFT size, $L$ is the number of paths for the selected TDL model,
$P_\ell$ and $\tau_\ell$ are the average power and delay for the
$\ell^{\text{th}}$ path, respectively, and $\Delta_f$ is the sub-carrier spacing.
Input
 
- **model** (<em>str</em>) – TDL model for which to return the covariance matrix.
Should be one of “A”, “B”, “C”, “D”, or “E”.
- **subcarrier_spacing** (<em>float</em>) – Sub-carrier spacing [Hz]
- **fft_size** (<em>float</em>) – FFT size
- **delay_spread** (<em>float</em>) – Delay spread [s]
- **dtype** (<em>tf.DType</em>) – Datatype to use for the output.
Should be one of <cite>tf.complex64</cite> or <cite>tf.complex128</cite>.
Defaults to <cite>tf.complex64</cite>.


Output
    
**cov_mat** (<em>[fft_size, fft_size], tf.complex</em>) – Channel frequency covariance matrix



## Precoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#precoding" title="Permalink to this headline"></a>

### ZFPrecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#zfprecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``ZFPrecoder`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`return_effective_channel``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/precoding.html#ZFPrecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ZFPrecoder" title="Permalink to this definition"></a>
    
Zero-forcing precoding for multi-antenna transmissions.
    
This layer precodes a tensor containing OFDM resource grids using
the <a class="reference internal" href="mimo.html#sionna.mimo.zero_forcing_precoder" title="sionna.mimo.zero_forcing_precoder">`zero_forcing_precoder()`</a>. For every
transmitter, the channels to all intended receivers are gathered
into a channel matrix, based on the which the precoding matrix
is computed and the input tensor is precoded. The layer also outputs
optionally the effective channel after precoding for each stream.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – An instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>.
- **return_effective_channel** (<em>bool</em>) – Indicates if the effective channel after precoding should be returned.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(x, h)** – Tuple:
- **x** (<em>[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex</em>) – Tensor containing the resource grid to be precoded.
- **h** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm, fft_size], tf.complex</em>) – Tensor containing the channel knowledge based on which the precoding
is computed.


Output
 
- **x_precoded** (<em>[batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – The precoded resource grids.
- **h_eff** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm, num_effective_subcarriers], tf.complex</em>) – Only returned if `return_effective_channel=True`.
The effectice channels for all streams after precoding. Can be used to
simulate perfect channel state information (CSI) at the receivers.
Nulled subcarriers are automatically removed to be compliant with the
behavior of a channel estimator.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

## Equalization<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#equalization" title="Permalink to this headline"></a>

### OFDMEqualizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmequalizer" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``OFDMEqualizer`(<em class="sig-param">`equalizer`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/equalization.html#OFDMEqualizer">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMEqualizer" title="Permalink to this definition"></a>
    
Layer that wraps a MIMO equalizer for use with the OFDM waveform.
    
The parameter `equalizer` is a callable (e.g., a function) that
implements a MIMO equalization algorithm for arbitrary batch dimensions.
    
This class pre-processes the received resource grid `y` and channel
estimate `h_hat`, and computes for each receiver the
noise-plus-interference covariance matrix according to the OFDM and stream
configuration provided by the `resource_grid` and
`stream_management`, which also accounts for the channel
estimation error variance `err_var`. These quantities serve as input
to the equalization algorithm that is implemented by the callable `equalizer`.
This layer computes soft-symbol estimates together with effective noise
variances for all streams which can, e.g., be used by a
<a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a> to obtain LLRs.

**Note**
    
The callable `equalizer` must take three inputs:
 
- **y** ([…,num_rx_ant], tf.complex) – 1+D tensor containing the received signals.
- **h** ([…,num_rx_ant,num_streams_per_rx], tf.complex) – 2+D tensor containing the channel matrices.
- **s** ([…,num_rx_ant,num_rx_ant], tf.complex) – 2+D tensor containing the noise-plus-interference covariance matrices.

    
It must generate two outputs:
 
- **x_hat** ([…,num_streams_per_rx], tf.complex) – 1+D tensor representing the estimated symbol vectors.
- **no_eff** (tf.float) – Tensor of the same shape as `x_hat` containing the effective noise variance estimates.
Parameters
 
- **equalizer** (<em>Callable</em>) – Callable object (e.g., a function) that implements a MIMO equalization
algorithm for arbitrary batch dimensions
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(y, h_hat, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **x_hat** (<em>[batch_size, num_tx, num_streams, num_data_symbols], tf.complex</em>) – Estimated symbols
- **no_eff** (<em>[batch_size, num_tx, num_streams, num_data_symbols], tf.float</em>) – Effective noise variance for each estimated symbol




### LMMSEEqualizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#lmmseequalizer" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``LMMSEEqualizer`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`whiten_interference``=``True`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/equalization.html#LMMSEEqualizer">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEEqualizer" title="Permalink to this definition"></a>
    
LMMSE equalization for OFDM MIMO transmissions.
    
This layer computes linear minimum mean squared error (LMMSE) equalization
for OFDM MIMO transmissions. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
detection algorithm is the <a class="reference internal" href="mimo.html#sionna.mimo.lmmse_equalizer" title="sionna.mimo.lmmse_equalizer">`lmmse_equalizer()`</a>. The layer
computes soft-symbol estimates together with effective noise variances
for all streams which can, e.g., be used by a
<a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a> to obtain LLRs.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **whiten_interference** (<em>bool</em>) – If <cite>True</cite> (default), the interference is first whitened before equalization.
In this case, an alternative expression for the receive filter is used which
can be numerically more stable.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **(y, h_hat, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **x_hat** (<em>[batch_size, num_tx, num_streams, num_data_symbols], tf.complex</em>) – Estimated symbols
- **no_eff** (<em>[batch_size, num_tx, num_streams, num_data_symbols], tf.float</em>) – Effective noise variance for each estimated symbol




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

