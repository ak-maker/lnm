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
## Pilot Pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#pilot-pattern" title="Permalink to this headline"></a>
### KroneckerPilotPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#kroneckerpilotpattern" title="Permalink to this headline"></a>
## Channel Estimation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#channel-estimation" title="Permalink to this headline"></a>
### BaseChannelEstimator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#basechannelestimator" title="Permalink to this headline"></a>
### BaseChannelInterpolator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#basechannelinterpolator" title="Permalink to this headline"></a>
### LSChannelEstimator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#lschannelestimator" title="Permalink to this headline"></a>
### LinearInterpolator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#linearinterpolator" title="Permalink to this headline"></a>
  
  

### KroneckerPilotPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#kroneckerpilotpattern" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``KroneckerPilotPattern`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`pilot_ofdm_symbol_indices`</em>, <em class="sig-param">`normalize``=``True`</em>, <em class="sig-param">`seed``=``0`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/pilot_pattern.html#KroneckerPilotPattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.KroneckerPilotPattern" title="Permalink to this definition"></a>
    
Simple orthogonal pilot pattern with Kronecker structure.
    
This function generates an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>
that allocates non-overlapping pilot sequences for all transmitters and
streams on specified OFDM symbols. As the same pilot sequences are reused
across those OFDM symbols, the resulting pilot pattern has a frequency-time
Kronecker structure. This structure enables a very efficient implementation
of the LMMSE channel estimator. Each pilot sequence is constructed from
randomly drawn QPSK constellation points.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **pilot_ofdm_symbol_indices** (<em>list</em><em>, </em><em>int</em>) – List of integers defining the OFDM symbol indices that are reserved
for pilots.
- **normalize** (<em>bool</em>) – Indicates if the `pilots` should be normalized to an average
energy of one across the last dimension.
Defaults to <cite>True</cite>.
- **seed** (<em>int</em>) – Seed for the generation of the pilot sequence. Different seed values
lead to different sequences. Defaults to 0.
- **dtype** (<em>tf.Dtype</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




**Note**
    
It is required that the `resource_grid`’s property
`num_effective_subcarriers` is an
integer multiple of `num_tx` `*` `num_streams_per_tx`. This condition is
required to ensure that all transmitters and streams get
non-overlapping pilot sequences. For a large number of streams and/or
transmitters, the pilot pattern becomes very sparse in the frequency
domain.
<p class="rubric">Examples
```python
>>> rg = ResourceGrid(num_ofdm_symbols=14,
...                   fft_size=64,
...                   subcarrier_spacing = 30e3,
...                   num_tx=4,
...                   num_streams_per_tx=2,
...                   pilot_pattern = "kronecker",
...                   pilot_ofdm_symbol_indices = [2, 11])
>>> rg.pilot_pattern.show();
```


## Channel Estimation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#channel-estimation" title="Permalink to this headline"></a>

### BaseChannelEstimator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#basechannelestimator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``BaseChannelEstimator`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`interpolation_type``=``'nn'`</em>, <em class="sig-param">`interpolator``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#BaseChannelEstimator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelEstimator" title="Permalink to this definition"></a>
    
Abstract layer for implementing an OFDM channel estimator.
    
Any layer that implements an OFDM channel estimator must implement this
class and its
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations" title="sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations">`estimate_at_pilot_locations()`</a>
abstract method.
    
This class extracts the pilots from the received resource grid `y`, calls
the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations" title="sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations">`estimate_at_pilot_locations()`</a>
method to estimate the channel for the pilot-carrying resource elements,
and then interpolates the channel to compute channel estimates for the
data-carrying resouce elements using the interpolation method specified by
`interpolation_type` or the `interpolator` object.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **interpolation_type** (<em>One of</em><em> [</em><em>"nn"</em><em>, </em><em>"lin"</em><em>, </em><em>"lin_time_avg"</em><em>]</em><em>, </em><em>string</em>) – The interpolation method to be used.
It is ignored if `interpolator` is not <cite>None</cite>.
Available options are <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.NearestNeighborInterpolator" title="sionna.ofdm.NearestNeighborInterpolator">`NearestNeighborInterpolator`</a> (<cite>“nn</cite>”)
or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LinearInterpolator" title="sionna.ofdm.LinearInterpolator">`LinearInterpolator`</a> without (<cite>“lin”</cite>) or with
averaging across OFDM symbols (<cite>“lin_time_avg”</cite>).
Defaults to “nn”.
- **interpolator** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelInterpolator" title="sionna.ofdm.BaseChannelInterpolator"><em>BaseChannelInterpolator</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelInterpolator" title="sionna.ofdm.BaseChannelInterpolator">`BaseChannelInterpolator`</a>,
such as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEInterpolator" title="sionna.ofdm.LMMSEInterpolator">`LMMSEInterpolator`</a>,
or <cite>None</cite>. In the latter case, the interpolator specfied
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
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex</em>) – Channel estimates accross the entire resource grid for all
transmitters and streams
- **err_var** (Same shape as `h_hat`, tf.float) – Channel estimation error variance accross the entire resource grid
for all transmitters and streams




<em class="property">`abstract` </em>`estimate_at_pilot_locations`(<em class="sig-param">`y_pilots`</em>, <em class="sig-param">`no`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#BaseChannelEstimator.estimate_at_pilot_locations">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations" title="Permalink to this definition"></a>
    
Estimates the channel for the pilot-carrying resource elements.
    
This is an abstract method that must be implemented by a concrete
OFDM channel estimator that implement this class.
Input
 
- **y_pilots** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], tf.complex</em>) – Observed signals for the pilot-carrying resource elements
- **no** (<em>[batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float</em>) – Variance of the AWGN


Output
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], tf.complex</em>) – Channel estimates for the pilot-carrying resource elements
- **err_var** (Same shape as `h_hat`, tf.float) – Channel estimation error variance for the pilot-carrying
resource elements





### BaseChannelInterpolator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#basechannelinterpolator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``BaseChannelInterpolator`<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#BaseChannelInterpolator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelInterpolator" title="Permalink to this definition"></a>
    
Abstract layer for implementing an OFDM channel interpolator.
    
Any layer that implements an OFDM channel interpolator must implement this
callable class.
    
A channel interpolator is used by an OFDM channel estimator
(<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelEstimator" title="sionna.ofdm.BaseChannelEstimator">`BaseChannelEstimator`</a>) to compute channel estimates
for the data-carrying resource elements from the channel estimates for the
pilot-carrying resource elements.
Input
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimates for the pilot-carrying resource elements
- **err_var** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimation error variances for the pilot-carrying resource elements


Output
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel estimates accross the entire resource grid for all
transmitters and streams
- **err_var** (Same shape as `h_hat`, tf.float) – Channel estimation error variance accross the entire resource grid
for all transmitters and streams




### LSChannelEstimator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#lschannelestimator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``LSChannelEstimator`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`interpolation_type``=``'nn'`</em>, <em class="sig-param">`interpolator``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#LSChannelEstimator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LSChannelEstimator" title="Permalink to this definition"></a>
    
Layer implementing least-squares (LS) channel estimation for OFDM MIMO systems.
    
After LS channel estimation at the pilot positions, the channel estimates
and error variances are interpolated accross the entire resource grid using
a specified interpolation function.
    
For simplicity, the underlying algorithm is described for a vectorized observation,
where we have a nonzero pilot for all elements to be estimated.
The actual implementation works on a full OFDM resource grid with sparse
pilot patterns. The following model is assumed:

$$
\mathbf{y} = \mathbf{h}\odot\mathbf{p} + \mathbf{n}
$$
    
where $\mathbf{y}\in\mathbb{C}^{M}$ is the received signal vector,
$\mathbf{p}\in\mathbb{C}^M$ is the vector of pilot symbols,
$\mathbf{h}\in\mathbb{C}^{M}$ is the channel vector to be estimated,
and $\mathbf{n}\in\mathbb{C}^M$ is a zero-mean noise vector whose
elements have variance $N_0$. The operator $\odot$ denotes
element-wise multiplication.
    
The channel estimate $\hat{\mathbf{h}}$ and error variances
$\sigma^2_i$, $i=0,\dots,M-1$, are computed as

$$
\begin{split}\hat{\mathbf{h}} &= \mathbf{y} \odot
                   \frac{\mathbf{p}^\star}{\left|\mathbf{p}\right|^2}
                 = \mathbf{h} + \tilde{\mathbf{h}}\\
     \sigma^2_i &= \mathbb{E}\left[\tilde{h}_i \tilde{h}_i^\star \right]
                 = \frac{N_0}{\left|p_i\right|^2}.\end{split}
$$
    
The channel estimates and error variances are then interpolated accross
the entire resource grid.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **interpolation_type** (<em>One of</em><em> [</em><em>"nn"</em><em>, </em><em>"lin"</em><em>, </em><em>"lin_time_avg"</em><em>]</em><em>, </em><em>string</em>) – The interpolation method to be used.
It is ignored if `interpolator` is not <cite>None</cite>.
Available options are <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.NearestNeighborInterpolator" title="sionna.ofdm.NearestNeighborInterpolator">`NearestNeighborInterpolator`</a> (<cite>“nn</cite>”)
or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LinearInterpolator" title="sionna.ofdm.LinearInterpolator">`LinearInterpolator`</a> without (<cite>“lin”</cite>) or with
averaging across OFDM symbols (<cite>“lin_time_avg”</cite>).
Defaults to “nn”.
- **interpolator** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelInterpolator" title="sionna.ofdm.BaseChannelInterpolator"><em>BaseChannelInterpolator</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.BaseChannelInterpolator" title="sionna.ofdm.BaseChannelInterpolator">`BaseChannelInterpolator`</a>,
such as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEInterpolator" title="sionna.ofdm.LMMSEInterpolator">`LMMSEInterpolator`</a>,
or <cite>None</cite>. In the latter case, the interpolator specfied
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
 
- **h_ls** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex</em>) – Channel estimates accross the entire resource grid for all
transmitters and streams
- **err_var** (Same shape as `h_ls`, tf.float) – Channel estimation error variance accross the entire resource grid
for all transmitters and streams




### LinearInterpolator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#linearinterpolator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``LinearInterpolator`(<em class="sig-param">`pilot_pattern`</em>, <em class="sig-param">`time_avg``=``False`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#LinearInterpolator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LinearInterpolator" title="Permalink to this definition"></a>
    
Linear channel estimate interpolation on a resource grid.
    
This class computes for each element of an OFDM resource grid
a channel estimate based on `num_pilots` provided channel estimates and
error variances through linear interpolation.
It is assumed that the measurements were taken at the nonzero positions
of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>.
    
The interpolation is done first across sub-carriers and then
across OFDM symbols.
Parameters
 
- **pilot_pattern** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern"><em>PilotPattern</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>
- **time_avg** (<em>bool</em>) – If enabled, measurements will be averaged across OFDM symbols
(i.e., time). This is useful for channels that do not vary
substantially over the duration of an OFDM frame. Defaults to <cite>False</cite>.


Input
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimates for the pilot-carrying resource elements
- **err_var** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimation error variances for the pilot-carrying resource elements


Output
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel estimates accross the entire resource grid for all
transmitters and streams
- **err_var** (Same shape as `h_hat`, tf.float) – Channel estimation error variances accross the entire resource grid
for all transmitters and streams




