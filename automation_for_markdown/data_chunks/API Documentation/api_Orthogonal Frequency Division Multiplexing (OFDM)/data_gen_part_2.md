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
## Modulation & Demodulation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#modulation-demodulation" title="Permalink to this headline"></a>
### OFDMModulator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmmodulator" title="Permalink to this headline"></a>
### OFDMDemodulator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmdemodulator" title="Permalink to this headline"></a>
## Pilot Pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#pilot-pattern" title="Permalink to this headline"></a>
### PilotPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#pilotpattern" title="Permalink to this headline"></a>
### EmptyPilotPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#emptypilotpattern" title="Permalink to this headline"></a>
  
  

### OFDMModulator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmmodulator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``OFDMModulator`(<em class="sig-param">`cyclic_prefix_length`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/modulator.html#OFDMModulator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMModulator" title="Permalink to this definition"></a>
    
Computes the time-domain representation of an OFDM resource grid
with (optional) cyclic prefix.
Parameters
    
**cyclic_prefix_length** (<em>int</em>) – Integer indicating the length of the
cyclic prefix that it prepended to each OFDM symbol. It cannot
be longer than the FFT size.

Input
    
<em>[…,num_ofdm_symbols,fft_size], tf.complex</em> – A resource grid in the frequency domain.

Output
    
<em>[…,num_ofdm_symbols*(fft_size+cyclic_prefix_length)], tf.complex</em> – Time-domain OFDM signal.



### OFDMDemodulator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmdemodulator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``OFDMDemodulator`(<em class="sig-param">`fft_size`</em>, <em class="sig-param">`l_min`</em>, <em class="sig-param">`cyclic_prefix_length`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/demodulator.html#OFDMDemodulator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMDemodulator" title="Permalink to this definition"></a>
    
Computes the frequency-domain representation of an OFDM waveform
with cyclic prefix removal.
    
The demodulator assumes that the input sequence is generated by the
<a class="reference internal" href="channel.wireless.html#sionna.channel.TimeChannel" title="sionna.channel.TimeChannel">`TimeChannel`</a>. For a single pair of antennas,
the received signal sequence is given as:

$$
y_b = \sum_{\ell =L_\text{min}}^{L_\text{max}} \bar{h}_\ell x_{b-\ell} + w_b, \quad b \in[L_\text{min}, N_B+L_\text{max}-1]
$$
    
where $\bar{h}_\ell$ are the discrete-time channel taps,
$x_{b}$ is the the transmitted signal,
and $w_\ell$ Gaussian noise.
    
Starting from the first symbol, the demodulator cuts the input
sequence into pieces of size `cyclic_prefix_length` `+` `fft_size`,
and throws away any trailing symbols. For each piece, the cyclic
prefix is removed and the `fft_size`-point discrete Fourier
transform is computed.
    
Since the input sequence starts at time $L_\text{min}$,
the FFT-window has a timing offset of $L_\text{min}$ symbols,
which leads to a subcarrier-dependent phase shift of
$e^{\frac{j2\pi k L_\text{min}}{N}}$, where $k$
is the subcarrier index, $N$ is the FFT size,
and $L_\text{min} \le 0$ is the largest negative time lag of
the discrete-time channel impulse response. This phase shift
is removed in this layer, by explicitly multiplying
each subcarrier by  $e^{\frac{-j2\pi k L_\text{min}}{N}}$.
This is a very important step to enable channel estimation with
sparse pilot patterns that needs to interpolate the channel frequency
response accross subcarriers. It also ensures that the
channel frequency response <cite>seen</cite> by the time-domain channel
is close to the <a class="reference internal" href="channel.wireless.html#sionna.channel.OFDMChannel" title="sionna.channel.OFDMChannel">`OFDMChannel`</a>.
Parameters
 
- **fft_size** (<em>int</em>) – FFT size (, i.e., the number of subcarriers).
- **l_min** (<em>int</em>) – The largest negative time lag of the discrete-time channel
impulse response. It should be the same value as that used by the
<cite>cir_to_time_channel</cite> function.
- **cyclic_prefix_length** (<em>int</em>) – Integer indicating the length of the cyclic prefix that
is prepended to each OFDM symbol.


Input
    
<em>[…,num_ofdm_symbols*(fft_size+cyclic_prefix_length)+n], tf.complex</em> – Tensor containing the time-domain signal along the last dimension.
<cite>n</cite> is a nonnegative integer.

Output
    
<em>[…,num_ofdm_symbols,fft_size], tf.complex</em> – Tensor containing the OFDM resource grid along the last
two dimension.



## Pilot Pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#pilot-pattern" title="Permalink to this headline"></a>
    
A <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a> defines how transmitters send pilot
sequences for each of their antennas or streams over an OFDM resource grid.
It consists of two components,
a `mask` and `pilots`. The `mask` indicates which resource elements are
reserved for pilot transmissions by each transmitter and its respective
streams. In some cases, the number of streams is equal to the number of
transmit antennas, but this does not need to be the case, e.g., for precoded
transmissions. The `pilots` contains the pilot symbols that are transmitted
at the positions indicated by the `mask`. Separating a pilot pattern into
`mask` and `pilots` enables the implementation of a wide range of pilot
configurations, including trainable pilot sequences.
    
The following code snippet shows how to define a simple custom
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a> for single transmitter, sending two streams
Note that `num_effective_subcarriers` is the number of subcarriers that
can be used for data or pilot transmissions. Due to guard
carriers or a nulled DC carrier, this number can be smaller than the
`fft_size` of the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
```python
num_tx = 1
num_streams_per_tx = 2
num_ofdm_symbols = 14
num_effective_subcarriers = 12
# Create a pilot mask
mask = np.zeros([num_tx,
                 num_streams_per_tx,
                 num_ofdm_symbols,
                 num_effective_subcarriers])
mask[0, :, [2,11], :] = 1
num_pilot_symbols = int(np.sum(mask[0,0]))
# Define pilot sequences
pilots = np.zeros([num_tx,
                   num_streams_per_tx,
                   num_pilot_symbols], np.complex64)
pilots[0, 0, 0:num_pilot_symbols:2] = (1+1j)/np.sqrt(2)
pilots[0, 1, 1:num_pilot_symbols:2] = (1+1j)/np.sqrt(2)
# Create a PilotPattern instance
pp = PilotPattern(mask, pilots)
# Visualize non-zero elements of the pilot sequence
pp.show(show_pilot_ind=True);
```

    
As shown in the figures above, the pilots are mapped onto the mask from
the smallest effective subcarrier and OFDM symbol index to the highest
effective subcarrier and OFDM symbol index. Here, boths stream have 24
pilot symbols, out of which only 12 are nonzero. It is important to keep
this order of mapping in mind when designing more complex pilot sequences.

### PilotPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#pilotpattern" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``PilotPattern`(<em class="sig-param">`mask`</em>, <em class="sig-param">`pilots`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`normalize``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/pilot_pattern.html#PilotPattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="Permalink to this definition"></a>
    
Class defining a pilot pattern for an OFDM ResourceGrid.
    
This class defines a pilot pattern object that is used to configure
an OFDM <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
Parameters
 
- **mask** (<em>[</em><em>num_tx</em><em>, </em><em>num_streams_per_tx</em><em>, </em><em>num_ofdm_symbols</em><em>, </em><em>num_effective_subcarriers</em><em>]</em><em>, </em><em>bool</em>) – Tensor indicating resource elements that are reserved for pilot transmissions.
- **pilots** (<em>[</em><em>num_tx</em><em>, </em><em>num_streams_per_tx</em><em>, </em><em>num_pilots</em><em>]</em><em>, </em><em>tf.complex</em>) – The pilot symbols to be mapped onto the `mask`.
- **trainable** (<em>bool</em>) – Indicates if `pilots` is a trainable <cite>Variable</cite>.
Defaults to <cite>False</cite>.
- **normalize** (<em>bool</em>) – Indicates if the `pilots` should be normalized to an average
energy of one across the last dimension. This can be useful to
ensure that trainable `pilots` have a finite energy.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.Dtype</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`mask`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.mask" title="Permalink to this definition"></a>
    
Mask of the pilot pattern


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.normalize" title="Permalink to this definition"></a>
    
Returns or sets the flag indicating if the pilots
are normalized or not


<em class="property">`property` </em>`num_data_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.num_data_symbols" title="Permalink to this definition"></a>
    
Number of data symbols per transmit stream.


<em class="property">`property` </em>`num_effective_subcarriers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.num_effective_subcarriers" title="Permalink to this definition"></a>
    
Number of effectvie subcarriers


<em class="property">`property` </em>`num_ofdm_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.num_ofdm_symbols" title="Permalink to this definition"></a>
    
Number of OFDM symbols


<em class="property">`property` </em>`num_pilot_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.num_pilot_symbols" title="Permalink to this definition"></a>
    
Number of pilot symbols per transmit stream.


<em class="property">`property` </em>`num_streams_per_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.num_streams_per_tx" title="Permalink to this definition"></a>
    
Number of streams per transmitter


<em class="property">`property` </em>`num_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.num_tx" title="Permalink to this definition"></a>
    
Number of transmitters


<em class="property">`property` </em>`pilots`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.pilots" title="Permalink to this definition"></a>
    
Returns or sets the possibly normalized tensor of pilot symbols.
If pilots are normalized, the normalization will be applied
after new values for pilots have been set. If this is
not the desired behavior, turn normalization off.


`show`(<em class="sig-param">`tx_ind``=``None`</em>, <em class="sig-param">`stream_ind``=``None`</em>, <em class="sig-param">`show_pilot_ind``=``False`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/pilot_pattern.html#PilotPattern.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.show" title="Permalink to this definition"></a>
    
Visualizes the pilot patterns for some transmitters and streams.
Input
 
- **tx_ind** (<em>list, int</em>) – Indicates the indices of transmitters to be included.
Defaults to <cite>None</cite>, i.e., all transmitters included.
- **stream_ind** (<em>list, int</em>) – Indicates the indices of streams to be included.
Defaults to <cite>None</cite>, i.e., all streams included.
- **show_pilot_ind** (<em>bool</em>) – Indicates if the indices of the pilot symbols should be shown.


Output
    
**list** (<em>matplotlib.figure.Figure</em>) – List of matplot figure objects showing each the pilot pattern
from a specific transmitter and stream.




<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern.trainable" title="Permalink to this definition"></a>
    
Returns if pilots are trainable or not


### EmptyPilotPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#emptypilotpattern" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``EmptyPilotPattern`(<em class="sig-param">`num_tx`</em>, <em class="sig-param">`num_streams_per_tx`</em>, <em class="sig-param">`num_ofdm_symbols`</em>, <em class="sig-param">`num_effective_subcarriers`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/pilot_pattern.html#EmptyPilotPattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.EmptyPilotPattern" title="Permalink to this definition"></a>
    
Creates an empty pilot pattern.
    
Generates a instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a> with
an empty `mask` and `pilots`.
Parameters
 
- **num_tx** (<em>int</em>) – Number of transmitters.
- **num_streams_per_tx** (<em>int</em>) – Number of streams per transmitter.
- **num_ofdm_symbols** (<em>int</em>) – Number of OFDM symbols.
- **num_effective_subcarriers** (<em>int</em>) – Number of effective subcarriers
that are available for the transmission of data and pilots.
Note that this number is generally smaller than the `fft_size`
due to nulled subcarriers.
- **dtype** (<em>tf.Dtype</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




