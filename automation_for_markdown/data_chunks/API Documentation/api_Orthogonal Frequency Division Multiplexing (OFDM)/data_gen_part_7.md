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
## Detection<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#detection" title="Permalink to this headline"></a>
### KBestDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#kbestdetector" title="Permalink to this headline"></a>
### LinearDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#lineardetector" title="Permalink to this headline"></a>
### MaximumLikelihoodDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#maximumlikelihooddetector" title="Permalink to this headline"></a>
### MaximumLikelihoodDetectorWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#maximumlikelihooddetectorwithprior" title="Permalink to this headline"></a>
  
  

### KBestDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#kbestdetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``KBestDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`num_streams`</em>, <em class="sig-param">`k`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`use_real_rep``=``False`</em>, <em class="sig-param">`list2llr``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#KBestDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.KBestDetector" title="Permalink to this definition"></a>
    
This layer wraps the MIMO K-Best detector for use with the OFDM waveform.
    
Both detection of symbols or bits with either
soft- or hard-decisions are supported. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
actual detector is an instance of <a class="reference internal" href="mimo.html#sionna.mimo.KBestDetector" title="sionna.mimo.KBestDetector">`KBestDetector`</a>.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **num_streams** (<em>tf.int</em>) – Number of transmitted streams
- **k** (<em>tf.int</em>) – Number of paths to keep. Cannot be larger than the
number of constellation points to the power of the number of
streams.
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – Instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **use_real_rep** (<em>bool</em>) – If <cite>True</cite>, the detector use the real-valued equivalent representation
of the channel. Note that this only works with a QAM constellation.
Defaults to <cite>False</cite>.
- **list2llr** (<cite>None</cite> or instance of <a class="reference internal" href="mimo.html#sionna.mimo.List2LLR" title="sionna.mimo.List2LLR">`List2LLR`</a>) – The function to be used to compute LLRs from a list of candidate solutions.
If <cite>None</cite>, the default solution <a class="reference internal" href="mimo.html#sionna.mimo.List2LLRSimple" title="sionna.mimo.List2LLRSimple">`List2LLRSimple`</a>
is used.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h_hat, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **One of**
- <em>[batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### LinearDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#lineardetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``LinearDetector`(<em class="sig-param">`equalizer`</em>, <em class="sig-param">`output`</em>, <em class="sig-param">`demapping_method`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#LinearDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LinearDetector" title="Permalink to this definition"></a>
    
This layer wraps a MIMO linear equalizer and a <a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a>
for use with the OFDM waveform.
    
Both detection of symbols or bits with either
soft- or hard-decisions are supported. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
actual detector is an instance of <a class="reference internal" href="mimo.html#sionna.mimo.LinearDetector" title="sionna.mimo.LinearDetector">`LinearDetector`</a>.
Parameters
 
- **equalizer** (<em>str</em><em>, </em><em>one of</em><em> [</em><em>"lmmse"</em><em>, </em><em>"zf"</em><em>, </em><em>"mf"</em><em>]</em><em>, or </em><em>an equalizer function</em>) – Equalizer to be used. Either one of the existing equalizers, e.g.,
<a class="reference internal" href="mimo.html#sionna.mimo.lmmse_equalizer" title="sionna.mimo.lmmse_equalizer">`lmmse_equalizer()`</a>, <a class="reference internal" href="mimo.html#sionna.mimo.zf_equalizer" title="sionna.mimo.zf_equalizer">`zf_equalizer()`</a>, or
<a class="reference internal" href="mimo.html#sionna.mimo.mf_equalizer" title="sionna.mimo.mf_equalizer">`mf_equalizer()`</a> can be used, or a custom equalizer
function provided that has the same input/output specification.
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – Demapping method used
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – Instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h_hat, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **One of**
- <em>[batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### MaximumLikelihoodDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#maximumlikelihooddetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``MaximumLikelihoodDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`demapping_method`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#MaximumLikelihoodDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.MaximumLikelihoodDetector" title="Permalink to this definition"></a>
    
Maximum-likelihood (ML) detection for OFDM MIMO transmissions.
    
This layer implements maximum-likelihood (ML) detection
for OFDM MIMO transmissions. Both ML detection of symbols or bits with either
soft- or hard-decisions are supported. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
actual detector is an instance of <a class="reference internal" href="mimo.html#sionna.mimo.MaximumLikelihoodDetector" title="sionna.mimo.MaximumLikelihoodDetector">`MaximumLikelihoodDetector`</a>.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – Demapping method used
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – Instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h_hat, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN noise


Output
 
- **One of**
- <em>[batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### MaximumLikelihoodDetectorWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#maximumlikelihooddetectorwithprior" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``MaximumLikelihoodDetectorWithPrior`(<em class="sig-param">`output`</em>, <em class="sig-param">`demapping_method`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#MaximumLikelihoodDetectorWithPrior">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.MaximumLikelihoodDetectorWithPrior" title="Permalink to this definition"></a>
    
Maximum-likelihood (ML) detection for OFDM MIMO transmissions, assuming prior
knowledge of the bits or constellation points is available.
    
This layer implements maximum-likelihood (ML) detection
for OFDM MIMO transmissions assuming prior knowledge on the transmitted data is available.
Both ML detection of symbols or bits with either
soft- or hard-decisions are supported. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
actual detector is an instance of <a class="reference internal" href="mimo.html#sionna.mimo.MaximumLikelihoodDetectorWithPrior" title="sionna.mimo.MaximumLikelihoodDetectorWithPrior">`MaximumLikelihoodDetectorWithPrior`</a>.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – Demapping method used
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – Instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h_hat, prior, err_var, no)** – Tuple:
- **y** (<em>[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex</em>) – Received OFDM resource grid after cyclic prefix removal and FFT
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex</em>) – Channel estimates for all streams from all transmitters
- **prior** (<em>[batch_size, num_tx, num_streams, num_data_symbols x num_bits_per_symbol] or [batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float</em>) – Prior of the transmitted signals.
If `output` equals “bit”, LLRs of the transmitted bits are expected.
If `output` equals “symbol”, logits of the transmitted constellation points are expected.
- **err_var** ([Broadcastable to shape of `h_hat`], tf.float) – Variance of the channel estimation error
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN noise


Output
 
- **One of**
- <em>[batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

