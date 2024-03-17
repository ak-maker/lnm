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
## Equalization<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#equalization" title="Permalink to this headline"></a>
### MFEqualizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#mfequalizer" title="Permalink to this headline"></a>
### ZFEqualizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#zfequalizer" title="Permalink to this headline"></a>
## Detection<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#detection" title="Permalink to this headline"></a>
### OFDMDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmdetector" title="Permalink to this headline"></a>
### OFDMDetectorWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmdetectorwithprior" title="Permalink to this headline"></a>
### EPDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#epdetector" title="Permalink to this headline"></a>
  
  

### MFEqualizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#mfequalizer" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``MFEqualizer`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/equalization.html#MFEqualizer">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.MFEqualizer" title="Permalink to this definition"></a>
    
MF equalization for OFDM MIMO transmissions.
    
This layer computes matched filter (MF) equalization
for OFDM MIMO transmissions. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
detection algorithm is the <a class="reference internal" href="mimo.html#sionna.mimo.mf_equalizer" title="sionna.mimo.mf_equalizer">`mf_equalizer()`</a>. The layer
computes soft-symbol estimates together with effective noise variances
for all streams which can, e.g., be used by a
<a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a> to obtain LLRs.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – An instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>.
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

### ZFEqualizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#zfequalizer" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``ZFEqualizer`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/equalization.html#ZFEqualizer">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ZFEqualizer" title="Permalink to this definition"></a>
    
ZF equalization for OFDM MIMO transmissions.
    
This layer computes zero-forcing (ZF) equalization
for OFDM MIMO transmissions. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
detection algorithm is the <a class="reference internal" href="mimo.html#sionna.mimo.zf_equalizer" title="sionna.mimo.zf_equalizer">`zf_equalizer()`</a>. The layer
computes soft-symbol estimates together with effective noise variances
for all streams which can, e.g., be used by a
<a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a> to obtain LLRs.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – An instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>.
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

## Detection<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#detection" title="Permalink to this headline"></a>

### OFDMDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmdetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``OFDMDetector`(<em class="sig-param">`detector`</em>, <em class="sig-param">`output`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#OFDMDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMDetector" title="Permalink to this definition"></a>
    
Layer that wraps a MIMO detector for use with the OFDM waveform.
    
The parameter `detector` is a callable (e.g., a function) that
implements a MIMO detection algorithm for arbitrary batch dimensions.
    
This class pre-processes the received resource grid `y` and channel
estimate `h_hat`, and computes for each receiver the
noise-plus-interference covariance matrix according to the OFDM and stream
configuration provided by the `resource_grid` and
`stream_management`, which also accounts for the channel
estimation error variance `err_var`. These quantities serve as input to the detection
algorithm that is implemented by `detector`.
Both detection of symbols or bits with either soft- or hard-decisions are supported.

**Note**
    
The callable `detector` must take as input a tuple $(\mathbf{y}, \mathbf{h}, \mathbf{s})$ such that:
 
- **y** ([…,num_rx_ant], tf.complex) – 1+D tensor containing the received signals.
- **h** ([…,num_rx_ant,num_streams_per_rx], tf.complex) – 2+D tensor containing the channel matrices.
- **s** ([…,num_rx_ant,num_rx_ant], tf.complex) – 2+D tensor containing the noise-plus-interference covariance matrices.

    
It must generate one of following outputs depending on the value of `output`:
 
- **b_hat** ([…, num_streams_per_rx, num_bits_per_symbol], tf.float) – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- **x_hat** ([…, num_streams_per_rx, num_points], tf.float) or ([…, num_streams_per_rx], tf.int) – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>. Hard-decisions correspond to the symbol indices.
Parameters
 
- **detector** (<em>Callable</em>) – Callable object (e.g., a function) that implements a MIMO detection
algorithm for arbitrary batch dimensions. Either one of the existing detectors, e.g.,
<a class="reference internal" href="mimo.html#sionna.mimo.LinearDetector" title="sionna.mimo.LinearDetector">`LinearDetector`</a>, <a class="reference internal" href="mimo.html#sionna.mimo.MaximumLikelihoodDetector" title="sionna.mimo.MaximumLikelihoodDetector">`MaximumLikelihoodDetector`</a>, or
<a class="reference internal" href="mimo.html#sionna.mimo.KBestDetector" title="sionna.mimo.KBestDetector">`KBestDetector`</a> can be used, or a custom detector
callable provided that has the same input/output specification.
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
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
- <em>[batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>
- <em>[batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




### OFDMDetectorWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#ofdmdetectorwithprior" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``OFDMDetectorWithPrior`(<em class="sig-param">`detector`</em>, <em class="sig-param">`output`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`constellation_type`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`constellation`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#OFDMDetectorWithPrior">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMDetectorWithPrior" title="Permalink to this definition"></a>
    
Layer that wraps a MIMO detector that assumes prior knowledge of the bits or
constellation points is available, for use with the OFDM waveform.
    
The parameter `detector` is a callable (e.g., a function) that
implements a MIMO detection algorithm with prior for arbitrary batch
dimensions.
    
This class pre-processes the received resource grid `y`, channel
estimate `h_hat`, and the prior information `prior`, and computes for each receiver the
noise-plus-interference covariance matrix according to the OFDM and stream
configuration provided by the `resource_grid` and
`stream_management`, which also accounts for the channel
estimation error variance `err_var`. These quantities serve as input to the detection
algorithm that is implemented by `detector`.
Both detection of symbols or bits with either soft- or hard-decisions are supported.

**Note**
    
The callable `detector` must take as input a tuple $(\mathbf{y}, \mathbf{h}, \mathbf{prior}, \mathbf{s})$ such that:
 
- **y** ([…,num_rx_ant], tf.complex) – 1+D tensor containing the received signals.
- **h** ([…,num_rx_ant,num_streams_per_rx], tf.complex) – 2+D tensor containing the channel matrices.
- **prior** ([…,num_streams_per_rx,num_bits_per_symbol] or […,num_streams_per_rx,num_points], tf.float) – Prior for the transmitted signals. If `output` equals “bit”, then LLRs for the transmitted bits are expected. If `output` equals “symbol”, then logits for the transmitted constellation points are expected.
- **s** ([…,num_rx_ant,num_rx_ant], tf.complex) – 2+D tensor containing the noise-plus-interference covariance matrices.

    
It must generate one of the following outputs depending on the value of `output`:
 
- **b_hat** ([…, num_streams_per_rx, num_bits_per_symbol], tf.float) – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- **x_hat** ([…, num_streams_per_rx, num_points], tf.float) or ([…, num_streams_per_rx], tf.int) – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>. Hard-decisions correspond to the symbol indices.
Parameters
 
- **detector** (<em>Callable</em>) – Callable object (e.g., a function) that implements a MIMO detection
algorithm with prior for arbitrary batch dimensions. Either the existing detector
<a class="reference internal" href="mimo.html#sionna.mimo.MaximumLikelihoodDetectorWithPrior" title="sionna.mimo.MaximumLikelihoodDetectorWithPrior">`MaximumLikelihoodDetectorWithPrior`</a> can be used, or a custom detector
callable provided that has the same input/output specification.
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – Instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
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
- **no** (<em>[batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float</em>) – Variance of the AWGN


Output
 
- **One of**
- <em>[batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




### EPDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#epdetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``EPDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`l``=``10`</em>, <em class="sig-param">`beta``=``0.9`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/detection.html#EPDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.EPDetector" title="Permalink to this definition"></a>
    
This layer wraps the MIMO EP detector for use with the OFDM waveform.
    
Both detection of symbols or bits with either
soft- or hard-decisions are supported. The OFDM and stream configuration are provided
by a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> instance, respectively. The
actual detector is an instance of <a class="reference internal" href="mimo.html#sionna.mimo.EPDetector" title="sionna.mimo.EPDetector">`EPDetector`</a>.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – Type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – Instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **l** (<em>int</em>) – Number of iterations. Defaults to 10.
- **beta** (<em>float</em>) – Parameter $\beta\in[0,1]$ for update smoothing.
Defaults to 0.9.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – Precision used for internal computations. Defaults to `tf.complex64`.
Especially for large MIMO setups, the precision can make a significant
performance difference.


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
    
For numerical stability, we do not recommend to use this function in Graph
mode with XLA, i.e., within a function that is decorated with
`@tf.function(jit_compile=True)`.
However, it is possible to do so by setting
`sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

