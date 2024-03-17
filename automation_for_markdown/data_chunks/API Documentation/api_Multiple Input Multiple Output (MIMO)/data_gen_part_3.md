# Multiple-Input Multiple-Output (MIMO)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#multiple-input-multiple-output-mimo" title="Permalink to this headline"></a>
    
This module provides layers and functions to support simulation of multicell
MIMO transmissions.

# Table of Content
## Detection<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#detection" title="Permalink to this headline"></a>
### EPDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#epdetector" title="Permalink to this headline"></a>
### KBestDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#kbestdetector" title="Permalink to this headline"></a>
### LinearDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#lineardetector" title="Permalink to this headline"></a>
  
  

### EPDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#epdetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``EPDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`l``=``10`</em>, <em class="sig-param">`beta``=``0.9`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/mimo/detection.html#EPDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.EPDetector" title="Permalink to this definition"></a>
    
MIMO Expectation Propagation (EP) detector
    
This layer implements Expectation Propagation (EP) MIMO detection as described
in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#ep2014" id="id5">[EP2014]</a>. It can generate hard- or soft-decisions for symbols or bits.
    
This layer assumes the following channel model:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$
    
where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathcal{C}^S$ is the vector of transmitted symbols which
are uniformly and independently drawn from the constellation $\mathcal{C}$,
$\mathbf{H}\in\mathbb{C}^{M\times S}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a complex Gaussian noise vector.
It is assumed that $\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}$ and
$\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}$,
where $\mathbf{S}$ has full rank.
    
The channel model is first whitened using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.whiten_channel" title="sionna.mimo.whiten_channel">`whiten_channel()`</a>
and then converted to its real-valued equivalent,
see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_channel" title="sionna.mimo.complex2real_channel">`complex2real_channel()`</a>, prior to MIMO detection.
    
The computation of LLRs is done by converting the symbol logits
that naturally arise in the algorithm to LLRs using
<a class="reference internal" href="mapping.html#sionna.mapping.PAM2QAM" title="sionna.mapping.PAM2QAM">`PAM2QAM()`</a>. Custom conversions of symbol logits to LLRs
can be implemented by using the soft-symbol output.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – The type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per QAM constellation symbol, e.g., 4 for QAM16.
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
 
- **(y, h, s)** – Tuple:
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals
- **h** (<em>[…,M,num_streams], tf.complex</em>) – 2+D tensor containing the channel matrices
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices


Output
 
- **One of**
- <em>[…,num_streams,num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>
- <em>[…,num_streams,2**num_bits_per_symbol], tf.float or […,num_streams], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>




**Note**
    
For numerical stability, we do not recommend to use this function in Graph
mode with XLA, i.e., within a function that is decorated with
`@tf.function(jit_compile=True)`.
However, it is possible to do so by setting
`sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### KBestDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#kbestdetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``KBestDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`num_streams`</em>, <em class="sig-param">`k`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`use_real_rep``=``False`</em>, <em class="sig-param">`list2llr``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/mimo/detection.html#KBestDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.KBestDetector" title="Permalink to this definition"></a>
    
MIMO K-Best detector
    
This layer implements K-Best MIMO detection as described
in (Eq. 4-5) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#ft2015" id="id6">[FT2015]</a>. It can either generate hard decisions (for symbols
or bits) or compute LLRs.
    
The algorithm operates in either the complex or real-valued domain.
Although both options produce identical results, the former has the advantage
that it can be applied to arbitrary non-QAM constellations. It also reduces
the number of streams (or depth) by a factor of two.
    
The way soft-outputs (i.e., LLRs) are computed is determined by the
`list2llr` function. The default solution
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLRSimple" title="sionna.mimo.List2LLRSimple">`List2LLRSimple`</a> assigns a predetermined
value to all LLRs without counter-hypothesis.
    
This layer assumes the following channel model:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$
    
where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathcal{C}^S$ is the vector of transmitted symbols which
are uniformly and independently drawn from the constellation $\mathcal{C}$,
$\mathbf{H}\in\mathbb{C}^{M\times S}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a complex Gaussian noise vector.
It is assumed that $\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}$ and
$\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}$,
where $\mathbf{S}$ has full rank.
    
In a first optional step, the channel model is converted to its real-valued equivalent,
see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_channel" title="sionna.mimo.complex2real_channel">`complex2real_channel()`</a>. We assume in the sequel the complex-valued
representation. Then, the channel is whitened using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.whiten_channel" title="sionna.mimo.whiten_channel">`whiten_channel()`</a>:

$$
\begin{split}\tilde{\mathbf{y}} &= \mathbf{S}^{-\frac{1}{2}}\mathbf{y}\\
&=  \mathbf{S}^{-\frac{1}{2}}\mathbf{H}\mathbf{x} + \mathbf{S}^{-\frac{1}{2}}\mathbf{n}\\
&= \tilde{\mathbf{H}}\mathbf{x} + \tilde{\mathbf{n}}.\end{split}
$$
    
Next, the columns of $\tilde{\mathbf{H}}$ are sorted according
to their norm in descending order. Then, the QR decomposition of the
resulting channel matrix is computed:

$$
\tilde{\mathbf{H}} = \mathbf{Q}\mathbf{R}
$$
    
where $\mathbf{Q}\in\mathbb{C}^{M\times S}$ is unitary and
$\mathbf{R}\in\mathbb{C}^{S\times S}$ is upper-triangular.
The channel outputs are then pre-multiplied by $\mathbf{Q}^{\mathsf{H}}$.
This leads to the final channel model on which the K-Best detection algorithm operates:

$$
\bar{\mathbf{y}} = \mathbf{R}\bar{\mathbf{x}} + \bar{\mathbf{n}}
$$
    
where $\bar{\mathbf{y}}\in\mathbb{C}^S$,
$\bar{\mathbf{x}}\in\mathbb{C}^S$, and $\bar{\mathbf{n}}\in\mathbb{C}^S$
with $\mathbb{E}\left[\bar{\mathbf{n}}\right]=\mathbf{0}$ and
$\mathbb{E}\left[\bar{\mathbf{n}}\bar{\mathbf{n}}^{\mathsf{H}}\right]=\mathbf{I}$.
    
**LLR Computation**
    
The K-Best algorithm produces $K$ candidate solutions $\bar{\mathbf{x}}_k\in\mathcal{C}^S$
and their associated distance metrics $d_k=\lVert \bar{\mathbf{y}} - \mathbf{R}\bar{\mathbf{x}}_k \rVert^2$
for $k=1,\dots,K$. If the real-valued channel representation is used, the distance
metrics are scaled by 0.5 to account for the reduced noise power in each complex dimension.
A hard-decision is simply the candidate with the shortest distance.
Various ways to compute LLRs from this list (and possibly
additional side-information) are possible. The (sub-optimal) default solution
is <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLRSimple" title="sionna.mimo.List2LLRSimple">`List2LLRSimple`</a>. Custom solutions can be provided.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – The type of output, either bits or symbols. Whether soft- or
hard-decisions are returned can be configured with the
`hard_out` flag.
- **num_streams** (<em>tf.int</em>) – Number of transmitted streams
- **k** (<em>tf.int</em>) – The number of paths to keep. Cannot be larger than the
number of constellation points to the power of the number of
streams.
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>. The detector cannot compute soft-symbols.
- **use_real_rep** (<em>bool</em>) – If <cite>True</cite>, the detector use the real-valued equivalent representation
of the channel. Note that this only works with a QAM constellation.
Defaults to <cite>False</cite>.
- **list2llr** (<cite>None</cite> or instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLR" title="sionna.mimo.List2LLR">`List2LLR`</a>) – The function to be used to compute LLRs from a list of candidate solutions.
If <cite>None</cite>, the default solution <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLRSimple" title="sionna.mimo.List2LLRSimple">`List2LLRSimple`</a>
is used.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of `y`. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h, s)** – Tuple:
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals
- **h** (<em>[…,M,num_streams], tf.complex</em>) – 2+D tensor containing the channel matrices
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices


Output
 
- **One of**
- <em>[…,num_streams,num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>
- <em>[…,num_streams,2**num_points], tf.float or […,num_streams], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### LinearDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#lineardetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``LinearDetector`(<em class="sig-param">`equalizer`</em>, <em class="sig-param">`output`</em>, <em class="sig-param">`demapping_method`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mimo/detection.html#LinearDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.LinearDetector" title="Permalink to this definition"></a>
    
Convenience class that combines an equalizer,
such as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.lmmse_equalizer" title="sionna.mimo.lmmse_equalizer">`lmmse_equalizer()`</a>, and a <a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a>.
Parameters
 
- **equalizer** (<em>str</em><em>, </em><em>one of</em><em> [</em><em>"lmmse"</em><em>, </em><em>"zf"</em><em>, </em><em>"mf"</em><em>]</em><em>, or </em><em>an equalizer function</em>) – The equalizer to be used. Either one of the existing equalizers
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.lmmse_equalizer" title="sionna.mimo.lmmse_equalizer">`lmmse_equalizer()`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.zf_equalizer" title="sionna.mimo.zf_equalizer">`zf_equalizer()`</a>, or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.mf_equalizer" title="sionna.mimo.mf_equalizer">`mf_equalizer()`</a> can be used, or a custom equalizer
callable provided that has the same input/output specification.
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – The type of output, either LLRs on bits or logits on constellation symbols.
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The demapping method used.
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the detector computes hard-decided bit values or
constellation point indices instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of `y`. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h, s)** – Tuple:
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals
- **h** (<em>[…,M,num_streams], tf.complex</em>) – 2+D tensor containing the channel matrices
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices


Output
 
- **One of**
- <em>[…, num_streams, num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>
- <em>[…, num_streams, num_points], tf.float or […, num_streams], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you might need to set `sionna.Config.xla_compat=true`. This depends on the
chosen equalizer function. See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

