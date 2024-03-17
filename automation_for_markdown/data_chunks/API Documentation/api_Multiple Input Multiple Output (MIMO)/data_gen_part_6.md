# Multiple-Input Multiple-Output (MIMO)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#multiple-input-multiple-output-mimo" title="Permalink to this headline"></a>
    
This module provides layers and functions to support simulation of multicell
MIMO transmissions.

# Table of Content
## Detection<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#detection" title="Permalink to this headline"></a>
### MMSE-PIC<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#mmse-pic" title="Permalink to this headline"></a>
## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#utility-functions" title="Permalink to this headline"></a>
### List2LLR<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#list2llr" title="Permalink to this headline"></a>
### List2LLRSimple<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#list2llrsimple" title="Permalink to this headline"></a>
### complex2real_vector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#complex2real-vector" title="Permalink to this headline"></a>
### real2complex_vector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#real2complex-vector" title="Permalink to this headline"></a>
  
  

### MMSE-PIC<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#mmse-pic" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``MMSEPICDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`demapping_method``=``'maxlog'`</em>, <em class="sig-param">`num_iter``=``1`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mimo/detection.html#MMSEPICDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.MMSEPICDetector" title="Permalink to this definition"></a>
    
Minimum mean square error (MMSE) with parallel interference cancellation (PIC) detector
    
This layer implements the MMSE PIC detector, as proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#cst2011" id="id7">[CST2011]</a>.
For `num_iter`>1, this implementation performs MMSE PIC self-iterations.
MMSE PIC self-iterations can be understood as a concatenation of MMSE PIC
detectors from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#cst2011" id="id8">[CST2011]</a>, which forward intrinsic LLRs to the next
self-iteration.
    
Compared to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#cst2011" id="id9">[CST2011]</a>, this implementation also accepts priors on the
constellation symbols as an alternative to priors on the bits.
    
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
    
The algorithm starts by computing the soft symbols
$\bar{x}_s=\mathbb{E}\left[ x_s \right]$ and
variances $v_s=\mathbb{E}\left[ |e_s|^2\right]$ from the priors,
where $e_s = x_s - \bar{x}_s$, for all $s=1,\dots,S$.
    
Next, for each stream, the interference caused by all other streams is cancelled
from the observation $\mathbf{y}$, leading to

$$
\hat{\mathbf{y}}_s = \mathbf{y} - \sum_{j\neq s} \mathbf{h}_j x_j = \mathbf{h}_s x_s + \tilde{\mathbf{n}}_s,\quad s=1,\dots,S
$$
    
where $\tilde{\mathbf{n}}_s=\sum_{j\neq s} \mathbf{h}_j e_j + \mathbf{n}$.
    
Then, a linear MMSE filter $\mathbf{w}_s$ is computed to reduce the resdiual noise
for each observation $\hat{\mathbf{y}}_s$, which is given as

$$
\mathbf{w}_s = \mathbf{h}_s^{\mathsf{H}}\left( \mathbf{H} \mathbf{D}_s\mathbf{H}^{\mathsf{H}} +\mathbf{S} \right)^{-1}
$$
    
where $\mathbf{D}_s \in \mathbb{C}^{S\times S}$ is diagonal with entries

$$
\begin{split}\left[\mathbf{D}_s\right]_{i,i} = \begin{cases}
                                    v_i & i\neq s \\
                                    1 & i=s.
                                  \end{cases}\end{split}
$$
    
The filtered observations

$$
\tilde{z}_s = \mathbf{w}_s^{\mathsf{H}} \hat{\mathbf{y}}_s = \tilde{\mu}_s x_s + \mathbf{w}_s^{\mathsf{H}}\tilde{\mathbf{n}}_s
$$
    
where $\tilde{\mu}_s=\mathbf{w}_s^{\mathsf{H}} \mathbf{h}_s$, are then demapped to either symbol logits or LLRs, assuming that the remaining noise is Gaussian with variance

$$
\nu_s^2 = \mathop{\text{Var}}\left[\tilde{z}_s\right] = \mathbf{w}_s^{\mathsf{H}} \left(\sum_{j\neq s} \mathbf{h}_j \mathbf{h}_j^{\mathsf{H}} v_j +\mathbf{S} \right)\mathbf{w}_s.
$$
    
The resulting soft-symbols can then be used for the next self-iteration of the algorithm.
    
Note that this algorithm can be substantially simplified as described in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#cst2011" id="id10">[CST2011]</a> to avoid
the computation of different matrix inverses for each stream. This is the version which is
implemented.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – The type of output, either LLRs on bits or logits on constellation
symbols.
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The demapping method used.
Defaults to “maxlog”.
- **num_iter** (<em>int</em>) – Number of MMSE PIC iterations.
Defaults to 1.
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
The output dtype is the corresponding real dtype
(tf.float32 or tf.float64).


Input
 
- **(y, h, prior, s)** – Tuple:
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals
- **h** (<em>[…,M,S], tf.complex</em>) – 2+D tensor containing the channel matrices
- **prior** (<em>[…,S,num_bits_per_symbol] or […,S,num_points], tf.float</em>) – Prior of the transmitted signals.
If `output` equals “bit”, then LLRs of the transmitted bits are expected.
If `output` equals “symbol”, then logits of the transmitted constellation points are expected.
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices


Output
 
- **One of**
- <em>[…,S,num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>
- <em>[…,S,2**num_bits_per_symbol], tf.float or […,S], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>




**Note**
    
For numerical stability, we do not recommend to use this function in Graph
mode with XLA, i.e., within a function that is decorated with
`@tf.function(jit_compile=True)`.
However, it is possible to do so by setting
`sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#utility-functions" title="Permalink to this headline"></a>

### List2LLR<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#list2llr" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``List2LLR`<a class="reference internal" href="../_modules/sionna/mimo/utils.html#List2LLR">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLR" title="Permalink to this definition"></a>
    
Abstract class defining a callable to compute LLRs from a list of
candidate vectors (or paths) provided by a MIMO detector.
    
The following channel model is assumed

$$
\bar{\mathbf{y}} = \mathbf{R}\bar{\mathbf{x}} + \bar{\mathbf{n}}
$$
    
where $\bar{\mathbf{y}}\in\mathbb{C}^S$ are the channel outputs,
$\mathbf{R}\in\mathbb{C}^{S\times S}$ is an upper-triangular matrix,
$\bar{\mathbf{x}}\in\mathbb{C}^S$ is the transmitted vector whose entries
are uniformly and independently drawn from the constellation $\mathcal{C}$,
and $\bar{\mathbf{n}}\in\mathbb{C}^S$ is white noise
with $\mathbb{E}\left[\bar{\mathbf{n}}\right]=\mathbf{0}$ and
$\mathbb{E}\left[\bar{\mathbf{n}}\bar{\mathbf{n}}^{\mathsf{H}}\right]=\mathbf{I}$.
    
It is assumed that a MIMO detector such as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.KBestDetector" title="sionna.mimo.KBestDetector">`KBestDetector`</a>
produces $K$ candidate solutions $\bar{\mathbf{x}}_k\in\mathcal{C}^S$
and their associated distance metrics $d_k=\lVert \bar{\mathbf{y}} - \mathbf{R}\bar{\mathbf{x}}_k \rVert^2$
for $k=1,\dots,K$. This layer can also be used with the real-valued representation of the channel.
Input
 
- **(y, r, dists, path_inds, path_syms)** – Tuple:
- **y** (<em>[…,M], tf.complex or tf.float</em>) – Channel outputs of the whitened channel
- **r** ([…,num_streams, num_streams], same dtype as `y`) – Upper triangular channel matrix of the whitened channel
- **dists** (<em>[…,num_paths], tf.float</em>) – Distance metric for each path (or candidate)
- **path_inds** (<em>[…,num_paths,num_streams], tf.int32</em>) – Symbol indices for every stream of every path (or candidate)
- **path_syms** ([…,num_path,num_streams], same dtype as `y`) – Constellation symbol for every stream of every path (or candidate)


Output
    
**llr** (<em>[…num_streams,num_bits_per_symbol], tf.float</em>) – LLRs for all bits of every stream



**Note**
    
An implementation of this class does not need to make use of all of
the provided inputs which enable various different implementations.

### List2LLRSimple<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#list2llrsimple" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``List2LLRSimple`(<em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`llr_clip_val``=``20.0`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#List2LLRSimple">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.List2LLRSimple" title="Permalink to this definition"></a>
    
Computes LLRs from a list of candidate vectors (or paths) provided by a MIMO detector.
    
The following channel model is assumed:

$$
\bar{\mathbf{y}} = \mathbf{R}\bar{\mathbf{x}} + \bar{\mathbf{n}}
$$
    
where $\bar{\mathbf{y}}\in\mathbb{C}^S$ are the channel outputs,
$\mathbf{R}\in\mathbb{C}^{S\times S}$ is an upper-triangular matrix,
$\bar{\mathbf{x}}\in\mathbb{C}^S$ is the transmitted vector whose entries
are uniformly and independently drawn from the constellation $\mathcal{C}$,
and $\bar{\mathbf{n}}\in\mathbb{C}^S$ is white noise
with $\mathbb{E}\left[\bar{\mathbf{n}}\right]=\mathbf{0}$ and
$\mathbb{E}\left[\bar{\mathbf{n}}\bar{\mathbf{n}}^{\mathsf{H}}\right]=\mathbf{I}$.
    
It is assumed that a MIMO detector such as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.KBestDetector" title="sionna.mimo.KBestDetector">`KBestDetector`</a>
produces $K$ candidate solutions $\bar{\mathbf{x}}_k\in\mathcal{C}^S$
and their associated distance metrics $d_k=\lVert \bar{\mathbf{y}} - \mathbf{R}\bar{\mathbf{x}}_k \rVert^2$
for $k=1,\dots,K$. This layer can also be used with the real-valued representation of the channel.
    
The LLR for the $i\text{th}$ bit of the $k\text{th}$ stream is computed as

$$
\begin{split}\begin{align}
    LLR(k,i) &= \log\left(\frac{\Pr(b_{k,i}=1|\bar{\mathbf{y}},\mathbf{R})}{\Pr(b_{k,i}=0|\bar{\mathbf{y}},\mathbf{R})}\right)\\
        &\approx \min_{j \in  \mathcal{C}_{k,i,0}}d_j - \min_{j \in  \mathcal{C}_{k,i,1}}d_j
\end{align}\end{split}
$$
    
where $\mathcal{C}_{k,i,1}$ and $\mathcal{C}_{k,i,0}$ are the set of indices
in the list of candidates for which the $i\text{th}$ bit of the $k\text{th}$
stream is equal to 1 and 0, respectively. The LLRs are clipped to $\pm LLR_\text{clip}$
which can be configured through the parameter `llr_clip_val`.
    
If $\mathcal{C}_{k,i,0}$ is empty, $LLR(k,i)=LLR_\text{clip}$;
if $\mathcal{C}_{k,i,1}$ is empty, $LLR(k,i)=-LLR_\text{clip}$.
Parameters
 
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol
- **llr_clip_val** (<em>float</em>) – The absolute values of LLRs are clipped to this value.
Defaults to 20.0. Can also be a trainable variable.


Input
 
- **(y, r, dists, path_inds, path_syms)** – Tuple:
- **y** (<em>[…,M], tf.complex or tf.float</em>) – Channel outputs of the whitened channel
- **r** ([…,num_streams, num_streams], same dtype as `y`) – Upper triangular channel matrix of the whitened channel
- **dists** (<em>[…,num_paths], tf.float</em>) – Distance metric for each path (or candidate)
- **path_inds** (<em>[…,num_paths,num_streams], tf.int32</em>) – Symbol indices for every stream of every path (or candidate)
- **path_syms** ([…,num_path,num_streams], same dtype as `y`) – Constellation symbol for every stream of every path (or candidate)


Output
    
**llr** (<em>[…num_streams,num_bits_per_symbol], tf.float</em>) – LLRs for all bits of every stream



### complex2real_vector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#complex2real-vector" title="Permalink to this headline"></a>

`sionna.mimo.``complex2real_vector`(<em class="sig-param">`z`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#complex2real_vector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_vector" title="Permalink to this definition"></a>
    
Transforms a complex-valued vector into its real-valued equivalent.
    
Transforms the last dimension of a complex-valued tensor into
its real-valued equivalent by stacking the real and imaginary
parts on top of each other.
    
For a vector $\mathbf{z}\in \mathbb{C}^M$ with real and imaginary
parts $\mathbf{x}\in \mathbb{R}^M$ and
$\mathbf{y}\in \mathbb{R}^M$, respectively, this function returns
the vector $\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}$.
Input
    
<em>[…,M], tf.complex</em>

Output
    
<em>[…,2M], tf.complex.real_dtype</em>



### real2complex_vector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#real2complex-vector" title="Permalink to this headline"></a>

`sionna.mimo.``real2complex_vector`(<em class="sig-param">`z`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#real2complex_vector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.real2complex_vector" title="Permalink to this definition"></a>
    
Transforms a real-valued vector into its complex-valued equivalent.
    
Transforms the last dimension of a real-valued tensor into
its complex-valued equivalent by interpreting the first half
as the real and the second half as the imaginary part.
    
For a vector $\mathbf{z}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in \mathbb{R}^{2M}$
with $\mathbf{x}\in \mathbb{R}^M$ and $\mathbf{y}\in \mathbb{R}^M$,
this function returns
the vector $\mathbf{x}+j\mathbf{y}\in\mathbb{C}^M$.
Input
    
<em>[…,2M], tf.float</em>

Output
    
<em>[…,M], tf.complex</em>



