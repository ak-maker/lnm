# Multiple-Input Multiple-Output (MIMO)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#multiple-input-multiple-output-mimo" title="Permalink to this headline"></a>
    
This module provides layers and functions to support simulation of multicell
MIMO transmissions.

# Table of Content
## Detection
### MaximumLikelihoodDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#maximumlikelihooddetector" title="Permalink to this headline"></a>
  
  

### MaximumLikelihoodDetector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#maximumlikelihooddetector" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mimo.``MaximumLikelihoodDetector`(<em class="sig-param">`output`</em>, <em class="sig-param">`demapping_method`</em>, <em class="sig-param">`num_streams`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`with_prior``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mimo/detection.html#MaximumLikelihoodDetector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.MaximumLikelihoodDetector" title="Permalink to this definition"></a>
    
MIMO maximum-likelihood (ML) detector.
If the `with_prior` flag is set, prior knowledge on the bits or constellation points is assumed to be available.
    
This layer implements MIMO maximum-likelihood (ML) detection assuming the
following channel model:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$
    
where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathcal{C}^K$ is the vector of transmitted symbols which
are uniformly and independently drawn from the constellation $\mathcal{C}$,
$\mathbf{H}\in\mathbb{C}^{M\times K}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a complex Gaussian noise vector.
It is assumed that $\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}$ and
$\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}$,
where $\mathbf{S}$ has full rank.
If the `with_prior` flag is set, it is assumed that prior information of the transmitted signal $\mathbf{x}$ is available,
provided either as LLRs on the bits mapped onto $\mathbf{x}$ or as logits on the individual
constellation points forming $\mathbf{x}$.
    
Prior to demapping, the received signal is whitened:

$$
\begin{split}\tilde{\mathbf{y}} &= \mathbf{S}^{-\frac{1}{2}}\mathbf{y}\\
&=  \mathbf{S}^{-\frac{1}{2}}\mathbf{H}\mathbf{x} + \mathbf{S}^{-\frac{1}{2}}\mathbf{n}\\
&= \tilde{\mathbf{H}}\mathbf{x} + \tilde{\mathbf{n}}\end{split}
$$
    
The layer can compute ML detection of symbols or bits with either
soft- or hard-decisions. Note that decisions are computed symbol-/bit-wise
and not jointly for the entire vector $\textbf{x}$ (or the underlying vector
of bits).
    
**ML detection of bits:**
    
Soft-decisions on bits are called log-likelihood ratios (LLR).
With the “app” demapping method, the LLR for the $i\text{th}$ bit
of the $k\text{th}$ user is then computed according to

$$
\begin{split}\begin{align}
    LLR(k,i)&= \ln\left(\frac{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}\right)\\
            &=\ln\left(\frac{
            \sum_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \exp\left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right) \Pr\left( \mathbf{x} \right)
            }{
            \sum_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \exp\left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right) \Pr\left( \mathbf{x} \right)
            }\right)
\end{align}\end{split}
$$
    
where $\mathcal{C}_{k,i,1}$ and $\mathcal{C}_{k,i,0}$ are the
sets of vectors of constellation points for which the $i\text{th}$ bit
of the $k\text{th}$ user is equal to 1 and 0, respectively.
$\Pr\left( \mathbf{x} \right)$ is the prior distribution of the vector of
constellation points $\mathbf{x}$. Assuming that the constellation points and
bit levels are independent, it is computed from the prior of the bits according to

$$
\Pr\left( \mathbf{x} \right) = \prod_{k=1}^K \prod_{i=1}^{I} \sigma \left( LLR_p(k,i) \right)
$$
    
where $LLR_p(k,i)$ is the prior knowledge of the $i\text{th}$ bit of the
$k\text{th}$ user given as an LLR and which is set to $0$ if no prior knowledge is assumed to be available,
and $\sigma\left(\cdot\right)$ is the sigmoid function.
The definition of the LLR has been chosen such that it is equivalent with that of logit. This is
different from many textbooks in communications, where the LLR is
defined as $LLR(k,i) = \ln\left(\frac{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}\right)$.
    
With the “maxlog” demapping method, the LLR for the $i\text{th}$ bit
of the $k\text{th}$ user is approximated like

$$
\begin{split}\begin{align}
    LLR(k,i) \approx&\ln\left(\frac{
        \max_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \exp\left(
            -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
            \right) \Pr\left( \mathbf{x} \right) \right)
        }{
        \max_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \exp\left(
            -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
            \right) \Pr\left( \mathbf{x} \right) \right)
        }\right)\\
        = &\min_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left(\Pr\left( \mathbf{x} \right) \right) \right) -
            \min_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left( \Pr\left( \mathbf{x} \right) \right) \right).
    \end{align}\end{split}
$$
    
**ML detection of symbols:**
    
Soft-decisions on symbols are called logits (i.e., unnormalized log-probability).
    
With the “app” demapping method, the logit for the
constellation point $c \in \mathcal{C}$ of the $k\text{th}$ user  is computed according to

$$
\begin{align}
    \text{logit}(k,c) &= \ln\left(\sum_{\mathbf{x} : x_k = c} \exp\left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right)\Pr\left( \mathbf{x} \right)\right).
\end{align}
$$
    
With the “maxlog” demapping method, the logit for the constellation point $c \in \mathcal{C}$
of the $k\text{th}$ user  is approximated like

$$
\text{logit}(k,c) \approx \max_{\mathbf{x} : x_k = c} \left(
        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 + \ln \left( \Pr\left( \mathbf{x} \right) \right)
        \right).
$$
    
When hard decisions are requested, this layer returns for the $k$ th stream

$$
\hat{c}_k = \underset{c \in \mathcal{C}}{\text{argmax}} \left( \sum_{\mathbf{x} : x_k = c} \exp\left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right)\Pr\left( \mathbf{x} \right) \right)
$$
    
where $\mathcal{C}$ is the set of constellation points.
Parameters
 
- **output** (<em>One of</em><em> [</em><em>"bit"</em><em>, </em><em>"symbol"</em><em>]</em><em>, </em><em>str</em>) – The type of output, either LLRs on bits or logits on constellation symbols.
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The demapping method used.
- **num_streams** (<em>tf.int</em>) – Number of transmitted streams
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
- **with_prior** (<em>bool</em>) – If <cite>True</cite>, it is assumed that prior knowledge on the bits or constellation points is available.
This prior information is given as LLRs (for bits) or log-probabilities (for constellation points) as an
additional input to the layer.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of `y`. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, h, s) or (y, h, prior, s)** – Tuple:
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals.
- **h** (<em>[…,M,num_streams], tf.complex</em>) – 2+D tensor containing the channel matrices.
- **prior** (<em>[…,num_streams,num_bits_per_symbol] or […,num_streams,num_points], tf.float</em>) – Prior of the transmitted signals.
If `output` equals “bit”, then LLRs of the transmitted bits are expected.
If `output` equals “symbol”, then logits of the transmitted constellation points are expected.
Only required if the `with_prior` flag is set.
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices.


Output
 
- **One of**
- <em>[…, num_streams, num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit of every stream, if `output` equals <cite>“bit”</cite>.
- <em>[…, num_streams, num_points], tf.float or […, num_streams], tf.int</em> – Logits or hard-decisions for constellation symbols for every stream, if `output` equals <cite>“symbol”</cite>.
Hard-decisions correspond to the symbol indices.




**Note**
    
If you want to use this layer in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

