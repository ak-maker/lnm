# Mapping<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#mapping" title="Permalink to this headline"></a>
    
This module contains classes and functions related to mapping
of bits to constellation symbols and demapping of soft-symbols
to log-likelihood ratios (LLRs). The key components are the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Mapper" title="sionna.mapping.Mapper">`Mapper`</a>,
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a>. A <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
can be made trainable to enable learning of geometric shaping.

# Table of Content
## Demapping<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#demapping" title="Permalink to this headline"></a>
### Demapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#demapper" title="Permalink to this headline"></a>
### DemapperWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#demapperwithprior" title="Permalink to this headline"></a>
### SymbolDemapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symboldemapper" title="Permalink to this headline"></a>
  
  

### Demapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#demapper" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``Demapper`(<em class="sig-param">`demapping_method`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`with_prior``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#Demapper">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Demapper" title="Permalink to this definition"></a>
    
Computes log-likelihood ratios (LLRs) or hard-decisions on bits
for a tensor of received symbols.
If the flag `with_prior` is set, prior knowledge on the bits is assumed to be available.
    
This class defines a layer implementing different demapping
functions. All demapping functions are fully differentiable when soft-decisions
are computed.
Parameters
 
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The demapping method used.
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the demapper provides hard-decided bits instead of soft-values.
Defaults to <cite>False</cite>.
- **with_prior** (<em>bool</em>) – If <cite>True</cite>, it is assumed that prior knowledge on the bits is available.
This prior information is given as LLRs as an additional input to the layer.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y,no) or (y, prior, no)** – Tuple:
- **y** (<em>[…,n], tf.complex</em>) – The received symbols.
- **prior** (<em>[num_bits_per_symbol] or […,num_bits_per_symbol], tf.float</em>) – Prior for every bit as LLRs.
It can be provided either as a tensor of shape <cite>[num_bits_per_symbol]</cite> for the
entire input batch, or as a tensor that is “broadcastable”
to <cite>[…, n, num_bits_per_symbol]</cite>.
Only required if the `with_prior` flag is set.
- **no** (<em>Scalar or […,n], tf.float</em>) – The noise variance estimate. It can be provided either as scalar
for the entire input batch or as a tensor that is “broadcastable” to
`y`.


Output
    
<em>[…,n*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit.



**Note**
    
With the “app” demapping method, the LLR for the $i\text{th}$ bit
is computed according to

$$
LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert y,\mathbf{p}\right)}{\Pr\left(b_i=0\lvert y,\mathbf{p}\right)}\right) =\ln\left(\frac{
        \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
        \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }{
        \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
        \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }\right)
$$
    
where $\mathcal{C}_{i,1}$ and $\mathcal{C}_{i,0}$ are the
sets of constellation points for which the $i\text{th}$ bit is
equal to 1 and 0, respectively. $\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]$
is the vector of LLRs that serves as prior knowledge on the $K$ bits that are mapped to
a constellation point and is set to $\mathbf{0}$ if no prior knowledge is assumed to be available,
and $\Pr(c\lvert\mathbf{p})$ is the prior probability on the constellation symbol $c$:

$$
\Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)
$$
    
where $\ell(c)_k$ is the $k^{th}$ bit label of $c$, where 0 is
replaced by -1.
The definition of the LLR has been
chosen such that it is equivalent with that of logits. This is
different from many textbooks in communications, where the LLR is
defined as $LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)$.
    
With the “maxlog” demapping method, LLRs for the $i\text{th}$ bit
are approximated like

$$
\begin{split}\begin{align}
    LLR(i) &\approx\ln\left(\frac{
        \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
            \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }{
        \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
            \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }\right)\\
        &= \max_{c\in\mathcal{C}_{i,0}}
            \left(\ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right)-\frac{|y-c|^2}{N_o}\right) -
         \max_{c\in\mathcal{C}_{i,1}}\left( \ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right) - \frac{|y-c|^2}{N_o}\right)
        .
\end{align}\end{split}
$$


### DemapperWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#demapperwithprior" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``DemapperWithPrior`(<em class="sig-param">`demapping_method`</em>, <em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#DemapperWithPrior">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.DemapperWithPrior" title="Permalink to this definition"></a>
    
Computes log-likelihood ratios (LLRs) or hard-decisions on bits
for a tensor of received symbols, assuming that prior knowledge on the bits is available.
    
This class defines a layer implementing different demapping
functions. All demapping functions are fully differentiable when soft-decisions
are computed.
    
This class is deprecated as the functionality has been integrated
into <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a>.
Parameters
 
- **demapping_method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The demapping method used.
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the demapper provides hard-decided bits instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, prior, no)** – Tuple:
- **y** (<em>[…,n], tf.complex</em>) – The received symbols.
- **prior** (<em>[num_bits_per_symbol] or […,num_bits_per_symbol], tf.float</em>) – Prior for every bit as LLRs.
It can be provided either as a tensor of shape <cite>[num_bits_per_symbol]</cite> for the
entire input batch, or as a tensor that is “broadcastable”
to <cite>[…, n, num_bits_per_symbol]</cite>.
- **no** (<em>Scalar or […,n], tf.float</em>) – The noise variance estimate. It can be provided either as scalar
for the entire input batch or as a tensor that is “broadcastable” to
`y`.


Output
    
<em>[…,n*num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit.



**Note**
    
With the “app” demapping method, the LLR for the $i\text{th}$ bit
is computed according to

$$
LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert y,\mathbf{p}\right)}{\Pr\left(b_i=0\lvert y,\mathbf{p}\right)}\right) =\ln\left(\frac{
        \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
        \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }{
        \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
        \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }\right)
$$
    
where $\mathcal{C}_{i,1}$ and $\mathcal{C}_{i,0}$ are the
sets of constellation points for which the $i\text{th}$ bit is
equal to 1 and 0, respectively. $\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]$
is the vector of LLRs that serves as prior knowledge on the $K$ bits that are mapped to
a constellation point,
and $\Pr(c\lvert\mathbf{p})$ is the prior probability on the constellation symbol $c$:

$$
\Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)
$$
    
where $\ell(c)_k$ is the $k^{th}$ bit label of $c$, where 0 is
replaced by -1.
The definition of the LLR has been
chosen such that it is equivalent with that of logits. This is
different from many textbooks in communications, where the LLR is
defined as $LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)$.
    
With the “maxlog” demapping method, LLRs for the $i\text{th}$ bit
are approximated like

$$
\begin{split}\begin{align}
    LLR(i) &\approx\ln\left(\frac{
        \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
            \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }{
        \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
            \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
        }\right)\\
        &= \max_{c\in\mathcal{C}_{i,0}}
            \left(\ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right)-\frac{|y-c|^2}{N_o}\right) -
         \max_{c\in\mathcal{C}_{i,1}}\left( \ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right) - \frac{|y-c|^2}{N_o}\right)
        .
\end{align}\end{split}
$$


### SymbolDemapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symboldemapper" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``SymbolDemapper`(<em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`with_prior``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#SymbolDemapper">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolDemapper" title="Permalink to this definition"></a>
    
Computes normalized log-probabilities (logits) or hard-decisions on symbols
for a tensor of received symbols.
If the `with_prior` flag is set, prior knowldge on the transmitted constellation points is assumed to be available.
The demapping function is fully differentiable when soft-values are
computed.
Parameters
 
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the demapper provides hard-decided symbols instead of soft-values.
Defaults to <cite>False</cite>.
- **with_prior** (<em>bool</em>) – If <cite>True</cite>, it is assumed that prior knowledge on the constellation points is available.
This prior information is given as log-probabilities (logits) as an additional input to the layer.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, no) or (y, prior, no)** – Tuple:
- **y** (<em>[…,n], tf.complex</em>) – The received symbols.
- **prior** (<em>[num_points] or […,num_points], tf.float</em>) – Prior for every symbol as log-probabilities (logits).
It can be provided either as a tensor of shape <cite>[num_points]</cite> for the
entire input batch, or as a tensor that is “broadcastable”
to <cite>[…, n, num_points]</cite>.
Only required if the `with_prior` flag is set.
- **no** (<em>Scalar or […,n], tf.float</em>) – The noise variance estimate. It can be provided either as scalar
for the entire input batch or as a tensor that is “broadcastable” to
`y`.


Output
    
<em>[…,n, num_points] or […,n], tf.float</em> – A tensor of shape <cite>[…,n, num_points]</cite> of logits for every constellation
point if <cite>hard_out</cite> is set to <cite>False</cite>.
Otherwise, a tensor of shape <cite>[…,n]</cite> of hard-decisions on the symbols.



**Note**
    
The normalized log-probability for the constellation point $c$ is computed according to

$$
\ln\left(\Pr\left(c \lvert y,\mathbf{p}\right)\right) = \ln\left( \frac{\exp\left(-\frac{|y-c|^2}{N_0} + p_c \right)}{\sum_{c'\in\mathcal{C}} \exp\left(-\frac{|y-c'|^2}{N_0} + p_{c'} \right)} \right)
$$
    
where $\mathcal{C}$ is the set of constellation points used for modulation,
and $\mathbf{p} = \left\{p_c \lvert c \in \mathcal{C}\right\}$ the prior information on constellation points given as log-probabilities
and which is set to $\mathbf{0}$ if no prior information on the constellation points is assumed to be available.

