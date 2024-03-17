# Mapping<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#mapping" title="Permalink to this headline"></a>
    
This module contains classes and functions related to mapping
of bits to constellation symbols and demapping of soft-symbols
to log-likelihood ratios (LLRs). The key components are the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Mapper" title="sionna.mapping.Mapper">`Mapper`</a>,
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a>. A <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
can be made trainable to enable learning of geometric shaping.

# Table of Content
## Demapping<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#demapping" title="Permalink to this headline"></a>
### SymbolDemapperWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symboldemapperwithprior" title="Permalink to this headline"></a>
## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#utility-functions" title="Permalink to this headline"></a>
### SymbolLogits2LLRs<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symbollogits2llrs" title="Permalink to this headline"></a>
### LLRs2SymbolLogits<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#llrs2symbollogits" title="Permalink to this headline"></a>
### SymbolLogits2LLRsWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symbollogits2llrswithprior" title="Permalink to this headline"></a>
  
  

### SymbolDemapperWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symboldemapperwithprior" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``SymbolDemapperWithPrior`(<em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#SymbolDemapperWithPrior">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolDemapperWithPrior" title="Permalink to this definition"></a>
    
Computes normalized log-probabilities (logits) or hard-decisions on symbols
for a tensor of received symbols, assuming that prior knowledge on the constellation points is available.
The demapping function is fully differentiable when soft-values are
computed.
    
This class is deprecated as the functionality has been integrated
into <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolDemapper" title="sionna.mapping.SymbolDemapper">`SymbolDemapper`</a>.
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
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype of <cite>y</cite>. Defaults to tf.complex64.
The output dtype is the corresponding real dtype (tf.float32 or tf.float64).


Input
 
- **(y, prior, no)** – Tuple:
- **y** (<em>[…,n], tf.complex</em>) – The received symbols.
- **prior** (<em>[num_points] or […,num_points], tf.float</em>) – Prior for every symbol as log-probabilities (logits).
It can be provided either as a tensor of shape <cite>[num_points]</cite> for the
entire input batch, or as a tensor that is “broadcastable”
to <cite>[…, n, num_points]</cite>.
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
and $\mathbf{p} = \left\{p_c \lvert c \in \mathcal{C}\right\}$ the prior information on constellation points given as log-probabilities.

## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#utility-functions" title="Permalink to this headline"></a>

### SymbolLogits2LLRs<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symbollogits2llrs" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``SymbolLogits2LLRs`(<em class="sig-param">`method`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`with_prior``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#SymbolLogits2LLRs">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolLogits2LLRs" title="Permalink to this definition"></a>
    
Computes log-likelihood ratios (LLRs) or hard-decisions on bits
from a tensor of logits (i.e., unnormalized log-probabilities) on constellation points.
If the flag `with_prior` is set, prior knowledge on the bits is assumed to be available.
Parameters
 
- **method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The method used for computing the LLRs.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the layer provides hard-decided bits instead of soft-values.
Defaults to <cite>False</cite>.
- **with_prior** (<em>bool</em>) – If <cite>True</cite>, it is assumed that prior knowledge on the bits is available.
This prior information is given as LLRs as an additional input to the layer.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.float32</em><em>, </em><em>tf.float64</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype for the input and output.
Defaults to <cite>tf.float32</cite>.


Input
 
- **logits or (logits, prior)** – Tuple:
- **logits** (<em>[…,n, num_points], tf.float</em>) – Logits on constellation points.
- **prior** (<em>[num_bits_per_symbol] or […n, num_bits_per_symbol], tf.float</em>) – Prior for every bit as LLRs.
It can be provided either as a tensor of shape <cite>[num_bits_per_symbol]</cite>
for the entire input batch, or as a tensor that is “broadcastable”
to <cite>[…, n, num_bits_per_symbol]</cite>.
Only required if the `with_prior` flag is set.


Output
    
<em>[…,n, num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit.



**Note**
    
With the “app” method, the LLR for the $i\text{th}$ bit
is computed according to

$$
LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert \mathbf{z},\mathbf{p}\right)}{\Pr\left(b_i=0\lvert \mathbf{z},\mathbf{p}\right)}\right) =\ln\left(\frac{
        \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
        e^{z_c}
        }{
        \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
        e^{z_c}
        }\right)
$$
    
where $\mathcal{C}_{i,1}$ and $\mathcal{C}_{i,0}$ are the
sets of $2^K$ constellation points for which the $i\text{th}$ bit is
equal to 1 and 0, respectively. $\mathbf{z} = \left[z_{c_0},\dots,z_{c_{2^K-1}}\right]$ is the vector of logits on the constellation points, $\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]$
is the vector of LLRs that serves as prior knowledge on the $K$ bits that are mapped to
a constellation point and is set to $\mathbf{0}$ if no prior knowledge is assumed to be available,
and $\Pr(c\lvert\mathbf{p})$ is the prior probability on the constellation symbol $c$:

$$
\Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \Pr\left(b_k = \ell(c)_k \lvert\mathbf{p} \right)
= \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)
$$
    
where $\ell(c)_k$ is the $k^{th}$ bit label of $c$, where 0 is
replaced by -1.
The definition of the LLR has been
chosen such that it is equivalent with that of logits. This is
different from many textbooks in communications, where the LLR is
defined as $LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)$.
    
With the “maxlog” method, LLRs for the $i\text{th}$ bit
are approximated like

$$
\begin{align}
    LLR(i) &\approx\ln\left(\frac{
        \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
            e^{z_c}
        }{
        \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
            e^{z_c}
        }\right)
        .
\end{align}
$$


### LLRs2SymbolLogits<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#llrs2symbollogits" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``LLRs2SymbolLogits`(<em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#LLRs2SymbolLogits">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.LLRs2SymbolLogits" title="Permalink to this definition"></a>
    
Computes logits (i.e., unnormalized log-probabilities) or hard decisions
on constellation points from a tensor of log-likelihood ratios (LLRs) on bits.
Parameters
 
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the layer provides hard-decided constellation points instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.float32</em><em>, </em><em>tf.float64</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype for the input and output.
Defaults to <cite>tf.float32</cite>.


Input
    
**llrs** (<em>[…, n, num_bits_per_symbol], tf.float</em>) – LLRs for every bit.

Output
    
<em>[…,n, num_points], tf.float or […, n], tf.int32</em> – Logits or hard-decisions on constellation points.



**Note**
    
The logit for the constellation $c$ point
is computed according to

$$
\begin{split}\begin{align}
    \log{\left(\Pr\left(c\lvert LLRs \right)\right)}
        &= \log{\left(\prod_{k=0}^{K-1} \Pr\left(b_k = \ell(c)_k \lvert LLRs \right)\right)}\\
        &= \log{\left(\prod_{k=0}^{K-1} \text{sigmoid}\left(LLR(k) \ell(c)_k\right)\right)}\\
        &= \sum_{k=0}^{K-1} \log{\left(\text{sigmoid}\left(LLR(k) \ell(c)_k\right)\right)}
\end{align}\end{split}
$$
    
where $\ell(c)_k$ is the $k^{th}$ bit label of $c$, where 0 is
replaced by -1.
The definition of the LLR has been
chosen such that it is equivalent with that of logits. This is
different from many textbooks in communications, where the LLR is
defined as $LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)$.

### SymbolLogits2LLRsWithPrior<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symbollogits2llrswithprior" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``SymbolLogits2LLRsWithPrior`(<em class="sig-param">`method`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`hard_out``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#SymbolLogits2LLRsWithPrior">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolLogits2LLRsWithPrior" title="Permalink to this definition"></a>
    
Computes log-likelihood ratios (LLRs) or hard-decisions on bits
from a tensor of logits (i.e., unnormalized log-probabilities) on constellation points,
assuming that prior knowledge on the bits is available.
    
This class is deprecated as the functionality has been integrated
into <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolLogits2LLRs" title="sionna.mapping.SymbolLogits2LLRs">`SymbolLogits2LLRs`</a>.
Parameters
 
- **method** (<em>One of</em><em> [</em><em>"app"</em><em>, </em><em>"maxlog"</em><em>]</em><em>, </em><em>str</em>) – The method used for computing the LLRs.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
- **hard_out** (<em>bool</em>) – If <cite>True</cite>, the layer provides hard-decided bits instead of soft-values.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.float32</em><em>, </em><em>tf.float64</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype for the input and output.
Defaults to <cite>tf.float32</cite>.


Input
 
- **(logits, prior)** – Tuple:
- **logits** (<em>[…,n, num_points], tf.float</em>) – Logits on constellation points.
- **prior** (<em>[num_bits_per_symbol] or […n, num_bits_per_symbol], tf.float</em>) – Prior for every bit as LLRs.
It can be provided either as a tensor of shape <cite>[num_bits_per_symbol]</cite> for the
entire input batch, or as a tensor that is “broadcastable”
to <cite>[…, n, num_bits_per_symbol]</cite>.


Output
    
<em>[…,n, num_bits_per_symbol], tf.float</em> – LLRs or hard-decisions for every bit.



**Note**
    
With the “app” method, the LLR for the $i\text{th}$ bit
is computed according to

$$
LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert \mathbf{z},\mathbf{p}\right)}{\Pr\left(b_i=0\lvert \mathbf{z},\mathbf{p}\right)}\right) =\ln\left(\frac{
        \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
        e^{z_c}
        }{
        \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
        e^{z_c}
        }\right)
$$
    
where $\mathcal{C}_{i,1}$ and $\mathcal{C}_{i,0}$ are the
sets of $2^K$ constellation points for which the $i\text{th}$ bit is
equal to 1 and 0, respectively. $\mathbf{z} = \left[z_{c_0},\dots,z_{c_{2^K-1}}\right]$ is the vector of logits on the constellation points, $\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]$
is the vector of LLRs that serves as prior knowledge on the $K$ bits that are mapped to
a constellation point,
and $\Pr(c\lvert\mathbf{p})$ is the prior probability on the constellation symbol $c$:

$$
\Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \Pr\left(b_k = \ell(c)_k \lvert\mathbf{p} \right)
= \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)
$$
    
where $\ell(c)_k$ is the $k^{th}$ bit label of $c$, where 0 is
replaced by -1.
The definition of the LLR has been
chosen such that it is equivalent with that of logits. This is
different from many textbooks in communications, where the LLR is
defined as $LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)$.
    
With the “maxlog” method, LLRs for the $i\text{th}$ bit
are approximated like

$$
\begin{align}
    LLR(i) &\approx\ln\left(\frac{
        \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
            e^{z_c}
        }{
        \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
            e^{z_c}
        }\right)
        .
\end{align}
$$


