# Mapping<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#mapping" title="Permalink to this headline"></a>
    
This module contains classes and functions related to mapping
of bits to constellation symbols and demapping of soft-symbols
to log-likelihood ratios (LLRs). The key components are the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Mapper" title="sionna.mapping.Mapper">`Mapper`</a>,
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a>. A <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
can be made trainable to enable learning of geometric shaping.

# Table of Content
## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#utility-functions" title="Permalink to this headline"></a>
### SymbolLogits2Moments<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symbollogits2moments" title="Permalink to this headline"></a>
### SymbolInds2Bits<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symbolinds2bits" title="Permalink to this headline"></a>
### PAM2QAM<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#pam2qam" title="Permalink to this headline"></a>
### QAM2PAM<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#qam2pam" title="Permalink to this headline"></a>
  
  

### SymbolLogits2Moments<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symbollogits2moments" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``SymbolLogits2Moments`(<em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#SymbolLogits2Moments">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolLogits2Moments" title="Permalink to this definition"></a>
    
Computes the mean and variance of a constellation from logits (unnormalized log-probabilities) on the
constellation points.
    
More precisely, given a constellation $\mathcal{C} = \left[ c_0,\dots,c_{N-1} \right]$ of size $N$, this layer computes the mean and variance
according to

$$
\begin{split}\begin{align}
    \mu &= \sum_{n = 0}^{N-1} c_n \Pr \left(c_n \lvert \mathbf{\ell} \right)\\
    \nu &= \sum_{n = 0}^{N-1} \left( c_n - \mu \right)^2 \Pr \left(c_n \lvert \mathbf{\ell} \right)
\end{align}\end{split}
$$
    
where $\mathbf{\ell} = \left[ \ell_0, \dots, \ell_{N-1} \right]$ are the logits, and

$$
\Pr \left(c_n \lvert \mathbf{\ell} \right) = \frac{\exp \left( \ell_n \right)}{\sum_{i=0}^{N-1} \exp \left( \ell_i \right) }.
$$

Parameters
 
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or <cite>None</cite>.
In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **dtype** (<em>One of</em><em> [</em><em>tf.float32</em><em>, </em><em>tf.float64</em><em>] </em><em>tf.DType</em><em> (</em><em>dtype</em><em>)</em>) – The dtype for the input and output.
Defaults to <cite>tf.float32</cite>.


Input
    
**logits** (<em>[…,n, num_points], tf.float</em>) – Logits on constellation points.

Output
 
- **mean** (<em>[…,n], tf.float</em>) – Mean of the constellation.
- **var** (<em>[…,n], tf.float</em>) – Variance of the constellation




### SymbolInds2Bits<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#symbolinds2bits" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``SymbolInds2Bits`(<em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#SymbolInds2Bits">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.SymbolInds2Bits" title="Permalink to this definition"></a>
    
Transforms symbol indices to their binary representations.
Parameters
 
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per constellation symbol
- **dtype** (<em>tf.DType</em>) – Output dtype. Defaults to <cite>tf.float32</cite>.


Input
    
<em>Tensor, tf.int</em> – Symbol indices

Output
    
<em>input.shape + [num_bits_per_symbol], dtype</em> – Binary representation of symbol indices



### PAM2QAM<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#pam2qam" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``PAM2QAM`(<em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`hard_in_out``=``True`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#PAM2QAM">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.PAM2QAM" title="Permalink to this definition"></a>
    
Transforms PAM symbol indices/logits to QAM symbol indices/logits.
    
For two PAM constellation symbol indices or logits, corresponding to
the real and imaginary components of a QAM constellation,
compute the QAM symbol index or logits.
Parameters
 
- **num_bits_per_symbol** (<em>int</em>) – Number of bits per QAM constellation symbol, e.g., 4 for QAM16
- **hard_in_out** (<em>bool</em>) – Determines if inputs and outputs are indices or logits over
constellation symbols.
Defaults to <cite>True</cite>.


Input
 
- **pam1** (<em>Tensor, tf.int, or […,2**(num_bits_per_symbol/2)], tf.float</em>) – Indices or logits for the first PAM constellation
- **pam2** (<em>Tensor, tf.int, or […,2**(num_bits_per_symbol/2)], tf.float</em>) – Indices or logits for the second PAM constellation


Output
    
**qam** (<em>Tensor, tf.int, or […,2**num_bits_per_symbol], tf.float</em>) – Indices or logits for the corresponding QAM constellation



### QAM2PAM<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#qam2pam" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``QAM2PAM`(<em class="sig-param">`num_bits_per_symbol`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#QAM2PAM">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.QAM2PAM" title="Permalink to this definition"></a>
    
Transforms QAM symbol indices to PAM symbol indices.
    
For indices in a QAM constellation, computes the corresponding indices
for the two PAM constellations corresponding the real and imaginary
components of the QAM constellation.
Parameters
    
**num_bits_per_symbol** (<em>int</em>) – The number of bits per QAM constellation symbol, e.g., 4 for QAM16.

Input
    
**ind_qam** (<em>Tensor, tf.int</em>) – Indices in the QAM constellation

Output
 
- **ind_pam1** (<em>Tensor, tf.int</em>) – Indices for the first component of the corresponding PAM modulation
- **ind_pam2** (<em>Tensor, tf.int</em>) – Indices for the first component of the corresponding PAM modulation





References:
3GPPTS38211(<a href="https://nvlabs.github.io/sionna/api/mapping.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/mapping.html#id2">2</a>,<a href="https://nvlabs.github.io/sionna/api/mapping.html#id3">3</a>)
    
ETSI TS 38.211 “5G NR Physical channels and modulation”, V16.2.0, Jul. 2020
<a class="reference external" href="https://www.3gpp.org/ftp/Specs/archive/38_series/38.211/38211-h00.zip">https://www.3gpp.org/ftp/Specs/archive/38_series/38.211/38211-h00.zip</a>



