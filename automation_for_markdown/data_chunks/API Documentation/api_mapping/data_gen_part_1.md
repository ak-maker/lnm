# Mapping<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#mapping" title="Permalink to this headline"></a>
    
This module contains classes and functions related to mapping
of bits to constellation symbols and demapping of soft-symbols
to log-likelihood ratios (LLRs). The key components are the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Mapper" title="sionna.mapping.Mapper">`Mapper`</a>,
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a>. A <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
can be made trainable to enable learning of geometric shaping.

# Table of Content
## Constellations<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#constellations" title="Permalink to this headline"></a>
### Constellation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#constellation" title="Permalink to this headline"></a>
### qam<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#qam" title="Permalink to this headline"></a>
### pam<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#pam" title="Permalink to this headline"></a>
### pam_gray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#pam-gray" title="Permalink to this headline"></a>
## Mapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#mapper" title="Permalink to this headline"></a>
## Demapping<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#demapping" title="Permalink to this headline"></a>
  
  

## Constellations<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#constellations" title="Permalink to this headline"></a>

### Constellation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#constellation" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``Constellation`(<em class="sig-param">`constellation_type`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`initial_value``=``None`</em>, <em class="sig-param">`normalize``=``True`</em>, <em class="sig-param">`center``=``False`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#Constellation">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="Permalink to this definition"></a>
    
Constellation that can be used by a (de)mapper.
    
This class defines a constellation, i.e., a complex-valued vector of
constellation points. A constellation can be trainable. The binary
representation of the index of an element of this vector corresponds
to the bit label of the constellation point. This implicit bit
labeling is used by the `Mapper` and `Demapper` classes.
Parameters
 
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, the constellation points are randomly initialized
if no `initial_value` is provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
- **initial_value** ($[2^\text{num_bits_per_symbol}]$, NumPy array or Tensor) – Initial values of the constellation points. If `normalize` or
`center` are <cite>True</cite>, the initial constellation might be changed.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the constellation is normalized to have unit power.
Defaults to <cite>True</cite>.
- **center** (<em>bool</em>) – If <cite>True</cite>, the constellation is ensured to have zero mean.
Defaults to <cite>False</cite>.
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the constellation points are trainable variables.
Defaults to <cite>False</cite>.
- **dtype** (<em>[</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>]</em><em>, </em><em>tf.DType</em>) – The dtype of the constellation.


Output
    
$[2^\text{num_bits_per_symbol}]$, `dtype` – The constellation.



**Note**
    
One can create a trainable PAM/QAM constellation. This is
equivalent to creating a custom trainable constellation which is
initialized with PAM/QAM constellation points.

<em class="property">`property` </em>`center`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation.center" title="Permalink to this definition"></a>
    
Indicates if the constellation is centered.


`create_or_check_constellation`(<em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#Constellation.create_or_check_constellation">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation.create_or_check_constellation" title="Permalink to this definition"></a>
    
Static method for conviently creating a constellation object or checking that an existing one
is consistent with requested settings.
    
If `constellation` is <cite>None</cite>, then this method creates a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
object of type `constellation_type` and with `num_bits_per_symbol` bits per symbol.
Otherwise, this method checks that <cite>constellation</cite> is consistent with `constellation_type` and
`num_bits_per_symbol`. If it is, `constellation` is returned. Otherwise, an assertion is raised.
Input
 
- **constellation_type** (<em>One of [“qam”, “pam”, “custom”], str</em>) – For “custom”, an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<em>Constellation</em>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or
<cite>None</cite>. In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.


Output
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> – A constellation object.




<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation.normalize" title="Permalink to this definition"></a>
    
Indicates if the constellation is normalized or not.


<em class="property">`property` </em>`num_bits_per_symbol`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation.num_bits_per_symbol" title="Permalink to this definition"></a>
    
The number of bits per constellation symbol.


<em class="property">`property` </em>`points`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation.points" title="Permalink to this definition"></a>
    
The (possibly) centered and normalized constellation points.


`show`(<em class="sig-param">`labels``=``True`</em>, <em class="sig-param">`figsize``=``(7,` `7)`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#Constellation.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation.show" title="Permalink to this definition"></a>
    
Generate a scatter-plot of the constellation.
Input
 
- **labels** (<em>bool</em>) – If <cite>True</cite>, the bit labels will be drawn next to each constellation
point. Defaults to <cite>True</cite>.
- **figsize** (<em>Two-element Tuple, float</em>) – Width and height in inches. Defaults to <cite>(7,7)</cite>.


Output
    
<em>matplotlib.figure.Figure</em> – A handle to a matplot figure object.




### qam<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#qam" title="Permalink to this headline"></a>

`sionna.mapping.``qam`(<em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`normalize``=``True`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#qam">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.qam" title="Permalink to this definition"></a>
    
Generates a QAM constellation.
    
This function generates a complex-valued vector, where each element is
a constellation point of an M-ary QAM constellation. The bit
label of the `n` th point is given by the length-`num_bits_per_symbol`
binary represenation of `n`.
Input
 
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation point.
Must be a multiple of two, e.g., 2, 4, 6, 8, etc.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the constellation is normalized to have unit power.
Defaults to <cite>True</cite>.


Output
    
$[2^{\text{num_bits_per_symbol}}]$, np.complex64 – The QAM constellation.



**Note**
    
The bit label of the nth constellation point is given by the binary
representation of its position within the array and can be obtained
through `np.binary_repr(n,` `num_bits_per_symbol)`.
    
The normalization factor of a QAM constellation is given in
closed-form as:

$$
\sqrt{\frac{1}{2^{n-2}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}
$$
    
where $n= \text{num_bits_per_symbol}/2$ is the number of bits
per dimension.
    
This algorithm is a recursive implementation of the expressions found in
Section 5.1 of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#gppts38211" id="id1">[3GPPTS38211]</a>. It is used in the 5G standard.

### pam<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#pam" title="Permalink to this headline"></a>

`sionna.mapping.``pam`(<em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`normalize``=``True`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#pam">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.pam" title="Permalink to this definition"></a>
    
Generates a PAM constellation.
    
This function generates a real-valued vector, where each element is
a constellation point of an M-ary PAM constellation. The bit
label of the `n` th point is given by the length-`num_bits_per_symbol`
binary represenation of `n`.
Input
 
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation point.
Must be positive.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the constellation is normalized to have unit power.
Defaults to <cite>True</cite>.


Output
    
$[2^{\text{num_bits_per_symbol}}]$, np.float32 – The PAM constellation.



**Note**
    
The bit label of the nth constellation point is given by the binary
representation of its position within the array and can be obtained
through `np.binary_repr(n,` `num_bits_per_symbol)`.
    
The normalization factor of a PAM constellation is given in
closed-form as:

$$
\sqrt{\frac{1}{2^{n-1}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}
$$
    
where $n= \text{num_bits_per_symbol}$ is the number of bits
per symbol.
    
This algorithm is a recursive implementation of the expressions found in
Section 5.1 of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#gppts38211" id="id2">[3GPPTS38211]</a>. It is used in the 5G standard.

### pam_gray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#pam-gray" title="Permalink to this headline"></a>

`sionna.mapping.``pam_gray`(<em class="sig-param">`b`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#pam_gray">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.pam_gray" title="Permalink to this definition"></a>
    
Maps a vector of bits to a PAM constellation points with Gray labeling.
    
This recursive function maps a binary vector to Gray-labelled PAM
constellation points. It can be used to generated QAM constellations.
The constellation is not normalized.
Input
    
**b** (<em>[n], NumPy array</em>) – Tensor with with binary entries.

Output
    
<em>signed int</em> – The PAM constellation point taking values in
$\{\pm 1,\pm 3,\dots,\pm (2^n-1)\}$.



**Note**
    
This algorithm is a recursive implementation of the expressions found in
Section 5.1 of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#gppts38211" id="id3">[3GPPTS38211]</a>. It is used in the 5G standard.

## Mapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#mapper" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.mapping.``Mapper`(<em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`return_indices``=``False`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/mapping.html#Mapper">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Mapper" title="Permalink to this definition"></a>
    
Maps binary tensors to points of a constellation.
    
This class defines a layer that maps a tensor of binary values
to a tensor of points from a provided constellation.
Parameters
 
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or
<cite>None</cite>. In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **return_indices** (<em>bool</em>) – If enabled, symbol indices are additionally returned.
Defaults to <cite>False</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>]</em><em>, </em><em>tf.DType</em>) – The output dtype. Defaults to tf.complex64.


Input
    
<em>[…, n], tf.float or tf.int</em> – Tensor with with binary entries.

Output
 
- <em>[…,n/Constellation.num_bits_per_symbol], tf.complex</em> – The mapped constellation symbols.
- <em>[…,n/Constellation.num_bits_per_symbol], tf.int32</em> – The symbol indices corresponding to the constellation symbols.
Only returned if `return_indices` is set to True.




**Note**
    
The last input dimension must be an integer multiple of the
number of bits per constellation symbol.

<em class="property">`property` </em>`constellation`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#sionna.mapping.Mapper.constellation" title="Permalink to this definition"></a>
    
The Constellation used by the Mapper.


## Demapping<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mapping.html#demapping" title="Permalink to this headline"></a>

