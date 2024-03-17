# Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#utility-functions" title="Permalink to this headline"></a>
    
The utilities sub-package of the Sionna library contains many convenience
functions as well as extensions to existing TensorFlow functions.

# Table of Content
## Tensors<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#tensors" title="Permalink to this headline"></a>
### matrix_sqrt<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#matrix-sqrt" title="Permalink to this headline"></a>
### matrix_sqrt_inv<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#matrix-sqrt-inv" title="Permalink to this headline"></a>
### matrix_inv<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#matrix-inv" title="Permalink to this headline"></a>
### matrix_pinv<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#matrix-pinv" title="Permalink to this headline"></a>
## Miscellaneous<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#miscellaneous" title="Permalink to this headline"></a>
### BinarySource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#binarysource" title="Permalink to this headline"></a>
### SymbolSource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#symbolsource" title="Permalink to this headline"></a>
### QAMSource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#qamsource" title="Permalink to this headline"></a>
### PAMSource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#pamsource" title="Permalink to this headline"></a>
  
  

### matrix_sqrt<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#matrix-sqrt" title="Permalink to this headline"></a>

`sionna.utils.``matrix_sqrt`(<em class="sig-param">`tensor`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#matrix_sqrt">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.matrix_sqrt" title="Permalink to this definition"></a>
    
Computes the square root of a matrix.
    
Given a batch of Hermitian positive semi-definite matrices
$\mathbf{A}$, returns matrices $\mathbf{B}$,
such that $\mathbf{B}\mathbf{B}^H = \mathbf{A}$.
    
The two inner dimensions are assumed to correspond to the matrix rows
and columns, respectively.
Parameters
    
**tensor** (<em>[</em><em>...</em><em>, </em><em>M</em><em>, </em><em>M</em><em>]</em>) – A tensor of rank greater than or equal
to two.

Returns
    
A tensor of the same shape and type as `tensor` containing
the matrix square root of its last two dimensions.



**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.config.xla_compat=true`.
See `xla_compat`.

### matrix_sqrt_inv<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#matrix-sqrt-inv" title="Permalink to this headline"></a>

`sionna.utils.``matrix_sqrt_inv`(<em class="sig-param">`tensor`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#matrix_sqrt_inv">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.matrix_sqrt_inv" title="Permalink to this definition"></a>
    
Computes the inverse square root of a Hermitian matrix.
    
Given a batch of Hermitian positive definite matrices
$\mathbf{A}$, with square root matrices $\mathbf{B}$,
such that $\mathbf{B}\mathbf{B}^H = \mathbf{A}$, the function
returns $\mathbf{B}^{-1}$, such that
$\mathbf{B}^{-1}\mathbf{B}=\mathbf{I}$.
    
The two inner dimensions are assumed to correspond to the matrix rows
and columns, respectively.
Parameters
    
**tensor** (<em>[</em><em>...</em><em>, </em><em>M</em><em>, </em><em>M</em><em>]</em>) – A tensor of rank greater than or equal
to two.

Returns
    
A tensor of the same shape and type as `tensor` containing
the inverse matrix square root of its last two dimensions.



**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### matrix_inv<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#matrix-inv" title="Permalink to this headline"></a>

`sionna.utils.``matrix_inv`(<em class="sig-param">`tensor`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#matrix_inv">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.matrix_inv" title="Permalink to this definition"></a>
    
Computes the inverse of a Hermitian matrix.
    
Given a batch of Hermitian positive definite matrices
$\mathbf{A}$, the function
returns $\mathbf{A}^{-1}$, such that
$\mathbf{A}^{-1}\mathbf{A}=\mathbf{I}$.
    
The two inner dimensions are assumed to correspond to the matrix rows
and columns, respectively.
Parameters
    
**tensor** (<em>[</em><em>...</em><em>, </em><em>M</em><em>, </em><em>M</em><em>]</em>) – A tensor of rank greater than or equal
to two.

Returns
    
A tensor of the same shape and type as `tensor`, containing
the inverse of its last two dimensions.



**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### matrix_pinv<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#matrix-pinv" title="Permalink to this headline"></a>

`sionna.utils.``matrix_pinv`(<em class="sig-param">`tensor`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#matrix_pinv">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.matrix_pinv" title="Permalink to this definition"></a>
    
Computes the Moore–Penrose (or pseudo) inverse of a matrix.
    
Given a batch of $M \times K$ matrices $\mathbf{A}$ with rank
$K$ (i.e., linearly independent columns), the function returns
$\mathbf{A}^+$, such that
$\mathbf{A}^{+}\mathbf{A}=\mathbf{I}_K$.
    
The two inner dimensions are assumed to correspond to the matrix rows
and columns, respectively.
Parameters
    
**tensor** (<em>[</em><em>...</em><em>, </em><em>M</em><em>, </em><em>K</em><em>]</em>) – A tensor of rank greater than or equal
to two.

Returns
    
A tensor of shape ([…, K,K]) of the same type as `tensor`,
containing the pseudo inverse of its last two dimensions.



**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.config.xla_compat=true`.
See `xla_compat`.

## Miscellaneous<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#miscellaneous" title="Permalink to this headline"></a>

### BinarySource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#binarysource" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.utils.``BinarySource`(<em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`seed``=``None`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#BinarySource">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.BinarySource" title="Permalink to this definition"></a>
    
Layer generating random binary tensors.
Parameters
 
- **dtype** (<em>tf.DType</em>) – Defines the output datatype of the layer.
Defaults to <cite>tf.float32</cite>.
- **seed** (<em>int</em><em> or </em><em>None</em>) – Set the seed for the random generator used to generate the bits.
Set to <cite>None</cite> for random initialization of the RNG.


Input
    
**shape** (<em>1D tensor/array/list, int</em>) – The desired shape of the output tensor.

Output
    
`shape`, `dtype` – Tensor filled with random binary values.



### SymbolSource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#symbolsource" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.utils.``SymbolSource`(<em class="sig-param">`constellation_type``=``None`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`constellation``=``None`</em>, <em class="sig-param">`return_indices``=``False`</em>, <em class="sig-param">`return_bits``=``False`</em>, <em class="sig-param">`seed``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#SymbolSource">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.SymbolSource" title="Permalink to this definition"></a>
    
Layer generating a tensor of arbitrary shape filled with random constellation symbols.
Optionally, the symbol indices and/or binary representations of the
constellation symbols can be returned.
Parameters
 
- **constellation_type** (<em>One of</em><em> [</em><em>"qam"</em><em>, </em><em>"pam"</em><em>, </em><em>"custom"</em><em>]</em><em>, </em><em>str</em>) – For “custom”, an instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a>
must be provided.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol.
Only required for `constellation_type` in [“qam”, “pam”].
- **constellation** (<a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation"><em>Constellation</em></a>) – An instance of <a class="reference internal" href="mapping.html#sionna.mapping.Constellation" title="sionna.mapping.Constellation">`Constellation`</a> or
<cite>None</cite>. In the latter case, `constellation_type`
and `num_bits_per_symbol` must be provided.
- **return_indices** (<em>bool</em>) – If enabled, the function also returns the symbol indices.
Defaults to <cite>False</cite>.
- **return_bits** (<em>bool</em>) – If enabled, the function also returns the binary symbol
representations (i.e., bit labels).
Defaults to <cite>False</cite>.
- **seed** (<em>int</em><em> or </em><em>None</em>) – The seed for the random generator.
<cite>None</cite> leads to a random initialization of the RNG.
Defaults to <cite>None</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>]</em><em>, </em><em>tf.DType</em>) – The output dtype. Defaults to tf.complex64.


Input
    
**shape** (<em>1D tensor/array/list, int</em>) – The desired shape of the output tensor.

Output
 
- **symbols** (`shape`, `dtype`) – Tensor filled with random symbols of the chosen `constellation_type`.
- **symbol_indices** (`shape`, tf.int32) – Tensor filled with the symbol indices.
Only returned if `return_indices` is <cite>True</cite>.
- **bits** ([`shape`, `num_bits_per_symbol`], tf.int32) – Tensor filled with the binary symbol representations (i.e., bit labels).
Only returned if `return_bits` is <cite>True</cite>.




### QAMSource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#qamsource" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.utils.``QAMSource`(<em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`return_indices``=``False`</em>, <em class="sig-param">`return_bits``=``False`</em>, <em class="sig-param">`seed``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#QAMSource">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.QAMSource" title="Permalink to this definition"></a>
    
Layer generating a tensor of arbitrary shape filled with random QAM symbols.
Optionally, the symbol indices and/or binary representations of the
constellation symbols can be returned.
Parameters
 
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 4 for QAM16.
- **return_indices** (<em>bool</em>) – If enabled, the function also returns the symbol indices.
Defaults to <cite>False</cite>.
- **return_bits** (<em>bool</em>) – If enabled, the function also returns the binary symbol
representations (i.e., bit labels).
Defaults to <cite>False</cite>.
- **seed** (<em>int</em><em> or </em><em>None</em>) – The seed for the random generator.
<cite>None</cite> leads to a random initialization of the RNG.
Defaults to <cite>None</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>]</em><em>, </em><em>tf.DType</em>) – The output dtype. Defaults to tf.complex64.


Input
    
**shape** (<em>1D tensor/array/list, int</em>) – The desired shape of the output tensor.

Output
 
- **symbols** (`shape`, `dtype`) – Tensor filled with random QAM symbols.
- **symbol_indices** (`shape`, tf.int32) – Tensor filled with the symbol indices.
Only returned if `return_indices` is <cite>True</cite>.
- **bits** ([`shape`, `num_bits_per_symbol`], tf.int32) – Tensor filled with the binary symbol representations (i.e., bit labels).
Only returned if `return_bits` is <cite>True</cite>.




### PAMSource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#pamsource" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.utils.``PAMSource`(<em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`return_indices``=``False`</em>, <em class="sig-param">`return_bits``=``False`</em>, <em class="sig-param">`seed``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#PAMSource">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.PAMSource" title="Permalink to this definition"></a>
    
Layer generating a tensor of arbitrary shape filled with random PAM symbols.
Optionally, the symbol indices and/or binary representations of the
constellation symbols can be returned.
Parameters
 
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per constellation symbol, e.g., 1 for BPSK.
- **return_indices** (<em>bool</em>) – If enabled, the function also returns the symbol indices.
Defaults to <cite>False</cite>.
- **return_bits** (<em>bool</em>) – If enabled, the function also returns the binary symbol
representations (i.e., bit labels).
Defaults to <cite>False</cite>.
- **seed** (<em>int</em><em> or </em><em>None</em>) – The seed for the random generator.
<cite>None</cite> leads to a random initialization of the RNG.
Defaults to <cite>None</cite>.
- **dtype** (<em>One of</em><em> [</em><em>tf.complex64</em><em>, </em><em>tf.complex128</em><em>]</em><em>, </em><em>tf.DType</em>) – The output dtype. Defaults to tf.complex64.


Input
    
**shape** (<em>1D tensor/array/list, int</em>) – The desired shape of the output tensor.

Output
 
- **symbols** (`shape`, `dtype`) – Tensor filled with random PAM symbols.
- **symbol_indices** (`shape`, tf.int32) – Tensor filled with the symbol indices.
Only returned if `return_indices` is <cite>True</cite>.
- **bits** ([`shape`, `num_bits_per_symbol`], tf.int32) – Tensor filled with the binary symbol representations (i.e., bit labels).
Only returned if `return_bits` is <cite>True</cite>.




