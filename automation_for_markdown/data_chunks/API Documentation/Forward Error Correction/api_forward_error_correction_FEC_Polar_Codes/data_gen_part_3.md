# Polar Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar-codes" title="Permalink to this headline"></a>
    
The Polar code module supports 5G-compliant Polar codes and includes successive cancellation (SC), successive cancellation list (SCL), and belief propagation (BP) decoding.
    
The module supports rate-matching and CRC-aided decoding.
Further, Reed-Muller (RM) code design is available and can be used in combination with the Polar encoding/decoding algorithms.
    
The following code snippets show how to setup and run a rate-matched 5G compliant Polar encoder and a corresponding successive cancellation list (SCL) decoder.
    
First, we need to create instances of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder" title="sionna.fec.polar.encoding.Polar5GEncoder">`Polar5GEncoder`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder" title="sionna.fec.polar.decoding.Polar5GDecoder">`Polar5GDecoder`</a>:
```python
encoder = Polar5GEncoder(k          = 100, # number of information bits (input)
                         n          = 200) # number of codeword bits (output)

decoder = Polar5GDecoder(encoder    = encoder, # connect the Polar decoder to the encoder
                         dec_type   = "SCL", # can be also "SC" or "BP"
                         list_size  = 8)
```

    
Now, the encoder and decoder can be used by:
```python
# --- encoder ---
# u contains the information bits to be encoded and has shape [...,k].
# c contains the polar encoded codewords and has shape [...,n].
c = encoder(u)
# --- decoder ---
# llr contains the log-likelihood ratios from the demapper and has shape [...,n].
# u_hat contains the estimated information bits and has shape [...,k].
u_hat = decoder(llr)
```
# Table of Content
## Polar Decoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar-decoding" title="Permalink to this headline"></a>
### PolarSCLDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polarscldecoder" title="Permalink to this headline"></a>
### PolarBPDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polarbpdecoder" title="Permalink to this headline"></a>
## Polar Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar-utility-functions" title="Permalink to this headline"></a>
### generate_5g_ranking<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#generate-5g-ranking" title="Permalink to this headline"></a>
### generate_polar_transform_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#generate-polar-transform-mat" title="Permalink to this headline"></a>
  
  

### PolarSCLDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polarscldecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.polar.decoding.``PolarSCLDecoder`(<em class="sig-param">`frozen_pos`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`list_size``=``8`</em>, <em class="sig-param">`crc_degree``=``None`</em>, <em class="sig-param">`use_hybrid_sc``=``False`</em>, <em class="sig-param">`use_fast_scl``=``True`</em>, <em class="sig-param">`cpu_only``=``False`</em>, <em class="sig-param">`use_scatter``=``False`</em>, <em class="sig-param">`ind_iil_inv``=``None`</em>, <em class="sig-param">`return_crc_status``=``False`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/decoding.html#PolarSCLDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder" title="Permalink to this definition"></a>
    
Successive cancellation list (SCL) decoder <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#tal-scl" id="id22">[Tal_SCL]</a> for Polar codes
and Polar-like codes.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **frozen_pos** (<em>ndarray</em>) – Array of <cite>int</cite> defining the `n-k` indices of the frozen positions.
- **n** (<em>int</em>) – Defining the codeword length.
- **list_size** (<em>int</em>) – Defaults to 8. Defines the list size of the decoder.
- **crc_degree** (<em>str</em>) – Defining the CRC polynomial to be used. Can be any value from
<cite>{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}</cite>.
- **use_hybrid_sc** (<em>bool</em>) – Defaults to False. If True, SC decoding is applied and only the
codewords with invalid CRC are decoded with SCL. This option
requires an outer CRC specified via `crc_degree`.
Remark: hybrid_sc does not support XLA optimization, i.e.,
<cite>@tf.function(jit_compile=True)</cite>.
- **use_fast_scl** (<em>bool</em>) – Defaults to True. If True, Tree pruning is used to
reduce the decoding complexity. The output is equivalent to the
non-pruned version (besides numerical differences).
- **cpu_only** (<em>bool</em>) – Defaults to False. If True, <cite>tf.py_function</cite> embedding
is used and the decoder runs on the CPU. This option is usually
slower, but also more memory efficient and, in particular,
recommended for larger blocklengths. Remark: cpu_only does not
support XLA optimization <cite>@tf.function(jit_compile=True)</cite>.
- **use_scatter** (<em>bool</em>) – Defaults to False. If True, <cite>tf.tensor_scatter_update</cite> is used for
tensor updates. This option is usually slower, but more memory
efficient.
- **ind_iil_inv** (<em>None</em><em> or </em><em>[</em><em>k+k_crc</em><em>]</em><em>, </em><em>int</em><em> or </em><em>tf.int</em>) – Defaults to None. If not <cite>None</cite>, the sequence is used as inverse
input bit interleaver before evaluating the CRC.
Remark: this only effects the CRC evaluation but the output
sequence is not permuted.
- **return_crc_status** (<em>bool</em>) – Defaults to False. If True, the decoder additionally returns the
CRC status indicating if a codeword was (most likely) correctly
recovered. This is only available if `crc_degree` is not None.
- **output_dtype** (<em>tf.DType</em>) – Defaults to tf.float32. Defines the output datatype of the layer
(internal precision remains tf.float32).


Input
    
**inputs** (<em>[…,n], tf.float32</em>) – 2+D tensor containing the channel LLR values (as logits).

Output
 
- **b_hat** (<em>[…,k], tf.float32</em>) – 2+D tensor containing hard-decided estimations of all <cite>k</cite>
information bits.
- **crc_status** (<em>[…], tf.bool</em>) – CRC status indicating if a codeword was (most likely) correctly
recovered. This is only returned if `return_crc_status` is True.
Note that false positives are possible.


Raises
 
- **AssertionError** – If `n` is not <cite>int</cite>.
- **AssertionError** – If `n` is not a power of 2.
- **AssertionError** – If the number of elements in `frozen_pos` is greater than `n`.
- **AssertionError** – If `frozen_pos` does not consists of <cite>int</cite>.
- **AssertionError** – If `list_size` is not <cite>int</cite>.
- **AssertionError** – If `cpu_only` is not <cite>bool</cite>.
- **AssertionError** – If `use_scatter` is not <cite>bool</cite>.
- **AssertionError** – If `use_fast_scl` is not <cite>bool</cite>.
- **AssertionError** – If `use_hybrid_sc` is not <cite>bool</cite>.
- **AssertionError** – If `list_size` is not a power of 2.
- **ValueError** – If `output_dtype` is not {tf.float16, tf.float32, tf.
    float64}.
- **ValueError** – If `inputs` is not of shape <cite>[…, n]</cite> or <cite>dtype</cite> is not
    correct.
- **InvalidArgumentError** – When rank(`inputs`)<2.




**Note**
    
This layer implements the successive cancellation list (SCL) decoder
as described in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#tal-scl" id="id23">[Tal_SCL]</a> but uses LLR-based message updates
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#stimming-llr" id="id24">[Stimming_LLR]</a>. The implementation follows the notation from
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gross-fast-scl" id="id25">[Gross_Fast_SCL]</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#hashemi-sscl" id="id26">[Hashemi_SSCL]</a>. If option <cite>use_fast_scl</cite> is active
tree pruning is used and tree nodes are combined if possible (see
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#hashemi-sscl" id="id27">[Hashemi_SSCL]</a> for details).
    
Implementing SCL decoding as TensorFlow graph is a difficult task that
requires several design tradeoffs to match the TF constraints while
maintaining a reasonable throughput. Thus, the decoder minimizes
the <cite>control flow</cite> as much as possible, leading to a strong memory
occupation (e.g., due to full path duplication after each decision).
For longer code lengths, the complexity of the decoding graph becomes
large and we recommend to use the <cite>CPU_only</cite> option that uses an
embedded Numpy decoder. Further, this function recursively unrolls the
SCL decoding tree, thus, for larger values of `n` building the
decoding graph can become time consuming. Please consider the
`cpu_only` option if building the graph takes to long.
    
A hybrid SC/SCL decoder as proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#cammerer-hybrid-scl" id="id28">[Cammerer_Hybrid_SCL]</a> (using SC
instead of BP) can be activated with option `use_hybrid_sc` iff an
outer CRC is available. Please note that the results are not exactly
SCL performance caused by the false positive rate of the CRC.
    
As commonly done, we assume frozen bits are set to <cite>0</cite>. Please note
that - although its practical relevance is only little - setting frozen
bits to <cite>1</cite> may result in <cite>affine</cite> codes instead of linear code as the
<cite>all-zero</cite> codeword is not necessarily part of the code any more.

<em class="property">`property` </em>`frozen_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.frozen_pos" title="Permalink to this definition"></a>
    
Frozen positions for Polar decoding.


<em class="property">`property` </em>`info_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.info_pos" title="Permalink to this definition"></a>
    
Information bit positions for Polar encoding.


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.k" title="Permalink to this definition"></a>
    
Number of information bits.


<em class="property">`property` </em>`k_crc`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.k_crc" title="Permalink to this definition"></a>
    
Number of CRC bits.


<em class="property">`property` </em>`list_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.list_size" title="Permalink to this definition"></a>
    
List size for SCL decoding.


<em class="property">`property` </em>`llr_max`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.llr_max" title="Permalink to this definition"></a>
    
Maximum LLR value for internal calculations.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.n" title="Permalink to this definition"></a>
    
Codeword length.


<em class="property">`property` </em>`output_dtype`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder.output_dtype" title="Permalink to this definition"></a>
    
Output dtype of decoder.


### PolarBPDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polarbpdecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.polar.decoding.``PolarBPDecoder`(<em class="sig-param">`frozen_pos`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`num_iter``=``20`</em>, <em class="sig-param">`hard_out``=``True`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/decoding.html#PolarBPDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder" title="Permalink to this definition"></a>
    
Belief propagation (BP) decoder for Polar codes <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#arikan-polar" id="id29">[Arikan_Polar]</a> and
Polar-like codes based on <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#arikan-bp" id="id30">[Arikan_BP]</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#forney-graphs" id="id31">[Forney_Graphs]</a>.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
    
Remark: The PolarBPDecoder does currently not support XLA.
Parameters
 
- **frozen_pos** (<em>ndarray</em>) – Array of <cite>int</cite> defining the `n-k` indices of the frozen positions.
- **n** (<em>int</em>) – Defining the codeword length.
- **num_iter** (<em>int</em>) – Defining the number of decoder iterations (no early stopping used
at the moment).
- **hard_out** (<em>bool</em>) – Defaults to True. If True, the decoder provides hard-decided
information bits instead of soft-values.
- **output_dtype** (<em>tf.DType</em>) – Defaults to tf.float32. Defines the output datatype of the layer
(internal precision remains tf.float32).


Input
    
**inputs** (<em>[…,n], tf.float32</em>) – 2+D tensor containing the channel logits/llr values.

Output
    
<em>[…,k], tf.float32</em> – 2+D tensor containing bit-wise soft-estimates
(or hard-decided bit-values) of all `k` information bits.

Raises
 
- **AssertionError** – If `n` is not <cite>int</cite>.
- **AssertionError** – If `n` is not a power of 2.
- **AssertionError** – If the number of elements in `frozen_pos` is greater than `n`.
- **AssertionError** – If `frozen_pos` does not consists of <cite>int</cite>.
- **AssertionError** – If `hard_out` is not <cite>bool</cite>.
- **ValueError** – If `output_dtype` is not {tf.float16, tf.float32, tf.float64}.
- **AssertionError** – If `num_iter` is not <cite>int</cite>.
- **AssertionError** – If `num_iter` is not a positive value.




**Note**
    
This decoder is fully differentiable and, thus, well-suited for
gradient descent-based learning tasks such as <cite>learned code design</cite>
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#ebada-design" id="id32">[Ebada_Design]</a>.
    
As commonly done, we assume frozen bits are set to <cite>0</cite>. Please note
that - although its practical relevance is only little - setting frozen
bits to <cite>1</cite> may result in <cite>affine</cite> codes instead of linear code as the
<cite>all-zero</cite> codeword is not necessarily part of the code any more.

<em class="property">`property` </em>`frozen_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.frozen_pos" title="Permalink to this definition"></a>
    
Frozen positions for Polar decoding.


<em class="property">`property` </em>`hard_out`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.hard_out" title="Permalink to this definition"></a>
    
Indicates if decoder hard-decides outputs.


<em class="property">`property` </em>`info_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.info_pos" title="Permalink to this definition"></a>
    
Information bit positions for Polar encoding.


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.k" title="Permalink to this definition"></a>
    
Number of information bits.


<em class="property">`property` </em>`llr_max`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.llr_max" title="Permalink to this definition"></a>
    
Maximum LLR value for internal calculations.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.n" title="Permalink to this definition"></a>
    
Codeword length.


<em class="property">`property` </em>`num_iter`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.num_iter" title="Permalink to this definition"></a>
    
Number of decoding iterations.


<em class="property">`property` </em>`output_dtype`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarBPDecoder.output_dtype" title="Permalink to this definition"></a>
    
Output dtype of decoder.


## Polar Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar-utility-functions" title="Permalink to this headline"></a>

### generate_5g_ranking<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#generate-5g-ranking" title="Permalink to this headline"></a>

`sionna.fec.polar.utils.``generate_5g_ranking`(<em class="sig-param">`k`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`sort``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/utils.html#generate_5g_ranking">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.utils.generate_5g_ranking" title="Permalink to this definition"></a>
    
Returns information and frozen bit positions of the 5G Polar code
as defined in Tab. 5.3.1.2-1 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id33">[3GPPTS38212]</a> for given values of `k`
and `n`.
Input
 
- **k** (<em>int</em>) – The number of information bit per codeword.
- **n** (<em>int</em>) – The desired codeword length. Must be a power of two.
- **sort** (<em>bool</em>) – Defaults to True. Indicates if the returned indices are
sorted.


Output
 
- **[frozen_pos, info_pos]** – List:
- **frozen_pos** (<em>ndarray</em>) – An array of ints of shape <cite>[n-k]</cite> containing the frozen
position indices.
- **info_pos** (<em>ndarray</em>) – An array of ints of shape <cite>[k]</cite> containing the information
position indices.


Raises
 
- **AssertionError** – If `k` or `n` are not positve ints.
- **AssertionError** – If `sort` is not bool.
- **AssertionError** – If `k` or `n` are larger than 1024
- **AssertionError** – If `n` is less than 32.
- **AssertionError** – If the resulting coderate is invalid (<cite>>1.0</cite>).
- **AssertionError** – If `n` is not a power of 2.




### generate_polar_transform_mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#generate-polar-transform-mat" title="Permalink to this headline"></a>

`sionna.fec.polar.utils.``generate_polar_transform_mat`(<em class="sig-param">`n_lift`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/utils.html#generate_polar_transform_mat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.utils.generate_polar_transform_mat" title="Permalink to this definition"></a>
    
Generate the polar transformation matrix (Kronecker product).
Input
    
**n_lift** (<em>int</em>) – Defining the Kronecker power, i.e., how often is the kernel lifted.

Output
    
<em>ndarray</em> – Array of <cite>0s</cite> and <cite>1s</cite> of shape <cite>[2^n_lift , 2^n_lift]</cite> containing
the Polar transformation matrix.



