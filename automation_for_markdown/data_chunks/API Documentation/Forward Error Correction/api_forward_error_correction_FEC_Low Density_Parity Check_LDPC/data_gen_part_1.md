# Low-Density Parity-Check (LDPC)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#low-density-parity-check-ldpc" title="Permalink to this headline"></a>
    
The low-density parity-check (LDPC) code module supports 5G compliant LDPC codes and allows iterative belief propagation (BP) decoding.
Further, the module supports rate-matching for 5G and provides a generic linear encoder.
    
The following code snippets show how to setup and run a rate-matched 5G compliant LDPC encoder and a corresponding belief propagation (BP) decoder.
    
First, we need to create instances of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder" title="sionna.fec.ldpc.encoding.LDPC5GEncoder">`LDPC5GEncoder`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPC5GDecoder" title="sionna.fec.ldpc.decoding.LDPC5GDecoder">`LDPC5GDecoder`</a>:
```python
encoder = LDPC5GEncoder(k                 = 100, # number of information bits (input)
                        n                 = 200) # number of codeword bits (output)

decoder = LDPC5GDecoder(encoder           = encoder,
                        num_iter          = 20, # number of BP iterations
                        return_infobits   = True)
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
## LDPC Encoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ldpc-encoder" title="Permalink to this headline"></a>
### LDPC5GEncoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ldpc5gencoder" title="Permalink to this headline"></a>
## LDPC Decoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ldpc-decoder" title="Permalink to this headline"></a>
  
  

## LDPC Encoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ldpc-encoder" title="Permalink to this headline"></a>

### LDPC5GEncoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ldpc5gencoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.ldpc.encoding.``LDPC5GEncoder`(<em class="sig-param">`k`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`num_bits_per_symbol``=``None`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/ldpc/encoding.html#LDPC5GEncoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder" title="Permalink to this definition"></a>
    
5G NR LDPC Encoder following the 3GPP NR Initiative <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#gppts38212-ldpc" id="id1">[3GPPTS38212_LDPC]</a>
including rate-matching.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **k** (<em>int</em>) – Defining the number of information bit per codeword.
- **n** (<em>int</em>) – Defining the desired codeword length.
- **num_bits_per_symbol** (<em>int</em><em> or </em><em>None</em>) – Defining the number of bits per QAM symbol. If this parameter is
explicitly provided, the codeword will be interleaved after
rate-matching as specified in Sec. 5.4.2.2 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#gppts38212-ldpc" id="id2">[3GPPTS38212_LDPC]</a>.
- **dtype** (<em>tf.DType</em>) – Defaults to <cite>tf.float32</cite>. Defines the output datatype of the layer
(internal precision remains <cite>tf.uint8</cite>).


Input
    
**inputs** (<em>[…,k], tf.float32</em>) – 2+D tensor containing the information bits to be
encoded.

Output
    
<em>[…,n], tf.float32</em> – 2+D tensor of same shape as inputs besides last dimension has
changed to <cite>n</cite> containing the encoded codeword bits.

Attributes
 
- **k** (<em>int</em>) – Defining the number of information bit per codeword.
- **n** (<em>int</em>) – Defining the desired codeword length.
- **coderate** (<em>float</em>) – Defining the coderate r= `k` / `n`.
- **n_ldpc** (<em>int</em>) – An integer defining the total codeword length (before
punturing) of the lifted parity-check matrix.
- **k_ldpc** (<em>int</em>) – An integer defining the total information bit length
(before zero removal) of the lifted parity-check matrix. Gap to
`k` must be filled with so-called filler bits.
- **num_bits_per_symbol** (<em>int or None.</em>) – Defining the number of bits per QAM symbol. If this parameter is
explicitly provided, the codeword will be interleaved after
rate-matching as specified in Sec. 5.4.2.2 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#gppts38212-ldpc" id="id3">[3GPPTS38212_LDPC]</a>.
- **out_int** (<em>[n], ndarray of int</em>) – Defining the rate-matching output interleaver sequence.
- **out_int_inv** (<em>[n], ndarray of int</em>) – Defining the inverse rate-matching output interleaver sequence.
- **_check_input** (<em>bool</em>) – A boolean that indicates whether the input vector
during call of the layer should be checked for consistency (i.e.,
binary).
- **_bg** (<em>str</em>) – Denoting the selected basegraph (either <cite>bg1</cite> or <cite>bg2</cite>).
- **_z** (<em>int</em>) – Denoting the lifting factor.
- **_i_ls** (<em>int</em>) – Defining which version of the basegraph to load.
Can take values between 0 and 7.
- **_k_b** (<em>int</em>) – Defining the number of <cite>information bit columns</cite> in the
basegraph. Determined by the code design procedure in
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#gppts38212-ldpc" id="id4">[3GPPTS38212_LDPC]</a>.
- **_bm** (<em>ndarray</em>) – An ndarray defining the basegraph.
- **_pcm** (<em>sp.sparse.csr_matrix</em>) – A sparse matrix of shape <cite>[k_ldpc-n_ldpc, n_ldpc]</cite>
containing the sparse parity-check matrix.


Raises
 
- **AssertionError** – If `k` is not <cite>int</cite>.
- **AssertionError** – If `n` is not <cite>int</cite>.
- **ValueError** – If `code_length` is not supported.
- **ValueError** – If <cite>dtype</cite> is not supported.
- **ValueError** – If `inputs` contains other values than <cite>0</cite> or <cite>1</cite>.
- **InvalidArgumentError** – When rank(`inputs`)<2.
- **InvalidArgumentError** – When shape of last dim is not `k`.




**Note**
    
As specified in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#gppts38212-ldpc" id="id5">[3GPPTS38212_LDPC]</a>, the encoder also performs
puncturing and shortening. Thus, the corresponding decoder needs to
<cite>invert</cite> these operations, i.e., must be compatible with the 5G
encoding scheme.

<em class="property">`property` </em>`coderate`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder.coderate" title="Permalink to this definition"></a>
    
Coderate of the LDPC code after rate-matching.


`generate_out_int`(<em class="sig-param">`n`</em>, <em class="sig-param">`num_bits_per_symbol`</em>)<a class="reference internal" href="../_modules/sionna/fec/ldpc/encoding.html#LDPC5GEncoder.generate_out_int">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder.generate_out_int" title="Permalink to this definition"></a>
    
“Generates LDPC output interleaver sequence as defined in
Sec 5.4.2.2 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#gppts38212-ldpc" id="id6">[3GPPTS38212_LDPC]</a>.
Parameters
 
- **n** (<em>int</em>) – Desired output sequence length.
- **num_bits_per_symbol** (<em>int</em>) – Number of symbols per QAM symbol, i.e., the modulation order.
- **Outputs** – 
- **-------** – 
- **(****perm_seq** – Tuple:
- **perm_seq_inv****)** – Tuple:
- **perm_seq** (<em>ndarray of length n</em>) – Containing the permuted indices.
- **perm_seq_inv** (<em>ndarray of length n</em>) – Containing the inverse permuted indices.




**Note**
    
The interleaver pattern depends on the modulation order and helps to
reduce dependencies in bit-interleaved coded modulation (BICM) schemes.


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder.k" title="Permalink to this definition"></a>
    
Number of input information bits.


<em class="property">`property` </em>`k_ldpc`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder.k_ldpc" title="Permalink to this definition"></a>
    
Number of LDPC information bits after rate-matching.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder.n" title="Permalink to this definition"></a>
    
Number of output codeword bits.


<em class="property">`property` </em>`n_ldpc`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder.n_ldpc" title="Permalink to this definition"></a>
    
Number of LDPC codeword bits before rate-matching.


<em class="property">`property` </em>`num_bits_per_symbol`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder.num_bits_per_symbol" title="Permalink to this definition"></a>
    
Modulation order used for the rate-matching output interleaver.


<em class="property">`property` </em>`out_int`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder.out_int" title="Permalink to this definition"></a>
    
Output interleaver sequence as defined in 5.4.2.2.


<em class="property">`property` </em>`out_int_inv`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder.out_int_inv" title="Permalink to this definition"></a>
    
Inverse output interleaver sequence as defined in 5.4.2.2.


<em class="property">`property` </em>`pcm`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder.pcm" title="Permalink to this definition"></a>
    
Parity-check matrix for given code parameters.


<em class="property">`property` </em>`z`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder.z" title="Permalink to this definition"></a>
    
Lifting factor of the basegraph.


## LDPC Decoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ldpc-decoder" title="Permalink to this headline"></a>

