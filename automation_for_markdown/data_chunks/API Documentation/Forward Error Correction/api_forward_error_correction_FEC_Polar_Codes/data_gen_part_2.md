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
### Polar5GDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar5gdecoder" title="Permalink to this headline"></a>
### PolarSCDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polarscdecoder" title="Permalink to this headline"></a>
  
  

### Polar5GDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar5gdecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.polar.decoding.``Polar5GDecoder`(<em class="sig-param">`enc_polar`</em>, <em class="sig-param">`dec_type``=``'SC'`</em>, <em class="sig-param">`list_size``=``8`</em>, <em class="sig-param">`num_iter``=``20`</em>, <em class="sig-param">`return_crc_status``=``False`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/decoding.html#Polar5GDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder" title="Permalink to this definition"></a>
    
Wrapper for 5G compliant decoding including rate-recovery and CRC removal.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **enc_polar** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder" title="sionna.fec.polar.encoding.Polar5GEncoder"><em>Polar5GEncoder</em></a>) – Instance of the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder" title="sionna.fec.polar.encoding.Polar5GEncoder">`Polar5GEncoder`</a>
used for encoding including rate-matching.
- **dec_type** (<em>str</em>) – Defaults to <cite>“SC”</cite>. Defining the decoder to be used.
Must be one of the following <cite>{“SC”, “SCL”, “hybSCL”, “BP”}</cite>.
- **list_size** (<em>int</em>) – Defaults to 8. Defining the list size <cite>iff</cite> list-decoding is used.
Only required for `dec_types` <cite>{“SCL”, “hybSCL”}</cite>.
- **num_iter** (<em>int</em>) – Defaults to 20. Defining the number of BP iterations. Only required
for `dec_type` <cite>“BP”</cite>.
- **return_crc_status** (<em>bool</em>) – Defaults to False. If True, the decoder additionally returns the
CRC status indicating if a codeword was (most likely) correctly
recovered.
- **output_dtype** (<em>tf.DType</em>) – Defaults to tf.float32. Defines the output datatype of the layer
(internal precision remains tf.float32).


Input
    
**inputs** (<em>[…,n], tf.float32</em>) – 2+D tensor containing the channel logits/llr values.

Output
 
- **b_hat** (<em>[…,k], tf.float32</em>) – 2+D tensor containing hard-decided estimations of all <cite>k</cite>
information bits.
- **crc_status** (<em>[…], tf.bool</em>) – CRC status indicating if a codeword was (most likely) correctly
recovered. This is only returned if `return_crc_status` is True.
Note that false positives are possible.


Raises
 
- **AssertionError** – If `enc_polar` is not <cite>Polar5GEncoder</cite>.
- **ValueError** – If `dec_type` is not <cite>{“SC”, “SCL”, “SCL8”, “SCL32”, “hybSCL”,
    “BP”}</cite>.
- **AssertionError** – If `dec_type` is not <cite>str</cite>.
- **ValueError** – If `inputs` is not of shape <cite>[…, n]</cite> or <cite>dtype</cite> is not
    the same as `output_dtype`.
- **InvalidArgumentError** – When rank(`inputs`)<2.




**Note**
    
This layer supports the uplink and downlink Polar rate-matching scheme
without <cite>codeword segmentation</cite>.
    
Although the decoding <cite>list size</cite> is not provided by 3GPP
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id17">[3GPPTS38212]</a>, the consortium has agreed on a <cite>list size</cite> of 8 for the
5G decoding reference curves <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#bioglio-design" id="id18">[Bioglio_Design]</a>.
    
All list-decoders apply <cite>CRC-aided</cite> decoding, however, the non-list
decoders (<cite>“SC”</cite> and <cite>“BP”</cite>) cannot materialize the CRC leading to an
effective rate-loss.

<em class="property">`property` </em>`dec_type`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.dec_type" title="Permalink to this definition"></a>
    
Decoder type used for decoding as str.


<em class="property">`property` </em>`frozen_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.frozen_pos" title="Permalink to this definition"></a>
    
Frozen positions for Polar decoding.


<em class="property">`property` </em>`info_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.info_pos" title="Permalink to this definition"></a>
    
Information bit positions for Polar encoding.


<em class="property">`property` </em>`k_polar`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.k_polar" title="Permalink to this definition"></a>
    
Number of information bits of mother Polar code.


<em class="property">`property` </em>`k_target`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.k_target" title="Permalink to this definition"></a>
    
Number of information bits including rate-matching.


<em class="property">`property` </em>`llr_max`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.llr_max" title="Permalink to this definition"></a>
    
Maximum LLR value for internal calculations.


<em class="property">`property` </em>`n_polar`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.n_polar" title="Permalink to this definition"></a>
    
Codeword length of mother Polar code.


<em class="property">`property` </em>`n_target`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.n_target" title="Permalink to this definition"></a>
    
Codeword length including rate-matching.


<em class="property">`property` </em>`output_dtype`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.output_dtype" title="Permalink to this definition"></a>
    
Output dtype of decoder.


<em class="property">`property` </em>`polar_dec`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder.polar_dec" title="Permalink to this definition"></a>
    
Decoder instance used for decoding.


### PolarSCDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polarscdecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.polar.decoding.``PolarSCDecoder`(<em class="sig-param">`frozen_pos`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/decoding.html#PolarSCDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder" title="Permalink to this definition"></a>
    
Successive cancellation (SC) decoder <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#arikan-polar" id="id19">[Arikan_Polar]</a> for Polar codes and
Polar-like codes.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **frozen_pos** (<em>ndarray</em>) – Array of <cite>int</cite> defining the `n-k` indices of the frozen positions.
- **n** – Defining the codeword length.


Input
    
**inputs** (<em>[…,n], tf.float32</em>) – 2+D tensor containing the channel LLR values (as logits).

Output
    
<em>[…,k], tf.float32</em> – 2+D tensor  containing hard-decided estimations of all `k`
information bits.

Raises
 
- **AssertionError** – If `n` is not <cite>int</cite>.
- **AssertionError** – If `n` is not a power of 2.
- **AssertionError** – If the number of elements in `frozen_pos` is greater than `n`.
- **AssertionError** – If `frozen_pos` does not consists of <cite>int</cite>.
- **ValueError** – If `output_dtype` is not {tf.float16, tf.float32, tf.float64}.




**Note**
    
This layer implements the SC decoder as described in
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#arikan-polar" id="id20">[Arikan_Polar]</a>. However, the implementation follows the <cite>recursive
tree</cite> <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gross-fast-scl" id="id21">[Gross_Fast_SCL]</a> terminology and combines nodes for increased
throughputs without changing the outcome of the algorithm.
    
As commonly done, we assume frozen bits are set to <cite>0</cite>. Please note
that - although its practical relevance is only little - setting frozen
bits to <cite>1</cite> may result in <cite>affine</cite> codes instead of linear code as the
<cite>all-zero</cite> codeword is not necessarily part of the code any more.

<em class="property">`property` </em>`frozen_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder.frozen_pos" title="Permalink to this definition"></a>
    
Frozen positions for Polar decoding.


<em class="property">`property` </em>`info_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder.info_pos" title="Permalink to this definition"></a>
    
Information bit positions for Polar encoding.


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder.k" title="Permalink to this definition"></a>
    
Number of information bits.


<em class="property">`property` </em>`llr_max`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder.llr_max" title="Permalink to this definition"></a>
    
Maximum LLR value for internal calculations.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder.n" title="Permalink to this definition"></a>
    
Codeword length.


<em class="property">`property` </em>`output_dtype`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder.output_dtype" title="Permalink to this definition"></a>
    
Output dtype of decoder.


