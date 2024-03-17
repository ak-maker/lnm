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
## Polar Encoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar-encoding" title="Permalink to this headline"></a>
### Polar5GEncoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar5gencoder" title="Permalink to this headline"></a>
### PolarEncoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polarencoder" title="Permalink to this headline"></a>
## Polar Decoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar-decoding" title="Permalink to this headline"></a>
  
  

## Polar Encoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar-encoding" title="Permalink to this headline"></a>

### Polar5GEncoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar5gencoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.polar.encoding.``Polar5GEncoder`(<em class="sig-param">`k`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`verbose``=``False`</em>, <em class="sig-param">`channel_type``=``'uplink'`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/encoding.html#Polar5GEncoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder" title="Permalink to this definition"></a>
    
5G compliant Polar encoder including rate-matching following <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id1">[3GPPTS38212]</a>
for the uplink scenario (<cite>UCI</cite>) and downlink scenario (<cite>DCI</cite>).
    
This layer performs polar encoding for `k` information bits and
rate-matching such that the codeword lengths is `n`. This includes the CRC
concatenation and the interleaving as defined in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id2">[3GPPTS38212]</a>.
    
Note: <cite>block segmentation</cite> is currently not supported (<cite>I_seq=False</cite>).
    
We follow the basic structure from Fig. 6 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#bioglio-design" id="id3">[Bioglio_Design]</a>.
<p class="caption">Fig. 6 Fig. 1: Implemented 5G Polar encoding chain following Fig. 6 in
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#bioglio-design" id="id4">[Bioglio_Design]</a> for the uplink (<cite>I_BIL</cite> = <cite>True</cite>) and the downlink
(<cite>I_IL</cite> = <cite>True</cite>) scenario without <cite>block segmentation</cite>.<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id35" title="Permalink to this image"></a>
    
For further details, we refer to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id5">[3GPPTS38212]</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#bioglio-design" id="id6">[Bioglio_Design]</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#hui-channelcoding" id="id7">[Hui_ChannelCoding]</a>.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model. Further, the class inherits from PolarEncoder.
Parameters
 
- **k** (<em>int</em>) – Defining the number of information bit per codeword.
- **n** (<em>int</em>) – Defining the codeword length.
- **channel_type** (<em>str</em>) – Defaults to “uplink”. Can be “uplink” or “downlink”.
- **verbose** (<em>bool</em>) – Defaults to False. If True, rate-matching parameters will be
printed.
- **dtype** (<em>tf.DType</em>) – Defaults to tf.float32. Defines the output datatype of the layer
(internal precision remains tf.uint8).


Input
    
**inputs** (<em>[…,k], tf.float32</em>) – 2+D tensor containing the information bits to be encoded.

Output
    
<em>[…,n], tf.float32</em> – 2+D tensor containing the codeword bits.

Raises
 
- **AssertionError** – `k` and `n` must be positive integers and `k` must be smaller
    (or equal) than `n`.
- **AssertionError** – If `n` and `k` are invalid code parameters (see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id8">[3GPPTS38212]</a>).
- **AssertionError** – If `verbose` is not <cite>bool</cite>.
- **ValueError** – If `dtype` is not supported.




**Note**
    
The encoder supports the <cite>uplink</cite> Polar coding (<cite>UCI</cite>) scheme from
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id9">[3GPPTS38212]</a> and the <cite>downlink</cite> Polar coding (<cite>DCI</cite>) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id10">[3GPPTS38212]</a>,
respectively.
    
For <cite>12 <= k <= 19</cite> the 3 additional parity bits as defined in
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id11">[3GPPTS38212]</a> are not implemented as it would also require a
modified decoding procedure to materialize the potential gains.
    
<cite>Code segmentation</cite> is currently not supported and, thus, `n` is
limited to a maximum length of 1088 codeword bits.
    
For the downlink scenario, the input length is limited to <cite>k <= 140</cite>
information bits due to the limited input bit interleaver size
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id12">[3GPPTS38212]</a>.
    
For simplicity, the implementation does not exactly re-implement the
<cite>DCI</cite> scheme from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id13">[3GPPTS38212]</a>. This implementation neglects the
<cite>all-one</cite> initialization of the CRC shift register and the scrambling of the CRC parity bits with the <cite>RNTI</cite>.

`channel_interleaver`(<em class="sig-param">`c`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/encoding.html#Polar5GEncoder.channel_interleaver">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.channel_interleaver" title="Permalink to this definition"></a>
    
Triangular interleaver following Sec. 5.4.1.3 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id14">[3GPPTS38212]</a>.
Input
    
**c** (<em>ndarray</em>) – 1D array to be interleaved.

Output
    
<em>ndarray</em> – Interleaved version of `c` with same shape and dtype as `c`.




<em class="property">`property` </em>`enc_crc`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.enc_crc" title="Permalink to this definition"></a>
    
CRC encoder layer used for CRC concatenation.


`input_interleaver`(<em class="sig-param">`c`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/encoding.html#Polar5GEncoder.input_interleaver">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.input_interleaver" title="Permalink to this definition"></a>
    
Input interleaver following Sec. 5.4.1.1 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id15">[3GPPTS38212]</a>.
Input
    
**c** (<em>ndarray</em>) – 1D array to be interleaved.

Output
    
<em>ndarray</em> – Interleaved version of `c` with same shape and dtype as `c`.




<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.k" title="Permalink to this definition"></a>
    
Number of information bits including rate-matching.


<em class="property">`property` </em>`k_polar`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.k_polar" title="Permalink to this definition"></a>
    
Number of information bits of the underlying Polar code.


<em class="property">`property` </em>`k_target`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.k_target" title="Permalink to this definition"></a>
    
Number of information bits including rate-matching.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.n" title="Permalink to this definition"></a>
    
Codeword length including rate-matching.


<em class="property">`property` </em>`n_polar`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.n_polar" title="Permalink to this definition"></a>
    
Codeword length of the underlying Polar code.


<em class="property">`property` </em>`n_target`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.n_target" title="Permalink to this definition"></a>
    
Codeword length including rate-matching.


`subblock_interleaving`(<em class="sig-param">`u`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/encoding.html#Polar5GEncoder.subblock_interleaving">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder.subblock_interleaving" title="Permalink to this definition"></a>
    
Input bit interleaving as defined in Sec 5.4.1.1 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#gppts38212" id="id16">[3GPPTS38212]</a>.
Input
    
**u** (<em>ndarray</em>) – 1D array to be interleaved. Length of `u` must be a multiple
of 32.

Output
    
<em>ndarray</em> – Interleaved version of `u` with same shape and dtype as `u`.

Raises
    
**AssertionError** – If length of `u` is not a multiple of 32.




### PolarEncoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polarencoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.polar.encoding.``PolarEncoder`(<em class="sig-param">`frozen_pos`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/encoding.html#PolarEncoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.PolarEncoder" title="Permalink to this definition"></a>
    
Polar encoder for given code parameters.
    
This layer performs polar encoding for the given `k` information bits and
the <cite>frozen set</cite> (i.e., indices of frozen positions) specified by
`frozen_pos`.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **frozen_pos** (<em>ndarray</em>) – Array of <cite>int</cite> defining the <cite>n-k</cite> frozen indices, i.e., information
bits are mapped onto the <cite>k</cite> complementary positions.
- **n** (<em>int</em>) – Defining the codeword length.
- **dtype** (<em>tf.DType</em>) – Defaults to <cite>tf.float32</cite>. Defines the output datatype of the layer
(internal precision is <cite>tf.uint8</cite>).


Input
    
**inputs** (<em>[…,k], tf.float32</em>) – 2+D tensor containing the information bits to be encoded.

Output
    
<em>[…,n], tf.float32</em> – 2+D tensor containing the codeword bits.

Raises
 
- **AssertionError** – `k` and `n` must be positive integers and `k` must be smaller
    (or equal) than `n`.
- **AssertionError** – If `n` is not a power of 2.
- **AssertionError** – If the number of elements in `frozen_pos` is great than `n`.
- **AssertionError** – If `frozen_pos` does not consists of <cite>int</cite>.
- **ValueError** – If `dtype` is not supported.
- **ValueError** – If `inputs` contains other values than <cite>0</cite> or <cite>1</cite>.
- **TypeError** – If `inputs` is not <cite>tf.float32</cite>.
- **InvalidArgumentError** – When rank(`inputs`)<2.
- **InvalidArgumentError** – When shape of last dim is not `k`.




**Note**
    
As commonly done, we assume frozen bits are set to <cite>0</cite>. Please note
that - although its practical relevance is only little - setting frozen
bits to <cite>1</cite> may result in <cite>affine</cite> codes instead of linear code as the
<cite>all-zero</cite> codeword is not necessarily part of the code any more.

<em class="property">`property` </em>`frozen_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.PolarEncoder.frozen_pos" title="Permalink to this definition"></a>
    
Frozen positions for Polar decoding.


<em class="property">`property` </em>`info_pos`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.PolarEncoder.info_pos" title="Permalink to this definition"></a>
    
Information bit positions for Polar encoding.


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.PolarEncoder.k" title="Permalink to this definition"></a>
    
Number of information bits.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.PolarEncoder.n" title="Permalink to this definition"></a>
    
Codeword length.


## Polar Decoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar-decoding" title="Permalink to this headline"></a>

