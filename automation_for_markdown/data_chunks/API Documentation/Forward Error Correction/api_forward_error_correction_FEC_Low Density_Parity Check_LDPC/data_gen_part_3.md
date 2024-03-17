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
### LDPC5GDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ldpc5gdecoder" title="Permalink to this headline"></a>
  
  

### LDPC5GDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ldpc5gdecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.ldpc.decoding.``LDPC5GDecoder`(<em class="sig-param">`encoder`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`cn_type``=``'boxplus-phi'`</em>, <em class="sig-param">`hard_out``=``True`</em>, <em class="sig-param">`track_exit``=``False`</em>, <em class="sig-param">`return_infobits``=``True`</em>, <em class="sig-param">`prune_pcm``=``True`</em>, <em class="sig-param">`num_iter``=``20`</em>, <em class="sig-param">`stateful``=``False`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/ldpc/decoding.html#LDPC5GDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPC5GDecoder" title="Permalink to this definition"></a>
    
(Iterative) belief propagation decoder for 5G NR LDPC codes.
    
Inherits from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPCBPDecoder" title="sionna.fec.ldpc.decoding.LDPCBPDecoder">`LDPCBPDecoder`</a> and provides
a wrapper for 5G compatibility, i.e., automatically handles puncturing and
shortening according to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#gppts38212-ldpc" id="id16">[3GPPTS38212_LDPC]</a>.
    
Note that for full 5G 3GPP NR compatibility, the correct puncturing and
shortening patterns must be applied and, thus, the encoder object is
required as input.
    
If required the decoder can be made trainable and is differentiable
(the training of some check node types may be not supported) following the
concept of “weighted BP” <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#nachmani" id="id17">[Nachmani]</a>.
    
For numerical stability, the decoder applies LLR clipping of
+/- 20 to the input LLRs.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **encoder** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder" title="sionna.fec.ldpc.encoding.LDPC5GEncoder"><em>LDPC5GEncoder</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder" title="sionna.fec.ldpc.encoding.LDPC5GEncoder">`LDPC5GEncoder`</a>
containing the correct code parameters.
- **trainable** (<em>bool</em>) – Defaults to False. If True, every outgoing variable node message is
scaled with a trainable scalar.
- **cn_type** (<em>str</em>) – A string defaults to ‘“boxplus-phi”’. One of
{<cite>“boxplus”</cite>, <cite>“boxplus-phi”</cite>, <cite>“minsum”</cite>} where
‘“boxplus”’ implements the single-parity-check APP decoding rule.
‘“boxplus-phi”’ implements the numerical more stable version of
boxplus <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ryan" id="id18">[Ryan]</a>.
‘“minsum”’ implements the min-approximation of the CN
update rule <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ryan" id="id19">[Ryan]</a>.
- **hard_out** (<em>bool</em>) – Defaults to True. If True, the decoder provides hard-decided
codeword bits instead of soft-values.
- **track_exit** (<em>bool</em>) – Defaults to False. If True, the decoder tracks EXIT characteristics.
Note that this requires the all-zero CW as input.
- **return_infobits** (<em>bool</em>) – Defaults to True. If True, only the <cite>k</cite> info bits (soft or
hard-decided) are returned. Otherwise all <cite>n</cite> positions are
returned.
- **prune_pcm** (<em>bool</em>) – Defaults to True. If True, all punctured degree-1 VNs and
connected check nodes are removed from the decoding graph (see
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#cammerer" id="id20">[Cammerer]</a> for details). Besides numerical differences, this should
yield the same decoding result but improved the decoding throughput
and reduces the memory footprint.
- **num_iter** (<em>int</em>) – Defining the number of decoder iteration (no early stopping used at
the moment!).
- **stateful** (<em>bool</em>) – Defaults to False. If True, the internal VN messages `msg_vn`
from the last decoding iteration are returned, and `msg_vn` or
<cite>None</cite> needs to be given as a second input when calling the decoder.
This is required for iterative demapping and decoding.
- **output_dtype** (<em>tf.DType</em>) – Defaults to tf.float32. Defines the output datatype of the layer
(internal precision remains tf.float32).


Input
 
- **llrs_ch or (llrs_ch, msg_vn)** – Tensor or Tuple (only required if `stateful` is True):
- **llrs_ch** (<em>[…,n], tf.float32</em>) – 2+D tensor containing the channel logits/llr values.
- **msg_vn** (<em>None or RaggedTensor, tf.float32</em>) – Ragged tensor of VN messages.
Required only if `stateful` is True.


Output
 
- <em>[…,n] or […,k], tf.float32</em> – 2+D Tensor of same shape as `inputs` containing
bit-wise soft-estimates (or hard-decided bit-values) of all
codeword bits. If `return_infobits` is True, only the <cite>k</cite>
information bits are returned.
- <em>RaggedTensor, tf.float32:</em> – Tensor of VN messages.
Returned only if `stateful` is set to True.


Raises
 
- **ValueError** – If the shape of `pcm` is invalid or contains other
    values than <cite>0</cite> or <cite>1</cite>.
- **AssertionError** – If `trainable` is not <cite>bool</cite>.
- **AssertionError** – If `track_exit` is not <cite>bool</cite>.
- **AssertionError** – If `hard_out` is not <cite>bool</cite>.
- **AssertionError** – If `return_infobits` is not <cite>bool</cite>.
- **AssertionError** – If `encoder` is not an instance of
    <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder" title="sionna.fec.ldpc.encoding.LDPC5GEncoder">`LDPC5GEncoder`</a>.
- **ValueError** – If `output_dtype` is not {tf.float16, tf.float32, tf.
    float64}.
- **ValueError** – If `inputs` is not of shape <cite>[batch_size, n]</cite>.
- **ValueError** – If `num_iter` is not an integer greater (or equal) <cite>0</cite>.
- **InvalidArgumentError** – When rank(`inputs`)<2.




**Note**
    
As decoding input logits
$\operatorname{log} \frac{p(x=1)}{p(x=0)}$ are assumed for
compatibility with the learning framework, but
internally llrs with definition
$\operatorname{log} \frac{p(x=0)}{p(x=1)}$ are used.
    
The decoder is not (particularly) optimized for Quasi-cyclic (QC) LDPC
codes and, thus, supports arbitrary parity-check matrices.
    
The decoder is implemented by using ‘“ragged Tensors”’ <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#tf-ragged" id="id21">[TF_ragged]</a> to
account for arbitrary node degrees. To avoid a performance degradation
caused by a severe indexing overhead, the batch-dimension is shifted to
the last dimension during decoding.
    
If the decoder is made trainable <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#nachmani" id="id22">[Nachmani]</a>, for performance
improvements only variable to check node messages are scaled as the VN
operation is linear and, thus, would not increase the expressive power
of the weights.

<em class="property">`property` </em>`encoder`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPC5GDecoder.encoder" title="Permalink to this definition"></a>
    
LDPC Encoder used for rate-matching/recovery.



References:
Pfister
    
J. Hou, P. H. Siegel, L. B. Milstein, and H. D. Pfister,
“Capacity-approaching bandwidth-efficient coded modulation schemes
based on low-density parity-check codes,” IEEE Trans. Inf. Theory,
Sep. 2003.

3GPPTS38212_LDPC(<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id2">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id3">3</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id4">4</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id5">5</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id6">6</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id16">7</a>)
    
ETSI 3GPP TS 38.212 “5G NR Multiplexing and channel
coding”, v.16.5.0, 2021-03.

Ryan(<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id7">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id8">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id12">3</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id13">4</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id18">5</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id19">6</a>)
    
W. Ryan, “An Introduction to LDPC codes”, CRC Handbook for
Coding and Signal Processing for Recording Systems, 2004.

TF_ragged(<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id14">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id21">2</a>)
    
<a class="reference external" href="https://www.tensorflow.org/guide/ragged_tensor">https://www.tensorflow.org/guide/ragged_tensor</a>

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id9">Richardson</a>
    
T. Richardson and S. Kudekar. “Design of low-density
parity-check codes for 5G new radio,” IEEE Communications
Magazine 56.3, 2018.

Nachmani(<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id10">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id11">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id15">3</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id17">4</a>,<a href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id22">5</a>)
    
E. Nachmani, Y. Be’ery, and D. Burshtein. “Learning to
decode linear codes using deep learning,” IEEE Annual Allerton
Conference on Communication, Control, and Computing (Allerton),
2016.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id20">Cammerer</a>
    
S. Cammerer, M. Ebada, A. Elkelesh, and S. ten Brink.
“Sparse graphs for belief propagation decoding of polar codes.”
IEEE International Symposium on Information Theory (ISIT), 2018.



