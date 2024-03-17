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
## LDPC Decoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ldpc-decoder" title="Permalink to this headline"></a>
### LDPCBPDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ldpcbpdecoder" title="Permalink to this headline"></a>
### Note
  
  

### LDPCBPDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ldpcbpdecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.ldpc.decoding.``LDPCBPDecoder`(<em class="sig-param">`pcm`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`cn_type``=``'boxplus-phi'`</em>, <em class="sig-param">`hard_out``=``True`</em>, <em class="sig-param">`track_exit``=``False`</em>, <em class="sig-param">`num_iter``=``20`</em>, <em class="sig-param">`stateful``=``False`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/ldpc/decoding.html#LDPCBPDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPCBPDecoder" title="Permalink to this definition"></a>
    
Iterative belief propagation decoder for low-density parity-check (LDPC)
codes and other <cite>codes on graphs</cite>.
    
This class defines a generic belief propagation decoder for decoding
with arbitrary parity-check matrices. It can be used to iteratively
estimate/recover the transmitted codeword (or information bits) based on the
LLR-values of the received noisy codeword observation.
    
The decoder implements the flooding SPA algorithm <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ryan" id="id7">[Ryan]</a>, i.e., all nodes
are updated in a parallel fashion. Different check node update functions are
available
<ol class="arabic">
- <cite>boxplus</cite>

$$
y_{j \to i} = 2 \operatorname{tanh}^{-1} \left( \prod_{i' \in \mathcal{N}_(j) \setminus i} \operatorname{tanh} \left( \frac{x_{i' \to j}}{2} \right) \right)
$$

- <cite>boxplus-phi</cite>

$$
y_{j \to i} = \alpha_{j \to i} \cdot \phi \left( \sum_{i' \in \mathcal{N}_(j) \setminus i} \phi \left( |x_{i' \to j}|\right) \right)
$$
    
with $\phi(x)=-\operatorname{log}(\operatorname{tanh} \left(\frac{x}{2}) \right)$

- <cite>minsum</cite>

$$
\qquad y_{j \to i} = \alpha_{j \to i} \cdot {min}_{i' \in \mathcal{N}_(j) \setminus i} \left(|x_{i' \to j}|\right)
$$

</ol>
    
where $y_{j \to i}$ denotes the message from check node (CN) <em>j</em> to
variable node (VN) <em>i</em> and $x_{i \to j}$ from VN <em>i</em> to CN <em>j</em>,
respectively. Further, $\mathcal{N}_(j)$ denotes all indices of
connected VNs to CN <em>j</em> and

$$
\alpha_{j \to i} = \prod_{i' \in \mathcal{N}_(j) \setminus i} \operatorname{sign}(x_{i' \to j})
$$
    
is the sign of the outgoing message. For further details we refer to
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ryan" id="id8">[Ryan]</a>.
    
Note that for full 5G 3GPP NR compatibility, the correct puncturing and
shortening patterns must be applied (cf. <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#richardson" id="id9">[Richardson]</a> for details), this
can be done by `LDPC5GEncoder` and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPC5GDecoder" title="sionna.fec.ldpc.decoding.LDPC5GDecoder">`LDPC5GDecoder`</a>, respectively.
    
If required, the decoder can be made trainable and is fully differentiable
by following the concept of <cite>weighted BP</cite> <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#nachmani" id="id10">[Nachmani]</a> as shown in Fig. 1
leading to

$$
y_{j \to i} = 2 \operatorname{tanh}^{-1} \left( \prod_{i' \in \mathcal{N}_(j) \setminus i} \operatorname{tanh} \left( \frac{\textcolor{red}{w_{i' \to j}} \cdot x_{i' \to j}}{2} \right) \right)
$$
    
where $w_{i \to j}$ denotes the trainable weight of message $x_{i \to j}$.
Please note that the training of some check node types may be not supported.
<p class="caption">Fig. 5 Fig. 1: Weighted BP as proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#nachmani" id="id11">[Nachmani]</a>.<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#id23" title="Permalink to this image"></a>
    
For numerical stability, the decoder applies LLR clipping of
+/- 20 to the input LLRs.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **pcm** (<em>ndarray</em>) – An ndarray of shape <cite>[n-k, n]</cite> defining the parity-check matrix
consisting only of <cite>0</cite> or <cite>1</cite> entries. Can be also of type <cite>scipy.
sparse.csr_matrix</cite> or <cite>scipy.sparse.csc_matrix</cite>.
- **trainable** (<em>bool</em>) – Defaults to False. If True, every outgoing variable node message is
scaled with a trainable scalar.
- **cn_type** (<em>str</em>) – A string defaults to ‘“boxplus-phi”’. One of
{<cite>“boxplus”</cite>, <cite>“boxplus-phi”</cite>, <cite>“minsum”</cite>} where
‘“boxplus”’ implements the single-parity-check APP decoding rule.
‘“boxplus-phi”’ implements the numerical more stable version of
boxplus <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ryan" id="id12">[Ryan]</a>.
‘“minsum”’ implements the min-approximation of the CN
update rule <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#ryan" id="id13">[Ryan]</a>.
- **hard_out** (<em>bool</em>) – Defaults to True. If True, the decoder provides hard-decided
codeword bits instead of soft-values.
- **track_exit** (<em>bool</em>) – Defaults to False. If True, the decoder tracks EXIT
characteristics. Note that this requires the all-zero
CW as input.
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
 
- <em>[…,n], tf.float32</em> – 2+D Tensor of same shape as `inputs` containing
bit-wise soft-estimates (or hard-decided bit-values) of all
codeword bits.
- <em>RaggedTensor, tf.float32:</em> – Tensor of VN messages.
Returned only if `stateful` is set to True.


Attributes
 
- **pcm** (<em>ndarray</em>) – An ndarray of shape <cite>[n-k, n]</cite> defining the parity-check matrix
consisting only of <cite>0</cite> or <cite>1</cite> entries. Can be also of type <cite>scipy.
sparse.csr_matrix</cite> or <cite>scipy.sparse.csc_matrix</cite>.
- **num_cns** (<em>int</em>) – Defining the number of check nodes.
- **num_vns** (<em>int</em>) – Defining the number of variable nodes.
- **num_edges** (<em>int</em>) – Defining the total number of edges.
- **trainable** (<em>bool</em>) – If True, the decoder uses trainable weights.
- **_atanh_clip_value** (<em>float</em>) – Defining the internal clipping value before the atanh is applied
(relates to the CN update).
- **_cn_type** (<em>str</em>) – Defining the CN update function type.
- **_cn_update** – A function defining the CN update.
- **_hard_out** (<em>bool</em>) – If True, the decoder outputs hard-decided bits.
- **_cn_con** (<em>ndarray</em>) – An ndarray of shape <cite>[num_edges]</cite> defining all edges from check
node perspective.
- **_vn_con** (<em>ndarray</em>) – An ndarray of shape <cite>[num_edges]</cite> defining all edges from variable
node perspective.
- **_vn_mask_tf** (<em>tf.float32</em>) – A ragged Tensor of shape <cite>[num_vns, None]</cite> defining the incoming
message indices per VN. The second dimension is ragged and depends
on the node degree.
- **_cn_mask_tf** (<em>tf.float32</em>) – A ragged Tensor of shape <cite>[num_cns, None]</cite> defining the incoming
message indices per CN. The second dimension is ragged and depends
on the node degree.
- **_ind_cn** (<em>ndarray</em>) – An ndarray of shape <cite>[num_edges]</cite> defining the permutation index to
rearrange messages from variable into check node perspective.
- **_ind_cn_inv** (<em>ndarray</em>) – An ndarray of shape <cite>[num_edges]</cite> defining the permutation index to
rearrange messages from check into variable node perspective.
- **_vn_row_splits** (<em>ndarray</em>) – An ndarray of shape <cite>[num_vns+1]</cite> defining the row split positions
of a 1D vector consisting of all edges messages. Used to build a
ragged Tensor of incoming VN messages.
- **_cn_row_splits** (<em>ndarray</em>) – An ndarray of shape <cite>[num_cns+1]</cite> defining the row split positions
of a 1D vector consisting of all edges messages. Used to build a
ragged Tensor of incoming CN messages.
- **_edge_weights** (<em>tf.float32</em>) – A Tensor of shape <cite>[num_edges]</cite> defining a (trainable) weight per
outgoing VN message.


Raises
 
- **ValueError** – If the shape of `pcm` is invalid or contains other values than
    <cite>0</cite> or <cite>1</cite> or dtype is not <cite>tf.float32</cite>.
- **ValueError** – If `num_iter` is not an integer greater (or equal) <cite>0</cite>.
- **ValueError** – If `output_dtype` is not
    {tf.float16, tf.float32, tf.float64}.
- **ValueError** – If `inputs` is not of shape <cite>[batch_size, n]</cite>.
- **InvalidArgumentError** – When rank(`inputs`)<2.




### Note
    
As decoding input logits
$\operatorname{log} \frac{p(x=1)}{p(x=0)}$ are
assumed for compatibility with the learning framework, but internally
log-likelihood ratios (LLRs) with definition $\operatorname{log} \frac{p(x=0)}{p(x=1)}$ are used.
    
The decoder is not (particularly) optimized for quasi-cyclic (QC) LDPC
codes and, thus, supports arbitrary parity-check matrices.
    
The decoder is implemented by using ‘“ragged Tensors”’ <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#tf-ragged" id="id14">[TF_ragged]</a> to
account for arbitrary node degrees. To avoid a performance degradation
caused by a severe indexing overhead, the batch-dimension is shifted to
the last dimension during decoding.
    
If the decoder is made trainable <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#nachmani" id="id15">[Nachmani]</a>, for performance
improvements only variable to check node messages are scaled as the VN
operation is linear and, thus, would not increase the expressive power
of the weights.

<em class="property">`property` </em>`edge_weights`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPCBPDecoder.edge_weights" title="Permalink to this definition"></a>
    
Trainable weights of the BP decoder.


<em class="property">`property` </em>`has_weights`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPCBPDecoder.has_weights" title="Permalink to this definition"></a>
    
Indicates if decoder has trainable weights.


<em class="property">`property` </em>`ie_c`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPCBPDecoder.ie_c" title="Permalink to this definition"></a>
    
Extrinsic mutual information at check node.


<em class="property">`property` </em>`ie_v`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPCBPDecoder.ie_v" title="Permalink to this definition"></a>
    
Extrinsic mutual information at variable node.


<em class="property">`property` </em>`llr_max`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPCBPDecoder.llr_max" title="Permalink to this definition"></a>
    
Max LLR value used for internal calculations and rate-matching.


<em class="property">`property` </em>`num_cns`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPCBPDecoder.num_cns" title="Permalink to this definition"></a>
    
Number of check nodes.


<em class="property">`property` </em>`num_edges`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPCBPDecoder.num_edges" title="Permalink to this definition"></a>
    
Number of edges in decoding graph.


<em class="property">`property` </em>`num_iter`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPCBPDecoder.num_iter" title="Permalink to this definition"></a>
    
Number of decoding iterations.


<em class="property">`property` </em>`num_vns`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPCBPDecoder.num_vns" title="Permalink to this definition"></a>
    
Number of variable nodes.


<em class="property">`property` </em>`output_dtype`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPCBPDecoder.output_dtype" title="Permalink to this definition"></a>
    
Output dtype of decoder.


<em class="property">`property` </em>`pcm`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPCBPDecoder.pcm" title="Permalink to this definition"></a>
    
Parity-check matrix of LDPC code.


`show_weights`(<em class="sig-param">`size``=``7`</em>)<a class="reference internal" href="../_modules/sionna/fec/ldpc/decoding.html#LDPCBPDecoder.show_weights">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.decoding.LDPCBPDecoder.show_weights" title="Permalink to this definition"></a>
    
Show histogram of trainable weights.
Input
    
**size** (<em>float</em>) – Figure size of the matplotlib figure.




