# Interleaving<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#interleaving" title="Permalink to this headline"></a>
    
The interleaver module allows to permute tensors with either pseudo-random permutations or by row/column swapping.
    
To simplify distributed graph execution (e.g., by running interleaver and deinterleaver in a different sub-graph/device), the interleavers are implemented stateless. Thus, the internal seed cannot be updated on runtime and does not change after the initialization. However, if required, an explicit random seed can be passed as additional input to the interleaver/deinterleaver pair when calling the layer.
    
The following code snippet shows how to setup and use an instance of the interleaver:
```python
# set-up system
interleaver = RandomInterleaver(seed=1234, # an explicit seed can be provided
                                keep_batch_constant=False, # if True, all samples in the batch are permuted with the same pattern
                                axis=-1) # axis which shall be permuted
deinterleaver = Deinterleaver(interleaver=interleaver) # connect interleaver and deinterleaver
# --- simplified usage with fixed seed ---
# c has arbitrary shape (rank>=2)
c_int = interleaver(c)
# call deinterleaver to reconstruct the original order
c_deint = deinterleaver(c_int)
# --- advanced usage ---
# provide explicit seed if a new random seed should be used for each call
s = tf.random.uniform((), minval=0, maxval=12345678, dtype=tf.int32)
c_int = interleaver([c, s])
c_deint = deinterleaver([c_int, s])
```
# Table of Content
## Interleaver<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#interleaver" title="Permalink to this headline"></a>
### Turbo3GPPInterleaver<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#turbo3gppinterleaver" title="Permalink to this headline"></a>
## Deinterleaver<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#deinterleaver" title="Permalink to this headline"></a>
  
  

### Turbo3GPPInterleaver<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#turbo3gppinterleaver" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.interleaving.``Turbo3GPPInterleaver`(<em class="sig-param">`inverse``=``False`</em>, <em class="sig-param">`axis``=``-` `1`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/interleaving.html#Turbo3GPPInterleaver">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#sionna.fec.interleaving.Turbo3GPPInterleaver" title="Permalink to this definition"></a>
    
Interleaver as used in the 3GPP Turbo codes <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#gppts36212-i" id="id1">[3GPPTS36212_I]</a> and, thus,
the maximum length is given as 6144 elements (only for the dimension as
specific by `axis`).
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **inverse** (<em>bool</em>) – A boolean defaults to False. If True, the inverse permutation is
performed.
- **axis** (<em>int</em>) – Defaults to <cite>-1</cite>. The dimension that should be interleaved.
First dimension (<cite>axis=0</cite>) is not allowed.
- **dtype** (<em>tf.DType</em>) – Defaults to <cite>tf.float32</cite>. Defines the datatype for internal
calculations and the output dtype.


Input
    
**x** (<em>tf.DType</em>) – 2+D tensor of arbitrary shape and dtype.

Output
    
<em>tf.DType</em> – 2+D tensor of same shape and dtype as the input `x`.

Raises
 
- **AssertionError** – If `axis` is not <cite>int</cite>.
- **AssertionError** – If `axis` > number of input dimensions.
- **AssertionError** – If `inverse` is not bool.
- **InvalidArgumentError** – When rank(`x`)<2.




**Note**
    
Note that this implementation slightly deviates from the 3GPP
standard <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#gppts36212-i" id="id2">[3GPPTS36212_I]</a> in a sense that zero-padding is introduced
for cases when the exact interleaver length is not supported by the
standard.

<em class="property">`property` </em>`axis`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#sionna.fec.interleaving.Turbo3GPPInterleaver.axis" title="Permalink to this definition"></a>
    
Axis to be permuted.


`call_inverse`(<em class="sig-param">`inputs`</em>)<a class="reference internal" href="../_modules/sionna/fec/interleaving.html#Turbo3GPPInterleaver.call_inverse">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#sionna.fec.interleaving.Turbo3GPPInterleaver.call_inverse" title="Permalink to this definition"></a>
    
Implements deinterleaver function corresponding to call().
Input
    
**x** (<em>tf.DType</em>) – 2+D tensor of arbitrary shape and dtype.

Output
    
<em>tf.DType</em> – 2+D tensor of same shape and dtype as the input `x`.

Raises
    
**InvalidArgumentError** – When rank(`x`)<2.




`find_s_min`(<em class="sig-param">`frame_size`</em>, <em class="sig-param">`s_min_stop``=``0`</em>)<a class="reference internal" href="../_modules/sionna/fec/interleaving.html#Turbo3GPPInterleaver.find_s_min">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#sionna.fec.interleaving.Turbo3GPPInterleaver.find_s_min" title="Permalink to this definition"></a>
    
Find $S$ parameter such that $\pi(i)-\pi(j)>S$ for all
$i-j<S$. This can be used to find optimized interleaver patterns.
    
`s_min_stop` is an additional stopping condition, i.e., stop if
current $S$ is already smaller than `s_min_stop`.
    
Please note that this is a Numpy utility function and usually not part
of the graph.
Input
 
- **frame_size** (<em>int</em>) – length of interleaver.
- **s_min_stop** (<em>int</em>) – Defaults to 0. Enables early stop if already current
s_min<`s_min_stop`.


Output
    
<em>float</em> – The S-parameter for the given `frame_size`.




## Deinterleaver<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#deinterleaver" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.interleaving.``Deinterleaver`(<em class="sig-param">`interleaver`</em>, <em class="sig-param">`dtype``=``None`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/interleaving.html#Deinterleaver">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#sionna.fec.interleaving.Deinterleaver" title="Permalink to this definition"></a>
    
Deinterleaver that reverts the interleaver for a given input sequence.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **interleaver** (<em>Interleaver</em>) – Associated Interleaver which shall be deinterleaved by this layer.
Can be either
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#sionna.fec.interleaving.RandomInterleaver" title="sionna.fec.interleaving.RandomInterleaver">`RandomInterleaver`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#sionna.fec.interleaving.RowColumnInterleaver" title="sionna.fec.interleaving.RowColumnInterleaver">`RowColumnInterleaver`</a>.
- **dtype** (<em>None</em><em> or </em><em>tf.DType</em>) – Defaults to <cite>None</cite>. Defines the datatype for internal calculations
and the output dtype. If no explicit dtype is provided the dtype
from the associated interleaver is used.


Input
 
- **(x, seed)** – Either Tuple `(x,` `seed)` or `x` only (no tuple) if the internal
seed should be used:
- **x** (<em>tf.DType</em>) – 2+D tensor of arbitrary shape.
- **seed** (<em>int</em>) – An integer defining the state of the random number
generator. If explicitly given, the global internal seed is
replaced by this seed. Can be used to realize random
interleaver/deinterleaver pairs (call with same random seed).


Output
    
<em>tf.DType</em> – 2+D tensor of same shape and dtype as the input `x`.

Raises
    
**AssertionError** – If `interleaver` is not a valid instance of Interleaver.



**Note**
    
This layer provides a wrapper of the inverse interleaver function.

<em class="property">`property` </em>`interleaver`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#sionna.fec.interleaving.Deinterleaver.interleaver" title="Permalink to this definition"></a>
    
Associated interleaver instance.


    
References:
<blockquote>
<div>
3GPPTS36212_I(<a href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.interleaving.html#id2">2</a>)
    
ETSI 3GPP TS 36.212 “Evolved Universal Terrestrial
Radio Access (EUTRA); Multiplexing and channel coding”, v.15.3.0, 2018-09.


</blockquote>