
# Scrambling<a class="headerlink" href="https://nvlabs.github.io/sionna/#scrambling" title="Permalink to this headline"></a>

The <a class="reference internal" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.Scrambler" title="sionna.fec.scrambling.Scrambler">`Scrambler`</a> module allows to (pseudo)
randomly flip bits in a binary sequence or the signs of a real-valued sequence,
respectively. The <a class="reference internal" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.Descrambler" title="sionna.fec.scrambling.Descrambler">`Descrambler`</a> implement the corresponding descrambling operation.

To simplify distributed graph execution (e.g., by running scrambler and
descrambler in a different sub-graph/device), the scramblers are implemented
stateless. Thus, the internal seed cannot be update on runtime and does not
change after the initialization.
However, if required an explicit random seed can be passed as additional input
the scrambler/descrambler pair when calling the layer.

Further, the <a class="reference internal" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.TB5GScrambler" title="sionna.fec.scrambling.TB5GScrambler">`TB5GScrambler`</a> enables 5G NR compliant
scrambling as specified in <a class="reference internal" href="https://nvlabs.github.io/sionna/#gppts38211-scr" id="id1">[3GPPTS38211_scr]</a>.

The following code snippet shows how to setup and use an instance of the
scrambler:
```python
# set-up system
scrambler = Scrambler(seed=1234, # an explicit seed can be provided
                     binary=True) # indicate if bits shall be flipped
descrambler = Descrambler(scrambler=scrambler) # connect scrambler and descrambler
# --- simplified usage with fixed seed ---
# c has arbitrary shape and contains 0s and 1s (otherwise set binary=False)
c_scr = scrambler(c)
# descramble to reconstruct the original order
c_descr = descrambler(c_scr)
# --- advanced usage ---
# provide explicite seed if a new random seed should be used for each call
s = tf.random.uniform((), minval=0, maxval=12345678, dtype=tf.int32)
c_scr = scrambler([c, s])
c_descr = descrambler([c_scr, s])
```
## Scrambler<a class="headerlink" href="https://nvlabs.github.io/sionna/#scrambler" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.scrambling.``Scrambler`(<em class="sig-param">`seed=None`</em>, <em class="sig-param">`keep_batch_constant=False`</em>, <em class="sig-param">`sequence=None`</em>, <em class="sig-param">`binary=True` `keep_state=True`</em>, <em class="sig-param">`dtype=tf.float32`</em>, <em class="sig-param">`**kwargs`</em>)<a class="reference internal" href="https://nvlabs.github.io/sionna/../_modules/sionna/fec/scrambling.html#Scrambler">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.Scrambler" title="Permalink to this definition"></a>

Randomly flips the state/sign of a sequence of bits or LLRs, respectively.

The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters

- 
**seed** (<em>int</em>) – Defaults to None. Defines the initial state of the
pseudo random generator to generate the scrambling sequence.
If None, a random integer will be generated. Only used
when called with `keep_state` is True.
- 
**keep_batch_constant** (<em>bool</em>) – Defaults to False. If True, all samples in the batch are scrambled
with the same scrambling sequence. Otherwise, per sample a random
sequence is generated.
- 
**sequence** (<em>Array of 0s and 1s</em><em> or </em><em>None</em>) – If provided, the seed will be ignored and the explicit scrambling
sequence is used. Shape must be broadcastable to `x`.
- 
**binary** (<em>bool</em>) – Defaults to True. Indicates whether bit-sequence should be flipped
(i.e., binary operations are performed) or the signs should be
flipped (i.e., soft-value/LLR domain-based).
- 
**keep_state** (<em>bool</em>) – Defaults to True. Indicates whether the scrambling sequence should
be kept constant.
- 
**dtype** (<em>tf.DType</em>) – Defaults to <cite>tf.float32</cite>. Defines the datatype for internal
calculations and the output dtype.


Input

- 
**(x, seed, binary)** – Either Tuple `(x,` `seed,` `binary)` or  `(x,` `seed)` or `x` only
(no tuple) if the internal  seed should be used:
- 
**x** (<em>tf.float</em>) – 1+D tensor of arbitrary shape.
- 
**seed** (<em>int</em>) – An integer defining the state of the random number
generator. If explicitly given, the global internal seed is
replaced by this seed. Can be used to realize random
scrambler/descrambler pairs (call with same random seed).
- 
**binary** (<em>bool</em>) – Overrules the init parameter <cite>binary</cite> iff explicitly given.
Indicates whether bit-sequence should be flipped
(i.e., binary operations are performed) or the signs should be
flipped (i.e., soft-value/LLR domain-based).


Output

<em>tf.float</em> – 1+D tensor of same shape as `x`.



### Note

For inverse scrambling, the same scrambler can be re-used (as the values
are flipped again, i.e., result in the original state). However,
`keep_state` must be set to True as a new sequence would be generated
otherwise.

The scrambler layer is stateless, i.e., the seed is either random
during each call or must be explicitly provided during init/call.
This simplifies XLA/graph execution.
If the seed is provided in the init() function, this fixed seed is used
for all calls. However, an explicit seed can be provided during
the call function to realize <cite>true</cite> random states.

Scrambling is typically used to ensure equal likely <cite>0</cite>  and <cite>1</cite> for
sources with unequal bit probabilities. As we have a perfect source in
the simulations, this is not required. However, for all-zero codeword
simulations and higher-order modulation, so-called “channel-adaptation”
<a class="reference internal" href="https://nvlabs.github.io/sionna/#pfister03" id="id2">[Pfister03]</a> is required.

Raises

- 
**AssertionError** – If `seed` is not int.
- 
**AssertionError** – If `keep_batch_constant` is not bool.
- 
**AssertionError** – If `binary` is not bool.
- 
**AssertionError** – If `keep_state` is not bool.
- 
**AssertionError** – If `seed` is provided to list of inputs but not an
    int.
- 
**TypeError** – If <cite>dtype</cite> of `x` is not as expected.




<em class="property">`property` </em>`keep_state`<a class="headerlink" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.Scrambler.keep_state" title="Permalink to this definition"></a>

Indicates if new random sequences are used per call.


<em class="property">`property` </em>`seed`<a class="headerlink" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.Scrambler.seed" title="Permalink to this definition"></a>

Seed used to generate random sequence.


<em class="property">`property` </em>`sequence`<a class="headerlink" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.Scrambler.sequence" title="Permalink to this definition"></a>

Explicit scrambling sequence if provided.


## TB5GScrambler<a class="headerlink" href="https://nvlabs.github.io/sionna/#tb5gscrambler" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.scrambling.``TB5GScrambler`(<em class="sig-param">`n_rnti``=``1`</em>, <em class="sig-param">`n_id``=``1`</em>, <em class="sig-param">`binary``=``True`</em>, <em class="sig-param">`channel_type``=``'PUSCH'`</em>, <em class="sig-param">`codeword_index``=``0`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="https://nvlabs.github.io/sionna/../_modules/sionna/fec/scrambling.html#TB5GScrambler">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.TB5GScrambler" title="Permalink to this definition"></a>

Implements the pseudo-random bit scrambling as defined in
<a class="reference internal" href="https://nvlabs.github.io/sionna/#gppts38211-scr" id="id3">[3GPPTS38211_scr]</a> Sec. 6.3.1.1 for the “PUSCH” channel and in Sec. 7.3.1.1
for the “PDSCH” channel.

Only for the “PDSCH” channel, the scrambler can be configured for two
codeword transmission mode. Hereby, `codeword_index` corresponds to the
index of the codeword to be scrambled.

If `n_rnti` are a list of ints, the scrambler assumes that the second
last axis contains <cite>len(</cite> `n_rnti` <cite>)</cite> elements. This allows independent
scrambling for multiple independent streams.

The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters

- 
**n_rnti** (<em>int</em><em> or </em><em>list of ints</em>) – RNTI identifier provided by higher layer. Defaults to 1 and must be
in range <cite>[0, 65335]</cite>. If a list is provided, every list element
defines a scrambling sequence for multiple independent streams.
- 
**n_id** (<em>int</em><em> or </em><em>list of ints</em>) – Scrambling ID related to cell id and provided by higher layer.
Defaults to 1 and must be in range <cite>[0, 1023]</cite>. If a list is
provided, every list element defines a scrambling sequence for
multiple independent streams.
- 
**binary** (<em>bool</em>) – Defaults to True. Indicates whether bit-sequence should be flipped
(i.e., binary operations are performed) or the signs should be
flipped (i.e., soft-value/LLR domain-based).
- 
**channel_type** (<em>str</em>) – Can be either “PUSCH” or “PDSCH”.
- 
**codeword_index** (<em>int</em>) – Scrambler can be configured for two codeword transmission.
`codeword_index` can be either 0 or 1.
- 
**dtype** (<em>tf.DType</em>) – Defaults to <cite>tf.float32</cite>. Defines the datatype for internal
calculations and the output dtype.


Input

- 
**(x, binary)** – Either Tuple `(x,` `binary)` or  `x` only
- 
**x** (<em>tf.float</em>) – 1+D tensor of arbitrary shape. If `n_rnti` and `n_id` are a
list, it is assumed that `x` has shape
<cite>[…,num_streams, n]</cite> where <cite>num_streams=len(</cite> `n_rnti` <cite>)</cite>.
- 
**binary** (<em>bool</em>) – Overrules the init parameter <cite>binary</cite> iff explicitly given.
Indicates whether bit-sequence should be flipped
(i.e., binary operations are performed) or the signs should be
flipped (i.e., soft-value/LLR domain-based).


Output

<em>tf.float</em> – 1+D tensor of same shape as `x`.



### Note

The parameters radio network temporary identifier (RNTI) `n_rnti` and
the datascrambling ID `n_id` are usually provided be the higher layer protocols.

For inverse scrambling, the same scrambler can be re-used (as the values
are flipped again, i.e., result in the original state).

<em class="property">`property` </em>`keep_state`<a class="headerlink" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.TB5GScrambler.keep_state" title="Permalink to this definition"></a>

Required for descrambler, is always <cite>True</cite> for the TB5GScrambler.


## Descrambler<a class="headerlink" href="https://nvlabs.github.io/sionna/#descrambler" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.scrambling.``Descrambler`(<em class="sig-param">`scrambler`</em>, <em class="sig-param">`binary``=``True`</em>, <em class="sig-param">`dtype``=``None`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="https://nvlabs.github.io/sionna/../_modules/sionna/fec/scrambling.html#Descrambler">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.Descrambler" title="Permalink to this definition"></a>

Descrambler for a given scrambler.

The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters

- 
**scrambler** (<a class="reference internal" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.Scrambler" title="sionna.fec.scrambling.Scrambler"><em>Scrambler</em></a><em>, </em><a class="reference internal" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.TB5GScrambler" title="sionna.fec.scrambling.TB5GScrambler"><em>TB5GScrambler</em></a>) – Associated <a class="reference internal" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.Scrambler" title="sionna.fec.scrambling.Scrambler">`Scrambler`</a> or
<a class="reference internal" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.TB5GScrambler" title="sionna.fec.scrambling.TB5GScrambler">`TB5GScrambler`</a> instance which
should be descrambled.
- 
**binary** (<em>bool</em>) – Defaults to True. Indicates whether bit-sequence should be flipped
(i.e., binary operations are performed) or the signs should be
flipped (i.e., soft-value/LLR domain-based).
- 
**dtype** (<em>None</em><em> or </em><em>tf.DType</em>) – Defaults to <cite>None</cite>. Defines the datatype for internal calculations
and the output dtype. If no explicit dtype is provided the dtype
from the associated interleaver is used.


Input

- 
**(x, seed)** – Either Tuple `(x,` `seed)` or `x` only (no tuple) if the internal
seed should be used:
- 
**x** (<em>tf.float</em>) – 1+D tensor of arbitrary shape.
- 
**seed** (<em>int</em>) – An integer defining the state of the random number
generator. If explicitly given, the global internal seed is
replaced by this seed. Can be used to realize random
scrambler/descrambler pairs (call with same random seed).


Output

<em>tf.float</em> – 1+D tensor of same shape as `x`.

Raises

- 
**AssertionError** – If `scrambler` is not an instance of <cite>Scrambler</cite>.
- 
**AssertionError** – If `seed` is provided to list of inputs but not an
    int.
- 
**TypeError** – If <cite>dtype</cite> of `x` is not as expected.




<em class="property">`property` </em>`scrambler`<a class="headerlink" href="https://nvlabs.github.io/sionna/#sionna.fec.scrambling.Descrambler.scrambler" title="Permalink to this definition"></a>

Associated scrambler instance.



References:
<a class="fn-backref" href="https://nvlabs.github.io/sionna/#id2">Pfister03</a>

J. Hou, P.Siegel, L. Milstein, and H. Pfister, “Capacity
approaching bandwidth-efficient coded modulation schemes
based on low-density parity-check codes,” IEEE Trans. Inf. Theory,
Sep. 2003.

3GPPTS38211_scr(<a href="https://nvlabs.github.io/sionna/#id1">1</a>,<a href="https://nvlabs.github.io/sionna/#id3">2</a>)

ETSI 3GPP TS 38.211 “Physical channels and modulation”,
v.16.2.0, 2020-07.



