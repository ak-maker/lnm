# Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#utility-functions" title="Permalink to this headline"></a>
    
The utilities sub-package of the Sionna library contains many convenience
functions as well as extensions to existing TensorFlow functions.

# Table of Content
## Metrics<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#metrics" title="Permalink to this headline"></a>
### BitErrorRate<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#biterrorrate" title="Permalink to this headline"></a>
### BitwiseMutualInformation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#bitwisemutualinformation" title="Permalink to this headline"></a>
### compute_ber<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#compute-ber" title="Permalink to this headline"></a>
### compute_bler<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#compute-bler" title="Permalink to this headline"></a>
### compute_ser<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#compute-ser" title="Permalink to this headline"></a>
### count_errors<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#count-errors" title="Permalink to this headline"></a>
### count_block_errors<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#count-block-errors" title="Permalink to this headline"></a>
## Tensors<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#tensors" title="Permalink to this headline"></a>
### expand_to_rank<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#expand-to-rank" title="Permalink to this headline"></a>
### flatten_dims<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#flatten-dims" title="Permalink to this headline"></a>
### flatten_last_dims<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#flatten-last-dims" title="Permalink to this headline"></a>
### insert_dims<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#insert-dims" title="Permalink to this headline"></a>
### split_dims<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#split-dims" title="Permalink to this headline"></a>
  
  

## Metrics<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#metrics" title="Permalink to this headline"></a>

### BitErrorRate<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#biterrorrate" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.utils.``BitErrorRate`(<em class="sig-param">`name``=``'bit_error_rate'`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/utils/metrics.html#BitErrorRate">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.BitErrorRate" title="Permalink to this definition"></a>
    
Computes the average bit error rate (BER) between two binary tensors.
    
This class implements a Keras metric for the bit error rate
between two tensors of bits.
Input
 
- **b** (<em>tf.float32</em>) – A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (<em>tf.float32</em>) – A tensor of the same shape as `b` filled with
ones and zeros.


Output
    
<em>tf.float32</em> – A scalar, the BER.



### BitwiseMutualInformation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#bitwisemutualinformation" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.utils.``BitwiseMutualInformation`(<em class="sig-param">`name``=``'bitwise_mutual_information'`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/utils/metrics.html#BitwiseMutualInformation">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.BitwiseMutualInformation" title="Permalink to this definition"></a>
    
Computes the bitwise mutual information between bits and LLRs.
    
This class implements a Keras metric for the bitwise mutual information
between a tensor of bits and LLR (logits).
Input
 
- **bits** (<em>tf.float32</em>) – A tensor of arbitrary shape filled with ones and zeros.
- **llr** (<em>tf.float32</em>) – A tensor of the same shape as `bits` containing logits.


Output
    
<em>tf.float32</em> – A scalar, the bit-wise mutual information.



### compute_ber<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#compute-ber" title="Permalink to this headline"></a>

`sionna.utils.``compute_ber`(<em class="sig-param">`b`</em>, <em class="sig-param">`b_hat`</em>)<a class="reference internal" href="../_modules/sionna/utils/metrics.html#compute_ber">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.compute_ber" title="Permalink to this definition"></a>
    
Computes the bit error rate (BER) between two binary tensors.
Input
 
- **b** (<em>tf.float32</em>) – A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (<em>tf.float32</em>) – A tensor of the same shape as `b` filled with
ones and zeros.


Output
    
<em>tf.float64</em> – A scalar, the BER.



### compute_bler<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#compute-bler" title="Permalink to this headline"></a>

`sionna.utils.``compute_bler`(<em class="sig-param">`b`</em>, <em class="sig-param">`b_hat`</em>)<a class="reference internal" href="../_modules/sionna/utils/metrics.html#compute_bler">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.compute_bler" title="Permalink to this definition"></a>
    
Computes the block error rate (BLER) between two binary tensors.
    
A block error happens if at least one element of `b` and `b_hat`
differ in one block. The BLER is evaluated over the last dimension of
the input, i. e., all elements of the last dimension are considered to
define a block.
    
This is also sometimes referred to as <cite>word error rate</cite> or <cite>frame error
rate</cite>.
Input
 
- **b** (<em>tf.float32</em>) – A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (<em>tf.float32</em>) – A tensor of the same shape as `b` filled with
ones and zeros.


Output
    
<em>tf.float64</em> – A scalar, the BLER.



### compute_ser<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#compute-ser" title="Permalink to this headline"></a>

`sionna.utils.``compute_ser`(<em class="sig-param">`s`</em>, <em class="sig-param">`s_hat`</em>)<a class="reference internal" href="../_modules/sionna/utils/metrics.html#compute_ser">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.compute_ser" title="Permalink to this definition"></a>
    
Computes the symbol error rate (SER) between two integer tensors.
Input
 
- **s** (<em>tf.int</em>) – A tensor of arbitrary shape filled with integers indicating
the symbol indices.
- **s_hat** (<em>tf.int</em>) – A tensor of the same shape as `s` filled with integers indicating
the estimated symbol indices.


Output
    
<em>tf.float64</em> – A scalar, the SER.



### count_errors<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#count-errors" title="Permalink to this headline"></a>

`sionna.utils.``count_errors`(<em class="sig-param">`b`</em>, <em class="sig-param">`b_hat`</em>)<a class="reference internal" href="../_modules/sionna/utils/metrics.html#count_errors">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.count_errors" title="Permalink to this definition"></a>
    
Counts the number of bit errors between two binary tensors.
Input
 
- **b** (<em>tf.float32</em>) – A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (<em>tf.float32</em>) – A tensor of the same shape as `b` filled with
ones and zeros.


Output
    
<em>tf.int64</em> – A scalar, the number of bit errors.



### count_block_errors<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#count-block-errors" title="Permalink to this headline"></a>

`sionna.utils.``count_block_errors`(<em class="sig-param">`b`</em>, <em class="sig-param">`b_hat`</em>)<a class="reference internal" href="../_modules/sionna/utils/metrics.html#count_block_errors">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.count_block_errors" title="Permalink to this definition"></a>
    
Counts the number of block errors between two binary tensors.
    
A block error happens if at least one element of `b` and `b_hat`
differ in one block. The BLER is evaluated over the last dimension of
the input, i. e., all elements of the last dimension are considered to
define a block.
    
This is also sometimes referred to as <cite>word error rate</cite> or <cite>frame error
rate</cite>.
Input
 
- **b** (<em>tf.float32</em>) – A tensor of arbitrary shape filled with ones and
zeros.
- **b_hat** (<em>tf.float32</em>) – A tensor of the same shape as `b` filled with
ones and zeros.


Output
    
<em>tf.int64</em> – A scalar, the number of block errors.



## Tensors<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#tensors" title="Permalink to this headline"></a>

### expand_to_rank<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#expand-to-rank" title="Permalink to this headline"></a>

`sionna.utils.``expand_to_rank`(<em class="sig-param">`tensor`</em>, <em class="sig-param">`target_rank`</em>, <em class="sig-param">`axis``=``-` `1`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#expand_to_rank">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.expand_to_rank" title="Permalink to this definition"></a>
    
Inserts as many axes to a tensor as needed to achieve a desired rank.
    
This operation inserts additional dimensions to a `tensor` starting at
`axis`, so that so that the rank of the resulting tensor has rank
`target_rank`. The dimension index follows Python indexing rules, i.e.,
zero-based, where a negative index is counted backward from the end.
Parameters
 
- **tensor** – A tensor.
- **target_rank** (<em>int</em>) – The rank of the output tensor.
If `target_rank` is smaller than the rank of `tensor`,
the function does nothing.
- **axis** (<em>int</em>) – The dimension index at which to expand the
shape of `tensor`. Given a `tensor` of <cite>D</cite> dimensions,
`axis` must be within the range <cite>[-(D+1), D]</cite> (inclusive).


Returns
    
A tensor with the same data as `tensor`, with
`target_rank`- rank(`tensor`) additional dimensions inserted at the
index specified by `axis`.
If `target_rank` <= rank(`tensor`), `tensor` is returned.



### flatten_dims<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#flatten-dims" title="Permalink to this headline"></a>

`sionna.utils.``flatten_dims`(<em class="sig-param">`tensor`</em>, <em class="sig-param">`num_dims`</em>, <em class="sig-param">`axis`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#flatten_dims">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.flatten_dims" title="Permalink to this definition"></a>
    
Flattens a specified set of dimensions of a tensor.
    
This operation flattens `num_dims` dimensions of a `tensor`
starting at a given `axis`.
Parameters
 
- **tensor** – A tensor.
- **num_dims** (<em>int</em>) – The number of dimensions
to combine. Must be larger than two and less or equal than the
rank of `tensor`.
- **axis** (<em>int</em>) – The index of the dimension from which to start.


Returns
    
A tensor of the same type as `tensor` with `num_dims`-1 lesser
dimensions, but the same number of elements.



### flatten_last_dims<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#flatten-last-dims" title="Permalink to this headline"></a>

`sionna.utils.``flatten_last_dims`(<em class="sig-param">`tensor`</em>, <em class="sig-param">`num_dims``=``2`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#flatten_last_dims">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.flatten_last_dims" title="Permalink to this definition"></a>
    
Flattens the last <cite>n</cite> dimensions of a tensor.
    
This operation flattens the last `num_dims` dimensions of a `tensor`.
It is a simplified version of the function `flatten_dims`.
Parameters
 
- **tensor** – A tensor.
- **num_dims** (<em>int</em>) – The number of dimensions
to combine. Must be greater than or equal to two and less or equal
than the rank of `tensor`.


Returns
    
A tensor of the same type as `tensor` with `num_dims`-1 lesser
dimensions, but the same number of elements.



### insert_dims<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#insert-dims" title="Permalink to this headline"></a>

`sionna.utils.``insert_dims`(<em class="sig-param">`tensor`</em>, <em class="sig-param">`num_dims`</em>, <em class="sig-param">`axis``=``-` `1`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#insert_dims">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.insert_dims" title="Permalink to this definition"></a>
    
Adds multiple length-one dimensions to a tensor.
    
This operation is an extension to TensorFlow`s `expand_dims` function.
It inserts `num_dims` dimensions of length one starting from the
dimension `axis` of a `tensor`. The dimension
index follows Python indexing rules, i.e., zero-based, where a negative
index is counted backward from the end.
Parameters
 
- **tensor** – A tensor.
- **num_dims** (<em>int</em>) – The number of dimensions to add.
- **axis** – The dimension index at which to expand the
shape of `tensor`. Given a `tensor` of <cite>D</cite> dimensions,
`axis` must be within the range <cite>[-(D+1), D]</cite> (inclusive).


Returns
    
A tensor with the same data as `tensor`, with `num_dims` additional
dimensions inserted at the index specified by `axis`.



### split_dims<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#split-dims" title="Permalink to this headline"></a>

`sionna.utils.``split_dim`(<em class="sig-param">`tensor`</em>, <em class="sig-param">`shape`</em>, <em class="sig-param">`axis`</em>)<a class="reference internal" href="../_modules/sionna/utils/tensors.html#split_dim">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.split_dim" title="Permalink to this definition"></a>
    
Reshapes a dimension of a tensor into multiple dimensions.
    
This operation splits the dimension `axis` of a `tensor` into
multiple dimensions according to `shape`.
Parameters
 
- **tensor** – A tensor.
- **shape** (<em>list</em><em> or </em><em>TensorShape</em>) – The shape to which the dimension should
be reshaped.
- **axis** (<em>int</em>) – The index of the axis to be reshaped.


Returns
    
A tensor of the same type as `tensor` with len(`shape`)-1
additional dimensions, but the same number of elements.



