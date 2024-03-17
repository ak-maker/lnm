# Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#utility-functions" title="Permalink to this headline"></a>
    
This module provides utility functions for the FEC package. It also provides serval functions to simplify EXIT analysis of iterative receivers.

# Table of Content
## Miscellaneous<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#miscellaneous" title="Permalink to this headline"></a>
### int2bin<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#int2bin" title="Permalink to this headline"></a>
### bin2int_tf<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#bin2int-tf" title="Permalink to this headline"></a>
### int2bin_tf<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#int2bin-tf" title="Permalink to this headline"></a>
### int_mod_2<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#int-mod-2" title="Permalink to this headline"></a>
### llr2mi<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#llr2mi" title="Permalink to this headline"></a>
### j_fun<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#j-fun" title="Permalink to this headline"></a>
### j_fun_inv<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#j-fun-inv" title="Permalink to this headline"></a>
### j_fun_tf<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#j-fun-tf" title="Permalink to this headline"></a>
### j_fun_inv_tf<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#j-fun-inv-tf" title="Permalink to this headline"></a>
  
  

### int2bin<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#int2bin" title="Permalink to this headline"></a>

`sionna.fec.utils.``int2bin`(<em class="sig-param">`num`</em>, <em class="sig-param">`len_`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#int2bin">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.int2bin" title="Permalink to this definition"></a>
    
Convert `num` of int type to list of length `len_` with 0’s and 1’s.
`num` and `len_` have to non-negative.
    
For e.g., `num` = <cite>5</cite>; <cite>int2bin(num</cite>, `len_` =4) = <cite>[0, 1, 0, 1]</cite>.
    
For e.g., `num` = <cite>12</cite>; <cite>int2bin(num</cite>, `len_` =3) = <cite>[1, 0, 0]</cite>.
Input
 
- **num** (<em>int</em>) – An integer to be converted into binary representation.
- **len_** (<em>int</em>) – An integer defining the length of the desired output.


Output
    
<em>list of int</em> – Binary representation of `num` of length `len_`.



### bin2int_tf<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#bin2int-tf" title="Permalink to this headline"></a>

`sionna.fec.utils.``bin2int_tf`(<em class="sig-param">`arr`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#bin2int_tf">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.bin2int_tf" title="Permalink to this definition"></a>
    
Converts binary tensor to int tensor. Binary representation in `arr`
is across the last dimension from most significant to least significant.
    
For example `arr` = <cite>[0, 1, 1]</cite> is converted to <cite>3</cite>.
Input
    
**arr** (<em>int or float</em>) – Tensor of  0’s and 1’s.

Output
    
<em>int</em> – Tensor containing the integer representation of `arr`.



### int2bin_tf<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#int2bin-tf" title="Permalink to this headline"></a>

`sionna.fec.utils.``int2bin_tf`(<em class="sig-param">`ints`</em>, <em class="sig-param">`len_`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#int2bin_tf">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.int2bin_tf" title="Permalink to this definition"></a>
    
Converts (int) tensor to (int) tensor with 0’s and 1’s. <cite>len_</cite> should be
to non-negative. Additional dimension of size <cite>len_</cite> is inserted at end.
Input
 
- **ints** (<em>int</em>) – Tensor of arbitrary shape <cite>[…,k]</cite> containing integer to be
converted into binary representation.
- **len_** (<em>int</em>) – An integer defining the length of the desired output.


Output
    
<em>int</em> – Tensor of same shape as `ints` except dimension of length
`len_` is added at the end <cite>[…,k, len_]</cite>. Contains the binary
representation of `ints` of length `len_`.



### int_mod_2<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#int-mod-2" title="Permalink to this headline"></a>

`sionna.fec.utils.``int_mod_2`(<em class="sig-param">`x`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#int_mod_2">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.int_mod_2" title="Permalink to this definition"></a>
    
Efficient implementation of modulo 2 operation for integer inputs.
    
This function assumes integer inputs or implicitly casts to int.
    
Remark: the function <cite>tf.math.mod(x, 2)</cite> is placed on the CPU and, thus,
causes unnecessary memory copies.
Parameters
    
**x** (<em>tf.Tensor</em>) – Tensor to which the modulo 2 operation is applied.



### llr2mi<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#llr2mi" title="Permalink to this headline"></a>

`sionna.fec.utils.``llr2mi`(<em class="sig-param">`llr`</em>, <em class="sig-param">`s``=``None`</em>, <em class="sig-param">`reduce_dims``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#llr2mi">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.llr2mi" title="Permalink to this definition"></a>
    
Implements an approximation of the mutual information based on LLRs.
    
The function approximates the mutual information for given `llr` as
derived in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#hagenauer" id="id16">[Hagenauer]</a> assuming an <cite>all-zero codeword</cite> transmission

$$
I \approx 1 - \sum \operatorname{log_2} \left( 1 + \operatorname{e}^{-\text{llr}} \right).
$$
    
This approximation assumes that the following <cite>symmetry condition</cite> is fulfilled

$$
p(\text{llr}|x=0) = p(\text{llr}|x=1) \cdot \operatorname{exp}(\text{llr}).
$$
    
For <cite>non-all-zero</cite> codeword transmissions, this methods requires knowledge
about the signs of the original bit sequence `s` and flips the signs
correspondingly (as if the all-zero codeword was transmitted).
    
Please note that we define LLRs as $\frac{p(x=1)}{p(x=0)}$, i.e.,
the sign of the LLRs differ to the solution in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#hagenauer" id="id17">[Hagenauer]</a>.
Input
 
- **llr** (<em>tf.float32</em>) – Tensor of arbitrary shape containing LLR-values.
- **s** (<em>None or tf.float32</em>) – Tensor of same shape as llr containing the signs of the
transmitted sequence (assuming BPSK), i.e., +/-1 values.
- **reduce_dims** (<em>bool</em>) – Defaults to True. If True, all dimensions are
reduced and the return is a scalar. Otherwise, <cite>reduce_mean</cite> is
only taken over the last dimension.


Output
    
**mi** (<em>tf.float32</em>) – A scalar tensor (if `reduce_dims` is True) or a tensor of same
shape as `llr` apart from the last dimensions that is removed.
It contains the approximated value of the mutual information.

Raises
    
**TypeError** – If dtype of `llr` is not a real-valued float.



### j_fun<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#j-fun" title="Permalink to this headline"></a>

`sionna.fec.utils.``j_fun`(<em class="sig-param">`mu`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#j_fun">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.j_fun" title="Permalink to this definition"></a>
    
Calculates the <cite>J-function</cite> in NumPy.
    
The so-called <cite>J-function</cite> relates mutual information to the mean of
Gaussian distributed LLRs (cf. Gaussian approximation). We use the
approximation as proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#brannstrom" id="id18">[Brannstrom]</a> which can be written as

$$
J(\mu) \approx \left( 1- 2^{H_\text{1}(2\mu)^{H_\text{2}}}\right)^{H_\text{2}}
$$
    
with $\mu$ denoting the mean value of the LLR distribution and
$H_\text{1}=0.3073$, $H_\text{2}=0.8935$ and
$H_\text{3}=1.1064$.
Input
    
**mu** (<em>float</em>) – float or <cite>ndarray</cite> of float.

Output
    
<em>float</em> – <cite>ndarray</cite> of same shape as the input.



### j_fun_inv<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#j-fun-inv" title="Permalink to this headline"></a>

`sionna.fec.utils.``j_fun_inv`(<em class="sig-param">`mi`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#j_fun_inv">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.j_fun_inv" title="Permalink to this definition"></a>
    
Calculates the inverse <cite>J-function</cite> in NumPy.
    
The so-called <cite>J-function</cite> relates mutual information to the mean of
Gaussian distributed LLRs (cf. Gaussian approximation). We use the
approximation as proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#brannstrom" id="id19">[Brannstrom]</a> which can be written as

$$
J(\mu) \approx \left( 1- 2^{H_\text{1}(2\mu)^{H_\text{2}}}\right)^{H_\text{2}}
$$
    
with $\mu$ denoting the mean value of the LLR distribution and
$H_\text{1}=0.3073$, $H_\text{2}=0.8935$ and
$H_\text{3}=1.1064$.
Input
    
**mi** (<em>float</em>) – float or <cite>ndarray</cite> of float.

Output
    
<em>float</em> – <cite>ndarray</cite> of same shape as the input.

Raises
    
**AssertionError** – If `mi` < 0.001 or `mi` > 0.999.



### j_fun_tf<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#j-fun-tf" title="Permalink to this headline"></a>

`sionna.fec.utils.``j_fun_tf`(<em class="sig-param">`mu`</em>, <em class="sig-param">`verify_inputs``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#j_fun_tf">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.j_fun_tf" title="Permalink to this definition"></a>
    
Calculates the <cite>J-function</cite> in Tensorflow.
    
The so-called <cite>J-function</cite> relates mutual information to the mean of
Gaussian distributed LLRs (cf. Gaussian approximation). We use the
approximation as proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#brannstrom" id="id20">[Brannstrom]</a> which can be written as

$$
J(\mu) \approx \left( 1- 2^{H_\text{1}(2\mu)^{H_\text{2}}}\right)^{H_\text{2}}
$$
    
with $\mu$ denoting the mean value of the LLR distribution and
$H_\text{1}=0.3073$, $H_\text{2}=0.8935$ and
$H_\text{3}=1.1064$.
Input
 
- **mu** (<em>tf.float32</em>) – Tensor of arbitrary shape.
- **verify_inputs** (<em>bool</em>) – A boolean defaults to True. If True, `mu` is clipped internally
to be numerical stable.


Output
    
<em>tf.float32</em> – Tensor of same shape and dtype as `mu`.

Raises
    
**InvalidArgumentError** – If `mu` is negative.



### j_fun_inv_tf<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#j-fun-inv-tf" title="Permalink to this headline"></a>

`sionna.fec.utils.``j_fun_inv_tf`(<em class="sig-param">`mi`</em>, <em class="sig-param">`verify_inputs``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#j_fun_inv_tf">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.j_fun_inv_tf" title="Permalink to this definition"></a>
    
Calculates the inverse <cite>J-function</cite> in Tensorflow.
    
The so-called <cite>J-function</cite> relates mutual information to the mean of
Gaussian distributed LLRs (cf. Gaussian approximation). We use the
approximation as proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#brannstrom" id="id21">[Brannstrom]</a> which can be written as

$$
J(\mu) \approx \left( 1- 2^{H_\text{1}(2\mu)^{H_\text{2}}}\right)^{H_\text{2}}
$$
    
with $\mu$ denoting the mean value of the LLR distribution and
$H_\text{1}=0.3073$, $H_\text{2}=0.8935$ and
$H_\text{3}=1.1064$.
Input
 
- **mi** (<em>tf.float32</em>) – Tensor of arbitrary shape.
- **verify_inputs** (<em>bool</em>) – A boolean defaults to True. If True, `mi` is clipped internally
to be numerical stable.


Output
    
<em>tf.float32</em> – Tensor of same shape and dtype as the `mi`.

Raises
    
**InvalidArgumentError** – If `mi` is not in <cite>(0,1)</cite>.




References:
tenBrinkEXIT(<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id10">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id11">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id13">3</a>)
    
S. ten Brink, “Convergence Behavior of Iteratively
Decoded Parallel Concatenated Codes,” IEEE Transactions on
Communications, vol. 49, no. 10, pp. 1727-1737, 2001.

Brannstrom(<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id15">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id18">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id19">3</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id20">4</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id21">5</a>)
    
F. Brannstrom, L. K. Rasmussen, and A. J. Grant,
“Convergence analysis and optimal scheduling for multiple
concatenated codes,” IEEE Trans. Inform. Theory, vol. 51, no. 9,
pp. 3354–3364, 2005.

Hagenauer(<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id16">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id17">2</a>)
    
J. Hagenauer, “The Turbo Principle in Mobile
Communications,” in Proc. IEEE Int. Symp. Inf. Theory and its Appl.
(ISITA), 2002.

tenBrink(<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id9">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id12">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id14">3</a>)
    
S. ten Brink, G. Kramer, and A. Ashikhmin, “Design of
low-density parity-check codes for modulation and detection,” IEEE
Trans. Commun., vol. 52, no. 4, pp. 670–678, Apr. 2004.

MacKay(<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id3">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id5">3</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id6">4</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id7">5</a>)
    
<a class="reference external" href="http://www.inference.org.uk/mackay/codes/alist.html">http://www.inference.org.uk/mackay/codes/alist.html</a>

UniKL(<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id2">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id4">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.utils.html#id8">3</a>)
    
<a class="reference external" href="https://www.uni-kl.de/en/channel-codes/">https://www.uni-kl.de/en/channel-codes/</a>



