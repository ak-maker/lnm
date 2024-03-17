# Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#utility-functions" title="Permalink to this headline"></a>
    
The utilities sub-package of the Sionna library contains many convenience
functions as well as extensions to existing TensorFlow functions.

# Table of Content
## Miscellaneous<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#miscellaneous" title="Permalink to this headline"></a>
### hard_decisions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#hard-decisions" title="Permalink to this headline"></a>
### plot_ber<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#plot-ber" title="Permalink to this headline"></a>
### complex_normal<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#complex-normal" title="Permalink to this headline"></a>
### log2<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#log2" title="Permalink to this headline"></a>
### log10<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#log10" title="Permalink to this headline"></a>
  
  

### hard_decisions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#hard-decisions" title="Permalink to this headline"></a>

`sionna.utils.``hard_decisions`(<em class="sig-param">`llr`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#hard_decisions">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.hard_decisions" title="Permalink to this definition"></a>
    
Transforms LLRs into hard decisions.
    
Positive values are mapped to $1$.
Nonpositive values are mapped to $0$.
Input
    
**llr** (<em>any non-complex tf.DType</em>) – Tensor of LLRs.

Output
    
Same shape and dtype as `llr` – The hard decisions.



### plot_ber<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#plot-ber" title="Permalink to this headline"></a>

`sionna.utils.plotting.``plot_ber`(<em class="sig-param">`snr_db`</em>, <em class="sig-param">`ber`</em>, <em class="sig-param">`legend``=``''`</em>, <em class="sig-param">`ylabel``=``'BER'`</em>, <em class="sig-param">`title``=``'Bit` `Error` `Rate'`</em>, <em class="sig-param">`ebno``=``True`</em>, <em class="sig-param">`is_bler``=``None`</em>, <em class="sig-param">`xlim``=``None`</em>, <em class="sig-param">`ylim``=``None`</em>, <em class="sig-param">`save_fig``=``False`</em>, <em class="sig-param">`path``=``''`</em>)<a class="reference internal" href="../_modules/sionna/utils/plotting.html#plot_ber">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.plot_ber" title="Permalink to this definition"></a>
    
Plot error-rates.
Input
 
- **snr_db** (<em>ndarray</em>) – Array of floats defining the simulated SNR points.
Can be also a list of multiple arrays.
- **ber** (<em>ndarray</em>) – Array of floats defining the BER/BLER per SNR point.
Can be also a list of multiple arrays.
- **legend** (<em>str</em>) – Defaults to “”. Defining the legend entries. Can be
either a string or a list of strings.
- **ylabel** (<em>str</em>) – Defaults to “BER”. Defining the y-label.
- **title** (<em>str</em>) – Defaults to “Bit Error Rate”. Defining the title of the figure.
- **ebno** (<em>bool</em>) – Defaults to True. If True, the x-label is set to
“EbNo [dB]” instead of “EsNo [dB]”.
- **is_bler** (<em>bool</em>) – Defaults to False. If True, the corresponding curve is dashed.
- **xlim** (<em>tuple of floats</em>) – Defaults to None. A tuple of two floats defining x-axis limits.
- **ylim** (<em>tuple of floats</em>) – Defaults to None. A tuple of two floats defining y-axis limits.
- **save_fig** (<em>bool</em>) – Defaults to False. If True, the figure is saved as <cite>.png</cite>.
- **path** (<em>str</em>) – Defaults to “”. Defining the path to save the figure
(iff `save_fig` is True).


Output
 
- **(fig, ax)** – Tuple:
- **fig** (<em>matplotlib.figure.Figure</em>) – A matplotlib figure handle.
- **ax** (<em>matplotlib.axes.Axes</em>) – A matplotlib axes object.




### complex_normal<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#complex-normal" title="Permalink to this headline"></a>

`sionna.utils.``complex_normal`(<em class="sig-param">`shape`</em>, <em class="sig-param">`var``=``1.0`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#complex_normal">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.complex_normal" title="Permalink to this definition"></a>
    
Generates a tensor of complex normal random variables.
Input
 
- **shape** (<em>tf.shape, or list</em>) – The desired shape.
- **var** (<em>float</em>) – The total variance., i.e., each complex dimension has
variance `var/2`.
- **dtype** (<em>tf.complex</em>) – The desired dtype. Defaults to <cite>tf.complex64</cite>.


Output
    
`shape`, `dtype` – Tensor of complex normal random variables.



### log2<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#log2" title="Permalink to this headline"></a>

`sionna.utils.``log2`(<em class="sig-param">`x`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#log2">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.log2" title="Permalink to this definition"></a>
    
TensorFlow implementation of NumPy’s <cite>log2</cite> function.
    
Simple extension to <cite>tf.experimental.numpy.log2</cite>
which casts the result to the <cite>dtype</cite> of the input.
For more details see the <a class="reference external" href="https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log2">TensorFlow</a> and <a class="reference external" href="https://numpy.org/doc/1.16/reference/generated/numpy.log2.html">NumPy</a> documentation.

### log10<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#log10" title="Permalink to this headline"></a>

`sionna.utils.``log10`(<em class="sig-param">`x`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#log10">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.log10" title="Permalink to this definition"></a>
    
TensorFlow implementation of NumPy’s <cite>log10</cite> function.
    
Simple extension to <cite>tf.experimental.numpy.log10</cite>
which casts the result to the <cite>dtype</cite> of the input.
For more details see the <a class="reference external" href="https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log10">TensorFlow</a> and <a class="reference external" href="https://numpy.org/doc/1.16/reference/generated/numpy.log10.html">NumPy</a> documentation.
