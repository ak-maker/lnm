# Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#utility-functions" title="Permalink to this headline"></a>
    
This module provides utility functions for the FEC package. It also provides serval functions to simplify EXIT analysis of iterative receivers.

# Table of Content
## (Binary) Linear Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#binary-linear-codes" title="Permalink to this headline"></a>
### pcm2gm<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#pcm2gm" title="Permalink to this headline"></a>
### verify_gm_pcm<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#verify-gm-pcm" title="Permalink to this headline"></a>
## EXIT Analysis<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#exit-analysis" title="Permalink to this headline"></a>
### plot_exit_chart<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#plot-exit-chart" title="Permalink to this headline"></a>
### get_exit_analytic<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#get-exit-analytic" title="Permalink to this headline"></a>
### plot_trajectory<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#plot-trajectory" title="Permalink to this headline"></a>
## Miscellaneous<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#miscellaneous" title="Permalink to this headline"></a>
### GaussianPriorSource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#gaussianpriorsource" title="Permalink to this headline"></a>
### bin2int<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#bin2int" title="Permalink to this headline"></a>
  
  

### pcm2gm<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#pcm2gm" title="Permalink to this headline"></a>

`sionna.fec.utils.``pcm2gm`(<em class="sig-param">`pcm`</em>, <em class="sig-param">`verify_results``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#pcm2gm">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.pcm2gm" title="Permalink to this definition"></a>
    
Generate the generator matrix for a given parity-check matrix.
    
This function brings `pcm` $\mathbf{H}$ in its systematic form and
uses the following relation to find the generator matrix
$\mathbf{G}$ in GF(2)

$$
\mathbf{G} = [\mathbf{I} |  \mathbf{M}]
\Leftrightarrow \mathbf{H} = [\mathbf{M} ^t | \mathbf{I}]. \tag{1}
$$
    
This follows from the fact that for an all-zero syndrome, it must hold that

$$
\mathbf{H} \mathbf{c}^t = \mathbf{H} * (\mathbf{u} * \mathbf{G})^t =
\mathbf{H} * \mathbf{G} ^t * \mathbf{u}^t =: \mathbf{0}
$$
    
where $\mathbf{c}$ denotes an arbitrary codeword and
$\mathbf{u}$ the corresponding information bits.
    
This leads to

$$
\mathbf{G} * \mathbf{H} ^t =: \mathbf{0}. \tag{2}
$$
    
It can be seen that (1) fulfills (2) as in GF(2) it holds that

$$
[\mathbf{I} |  \mathbf{M}] * [\mathbf{M} ^t | \mathbf{I}]^t
 = \mathbf{M} + \mathbf{M} = \mathbf{0}.
$$

Input
 
- **pcm** (<em>ndarray</em>) – Binary parity-check matrix of shape <cite>[n-k, n]</cite>.
- **verify_results** (<em>bool</em>) – Defaults to True. If True, it is verified that the generated
generator matrix is orthogonal to the parity-check matrix in GF(2).


Output
    
<em>ndarray</em> – Binary generator matrix of shape <cite>[k, n]</cite>.



**Note**
    
This algorithm only works if `pcm` has full rank. Otherwise an error is
raised.

### verify_gm_pcm<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#verify-gm-pcm" title="Permalink to this headline"></a>

`sionna.fec.utils.``verify_gm_pcm`(<em class="sig-param">`gm`</em>, <em class="sig-param">`pcm`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#verify_gm_pcm">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.verify_gm_pcm" title="Permalink to this definition"></a>
    
Verify that generator matrix $\mathbf{G}$ `gm` and parity-check
matrix $\mathbf{H}$ `pcm` are orthogonal in GF(2).
    
For an all-zero syndrome, it must hold that

$$
\mathbf{H} \mathbf{c}^t = \mathbf{H} * (\mathbf{u} * \mathbf{G})^t =
\mathbf{H} * \mathbf{G} ^t * \mathbf{u}^t =: \mathbf{0}
$$
    
where $\mathbf{c}$ denotes an arbitrary codeword and
$\mathbf{u}$ the corresponding information bits.
    
As $\mathbf{u}$ can be arbitrary it follows that

$$
\mathbf{H} * \mathbf{G} ^t =: \mathbf{0}.
$$

Input
 
- **gm** (<em>ndarray</em>) – Binary generator matrix of shape <cite>[k, n]</cite>.
- **pcm** (<em>ndarray</em>) – Binary parity-check matrix of shape <cite>[n-k, n]</cite>.


Output
    
<em>bool</em> – True if `gm` and `pcm` define a valid pair of parity-check and
generator matrices in GF(2).



## EXIT Analysis<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#exit-analysis" title="Permalink to this headline"></a>
    
The LDPC BP decoder allows to track the internal information flow (<cite>extrinsic information</cite>) during decoding. This can be plotted in so-called EXIT Charts <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#tenbrinkexit" id="id10">[tenBrinkEXIT]</a> to visualize the decoding convergence.
    
This short code snippet shows how to generate and plot EXIT charts:
```python
# parameters
ebno_db = 2.5 # simulation SNR
batch_size = 10000
num_bits_per_symbol = 2 # QPSK
pcm_id = 4 # decide which parity check matrix should be used (0-2: BCH; 3: (3,6)-LDPC 4: LDPC 802.11n
pcm, k, n , coderate = load_parity_check_examples(pcm_id, verbose=True)
noise_var = ebnodb2no(ebno_db=ebno_db,
                      num_bits_per_symbol=num_bits_per_symbol,
                      coderate=coderate)
# init components
decoder = LDPCBPDecoder(pcm,
                        hard_out=False,
                        cn_type="boxplus",
                        trainable=False,
                        track_exit=True, # if activated, the decoder stores the outgoing extrinsic mutual information per iteration
                        num_iter=20)
# generates fake llrs as if the all-zero codeword was transmitted over an AWNG channel with BPSK modulation
llr_source = GaussianPriorSource()

# generate fake LLRs (Gaussian approximation)
llr = llr_source([[batch_size, n], noise_var])
# simulate free running decoder (for EXIT trajectory)
decoder(llr)
# calculate analytical EXIT characteristics
# Hint: these curves assume asymptotic code length, i.e., may become inaccurate in the short length regime
Ia, Iev, Iec = get_exit_analytic(pcm, ebno_db)
# and plot the analytical exit curves
plt = plot_exit_chart(Ia, Iev, Iec)
# and add simulated trajectory (requires "track_exit=True")
plot_trajectory(plt, decoder.ie_v, decoder.ie_c, ebno_db)
```

    
Remark: for rate-matched 5G LDPC codes, the EXIT approximation becomes
inaccurate due to the rate-matching and the very specific structure of the
code.

### plot_exit_chart<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#plot-exit-chart" title="Permalink to this headline"></a>

`sionna.fec.utils.``plot_exit_chart`(<em class="sig-param">`mi_a``=``None`</em>, <em class="sig-param">`mi_ev``=``None`</em>, <em class="sig-param">`mi_ec``=``None`</em>, <em class="sig-param">`title``=``'EXIT-Chart'`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#plot_exit_chart">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.plot_exit_chart" title="Permalink to this definition"></a>
    
Utility function to plot EXIT-Charts <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#tenbrinkexit" id="id11">[tenBrinkEXIT]</a>.
    
If all inputs are <cite>None</cite> an empty EXIT chart is generated. Otherwise,
the mutual information curves are plotted.
Input
 
- **mi_a** (<em>float</em>) – An ndarray of floats containing the a priori mutual
information.
- **mi_v** (<em>float</em>) – An ndarray of floats containing the variable node mutual
information.
- **mi_c** (<em>float</em>) – An ndarray of floats containing the check node mutual
information.
- **title** (<em>str</em>) – A string defining the title of the EXIT chart.


Output
    
**plt** (<em>matplotlib.figure</em>) – A matplotlib figure handle

Raises
    
**AssertionError** – If `title` is not <cite>str</cite>.



### get_exit_analytic<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#get-exit-analytic" title="Permalink to this headline"></a>

`sionna.fec.utils.``get_exit_analytic`(<em class="sig-param">`pcm`</em>, <em class="sig-param">`ebno_db`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#get_exit_analytic">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.get_exit_analytic" title="Permalink to this definition"></a>
    
Calculate the analytic EXIT-curves for a given parity-check matrix.
    
This function extracts the degree profile from `pcm` and calculates the
variable (VN) and check node (CN) decoder EXIT curves. Please note that
this is an asymptotic tool which needs a certain codeword length for
accurate predictions.
    
Transmission over an AWGN channel with BPSK modulation and SNR `ebno_db`
is assumed. The detailed equations can be found in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#tenbrink" id="id12">[tenBrink]</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#tenbrinkexit" id="id13">[tenBrinkEXIT]</a>.
Input
 
- **pcm** (<em>ndarray</em>) – The parity-check matrix.
- **ebno_db** (<em>float</em>) – The channel SNR in dB.


Output
 
- **mi_a** (<em>ndarray of floats</em>) – NumPy array containing the <cite>a priori</cite> mutual information.
- **mi_ev** (<em>ndarray of floats</em>) – NumPy array containing the extrinsic mutual information of the
variable node decoder for the corresponding `mi_a`.
- **mi_ec** (<em>ndarray of floats</em>) – NumPy array containing the extrinsic mutual information of the check
node decoder for the corresponding `mi_a`.




**Note**
    
This function assumes random parity-check matrices without any imposed
structure. Thus, explicit code construction algorithms may lead
to inaccurate EXIT predictions. Further, this function is based
on asymptotic properties of the code, i.e., only works well for large
parity-check matrices. For details see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#tenbrink" id="id14">[tenBrink]</a>.

### plot_trajectory<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#plot-trajectory" title="Permalink to this headline"></a>

`sionna.fec.utils.``plot_trajectory`(<em class="sig-param">`plot`</em>, <em class="sig-param">`mi_v`</em>, <em class="sig-param">`mi_c`</em>, <em class="sig-param">`ebno``=``None`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#plot_trajectory">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.plot_trajectory" title="Permalink to this definition"></a>
    
Utility function to plot the trajectory of an EXIT-chart.
Input
 
- **plot** (<em>matplotlib.figure</em>) – A handle to a matplotlib figure.
- **mi_v** (<em>float</em>) – An ndarray of floats containing the variable node mutual
information.
- **mi_c** (<em>float</em>) – An ndarray of floats containing the check node mutual
information.
- **ebno** (<em>float</em>) – A float denoting the EbNo in dB for the legend entry.




## Miscellaneous<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#miscellaneous" title="Permalink to this headline"></a>

### GaussianPriorSource<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#gaussianpriorsource" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.utils.``GaussianPriorSource`(<em class="sig-param">`specified_by_mi``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#GaussianPriorSource">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.GaussianPriorSource" title="Permalink to this definition"></a>
    
Generates <cite>fake</cite> LLRs as if the all-zero codeword was transmitted
over an Bi-AWGN channel with noise variance `no` or mutual information
(if `specified_by_mi` is True). If selected, the mutual information
denotes the mutual information associated with a binary random variable
observed at the output of a corresponding AWGN channel (cf. Gaussian
approximation).
    
The generated LLRs are drawn from a Gaussian distribution with

$$
\sigma_{\text{llr}}^2 = \frac{4}{\sigma_\text{ch}^2}
$$
    
and

$$
\mu_{\text{llr}} = \frac{\sigma_\text{llr}^2}{2}
$$
    
where $\sigma_\text{ch}^2$ is the channel noise variance as defined by
`no`.
    
If `specified_by_mi` is True, this class uses the of the so-called
<cite>J-function</cite> (relates mutual information to Gaussian distributed LLRs) as
proposed in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#brannstrom" id="id15">[Brannstrom]</a>.
Parameters
 
- **specified_by_mi** (<em>bool</em>) – Defaults to False. If True, the second input parameter `no` is
interpreted as mutual information instead of noise variance.
- **dtype** (<em>tf.DType</em>) – Defaults to <cite>tf.float32</cite>. Defines the datatype for internal
calculations and the output. Must be one of the following
<cite>(tf.float16, tf.bfloat16, tf.float32, tf.float64)</cite>.


Input
 
- **(output_shape, no)** – Tuple:
- **output_shape** (<em>tf.int</em>) – Integer tensor or Python array defining the shape of the desired
output tensor.
- **no** (<em>tf.float32</em>) – Scalar defining the noise variance or mutual information (if
`specified_by_mi` is True) of the corresponding (fake) AWGN
channel.


Output
    
`dtype`, defaults to <cite>tf.float32</cite> – 1+D Tensor with shape as defined by `output_shape`.

Raises
 
- **InvalidArgumentError** – If mutual information is not in (0,1).
- **AssertionError** – If `inputs` is not a list with 2 elements.




### bin2int<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#bin2int" title="Permalink to this headline"></a>

`sionna.fec.utils.``bin2int`(<em class="sig-param">`arr`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#bin2int">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.bin2int" title="Permalink to this definition"></a>
    
Convert binary array to integer.
    
For example `arr` = <cite>[1, 0, 1]</cite> is converted to <cite>5</cite>.
Input
    
**arr** (<em>int or float</em>) – An iterable that yields 0’s and 1’s.

Output
    
<em>int</em> – Integer representation of `arr`.



