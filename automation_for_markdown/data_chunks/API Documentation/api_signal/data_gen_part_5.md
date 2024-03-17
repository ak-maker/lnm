# Signal<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#signal" title="Permalink to this headline"></a>
    
This module contains classes and functions for <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#filter">filtering</a> (pulse shaping), <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#window">windowing</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#upsampling">up-</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#downsampling">downsampling</a>.
The following figure shows the different components that can be implemented using this module.
<a class="reference internal image-reference" href="../_images/signal_module.png"><img alt="../_images/signal_module.png" src="https://nvlabs.github.io/sionna/_images/signal_module.png" style="width: 75%;" /></a>
    
This module also contains <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#utility">utility functions</a> for computing the (inverse) discrete Fourier transform (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#fft">FFT</a>/<a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#ifft">IFFT</a>), and for empirically computing the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#empirical-psd">power spectral density (PSD)</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#empirical-aclr">adjacent channel leakage ratio (ACLR)</a> of a signal.
    
The following code snippet shows how to filter a sequence of QAM baseband symbols using a root-raised-cosine filter with a Hann window:
```python
# Create batch of QAM-16 sequences
batch_size = 128
num_symbols = 1000
num_bits_per_symbol = 4
x = QAMSource(num_bits_per_symbol)([batch_size, num_symbols])
# Create a root-raised-cosine filter with Hann windowing
beta = 0.22 # Roll-off factor
span_in_symbols = 32 # Filter span in symbols
samples_per_symbol = 4 # Number of samples per symbol, i.e., the oversampling factor
rrcf_hann = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta, window="hann")
# Create instance of the Upsampling layer
us = Upsampling(samples_per_symbol)
# Upsample the baseband x
x_us = us(x)
# Filter the upsampled sequence
x_rrcf = rrcf_hann(x_us)
```

    
On the receiver side, one would recover the baseband symbols as follows:
```python
# Instantiate a downsampling layer
ds = Downsampling(samples_per_symbol, rrcf_hann.length-1, num_symbols)
# Apply the matched filter
x_mf = rrcf_hann(x_rrcf)
# Recover the transmitted symbol sequence
x_hat = ds(x_mf)
```
# Table of Content
## Window functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#window-functions" title="Permalink to this headline"></a>
### Window<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#id2" title="Permalink to this headline"></a>
## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#utility-functions" title="Permalink to this headline"></a>
### convolve<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#convolve" title="Permalink to this headline"></a>
### fft<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#fft" title="Permalink to this headline"></a>
### ifft<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#ifft" title="Permalink to this headline"></a>
### Upsampling<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#upsampling" title="Permalink to this headline"></a>
### Downsampling<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#downsampling" title="Permalink to this headline"></a>
### empirical_psd<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#empirical-psd" title="Permalink to this headline"></a>
### empirical_aclr<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#empirical-aclr" title="Permalink to this headline"></a>
  
  

### Window<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#id2" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``Window`(<em class="sig-param">`length`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`normalize``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/window.html#Window">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="Permalink to this definition"></a>
    
This is an abtract class for defining and applying a window function of length `length` to an input `x` of the same length.
    
The window function is applied through element-wise multiplication.
    
The window function is real-valued, i.e., has <cite>tf.float</cite> as <cite>dtype</cite>.
The <cite>dtype</cite> of the output is the same as the <cite>dtype</cite> of the input `x` to which the window function is applied.
The window function and the input must have the same precision.
Parameters
 
- **length** (<em>int</em>) – Window length (number of samples).
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the window coefficients are trainable variables.
Defaults to <cite>False</cite>.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the window is normalized to have unit average power
per coefficient.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – The <cite>dtype</cite> of the filter coefficients.
Must be either <cite>tf.float32</cite> or <cite>tf.float64</cite>.
Defaults to <cite>tf.float32</cite>.


Input
    
**x** (<em>[…, N], tf.complex or tf.float</em>) – The input to which the window function is applied.
The window function is applied along the last dimension.
The length of the last dimension `N` must be the same as the `length` of the window function.

Output
    
**y** (<em>[…,N], tf.complex or tf.float</em>) – Output of the windowing operation.
The output has the same shape and <cite>dtype</cite> as the input `x`.



<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window.coefficients" title="Permalink to this definition"></a>
    
The window coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window.length" title="Permalink to this definition"></a>
    
Window length in number of samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window is normalized to have unit average power per coefficient. <cite>False</cite>
otherwise.


`show`(<em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`domain``=``'time'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="reference internal" href="../_modules/sionna/signal/window.html#Window.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window.show" title="Permalink to this definition"></a>
    
Plot the window in time or frequency domain
    
For the computation of the Fourier transform, a minimum DFT size
of 1024 is assumed which is obtained through zero padding of
the window coefficients in the time domain.
Input
 
- **samples_per_symbol** (<em>int</em>) – Number of samples per symbol, i.e., the oversampling factor.
- **domain** (<em>str, one of [“time”, “frequency”]</em>) – The desired domain.
Defaults to “time”
- **scale** (<em>str, one of [“lin”, “db”]</em>) – The y-scale of the magnitude in the frequency domain.
Can be “lin” (i.e., linear) or “db” (, i.e., Decibel).
Defaults to “lin”.





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window coefficients are trainable. <cite>False</cite> otherwise.


## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#utility-functions" title="Permalink to this headline"></a>

### convolve<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#convolve" title="Permalink to this headline"></a>

`sionna.signal.``convolve`(<em class="sig-param">`inp`</em>, <em class="sig-param">`ker`</em>, <em class="sig-param">`padding``=``'full'`</em>, <em class="sig-param">`axis``=``-` `1`</em>)<a class="reference internal" href="../_modules/sionna/signal/utils.html#convolve">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.convolve" title="Permalink to this definition"></a>
    
Filters an input `inp` of length <cite>N</cite> by convolving it with a kernel `ker` of length <cite>K</cite>.
    
The length of the kernel `ker` must not be greater than the one of the input sequence `inp`.
    
The <cite>dtype</cite> of the output is <cite>tf.float</cite> only if both `inp` and `ker` are <cite>tf.float</cite>. It is <cite>tf.complex</cite> otherwise.
`inp` and `ker` must have the same precision.
    
Three padding modes are available:
 
- “full” (default): Returns the convolution at each point of overlap between `ker` and `inp`.
The length of the output is <cite>N + K - 1</cite>. Zero-padding of the input `inp` is performed to
compute the convolution at the border points.
- “same”: Returns an output of the same length as the input `inp`. The convolution is computed such
that the coefficients of the input `inp` are centered on the coefficient of the kernel `ker` with index
`(K-1)/2` for kernels of odd length, and `K/2` `-` `1` for kernels of even length.
Zero-padding of the input signal is performed to compute the convolution at the border points.
- “valid”: Returns the convolution only at points where `inp` and `ker` completely overlap.
The length of the output is <cite>N - K + 1</cite>.

Input
 
- **inp** (<em>[…,N], tf.complex or tf.real</em>) – Input to filter.
- **ker** (<em>[K], tf.complex or tf.real</em>) – Kernel of the convolution.
- **padding** (<em>string</em>) – Padding mode. Must be one of “full”, “valid”, or “same”. Case insensitive.
Defaults to “full”.
- **axis** (<em>int</em>) – Axis along which to perform the convolution.
Defaults to <cite>-1</cite>.


Output
    
**out** (<em>[…,M], tf.complex or tf.float</em>) – Convolution output.
It is <cite>tf.float</cite> only if both `inp` and `ker` are <cite>tf.float</cite>. It is <cite>tf.complex</cite> otherwise.
The length <cite>M</cite> of the output depends on the `padding`.



### fft<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#fft" title="Permalink to this headline"></a>

`sionna.signal.``fft`(<em class="sig-param">`tensor`</em>, <em class="sig-param">`axis``=``-` `1`</em>)<a class="reference internal" href="../_modules/sionna/signal/utils.html#fft">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.fft" title="Permalink to this definition"></a>
    
Computes the normalized DFT along a specified axis.
    
This operation computes the normalized one-dimensional discrete Fourier
transform (DFT) along the `axis` dimension of a `tensor`.
For a vector $\mathbf{x}\in\mathbb{C}^N$, the DFT
$\mathbf{X}\in\mathbb{C}^N$ is computed as

$$
X_m = \frac{1}{\sqrt{N}}\sum_{n=0}^{N-1} x_n \exp \left\{
    -j2\pi\frac{mn}{N}\right\},\quad m=0,\dots,N-1.
$$

Input
 
- **tensor** (<em>tf.complex</em>) – Tensor of arbitrary shape.
- **axis** (<em>int</em>) – Indicates the dimension along which the DFT is taken.


Output
    
<em>tf.complex</em> – Tensor of the same dtype and shape as `tensor`.



### ifft<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#ifft" title="Permalink to this headline"></a>

`sionna.signal.``ifft`(<em class="sig-param">`tensor`</em>, <em class="sig-param">`axis``=``-` `1`</em>)<a class="reference internal" href="../_modules/sionna/signal/utils.html#ifft">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.ifft" title="Permalink to this definition"></a>
    
Computes the normalized IDFT along a specified axis.
    
This operation computes the normalized one-dimensional discrete inverse
Fourier transform (IDFT) along the `axis` dimension of a `tensor`.
For a vector $\mathbf{X}\in\mathbb{C}^N$, the IDFT
$\mathbf{x}\in\mathbb{C}^N$ is computed as

$$
x_n = \frac{1}{\sqrt{N}}\sum_{m=0}^{N-1} X_m \exp \left\{
    j2\pi\frac{mn}{N}\right\},\quad n=0,\dots,N-1.
$$

Input
 
- **tensor** (<em>tf.complex</em>) – Tensor of arbitrary shape.
- **axis** (<em>int</em>) – Indicates the dimension along which the IDFT is taken.


Output
    
<em>tf.complex</em> – Tensor of the same dtype and shape as `tensor`.



### Upsampling<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#upsampling" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``Upsampling`(<em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`axis``=``-` `1`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/upsampling.html#Upsampling">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Upsampling" title="Permalink to this definition"></a>
    
Upsamples a tensor along a specified axis by inserting zeros
between samples.
Parameters
 
- **samples_per_symbol** (<em>int</em>) – The upsampling factor. If `samples_per_symbol` is equal to <cite>n</cite>,
then the upsampled axis will be <cite>n</cite>-times longer.
- **axis** (<em>int</em>) – The dimension to be up-sampled. Must not be the first dimension.


Input
    
**x** (<em>[…,n,…], tf.DType</em>) – The tensor to be upsampled. <cite>n</cite> is the size of the <cite>axis</cite> dimension.

Output
    
**y** ([…,n*samples_per_symbol,…], same dtype as `x`) – The upsampled tensor.



### Downsampling<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#downsampling" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``Downsampling`(<em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`offset``=``0`</em>, <em class="sig-param">`num_symbols``=``None`</em>, <em class="sig-param">`axis``=``-` `1`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/downsampling.html#Downsampling">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Downsampling" title="Permalink to this definition"></a>
    
Downsamples a tensor along a specified axis by retaining one out of
`samples_per_symbol` elements.
Parameters
 
- **samples_per_symbol** (<em>int</em>) – The downsampling factor. If `samples_per_symbol` is equal to <cite>n</cite>, then the
downsampled axis will be <cite>n</cite>-times shorter.
- **offset** (<em>int</em>) – Defines the index of the first element to be retained.
Defaults to zero.
- **num_symbols** (<em>int</em>) – Defines the total number of symbols to be retained after
downsampling.
Defaults to None (i.e., the maximum possible number).
- **axis** (<em>int</em>) – The dimension to be downsampled. Must not be the first dimension.


Input
    
**x** (<em>[…,n,…], tf.DType</em>) – The tensor to be downsampled. <cite>n</cite> is the size of the <cite>axis</cite> dimension.

Output
    
**y** ([…,k,…], same dtype as `x`) – The downsampled tensor, where `k`
is min((`n`-`offset`)//`samples_per_symbol`, `num_symbols`).



### empirical_psd<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#empirical-psd" title="Permalink to this headline"></a>

`sionna.signal.``empirical_psd`(<em class="sig-param">`x`</em>, <em class="sig-param">`show``=``True`</em>, <em class="sig-param">`oversampling``=``1.0`</em>, <em class="sig-param">`ylim``=``(-` `30,` `3)`</em>)<a class="reference internal" href="../_modules/sionna/signal/utils.html#empirical_psd">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.empirical_psd" title="Permalink to this definition"></a>
    
Computes the empirical power spectral density.
    
Computes the empirical power spectral density (PSD) of tensor `x`
along the last dimension by averaging over all other dimensions.
Note that this function
simply returns the averaged absolute squared discrete Fourier
spectrum of `x`.
Input
 
- **x** (<em>[…,N], tf.complex</em>) – The signal of which to compute the PSD.
- **show** (<em>bool</em>) – Indicates if a plot of the PSD should be generated.
Defaults to True,
- **oversampling** (<em>float</em>) – The oversampling factor. Defaults to 1.
- **ylim** (<em>tuple of floats</em>) – The limits of the y axis. Defaults to [-30, 3].
Only relevant if `show` is True.


Output
 
- **freqs** (<em>[N], float</em>) – The normalized frequencies at which the PSD was evaluated.
- **psd** (<em>[N], float</em>) – The PSD.




### empirical_aclr<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#empirical-aclr" title="Permalink to this headline"></a>

`sionna.signal.``empirical_aclr`(<em class="sig-param">`x`</em>, <em class="sig-param">`oversampling``=``1.0`</em>, <em class="sig-param">`f_min``=``-` `0.5`</em>, <em class="sig-param">`f_max``=``0.5`</em>)<a class="reference internal" href="../_modules/sionna/signal/utils.html#empirical_aclr">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.empirical_aclr" title="Permalink to this definition"></a>
    
Computes the empirical ACLR.
    
Computes the empirical adjacent channel leakgae ration (ACLR)
of tensor `x` based on its empirical power spectral density (PSD)
which is computed along the last dimension by averaging over
all other dimensions.
    
It is assumed that the in-band ranges from [`f_min`, `f_max`] in
normalized frequency. The ACLR is then defined as

$$
\text{ACLR} = \frac{P_\text{out}}{P_\text{in}}
$$
    
where $P_\text{in}$ and $P_\text{out}$ are the in-band
and out-of-band power, respectively.
Input
 
- **x** (<em>[…,N],  complex</em>) – The signal for which to compute the ACLR.
- **oversampling** (<em>float</em>) – The oversampling factor. Defaults to 1.
- **f_min** (<em>float</em>) – The lower border of the in-band in normalized frequency.
Defaults to -0.5.
- **f_max** (<em>float</em>) – The upper border of the in-band in normalized frequency.
Defaults to 0.5.


Output
    
**aclr** (<em>float</em>) – The ACLR in linear scale.


