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
### HammingWindow<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#hammingwindow" title="Permalink to this headline"></a>
### BlackmanWindow<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#blackmanwindow" title="Permalink to this headline"></a>
### CustomWindow<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#customwindow" title="Permalink to this headline"></a>
  
  

### HammingWindow<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#hammingwindow" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``HammingWindow`(<em class="sig-param">`length`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`normalize``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/window.html#HammingWindow">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HammingWindow" title="Permalink to this definition"></a>
    
Layer for applying a Hamming window function of length `length` to an input `x` of the same length.
    
The window function is applied through element-wise multiplication.
    
The window function is real-valued, i.e., has <cite>tf.float</cite> as <cite>dtype</cite>.
The <cite>dtype</cite> of the output is the same as the <cite>dtype</cite> of the input `x` to which the window function is applied.
The window function and the input must have the same precision.
    
The Hamming window is defined by

$$
w_n = a_0 - (1-a_0) \cos \left( \frac{2 \pi n}{N} \right), 0 \leq n \leq N-1
$$
    
where $N$ is the window length and $a_0 = \frac{25}{46}$.
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



<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HammingWindow.coefficients" title="Permalink to this definition"></a>
    
The window coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HammingWindow.length" title="Permalink to this definition"></a>
    
Window length in number of samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HammingWindow.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window is normalized to have unit average power per coefficient. <cite>False</cite>
otherwise.


`show`(<em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`domain``=``'time'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HammingWindow.show" title="Permalink to this definition"></a>
    
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





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.HammingWindow.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window coefficients are trainable. <cite>False</cite> otherwise.


### BlackmanWindow<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#blackmanwindow" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``BlackmanWindow`(<em class="sig-param">`length`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`normalize``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/window.html#BlackmanWindow">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.BlackmanWindow" title="Permalink to this definition"></a>
    
Layer for applying a Blackman window function of length `length` to an input `x` of the same length.
    
The window function is applied through element-wise multiplication.
    
The window function is real-valued, i.e., has <cite>tf.float</cite> as <cite>dtype</cite>.
The <cite>dtype</cite> of the output is the same as the <cite>dtype</cite> of the input `x` to which the window function is applied.
The window function and the input must have the same precision.
    
The Blackman window is defined by

$$
w_n = a_0 - a_1 \cos \left( \frac{2 \pi n}{N} \right) + a_2 \cos \left( \frac{4 \pi n}{N} \right), 0 \leq n \leq N-1
$$
    
where $N$ is the window length, $a_0 = \frac{7938}{18608}$, $a_1 = \frac{9240}{18608}$, and $a_2 = \frac{1430}{18608}$.
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



<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.BlackmanWindow.coefficients" title="Permalink to this definition"></a>
    
The window coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.BlackmanWindow.length" title="Permalink to this definition"></a>
    
Window length in number of samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.BlackmanWindow.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window is normalized to have unit average power per coefficient. <cite>False</cite>
otherwise.


`show`(<em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`domain``=``'time'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.BlackmanWindow.show" title="Permalink to this definition"></a>
    
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





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.BlackmanWindow.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window coefficients are trainable. <cite>False</cite> otherwise.


### CustomWindow<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#customwindow" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``CustomWindow`(<em class="sig-param">`length`</em>, <em class="sig-param">`coefficients``=``None`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`normalize``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/window.html#CustomWindow">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomWindow" title="Permalink to this definition"></a>
    
Layer for defining and applying a custom window function of length `length` to an input `x` of the same length.
    
The window function is applied through element-wise multiplication.
    
The window function is real-valued, i.e., has <cite>tf.float</cite> as <cite>dtype</cite>.
The <cite>dtype</cite> of the output is the same as the <cite>dtype</cite> of the input `x` to which the window function is applied.
The window function and the input must have the same precision.
    
The window coefficients can be set through the `coefficients` parameter.
If not provided, random window coefficients are generated by sampling a Gaussian distribution.
Parameters
 
- **length** (<em>int</em>) – Window length (number of samples).
- **coefficients** (<em>[</em><em>N</em><em>]</em><em>, </em><em>tf.float</em>) – Optional window coefficients.
If set to <cite>None</cite>, then a random window of length `length` is generated by sampling a Gaussian distribution.
Defaults to <cite>None</cite>.
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



<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomWindow.coefficients" title="Permalink to this definition"></a>
    
The window coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomWindow.length" title="Permalink to this definition"></a>
    
Window length in number of samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomWindow.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window is normalized to have unit average power per coefficient. <cite>False</cite>
otherwise.


`show`(<em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`domain``=``'time'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomWindow.show" title="Permalink to this definition"></a>
    
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





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomWindow.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the window coefficients are trainable. <cite>False</cite> otherwise.


