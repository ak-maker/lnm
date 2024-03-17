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
## Filters<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#filters" title="Permalink to this headline"></a>
### SincFilter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sincfilter" title="Permalink to this headline"></a>
### RaisedCosineFilter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#raisedcosinefilter" title="Permalink to this headline"></a>
  
  

## Filters<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#filters" title="Permalink to this headline"></a>

### SincFilter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sincfilter" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``SincFilter`(<em class="sig-param">`span_in_symbols`</em>, <em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`window``=``None`</em>, <em class="sig-param">`normalize``=``True`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/filter.html#SincFilter">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter" title="Permalink to this definition"></a>
    
Layer for applying a sinc filter of `length` K
to an input `x` of length N.
    
The sinc filter is defined by

$$
h(t) = \frac{1}{T}\text{sinc}\left(\frac{t}{T}\right)
$$
    
where $T$ the symbol duration.
    
The filter length K is equal to the filter span in symbols (`span_in_symbols`)
multiplied by the oversampling factor (`samples_per_symbol`).
If this product is even, a value of one will be added.
    
The filter is applied through discrete convolution.
    
An optional windowing function `window` can be applied to the filter.
    
The <cite>dtype</cite> of the output is <cite>tf.float</cite> if both `x` and the filter coefficients have dtype <cite>tf.float</cite>.
Otherwise, the dtype of the output is <cite>tf.complex</cite>.
    
Three padding modes are available for applying the filter:
 
- “full” (default): Returns the convolution at each point of overlap between `x` and the filter.
The length of the output is N + K - 1. Zero-padding of the input `x` is performed to
compute the convolution at the borders.
- “same”: Returns an output of the same length as the input `x`. The convolution is computed such
that the coefficients of the input `x` are centered on the coefficient of the filter with index
(K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
- “valid”: Returns the convolution only at points where `x` and the filter completely overlap.
The length of the output is N - K + 1.

Parameters
 
- **span_in_symbols** (<em>int</em>) – Filter span as measured by the number of symbols.
- **samples_per_symbol** (<em>int</em>) – Number of samples per symbol, i.e., the oversampling factor.
- **window** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="sionna.signal.Window"><em>Window</em></a><em> or </em><em>string</em><em> (</em><em>[</em><em>"hann"</em><em>, </em><em>"hamming"</em><em>, </em><em>"blackman"</em><em>]</em><em>)</em>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="sionna.signal.Window">`Window`</a> that is applied to the filter coefficients.
Alternatively, a string indicating the window name can be provided. In this case,
the chosen window will be instantiated with the default parameters. Custom windows
must be provided as instance.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the filter is normalized to have unit power.
Defaults to <cite>True</cite>.
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the filter coefficients are trainable variables.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – The <cite>dtype</cite> of the filter coefficients.
Defaults to <cite>tf.float32</cite>.


Input
 
- **x** (<em>[…, N], tf.complex or tf.float</em>) – The input to which the filter is applied.
The filter is applied along the last dimension.
- **padding** (<em>string ([“full”, “valid”, “same”])</em>) – Padding mode for convolving `x` and the filter.
Must be one of “full”, “valid”, or “same”. Case insensitive.
Defaults to “full”.
- **conjugate** (<em>bool</em>) – If <cite>True</cite>, the complex conjugate of the filter is applied.
Defaults to <cite>False</cite>.


Output
    
**y** (<em>[…,M], tf.complex or tf.float</em>) – Filtered input.
It is <cite>tf.float</cite> only if both `x` and the filter are <cite>tf.float</cite>.
It is <cite>tf.complex</cite> otherwise.
The length M depends on the `padding`.



<em class="property">`property` </em>`aclr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.aclr" title="Permalink to this definition"></a>
    
ACLR of the filter
    
This ACLR corresponds to what one would obtain from using
this filter as pulse shaping filter on an i.i.d. sequence of symbols.
The in-band is assumed to range from [-0.5, 0.5] in normalized
frequency.


<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.coefficients" title="Permalink to this definition"></a>
    
The filter coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.length" title="Permalink to this definition"></a>
    
The filter length in samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter is normalized to have unit power. <cite>False</cite> otherwise.


<em class="property">`property` </em>`sampling_times`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.sampling_times" title="Permalink to this definition"></a>
    
Sampling times in multiples of the symbol duration


`show`(<em class="sig-param">`response``=``'impulse'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.show" title="Permalink to this definition"></a>
    
Plot the impulse or magnitude response
    
Plots the impulse response (time domain) or magnitude response
(frequency domain) of the filter.
    
For the computation of the magnitude response, a minimum DFT size
of 1024 is assumed which is obtained through zero padding of
the filter coefficients in the time domain.
Input
 
- **response** (<em>str, one of [“impulse”, “magnitude”]</em>) – The desired response type.
Defaults to “impulse”
- **scale** (<em>str, one of [“lin”, “db”]</em>) – The y-scale of the magnitude response.
Can be “lin” (i.e., linear) or “db” (, i.e., Decibel).
Defaults to “lin”.





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter coefficients are trainable. <cite>False</cite> otherwise.


<em class="property">`property` </em>`window`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.SincFilter.window" title="Permalink to this definition"></a>
    
The window function that is applied to the filter coefficients. <cite>None</cite> if no window is applied.


### RaisedCosineFilter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#raisedcosinefilter" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``RaisedCosineFilter`(<em class="sig-param">`span_in_symbols`</em>, <em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`beta`</em>, <em class="sig-param">`window``=``None`</em>, <em class="sig-param">`normalize``=``True`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/filter.html#RaisedCosineFilter">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter" title="Permalink to this definition"></a>
    
Layer for applying a raised-cosine filter of `length` K
to an input `x` of length N.
    
The raised-cosine filter is defined by

$$
\begin{split}h(t) =
\begin{cases}
\frac{\pi}{4T} \text{sinc}\left(\frac{1}{2\beta}\right), & \text { if }t = \pm \frac{T}{2\beta}\\
\frac{1}{T}\text{sinc}\left(\frac{t}{T}\right)\frac{\cos\left(\frac{\pi\beta t}{T}\right)}{1-\left(\frac{2\beta t}{T}\right)^2}, & \text{otherwise}
\end{cases}\end{split}
$$
    
where $\beta$ is the roll-off factor and $T$ the symbol duration.
    
The filter length K is equal to the filter span in symbols (`span_in_symbols`)
multiplied by the oversampling factor (`samples_per_symbol`).
If this product is even, a value of one will be added.
    
The filter is applied through discrete convolution.
    
An optional windowing function `window` can be applied to the filter.
    
The <cite>dtype</cite> of the output is <cite>tf.float</cite> if both `x` and the filter coefficients have dtype <cite>tf.float</cite>.
Otherwise, the dtype of the output is <cite>tf.complex</cite>.
    
Three padding modes are available for applying the filter:
 
- “full” (default): Returns the convolution at each point of overlap between `x` and the filter.
The length of the output is N + K - 1. Zero-padding of the input `x` is performed to
compute the convolution at the borders.
- “same”: Returns an output of the same length as the input `x`. The convolution is computed such
that the coefficients of the input `x` are centered on the coefficient of the filter with index
(K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
- “valid”: Returns the convolution only at points where `x` and the filter completely overlap.
The length of the output is N - K + 1.

Parameters
 
- **span_in_symbols** (<em>int</em>) – Filter span as measured by the number of symbols.
- **samples_per_symbol** (<em>int</em>) – Number of samples per symbol, i.e., the oversampling factor.
- **beta** (<em>float</em>) – Roll-off factor.
Must be in the range $[0,1]$.
- **window** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="sionna.signal.Window"><em>Window</em></a><em> or </em><em>string</em><em> (</em><em>[</em><em>"hann"</em><em>, </em><em>"hamming"</em><em>, </em><em>"blackman"</em><em>]</em><em>)</em>) – Instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.Window" title="sionna.signal.Window">`Window`</a> that is applied to the filter coefficients.
Alternatively, a string indicating the window name can be provided. In this case,
the chosen window will be instantiated with the default parameters. Custom windows
must be provided as instance.
- **normalize** (<em>bool</em>) – If <cite>True</cite>, the filter is normalized to have unit power.
Defaults to <cite>True</cite>.
- **trainable** (<em>bool</em>) – If <cite>True</cite>, the filter coefficients are trainable variables.
Defaults to <cite>False</cite>.
- **dtype** (<em>tf.DType</em>) – The <cite>dtype</cite> of the filter coefficients.
Defaults to <cite>tf.float32</cite>.


Input
 
- **x** (<em>[…, N], tf.complex or tf.float</em>) – The input to which the filter is applied.
The filter is applied along the last dimension.
- **padding** (<em>string ([“full”, “valid”, “same”])</em>) – Padding mode for convolving `x` and the filter.
Must be one of “full”, “valid”, or “same”.
Defaults to “full”.
- **conjugate** (<em>bool</em>) – If <cite>True</cite>, the complex conjugate of the filter is applied.
Defaults to <cite>False</cite>.


Output
    
**y** (<em>[…,M], tf.complex or tf.float</em>) – Filtered input.
It is <cite>tf.float</cite> only if both `x` and the filter are <cite>tf.float</cite>.
It is <cite>tf.complex</cite> otherwise.
The length M depends on the `padding`.



<em class="property">`property` </em>`aclr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.aclr" title="Permalink to this definition"></a>
    
ACLR of the filter
    
This ACLR corresponds to what one would obtain from using
this filter as pulse shaping filter on an i.i.d. sequence of symbols.
The in-band is assumed to range from [-0.5, 0.5] in normalized
frequency.


<em class="property">`property` </em>`beta`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.beta" title="Permalink to this definition"></a>
    
Roll-off factor


<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.coefficients" title="Permalink to this definition"></a>
    
The filter coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.length" title="Permalink to this definition"></a>
    
The filter length in samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter is normalized to have unit power. <cite>False</cite> otherwise.


<em class="property">`property` </em>`sampling_times`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.sampling_times" title="Permalink to this definition"></a>
    
Sampling times in multiples of the symbol duration


`show`(<em class="sig-param">`response``=``'impulse'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.show" title="Permalink to this definition"></a>
    
Plot the impulse or magnitude response
    
Plots the impulse response (time domain) or magnitude response
(frequency domain) of the filter.
    
For the computation of the magnitude response, a minimum DFT size
of 1024 is assumed which is obtained through zero padding of
the filter coefficients in the time domain.
Input
 
- **response** (<em>str, one of [“impulse”, “magnitude”]</em>) – The desired response type.
Defaults to “impulse”
- **scale** (<em>str, one of [“lin”, “db”]</em>) – The y-scale of the magnitude response.
Can be “lin” (i.e., linear) or “db” (, i.e., Decibel).
Defaults to “lin”.





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter coefficients are trainable. <cite>False</cite> otherwise.


<em class="property">`property` </em>`window`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RaisedCosineFilter.window" title="Permalink to this definition"></a>
    
The window function that is applied to the filter coefficients. <cite>None</cite> if no window is applied.


