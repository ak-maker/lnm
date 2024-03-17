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
### RootRaisedCosineFilter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#rootraisedcosinefilter" title="Permalink to this headline"></a>
### CustomFilter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#customfilter" title="Permalink to this headline"></a>
  
  

### RootRaisedCosineFilter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#rootraisedcosinefilter" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``RootRaisedCosineFilter`(<em class="sig-param">`span_in_symbols`</em>, <em class="sig-param">`samples_per_symbol`</em>, <em class="sig-param">`beta`</em>, <em class="sig-param">`window``=``None`</em>, <em class="sig-param">`normalize``=``True`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/filter.html#RootRaisedCosineFilter">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter" title="Permalink to this definition"></a>
    
Layer for applying a root-raised-cosine filter of `length` K
to an input `x` of length N.
    
The root-raised-cosine filter is defined by

$$
\begin{split}h(t) =
\begin{cases}
\frac{1}{T} \left(1 + \beta\left(\frac{4}{\pi}-1\right) \right), & \text { if }t = 0\\
\frac{\beta}{T\sqrt{2}} \left[ \left(1+\frac{2}{\pi}\right)\sin\left(\frac{\pi}{4\beta}\right) + \left(1-\frac{2}{\pi}\right)\cos\left(\frac{\pi}{4\beta}\right) \right], & \text { if }t = \pm\frac{T}{4\beta} \\
\frac{1}{T} \frac{\sin\left(\pi\frac{t}{T}(1-\beta)\right) + 4\beta\frac{t}{T}\cos\left(\pi\frac{t}{T}(1+\beta)\right)}{\pi\frac{t}{T}\left(1-\left(4\beta\frac{t}{T}\right)^2\right)}, & \text { otherwise}
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
Must be one of “full”, “valid”, or “same”. Case insensitive.
Defaults to “full”.
- **conjugate** (<em>bool</em>) – If <cite>True</cite>, the complex conjugate of the filter is applied.
Defaults to <cite>False</cite>.


Output
    
**y** (<em>[…,M], tf.complex or tf.float</em>) – Filtered input.
It is <cite>tf.float</cite> only if both `x` and the filter are <cite>tf.float</cite>.
It is <cite>tf.complex</cite> otherwise.
The length M depends on the `padding`.



<em class="property">`property` </em>`aclr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.aclr" title="Permalink to this definition"></a>
    
ACLR of the filter
    
This ACLR corresponds to what one would obtain from using
this filter as pulse shaping filter on an i.i.d. sequence of symbols.
The in-band is assumed to range from [-0.5, 0.5] in normalized
frequency.


<em class="property">`property` </em>`beta`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.beta" title="Permalink to this definition"></a>
    
Roll-off factor


<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.coefficients" title="Permalink to this definition"></a>
    
The filter coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.length" title="Permalink to this definition"></a>
    
The filter length in samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter is normalized to have unit power. <cite>False</cite> otherwise.


<em class="property">`property` </em>`sampling_times`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.sampling_times" title="Permalink to this definition"></a>
    
Sampling times in multiples of the symbol duration


`show`(<em class="sig-param">`response``=``'impulse'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.show" title="Permalink to this definition"></a>
    
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





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter coefficients are trainable. <cite>False</cite> otherwise.


<em class="property">`property` </em>`window`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.RootRaisedCosineFilter.window" title="Permalink to this definition"></a>
    
The window function that is applied to the filter coefficients. <cite>None</cite> if no window is applied.


### CustomFilter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#customfilter" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.signal.``CustomFilter`(<em class="sig-param">`span_in_symbols``=``None`</em>, <em class="sig-param">`samples_per_symbol``=``None`</em>, <em class="sig-param">`coefficients``=``None`</em>, <em class="sig-param">`window``=``None`</em>, <em class="sig-param">`normalize``=``True`</em>, <em class="sig-param">`trainable``=``False`</em>, <em class="sig-param">`dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/signal/filter.html#CustomFilter">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter" title="Permalink to this definition"></a>
    
Layer for applying a custom filter of `length` K
to an input `x` of length N.
    
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
Only needs to be provided if `coefficients` is None.
- **samples_per_symbol** (<em>int</em>) – Number of samples per symbol, i.e., the oversampling factor.
Must always be provided.
- **coefficients** (<em>[</em><em>K</em><em>]</em><em>, </em><em>tf.float</em><em> or </em><em>tf.complex</em>) – Optional filter coefficients.
If set to <cite>None</cite>, then a random filter of K is generated
by sampling a Gaussian distribution. Defaults to <cite>None</cite>.
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



<em class="property">`property` </em>`aclr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.aclr" title="Permalink to this definition"></a>
    
ACLR of the filter
    
This ACLR corresponds to what one would obtain from using
this filter as pulse shaping filter on an i.i.d. sequence of symbols.
The in-band is assumed to range from [-0.5, 0.5] in normalized
frequency.


<em class="property">`property` </em>`coefficients`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.coefficients" title="Permalink to this definition"></a>
    
The filter coefficients (after normalization)


<em class="property">`property` </em>`length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.length" title="Permalink to this definition"></a>
    
The filter length in samples


<em class="property">`property` </em>`normalize`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.normalize" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter is normalized to have unit power. <cite>False</cite> otherwise.


<em class="property">`property` </em>`sampling_times`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.sampling_times" title="Permalink to this definition"></a>
    
Sampling times in multiples of the symbol duration


`show`(<em class="sig-param">`response``=``'impulse'`</em>, <em class="sig-param">`scale``=``'lin'`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.show" title="Permalink to this definition"></a>
    
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





<em class="property">`property` </em>`trainable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.trainable" title="Permalink to this definition"></a>
    
<cite>True</cite> if the filter coefficients are trainable. <cite>False</cite> otherwise.


<em class="property">`property` </em>`window`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/signal.html#sionna.signal.CustomFilter.window" title="Permalink to this definition"></a>
    
The window function that is applied to the filter coefficients. <cite>None</cite> if no window is applied.


