# Ray Tracing<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#ray-tracing" title="Permalink to this headline"></a>
    
This module provides a differentiable ray tracer for radio propagation modeling.
The best way to get started is by having a look at the <a class="reference external" href="../examples/Sionna_Ray_Tracing_Introduction.html">Sionna Ray Tracing Tutorial</a>.
The <a class="reference external" href="../em_primer.html">Primer on Electromagnetics</a> provides useful background knowledge and various definitions that are used throughout the API documentation.
    
The most important component of the ray tracer is the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a>.
It has methods for the computation of propagation <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a>) and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a> (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.coverage_map" title="sionna.rt.Scene.coverage_map">`coverage_map()`</a>).
Sionna has several integrated <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#example-scenes">Example Scenes</a> that you can use for your own experiments. In this <a class="reference external" href="https://youtu.be/7xHLDxUaQ7c">video</a>, we explain how you can create your own scenes using <a class="reference external" href="https://www.openstreetmap.org">OpenStreetMap</a> and <a class="reference external" href="https://www.blender.org">Blender</a>.
You can preview a scene within a Jupyter notebook (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>) or render it to a file from the viewpoint of a camera (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render" title="sionna.rt.Scene.render">`render()`</a> or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render_to_file" title="sionna.rt.Scene.render_to_file">`render_to_file()`</a>).
    
Propagation <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> can be transformed into time-varying channel impulse responses (CIRs) via <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.cir" title="sionna.rt.Paths.cir">`cir()`</a>. The CIRs can then be used for link-level simulations in Sionna via the functions <a class="reference internal" href="channel.wireless.html#sionna.channel.cir_to_time_channel" title="sionna.channel.cir_to_time_channel">`cir_to_time_channel()`</a> or <a class="reference internal" href="channel.wireless.html#sionna.channel.cir_to_ofdm_channel" title="sionna.channel.cir_to_ofdm_channel">`cir_to_ofdm_channel()`</a>. Alternatively, you can create a dataset of CIRs that can be used by a channel model with the help of <a class="reference internal" href="channel.wireless.html#sionna.channel.CIRDataset" title="sionna.channel.CIRDataset">`CIRDataset`</a>.
    
The paper <a class="reference external" href="https://nvlabs.github.io/sionna/made_with_sionna.html#sionna-rt-differentiable-ray-tracing-for-radio-propagation-modeling">Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling</a> shows how differentiable ray tracing can be used for various optimization tasks. The related <a class="reference external" href="https://nvlabs.github.io/sionna/made_with_sionna.html#sionna-rt-differentiable-ray-tracing-for-radio-propagation-modeling">notebooks</a> can be a good starting point for your own experiments.

# Table of Content
## Radio Materials<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#radio-materials" title="Permalink to this headline"></a>
### ScatteringPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#scatteringpattern" title="Permalink to this headline"></a>

  

### ScatteringPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#scatteringpattern" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``LambertianPattern`(<em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/scattering_pattern.html#LambertianPattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.LambertianPattern" title="Permalink to this definition"></a>
    
Lambertian scattering model from <a class="reference internal" href="../em_primer.html#degli-esposti07" id="id18">[Degli-Esposti07]</a> as given in <a class="reference internal" href="../em_primer.html#equation-lambertian-model">(40)</a>
Parameters
    
**dtype** (<em>tf.complex64</em><em> or </em><em>tf.complex128</em>) – Datatype used for all computations.
Defaults to <cite>tf.complex64</cite>.

Input
 
- **k_i** (<em>[batch_size, 3], dtype.real_dtype</em>) – Incoming directions
- **k_s** (<em>[batch_size,3], dtype.real_dtype</em>) – Outgoing directions


Output
    
**pattern** (<em>[batch_size], dtype.real_dtype</em>) – Scattering pattern


<p class="rubric">Example
```python
>>> LambertianPattern().visualize()
```



`visualize`(<em class="sig-param">`k_i``=``(0.7071,` `0.0,` `-` `0.7071)`</em>, <em class="sig-param">`show_directions``=``False`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.LambertianPattern.visualize" title="Permalink to this definition"></a>
    
Visualizes the scattering pattern
    
It is assumed that the surface normal points toward the
positive z-axis.
Input
 
- **k_i** (<em>[3], array_like</em>) – Incoming direction
- **show_directions** (<em>bool</em>) – If <cite>True</cite>, the incoming and specular reflection directions
are shown.
Defaults to <cite>False</cite>.


Output
 
- `matplotlib.pyplot.Figure` – 3D visualization of the scattering pattern
- `matplotlib.pyplot.Figure` – Visualization of the incident plane cut through
the scattering pattern






<em class="property">`class` </em>`sionna.rt.``DirectivePattern`(<em class="sig-param">`alpha_r`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/scattering_pattern.html#DirectivePattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.DirectivePattern" title="Permalink to this definition"></a>
    
Directive scattering model from <a class="reference internal" href="../em_primer.html#degli-esposti07" id="id19">[Degli-Esposti07]</a> as given in <a class="reference internal" href="../em_primer.html#equation-directive-model">(41)</a>
Parameters
 
- **alpha_r** (<em>int</em><em>, </em><em>[</em><em>1</em><em>,</em><em>2</em><em>,</em><em>...</em><em>]</em>) – Parameter related to the width of the scattering lobe in the
direction of the specular reflection.
- **dtype** (<em>tf.complex64</em><em> or </em><em>tf.complex128</em>) – Datatype used for all computations.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **k_i** (<em>[batch_size, 3], dtype.real_dtype</em>) – Incoming directions
- **k_s** (<em>[batch_size,3], dtype.real_dtype</em>) – Outgoing directions


Output
    
**pattern** (<em>[batch_size], dtype.real_dtype</em>) – Scattering pattern


<p class="rubric">Example
```python
>>> DirectivePattern(alpha_r=10).visualize()
```



<em class="property">`property` </em>`alpha_r`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.DirectivePattern.alpha_r" title="Permalink to this definition"></a>
    
Get/set `alpha_r`
Type
    
bool




`visualize`(<em class="sig-param">`k_i``=``(0.7071,` `0.0,` `-` `0.7071)`</em>, <em class="sig-param">`show_directions``=``False`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.DirectivePattern.visualize" title="Permalink to this definition"></a>
    
Visualizes the scattering pattern
    
It is assumed that the surface normal points toward the
positive z-axis.
Input
 
- **k_i** (<em>[3], array_like</em>) – Incoming direction
- **show_directions** (<em>bool</em>) – If <cite>True</cite>, the incoming and specular reflection directions
are shown.
Defaults to <cite>False</cite>.


Output
 
- `matplotlib.pyplot.Figure` – 3D visualization of the scattering pattern
- `matplotlib.pyplot.Figure` – Visualization of the incident plane cut through
the scattering pattern






<em class="property">`class` </em>`sionna.rt.``BackscatteringPattern`(<em class="sig-param">`alpha_r`</em>, <em class="sig-param">`alpha_i`</em>, <em class="sig-param">`lambda_`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/scattering_pattern.html#BackscatteringPattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.BackscatteringPattern" title="Permalink to this definition"></a>
    
Backscattering model from <a class="reference internal" href="../em_primer.html#degli-esposti07" id="id20">[Degli-Esposti07]</a> as given in <a class="reference internal" href="../em_primer.html#equation-backscattering-model">(42)</a>
    
The parameter `lambda_` can be assigned to a TensorFlow variable
or tensor.  In the latter case, the tensor can be the output of a callable, such as
a Keras layer implementing a neural network.
In the former case, it can be set to a trainable variable:
```python
sp = BackscatteringPattern(alpha_r=3,
                           alpha_i=5,
                           lambda_=tf.Variable(0.3, dtype=tf.float32))
```

Parameters
 
- **alpha_r** (<em>int</em><em>, </em><em>[</em><em>1</em><em>,</em><em>2</em><em>,</em><em>...</em><em>]</em>) – Parameter related to the width of the scattering lobe in the
direction of the specular reflection.
- **alpha_i** (<em>int</em><em>, </em><em>[</em><em>1</em><em>,</em><em>2</em><em>,</em><em>...</em><em>]</em>) – Parameter related to the width of the scattering lobe in the
incoming direction.
- **lambda** (<em>float</em><em>, </em><em>[</em><em>0</em><em>,</em><em>1</em><em>]</em>) – Parameter determining the percentage of the diffusely
reflected energy in the lobe around the specular reflection.
- **dtype** (<em>tf.complex64</em><em> or </em><em>tf.complex128</em>) – Datatype used for all computations.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **k_i** (<em>[batch_size, 3], dtype.real_dtype</em>) – Incoming directions
- **k_s** (<em>[batch_size,3], dtype.real_dtype</em>) – Outgoing directions


Output
    
**pattern** (<em>[batch_size], dtype.real_dtype</em>) – Scattering pattern


<p class="rubric">Example
```python
>>> BackscatteringPattern(alpha_r=20, alpha_i=30, lambda_=0.7).visualize()
```



<em class="property">`property` </em>`alpha_i`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.BackscatteringPattern.alpha_i" title="Permalink to this definition"></a>
    
Get/set `alpha_i`
Type
    
bool




<em class="property">`property` </em>`alpha_r`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.BackscatteringPattern.alpha_r" title="Permalink to this definition"></a>
    
Get/set `alpha_r`
Type
    
bool




<em class="property">`property` </em>`lambda_`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.BackscatteringPattern.lambda_" title="Permalink to this definition"></a>
    
Get/set `lambda_`
Type
    
bool




`visualize`(<em class="sig-param">`k_i``=``(0.7071,` `0.0,` `-` `0.7071)`</em>, <em class="sig-param">`show_directions``=``False`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.BackscatteringPattern.visualize" title="Permalink to this definition"></a>
    
Visualizes the scattering pattern
    
It is assumed that the surface normal points toward the
positive z-axis.
Input
 
- **k_i** (<em>[3], array_like</em>) – Incoming direction
- **show_directions** (<em>bool</em>) – If <cite>True</cite>, the incoming and specular reflection directions
are shown.
Defaults to <cite>False</cite>.


Output
 
- `matplotlib.pyplot.Figure` – 3D visualization of the scattering pattern
- `matplotlib.pyplot.Figure` – Visualization of the incident plane cut through
the scattering pattern





