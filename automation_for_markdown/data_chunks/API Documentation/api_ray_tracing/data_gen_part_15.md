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
## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#utility-functions" title="Permalink to this headline"></a>
### sample_points_on_hemisphere<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sample-points-on-hemisphere" title="Permalink to this headline"></a>
### theta_hat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#theta-hat" title="Permalink to this headline"></a>
### theta_phi_from_unit_vec<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#theta-phi-from-unit-vec" title="Permalink to this headline"></a>
  
  

### sample_points_on_hemisphere<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sample-points-on-hemisphere" title="Permalink to this headline"></a>

`sionna.rt.``sample_points_on_hemisphere`(<em class="sig-param">`normals`</em>, <em class="sig-param">`num_samples``=``1`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#sample_points_on_hemisphere">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.sample_points_on_hemisphere" title="Permalink to this definition"></a>
    
Randomly sample points on hemispheres defined by their normal vectors
Input
 
- **normals** (<em>[batch_size, 3], tf.float</em>) – Normal vectors defining hemispheres
- **num_samples** (<em>int</em>) – Number of random samples to draw for each hemisphere
defined by its normal vector.
Defaults to 1.


Output
    
**points** (<em>[batch_size, num_samples, 3], tf.float or [batch_size, 3], tf.float if num_samples=1.</em>) – Random points on the hemispheres



### theta_hat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#theta-hat" title="Permalink to this headline"></a>

`sionna.rt.``theta_hat`(<em class="sig-param">`theta`</em>, <em class="sig-param">`phi`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#theta_hat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.theta_hat" title="Permalink to this definition"></a>
    
Computes the spherical unit vector
$\hat{\boldsymbol{\theta}}(\theta, \varphi)$
as defined in <a class="reference internal" href="../em_primer.html#equation-spherical-vecs">(1)</a>
Input
 
- **theta** (<em>arbitrary shape, tf.float</em>) – Zenith angles $\theta$ [rad]
- **phi** (same shape as `theta`, tf.float) – Azimuth angles $\varphi$ [rad]


Output
    
**theta_hat** (`phi.shape` + [3], tf.float) – Vector $\hat{\boldsymbol{\theta}}(\theta, \varphi)$



### theta_phi_from_unit_vec<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#theta-phi-from-unit-vec" title="Permalink to this headline"></a>

`sionna.rt.``theta_phi_from_unit_vec`(<em class="sig-param">`v`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#theta_phi_from_unit_vec">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.theta_phi_from_unit_vec" title="Permalink to this definition"></a>
    
Computes zenith and azimuth angles ($\theta,\varphi$)
from unit-norm vectors as described in <a class="reference internal" href="../em_primer.html#equation-theta-phi">(2)</a>
Input
    
**v** (<em>[…,3], tf.float</em>) – Tensor with unit-norm vectors in the last dimension

Output
 
- **theta** (<em>[…], tf.float</em>) – Zenith angles $\theta$
- **phi** (<em>[…], tf.float</em>) – Azimuth angles $\varphi$





References:
Balanis97(<a href="https://nvlabs.github.io/sionna/api/rt.html#id21">1</a>,<a href="https://nvlabs.github.io/sionna/api/rt.html#id22">2</a>)
<ol class="upperalpha simple">
- Balanis, “Antenna Theory: Analysis and Design,” 2nd Edition, John Wiley & Sons, 1997.
</ol>

ITUR_P2040_2(<a href="https://nvlabs.github.io/sionna/api/rt.html#id16">1</a>,<a href="https://nvlabs.github.io/sionna/api/rt.html#id17">2</a>)
    
ITU-R, “Effects of building materials and structures on radiowave propagation above about 100 MHz“, Recommendation ITU-R P.2040-2

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/rt.html#id2">SurfaceIntegral</a>
    
Wikipedia, “<a class="reference external" href="https://en.wikipedia.org/wiki/Surface_integral">Surface integral</a>”, accessed Jun. 22, 2023.



