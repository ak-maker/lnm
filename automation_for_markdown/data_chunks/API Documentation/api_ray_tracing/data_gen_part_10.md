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
## Radio Materials
### RadioMaterial<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#radiomaterial" title="Permalink to this headline"></a>
  
  

### RadioMaterial<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#radiomaterial" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``RadioMaterial`(<em class="sig-param">`name`</em>, <em class="sig-param">`relative_permittivity``=``1.0`</em>, <em class="sig-param">`conductivity``=``0.0`</em>, <em class="sig-param">`scattering_coefficient``=``0.0`</em>, <em class="sig-param">`xpd_coefficient``=``0.0`</em>, <em class="sig-param">`scattering_pattern``=``None`</em>, <em class="sig-param">`frequency_update_callback``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/radio_material.html#RadioMaterial">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="Permalink to this definition"></a>
    
Class implementing a radio material
    
A radio material is defined by its relative permittivity
$\varepsilon_r$ and conductivity $\sigma$ (see <a class="reference internal" href="../em_primer.html#equation-eta">(9)</a>),
as well as optional parameters related to diffuse scattering, such as the
scattering coefficient $S$, cross-polarization discrimination
coefficient $K_x$, and scattering pattern $f_\text{s}(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s})$.
    
We assume non-ionized and non-magnetic materials, and therefore the
permeability $\mu$ of the material is assumed to be equal
to the permeability of vacuum i.e., $\mu_r=1.0$.
    
For frequency-dependent materials, it is possible to
specify a callback function `frequency_update_callback` that computes
the material properties $(\varepsilon_r, \sigma)$ from the
frequency. If a callback function is specified, the material properties
cannot be set and the values specified at instantiation are ignored.
The callback should return <cite>-1</cite> for both the relative permittivity and
the conductivity if these are not defined for the given carrier frequency.
    
The material properties can be assigned to a TensorFlow variable or
tensor. In the latter case, the tensor could be the output of a callable,
such as a Keras layer implementing a neural network. In the former case, it
could be set to a trainable variable:
```python
mat = RadioMaterial("my_mat")
mat.conductivity = tf.Variable(0.0, dtype=tf.float32)
```

Parameters
 
- **name** (<em>str</em>) – Unique name of the material
- **relative_permittivity** (float | <cite>None</cite>) – Relative permittivity of the material.
Must be larger or equal to 1.
Defaults to 1. Ignored if `frequency_update_callback`
is provided.
- **conductivity** (float | <cite>None</cite>) – Conductivity of the material [S/m].
Must be non-negative.
Defaults to 0.
Ignored if `frequency_update_callback`
is provided.
- **scattering_coefficient** (<em>float</em>) – Scattering coefficient $S\in[0,1]$ as defined in
<a class="reference internal" href="../em_primer.html#equation-scattering-coefficient">(37)</a>.
Defaults to 0.
- **xpd_coefficient** (<em>float</em>) – Cross-polarization discrimination coefficient $K_x\in[0,1]$ as
defined in <a class="reference internal" href="../em_primer.html#equation-xpd">(39)</a>.
Only relevant if `scattering_coefficient`>0.
Defaults to 0.
- **scattering_pattern** (<em>ScatteringPattern</em>) – `ScatteringPattern` to be applied.
Only relevant if `scattering_coefficient`>0.
Defaults to <cite>None</cite>, which implies a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.LambertianPattern" title="sionna.rt.LambertianPattern">`LambertianPattern`</a>.
- **frequency_update_callback** (callable | <cite>None</cite>) –     
An optional callable object used to obtain the material parameters
from the scene’s <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.frequency" title="sionna.rt.Scene.frequency">`frequency`</a>.
This callable must take as input the frequency [Hz] and
must return the material properties as a tuple:
    
`(relative_permittivity,` `conductivity)`.
    
If set to <cite>None</cite>, the material properties are constant and equal
to `relative_permittivity` and `conductivity`.
Defaults to <cite>None</cite>.

- **dtype** (<em>tf.complex64</em><em> or </em><em>tf.complex128</em>) – Datatype.
Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`complex_relative_permittivity`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.complex_relative_permittivity" title="Permalink to this definition"></a>
    
Complex relative permittivity
$\eta$ <a class="reference internal" href="../em_primer.html#equation-eta">(9)</a>
Type
    
tf.complex (read-only)




<em class="property">`property` </em>`conductivity`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.conductivity" title="Permalink to this definition"></a>
    
Get/set the conductivity
$\sigma$ [S/m] <a class="reference internal" href="../em_primer.html#equation-eta">(9)</a>
Type
    
tf.float




<em class="property">`property` </em>`frequency_update_callback`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.frequency_update_callback" title="Permalink to this definition"></a>
    
Get/set frequency update callback function
Type
    
callable




<em class="property">`property` </em>`is_used`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.is_used" title="Permalink to this definition"></a>
    
Indicator if the material is used by at least one object of
the scene
Type
    
bool




<em class="property">`property` </em>`name`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.name" title="Permalink to this definition"></a>
    
Name of the radio material
Type
    
str (read-only)




<em class="property">`property` </em>`relative_permeability`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.relative_permeability" title="Permalink to this definition"></a>
    
Relative permeability
$\mu_r$ <a class="reference internal" href="../em_primer.html#equation-mu">(8)</a>.
Defaults to 1.
Type
    
tf.float (read-only)




<em class="property">`property` </em>`relative_permittivity`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.relative_permittivity" title="Permalink to this definition"></a>
    
Get/set the relative permittivity
$\varepsilon_r$ <a class="reference internal" href="../em_primer.html#equation-eta">(9)</a>
Type
    
tf.float




<em class="property">`property` </em>`scattering_coefficient`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.scattering_coefficient" title="Permalink to this definition"></a>
    
Get/set the scattering coefficient
$S\in[0,1]$ <a class="reference internal" href="../em_primer.html#equation-scattering-coefficient">(37)</a>.
Type
    
tf.float




<em class="property">`property` </em>`scattering_pattern`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.scattering_pattern" title="Permalink to this definition"></a>
    
Get/set the ScatteringPattern.
Type
    
ScatteringPattern




<em class="property">`property` </em>`use_counter`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.use_counter" title="Permalink to this definition"></a>
    
Number of scene objects using this material
Type
    
int




<em class="property">`property` </em>`using_objects`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.using_objects" title="Permalink to this definition"></a>
    
Identifiers of the objects using this
material
Type
    
[num_using_objects], tf.int




<em class="property">`property` </em>`well_defined`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.well_defined" title="Permalink to this definition"></a>
    
Get if the material is well-defined
Type
    
bool




<em class="property">`property` </em>`xpd_coefficient`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.xpd_coefficient" title="Permalink to this definition"></a>
    
Get/set the cross-polarization discrimination coefficient
$K_x\in[0,1]$ <a class="reference internal" href="../em_primer.html#equation-xpd">(39)</a>.
Type
    
tf.float




