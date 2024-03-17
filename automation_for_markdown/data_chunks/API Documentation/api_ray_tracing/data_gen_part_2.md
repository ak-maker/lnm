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
## Scene
### Scene<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#id1" title="Permalink to this headline"></a>
  
  

### Scene<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#id1" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``Scene`<a class="reference internal" href="../_modules/sionna/rt/scene.html#Scene">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="Permalink to this definition"></a>
    
The scene contains everything that is needed for radio propagation simulation
and rendering.
    
A scene is a collection of multiple instances of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="sionna.rt.SceneObject">`SceneObject`</a> which define
the geometry and materials of the objects in the scene.
The scene also includes transmitters (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>) and receivers (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>)
for which propagation <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>, channel impulse responses (CIRs) or coverage maps (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a>) can be computed,
as well as cameras (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>) for rendering.
    
The only way to instantiate a scene is by calling `load_scene()`.
Note that only a single scene can be loaded at a time.
    
Example scenes can be loaded as follows:
```python
scene = load_scene(sionna.rt.scene.munich)
scene.preview()
```


`add`(<em class="sig-param">`item`</em>)<a class="reference internal" href="../_modules/sionna/rt/scene.html#Scene.add">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.add" title="Permalink to this definition"></a>
    
Adds a transmitter, receiver, radio material, or camera to the scene.
    
If a different item with the same name as `item` is already part of the scene,
an error is raised.
Input
    
**item** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>) – Item to add to the scene




<em class="property">`property` </em>`cameras`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.cameras" title="Permalink to this definition"></a>
    
Dictionary
of cameras in the scene
Type
    
<cite>dict</cite> (read-only), { “name”, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>}




<em class="property">`property` </em>`center`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.center" title="Permalink to this definition"></a>
    
Get the center of the scene
Type
    
[3], tf.float




<em class="property">`property` </em>`dtype`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.dtype" title="Permalink to this definition"></a>
    
Datatype used in tensors
Type
    
<cite>tf.complex64 | tf.complex128</cite>




<em class="property">`property` </em>`frequency`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.frequency" title="Permalink to this definition"></a>
    
Get/set the carrier frequency [Hz]
    
Setting the frequency updates the parameters of frequency-dependent
radio materials. Defaults to 3.5e9.
Type
    
float




`get`(<em class="sig-param">`name`</em>)<a class="reference internal" href="../_modules/sionna/rt/scene.html#Scene.get">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.get" title="Permalink to this definition"></a>
    
Returns a scene object, transmitter, receiver, camera, or radio material
Input
    
**name** (<em>str</em>) – Name of the item to retrieve

Output
    
**item** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="sionna.rt.SceneObject">`SceneObject`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> | <cite>None</cite>) – Retrieved item. Returns <cite>None</cite> if no corresponding item was found in the scene.




<em class="property">`property` </em>`objects`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.objects" title="Permalink to this definition"></a>
    
Dictionary
of scene objects
Type
    
<cite>dict</cite> (read-only), { “name”, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="sionna.rt.SceneObject">`SceneObject`</a>}




<em class="property">`property` </em>`radio_material_callable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.radio_material_callable" title="Permalink to this definition"></a>
    
Get/set a callable that computes the radio material properties at the
points of intersection between the rays and the scene objects.
    
If set, then the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> of the objects are
not used and the callable is invoked instead to obtain the
electromagnetic properties required to simulate the propagation of radio
waves.
    
If not set, i.e., <cite>None</cite> (default), then the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> of objects are used to simulate the
propagation of radio waves in the scene.
    
This callable is invoked on batches of intersection points.
It takes as input the following tensors:
 
- `object_id` (<cite>[batch_dims]</cite>, <cite>int</cite>) : Integers uniquely identifying the intersected objects
- `points` (<cite>[batch_dims, 3]</cite>, <cite>float</cite>) : Positions of the intersection points

    
The callable must output a tuple/list of the following tensors:
 
- `complex_relative_permittivity` (<cite>[batch_dims]</cite>, <cite>complex</cite>) : Complex relative permittivities $\eta$ <a class="reference internal" href="../em_primer.html#equation-eta">(9)</a>
- `scattering_coefficient` (<cite>[batch_dims]</cite>, <cite>float</cite>) : Scattering coefficients $S\in[0,1]$ <a class="reference internal" href="../em_primer.html#equation-scattering-coefficient">(37)</a>
- `xpd_coefficient` (<cite>[batch_dims]</cite>, <cite>float</cite>) : Cross-polarization discrimination coefficients $K_x\in[0,1]$ <a class="reference internal" href="../em_primer.html#equation-xpd">(39)</a>. Only relevant for the scattered field.

    
**Note:** The number of batch dimensions is not necessarily equal to one.


<em class="property">`property` </em>`radio_materials`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.radio_materials" title="Permalink to this definition"></a>
    
Dictionary
of radio materials
Type
    
<cite>dict</cite> (read-only), { “name”, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a>}




<em class="property">`property` </em>`receivers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.receivers" title="Permalink to this definition"></a>
    
Dictionary
of receivers in the scene
Type
    
<cite>dict</cite> (read-only), { “name”, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>}




`remove`(<em class="sig-param">`name`</em>)<a class="reference internal" href="../_modules/sionna/rt/scene.html#Scene.remove">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.remove" title="Permalink to this definition"></a>
    
Removes a transmitter, receiver, camera, or radio material from the
scene.
    
In the case of a radio material, it must not be used by any object of
the scene.
Input
    
**name** (<em>str</em>) – Name of the item to remove




<em class="property">`property` </em>`rx_array`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.rx_array" title="Permalink to this definition"></a>
    
Get/set the antenna array used by
all receivers in the scene. Defaults to <cite>None</cite>.
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a>




<em class="property">`property` </em>`scattering_pattern_callable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.scattering_pattern_callable" title="Permalink to this definition"></a>
    
Get/set a callable that computes the scattering pattern at the
points of intersection between the rays and the scene objects.
    
If set, then the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.scattering_pattern" title="sionna.rt.RadioMaterial.scattering_pattern">`scattering_pattern`</a> of
the radio materials of the objects are not used and the callable is invoked
instead to evaluate the scattering pattern required to simulate the
propagation of diffusely reflected radio waves.
    
If not set, i.e., <cite>None</cite> (default), then the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.scattering_pattern" title="sionna.rt.RadioMaterial.scattering_pattern">`scattering_pattern`</a> of the objects’
radio materials are used to simulate the propagation of diffusely
reflected radio waves in the scene.
    
This callable is invoked on batches of intersection points.
It takes as input the following tensors:
 
- `object_id` (<cite>[batch_dims]</cite>, <cite>int</cite>) : Integers uniquely identifying the intersected objects
- `points` (<cite>[batch_dims, 3]</cite>, <cite>float</cite>) : Positions of the intersection points
- `k_i` (<cite>[batch_dims, 3]</cite>, <cite>float</cite>) : Unitary vector corresponding to the direction of incidence in the scene’s global coordinate system
- `k_s` (<cite>[batch_dims, 3]</cite>, <cite>float</cite>) : Unitary vector corresponding to the direction of the diffuse reflection in the scene’s global coordinate system
- `n` (<cite>[batch_dims, 3]</cite>, <cite>float</cite>) : Unitary vector corresponding to the normal to the surface at the intersection point

    
The callable must output the following tensor:
 
- `f_s` (<cite>[batch_dims]</cite>, <cite>float</cite>) : The scattering pattern evaluated for the previous inputs

    
**Note:** The number of batch dimensions is not necessarily equal to one.


<em class="property">`property` </em>`size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.size" title="Permalink to this definition"></a>
    
Get the size of the scene, i.e., the size of the
axis-aligned minimum bounding box for the scene
Type
    
[3], tf.float




<em class="property">`property` </em>`synthetic_array`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.synthetic_array" title="Permalink to this definition"></a>
    
Get/set if the antenna arrays are applied synthetically.
Defaults to <cite>True</cite>.
Type
    
bool




<em class="property">`property` </em>`transmitters`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.transmitters" title="Permalink to this definition"></a>
    
Dictionary
of transmitters in the scene
Type
    
<cite>dict</cite> (read-only), { “name”, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>}




<em class="property">`property` </em>`tx_array`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.tx_array" title="Permalink to this definition"></a>
    
Get/set the antenna array used by
all transmitters in the scene. Defaults to <cite>None</cite>.
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a>




<em class="property">`property` </em>`wavelength`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.wavelength" title="Permalink to this definition"></a>
    
Wavelength [m]
Type
    
float (read-only)




