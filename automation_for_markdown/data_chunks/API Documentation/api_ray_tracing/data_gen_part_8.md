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
## Cameras<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#cameras" title="Permalink to this headline"></a>
### Camera<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#camera" title="Permalink to this headline"></a>
## Scene Objects<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#scene-objects" title="Permalink to this headline"></a>
### SceneObject<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sceneobject" title="Permalink to this headline"></a>
  

## Cameras<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#cameras" title="Permalink to this headline"></a>
    
A <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> defines a position and view direction
for rendering the scene.
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.cameras" title="sionna.rt.Scene.cameras">`cameras`</a> property of the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a>
list all the cameras currently available for rendering. Cameras can be either
defined through the scene file or instantiated using the API.
The following code snippet shows how to load a scene and list the available
cameras:
```python
scene = load_scene(sionna.rt.scene.munich)
print(scene.cameras)
scene.render("scene-cam-0") # Use the first camera of the scene for rendering
```

    
A new camera can be instantiated as follows:
```python
cam = Camera("mycam", position=[200., 0.0, 50.])
scene.add(cam)
cam.look_at([0.0,0.0,0.0])
scene.render(cam) # Render using the Camera instance
scene.render("mycam") # or using the name of the camera
```
  

### Camera<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#camera" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``Camera`(<em class="sig-param">`name`</em>, <em class="sig-param">`position`</em>, <em class="sig-param">`orientation``=``[0.,` `0.,` `0.]`</em>, <em class="sig-param">`look_at``=``None`</em>)<a class="reference internal" href="../_modules/sionna/rt/camera.html#Camera">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="Permalink to this definition"></a>
    
A camera defines a position and view direction for rendering the scene.
    
In its local coordinate system, a camera looks toward the positive X-axis
with the positive Z-axis being the upward direction.
Input
 
- **name** (<em>str</em>) – Name.
Cannot be <cite>“preview”</cite>, as it is reserved for the viewpoint of the
interactive viewer.
- **position** (<em>[3], float</em>) – Position $(x,y,z)$ [m] as three-dimensional vector
- **orientation** (<em>[3], float</em>) – Orientation $(\alpha, \beta, \gamma)$ specified
through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.
This parameter is ignored if `look_at` is not <cite>None</cite>.
Defaults to <cite>[0,0,0]</cite>.
- **look_at** ([3], float | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> | None) – A position or instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>, or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> to look at.
If set to <cite>None</cite>, then `orientation` is used to orientate the camera.




`look_at`(<em class="sig-param">`target`</em>)<a class="reference internal" href="../_modules/sionna/rt/camera.html#Camera.look_at">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera.look_at" title="Permalink to this definition"></a>
    
Sets the orientation so that the camera looks at a position, radio
device, or another camera.
    
Given a point $\mathbf{x}\in\mathbb{R}^3$ with spherical angles
$\theta$ and $\varphi$, the orientation of the camera
will be set equal to $(\varphi, \frac{\pi}{2}-\theta, 0.0)$.
Input
    
**target** ([3], float | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> | str) – A position or the name or instance of a
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>, or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> in the scene to look at.




<em class="property">`property` </em>`orientation`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera.orientation" title="Permalink to this definition"></a>
    
Get/set the orientation $(\alpha, \beta, \gamma)$
specified through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.
Type
    
[3], float




<em class="property">`property` </em>`position`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera.position" title="Permalink to this definition"></a>
    
Get/set the position $(x,y,z)$ as three-dimensional
vector
Type
    
[3], float




## Scene Objects<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#scene-objects" title="Permalink to this headline"></a>
    
A scene is made of scene objects. Examples include cars, trees,
buildings, furniture, etc.
A scene object is characterized by its geometry and material (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a>)
and implemented as an instance of the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="sionna.rt.SceneObject">`SceneObject`</a> class.
    
Scene objects are uniquely identified by their name.
To access a scene object, the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.get" title="sionna.rt.Scene.get">`get()`</a> method of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a> may be used.
For example, the following code snippet shows how to load a scene and list its scene objects:
```python
scene = load_scene(sionna.rt.scene.munich)
print(scene.objects)
```

    
To select an object, e.g., named <cite>“Schrannenhalle-itu_metal”</cite>, you can run:
```python
my_object = scene.get("Schrannenhalle-itu_metal")
```

    
You can then set the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a>
of `my_object` as follows:
```python
my_object.radio_material = "itu_wood"
```

    
Most scene objects names have postfixes of the form “-material_name”. These are used during loading of a scene
to assign a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> to each of them. This <a class="reference external" href="https://youtu.be/7xHLDxUaQ7c">tutorial video</a>
explains how you can assign radio materials to objects when you create your own scenes.

### SceneObject<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sceneobject" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``SceneObject`<a class="reference internal" href="../_modules/sionna/rt/scene_object.html#SceneObject">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="Permalink to this definition"></a>
    
Every object in the scene is implemented by an instance of this class

<em class="property">`property` </em>`name`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject.name" title="Permalink to this definition"></a>
    
Name
Type
    
str (read-only)




<em class="property">`property` </em>`radio_material`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject.radio_material" title="Permalink to this definition"></a>
    
Get/set the radio material of the
object. Setting can be done by using either an instance of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> or the material name (<cite>str</cite>).
If the radio material is not part of the scene, it will be added. This
can raise an error if a different radio material with the same name was
already added to the scene.
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a>




