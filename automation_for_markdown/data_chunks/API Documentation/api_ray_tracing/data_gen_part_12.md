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
## Radio Devices<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#radio-devices" title="Permalink to this headline"></a>
### Transmitter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#transmitter" title="Permalink to this headline"></a>
### Receiver<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#receiver" title="Permalink to this headline"></a>
## Antenna Arrays<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#antenna-arrays" title="Permalink to this headline"></a>
### AntennaArray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#antennaarray" title="Permalink to this headline"></a>
  



## Radio Devices<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#radio-devices" title="Permalink to this headline"></a>
    
A radio device refers to a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> equipped
with an <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a> as specified by the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a>’s properties
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.tx_array" title="sionna.rt.Scene.tx_array">`tx_array`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.rx_array" title="sionna.rt.Scene.rx_array">`rx_array`</a>, respectively.
    
The following code snippet shows how to instantiate a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>
equipped with a $4 \times 2$ <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray" title="sionna.rt.PlanarArray">`PlanarArray`</a> with cross-polarized isotropic antennas:
```python
 scene.tx_array = PlanarArray(num_rows=4,
                              num_cols=2,
                              vertical_spacing=0.5,
                              horizontal_spacing=0.5,
                              pattern="iso",
                              polarization="cross")
 my_tx = Transmitter(name="my_tx",
                     position=(0,0,0),
                     orientation=(0,0,0))
scene.add(my_tx)
```

    
The position $(x,y,z)$ and orientation $(\alpha, \beta, \gamma)$ of a radio device
can be freely configured. The latter is specified through three angles corresponding to a 3D
rotation as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.
Both can be assigned to TensorFlow variables or tensors. In the latter case,
the tensor can be the output of a callable, such as a Keras layer implementing a neural network.
In the former case, it can be set to a trainable variable.
    
Radio devices need to be explicitly added to the scene using the scene’s method <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.add" title="sionna.rt.Scene.add">`add()`</a>
and can be removed from it using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.remove" title="sionna.rt.Scene.remove">`remove()`</a>:
```python
scene = load_scene()
scene.add(Transmitter("tx", [10.0, 0.0, 1.5], [0.0,0.0,0.0]))
scene.remove("tx")
```

### Transmitter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#transmitter" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``Transmitter`(<em class="sig-param">`name`</em>, <em class="sig-param">`position`</em>, <em class="sig-param">`orientation``=``(0.0,` `0.0,` `0.0)`</em>, <em class="sig-param">`look_at``=``None`</em>, <em class="sig-param">`color``=``(0.16,` `0.502,` `0.725)`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/transmitter.html#Transmitter">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="Permalink to this definition"></a>
    
Class defining a transmitter
    
The `position` and `orientation` properties can be assigned to a TensorFlow
variable or tensor. In the latter case, the tensor can be the output of a callable,
such as a Keras layer implementing a neural network. In the former case, it
can be set to a trainable variable:
```python
tx = Transmitter(name="my_tx",
                 position=tf.Variable([0, 0, 0], dtype=tf.float32),
                 orientation=tf.Variable([0, 0, 0], dtype=tf.float32))
```

Parameters
 
- **name** (<em>str</em>) – Name
- **position** (<em>[</em><em>3</em><em>]</em><em>, </em><em>float</em>) – Position $(x,y,z)$ [m] as three-dimensional vector
- **orientation** (<em>[</em><em>3</em><em>]</em><em>, </em><em>float</em>) – Orientation $(\alpha, \beta, \gamma)$ [rad] specified
through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.
This parameter is ignored if `look_at` is not <cite>None</cite>.
Defaults to [0,0,0].
- **look_at** ([3], float | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> | None) – A position or the instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>, or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> to look at.
If set to <cite>None</cite>, then `orientation` is used to orientate the device.
- **color** (<em>[</em><em>3</em><em>]</em><em>, </em><em>float</em>) – Defines the RGB (red, green, blue) `color` parameter for the device as displayed in the previewer and renderer.
Each RGB component must have a value within the range $\in [0,1]$.
Defaults to <cite>[0.160, 0.502, 0.725]</cite>.
- **dtype** (<em>tf.complex</em>) – Datatype to be used in internal calculations.
Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`color`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter.color" title="Permalink to this definition"></a>
    
Get/set the the RGB (red, green, blue) color for the device as displayed in the previewer and renderer.
Each RGB component must have a value within the range $\in [0,1]$.
Type
    
[3], float




`look_at`(<em class="sig-param">`target`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter.look_at" title="Permalink to this definition"></a>
    
Sets the orientation so that the x-axis points toward a
position, radio device, or camera.
    
Given a point $\mathbf{x}\in\mathbb{R}^3$ with spherical angles
$\theta$ and $\varphi$, the orientation of the radio device
will be set equal to $(\varphi, \frac{\pi}{2}-\theta, 0.0)$.
Input
    
**target** ([3], float | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> | str) – A position or the name or instance of a
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>, or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> in the scene to look at.




<em class="property">`property` </em>`name`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter.name" title="Permalink to this definition"></a>
    
Name
Type
    
str (read-only)




<em class="property">`property` </em>`orientation`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter.orientation" title="Permalink to this definition"></a>
    
Get/set the orientation
Type
    
[3], tf.float




<em class="property">`property` </em>`position`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter.position" title="Permalink to this definition"></a>
    
Get/set the position
Type
    
[3], tf.float




### Receiver<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#receiver" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``Receiver`(<em class="sig-param">`name`</em>, <em class="sig-param">`position`</em>, <em class="sig-param">`orientation``=``(0.0,` `0.0,` `0.0)`</em>, <em class="sig-param">`look_at``=``None`</em>, <em class="sig-param">`color``=``(0.153,` `0.682,` `0.375)`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/receiver.html#Receiver">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="Permalink to this definition"></a>
    
Class defining a receiver
    
The `position` and `orientation` properties can be assigned to a TensorFlow
variable or tensor. In the latter case, the tensor can be the output of a callable,
such as a Keras layer implementing a neural network. In the former case, it
can be set to a trainable variable:
```python
rx = Transmitter(name="my_rx",
                 position=tf.Variable([0, 0, 0], dtype=tf.float32),
                 orientation=tf.Variable([0, 0, 0], dtype=tf.float32))
```

Parameters
 
- **name** (<em>str</em>) – Name
- **position** (<em>[</em><em>3</em><em>]</em><em>, </em><em>float</em>) – Position $(x,y,z)$ as three-dimensional vector
- **orientation** (<em>[</em><em>3</em><em>]</em><em>, </em><em>float</em>) – Orientation $(\alpha, \beta, \gamma)$ [rad] specified
through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.
This parameter is ignored if `look_at` is not <cite>None</cite>.
Defaults to [0,0,0].
- **look_at** ([3], float | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> | None) – A position or the instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>, or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> to look at.
If set to <cite>None</cite>, then `orientation` is used to orientate the device.
- **color** (<em>[</em><em>3</em><em>]</em><em>, </em><em>float</em>) – Defines the RGB (red, green, blue) `color` parameter for the device as displayed in the previewer and renderer.
Each RGB component must have a value within the range $\in [0,1]$.
Defaults to <cite>[0.153, 0.682, 0.375]</cite>.
- **dtype** (<em>tf.complex</em>) – Datatype to be used in internal calculations.
Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`color`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver.color" title="Permalink to this definition"></a>
    
Get/set the the RGB (red, green, blue) color for the device as displayed in the previewer and renderer.
Each RGB component must have a value within the range $\in [0,1]$.
Type
    
[3], float




`look_at`(<em class="sig-param">`target`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver.look_at" title="Permalink to this definition"></a>
    
Sets the orientation so that the x-axis points toward a
position, radio device, or camera.
    
Given a point $\mathbf{x}\in\mathbb{R}^3$ with spherical angles
$\theta$ and $\varphi$, the orientation of the radio device
will be set equal to $(\varphi, \frac{\pi}{2}-\theta, 0.0)$.
Input
    
**target** ([3], float | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> | str) – A position or the name or instance of a
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>, or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> in the scene to look at.




<em class="property">`property` </em>`name`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver.name" title="Permalink to this definition"></a>
    
Name
Type
    
str (read-only)




<em class="property">`property` </em>`orientation`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver.orientation" title="Permalink to this definition"></a>
    
Get/set the orientation
Type
    
[3], tf.float




<em class="property">`property` </em>`position`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver.position" title="Permalink to this definition"></a>
    
Get/set the position
Type
    
[3], tf.float




## Antenna Arrays<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#antenna-arrays" title="Permalink to this headline"></a>
    
Transmitters (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>) and receivers (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>) are equipped with an <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a> that is composed of one or more antennas. All transmitters and all receivers share the same <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a> which can be set through the scene properties <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.tx_array" title="sionna.rt.Scene.tx_array">`tx_array`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.rx_array" title="sionna.rt.Scene.rx_array">`rx_array`</a>, respectively.

### AntennaArray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#antennaarray" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``AntennaArray`(<em class="sig-param">`antenna`</em>, <em class="sig-param">`positions`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna_array.html#AntennaArray">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="Permalink to this definition"></a>
    
Class implementing an antenna array
    
An antenna array is composed of identical antennas that are placed
at different positions. The `positions` parameter can be assigned
to a TensorFlow variable or tensor.
```python
array = AntennaArray(antenna=Antenna("tr38901", "V"),
                     positions=tf.Variable([[0,0,0], [0, 1, 1]]))
```

Parameters
 
- **antenna** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna" title="sionna.rt.Antenna">`Antenna`</a>) – Antenna instance
- **positions** (<em>[</em><em>array_size</em><em>, </em><em>3</em><em>]</em><em>, </em><em>array_like</em>) – Array of relative positions $(x,y,z)$ [m] of each
antenna (dual-polarized antennas are counted as a single antenna
and share the same position).
The absolute position of the antennas is obtained by
adding the position of the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>
or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> using it.
- **dtype** (<em>tf.complex64</em><em> or </em><em>tf.complex128</em>) – Data type used for all computations.
Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`antenna`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray.antenna" title="Permalink to this definition"></a>
    
Get/set the antenna
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna" title="sionna.rt.Antenna">`Antenna`</a>




<em class="property">`property` </em>`array_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray.array_size" title="Permalink to this definition"></a>
    
Number of antennas in the array.
Dual-polarized antennas are counted as a single antenna.
Type
    
int (read-only)




<em class="property">`property` </em>`num_ant`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray.num_ant" title="Permalink to this definition"></a>
    
Number of linearly polarized antennas in the array.
Dual-polarized antennas are counted as two linearly polarized
antennas.
Type
    
int (read-only)




<em class="property">`property` </em>`positions`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray.positions" title="Permalink to this definition"></a>
    
Get/set  array of relative positions
$(x,y,z)$ [m] of each antenna (dual-polarized antennas are
counted as a single antenna and share the same position).
Type
    
[array_size, 3], <cite>tf.float</cite>




`rotated_positions`(<em class="sig-param">`orientation`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna_array.html#AntennaArray.rotated_positions">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray.rotated_positions" title="Permalink to this definition"></a>
    
Get the antenna positions rotated according to `orientation`
Input
    
**orientation** (<em>[3], tf.float</em>) – Orientation $(\alpha, \beta, \gamma)$ [rad] specified
through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.

Output
    
<em>[array_size, 3]</em> – Rotated positions




