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
  
  

## Radio Materials<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#radio-materials" title="Permalink to this headline"></a>
    
A <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> contains everything that is needed to enable the simulation
of the interaction of a radio wave with an object made of a particular material.
More precisely, it consists of the real-valued relative permittivity $\varepsilon_r$,
the conductivity $\sigma$, and the relative
permeability $\mu_r$. For more details, see <a class="reference internal" href="../em_primer.html#equation-epsilon">(7)</a>, <a class="reference internal" href="../em_primer.html#equation-mu">(8)</a>, <a class="reference internal" href="../em_primer.html#equation-eta">(9)</a>.
These quantities can possibly depend on the frequency of the incident radio
wave. Note that Sionna currently only allows non-magnetic materials with $\mu_r=1$.
    
Additionally, a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> can have an effective roughness (ER)
associated with it, leading to diffuse reflections (see, e.g., <a class="reference internal" href="../em_primer.html#degli-esposti11" id="id15">[Degli-Esposti11]</a>).
The ER model requires a scattering coefficient $S\in[0,1]$ <a class="reference internal" href="../em_primer.html#equation-scattering-coefficient">(37)</a>,
a cross-polarization discrimination coefficient $K_x$ <a class="reference internal" href="../em_primer.html#equation-xpd">(39)</a>, as well as a scattering pattern
$f_\text{s}(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s})$ <a class="reference internal" href="../em_primer.html#equation-lambertian-model">(40)</a>–<a class="reference internal" href="../em_primer.html#equation-backscattering-model">(42)</a>, such as the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.LambertianPattern" title="sionna.rt.LambertianPattern">`LambertianPattern`</a> or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.DirectivePattern" title="sionna.rt.DirectivePattern">`DirectivePattern`</a>. The meaning of
these parameters is explained in <a class="reference internal" href="../em_primer.html#scattering">Scattering</a>.
    
Similarly to scene objects (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="sionna.rt.SceneObject">`SceneObject`</a>), all radio
materials are uniquely identified by their name.
For example, specifying that a scene object named <cite>“wall”</cite> is made of the
material named <cite>“itu-brick”</cite> is done as follows:
```python
obj = scene.get("wall") # obj is a SceneObject
obj.radio_material = "itu_brick" # "wall" is made of "itu_brick"
```

    
Sionna provides the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#provided-materials">ITU models of several materials</a> whose properties
are automatically updated according to the configured <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.frequency" title="sionna.rt.Scene.frequency">`frequency`</a>.
It is also possible to
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#custom-radio-materials">define custom radio materials</a>.
<p id="provided-materials">**Radio materials provided with Sionna**
    
Sionna provides the models of all of the materials defined in the ITU-R P.2040-2
recommendation <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#itur-p2040-2" id="id16">[ITUR_P2040_2]</a>. These models are based on curve fitting to
measurement results and assume non-ionized and non-magnetic materials
($\mu_r = 1$).
Frequency dependence is modeled by

$$
\begin{split}\begin{align}
   \varepsilon_r &= a f_{\text{GHz}}^b\\
   \sigma &= c f_{\text{GHz}}^d
\end{align}\end{split}
$$
    
where $f_{\text{GHz}}$ is the frequency in GHz, and the constants
$a$, $b$, $c$, and $d$ characterize the material.
The table below provides their values which are used in Sionna
(from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#itur-p2040-2" id="id17">[ITUR_P2040_2]</a>).
Note that the relative permittivity $\varepsilon_r$ and
conductivity $\sigma$ of all materials are updated automatically when
the frequency is set through the scene’s property <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.frequency" title="sionna.rt.Scene.frequency">`frequency`</a>.
Moreover, by default, the scattering coefficient, $S$, of these materials is set to
0, leading to no diffuse reflection.
<table class="docutils align-default">
<colgroup>
<col style="width: 25%" />
<col style="width: 17%" />
<col style="width: 15%" />
<col style="width: 14%" />
<col style="width: 9%" />
<col style="width: 21%" />
</colgroup>
<tbody>
<tr class="row-odd"><td rowspan="2">    
Material name</td>
<td colspan="2">    
Real part of relative permittivity</td>
<td colspan="2">    
Conductivity [S/m]</td>
<td rowspan="2">    
Frequency range (GHz)</td>
</tr>
<tr class="row-even"><td>    
a</td>
<td>    
b</td>
<td>    
c</td>
<td>    
d</td>
</tr>
<tr class="row-odd"><td>    
vacuum</td>
<td>    
1</td>
<td>    
0</td>
<td>    
0</td>
<td>    
0</td>
<td>    
0.001 – 100</td>
</tr>
<tr class="row-even"><td>    
itu_concrete</td>
<td>    
5.24</td>
<td>    
0</td>
<td>    
0.0462</td>
<td>    
0.7822</td>
<td>    
1 – 100</td>
</tr>
<tr class="row-odd"><td>    
itu_brick</td>
<td>    
3.91</td>
<td>    
0</td>
<td>    
0.0238</td>
<td>    
0.16</td>
<td>    
1 – 40</td>
</tr>
<tr class="row-even"><td>    
itu_plasterboard</td>
<td>    
2.73</td>
<td>    
0</td>
<td>    
0.0085</td>
<td>    
0.9395</td>
<td>    
1 – 100</td>
</tr>
<tr class="row-odd"><td>    
itu_wood</td>
<td>    
1.99</td>
<td>    
0</td>
<td>    
0.0047</td>
<td>    
1.0718</td>
<td>    
0.001 – 100</td>
</tr>
<tr class="row-even"><td rowspan="2">    
itu_glass</td>
<td>    
6.31</td>
<td>    
0</td>
<td>    
0.0036</td>
<td>    
1.3394</td>
<td>    
0.1 – 100</td>
</tr>
<tr class="row-odd"><td>    
5.79</td>
<td>    
0</td>
<td>    
0.0004</td>
<td>    
1.658</td>
<td>    
220 – 450</td>
</tr>
<tr class="row-even"><td rowspan="2">    
itu_ceiling_board</td>
<td>    
1.48</td>
<td>    
0</td>
<td>    
0.0011</td>
<td>    
1.0750</td>
<td>    
1 – 100</td>
</tr>
<tr class="row-odd"><td>    
1.52</td>
<td>    
0</td>
<td>    
0.0029</td>
<td>    
1.029</td>
<td>    
220 – 450</td>
</tr>
<tr class="row-even"><td>    
itu_chipboard</td>
<td>    
2.58</td>
<td>    
0</td>
<td>    
0.0217</td>
<td>    
0.7800</td>
<td>    
1 – 100</td>
</tr>
<tr class="row-odd"><td>    
itu_plywood</td>
<td>    
2.71</td>
<td>    
0</td>
<td>    
0.33</td>
<td>    
0</td>
<td>    
1 – 40</td>
</tr>
<tr class="row-even"><td>    
itu_marble</td>
<td>    
7.074</td>
<td>    
0</td>
<td>    
0.0055</td>
<td>    
0.9262</td>
<td>    
1 – 60</td>
</tr>
<tr class="row-odd"><td>    
itu_floorboard</td>
<td>    
3.66</td>
<td>    
0</td>
<td>    
0.0044</td>
<td>    
1.3515</td>
<td>    
50 – 100</td>
</tr>
<tr class="row-even"><td>    
itu_metal</td>
<td>    
1</td>
<td>    
0</td>
<td>    
$10^7$</td>
<td>    
0</td>
<td>    
1 – 100</td>
</tr>
<tr class="row-odd"><td>    
itu_very_dry_ground</td>
<td>    
3</td>
<td>    
0</td>
<td>    
0.00015</td>
<td>    
2.52</td>
<td>    
1 – 10</td>
</tr>
<tr class="row-even"><td>    
itu_medium_dry_ground</td>
<td>    
15</td>
<td>    
-0.1</td>
<td>    
0.035</td>
<td>    
1.63</td>
<td>    
1 – 10</td>
</tr>
<tr class="row-odd"><td>    
itu_wet_ground</td>
<td>    
30</td>
<td>    
-0.4</td>
<td>    
0.15</td>
<td>    
1.30</td>
<td>    
1 – 10</td>
</tr>
</tbody>
</table>
<p id="custom-radio-materials">**Defining custom radio materials**
    
Custom radio materials can be implemented using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> class by specifying a relative permittivity
$\varepsilon_r$ and conductivity $\sigma$, as well as optional
parameters related to diffuse scattering, such as the scattering coefficient $S$,
cross-polarization discrimination coefficient $K_x$, and scattering pattern $f_\text{s}(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s})$.
Note that only non-magnetic materials with $\mu_r=1$ are currently allowed.
The following code snippet shows how to create a custom radio material.
```python
load_scene() # Load empty scene
custom_material = RadioMaterial("my_material",
                                relative_permittivity=2.0,
                                conductivity=5.0,
                                scattering_coefficient=0.3,
                                xpd_coefficient=0.1,
                                scattering_pattern=LambertianPattern())
```

    
It is also possible to define the properties of a material through a callback
function that computes the material properties
$(\varepsilon_r, \sigma)$ from the frequency:
```python
def my_material_callback(f_hz):
   relative_permittivity = compute_relative_permittivity(f_hz)
   conductivity = compute_conductivity(f_hz)
   return (relative_permittivity, conductivity)
custom_material = RadioMaterial("my_material",
                                frequency_update_callback=my_material_callback)
scene.add(custom_material)
```

    
Once defined, the custom material can be assigned to a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="sionna.rt.SceneObject">`SceneObject`</a> using its name:
```python
obj = scene.get("my_object") # obj is a SceneObject
obj.radio_material = "my_material" # "my_object" is made of "my_material"
```

    
or the material instance:
```python
obj = scene.get("my_object") # obj is a SceneObject
obj.radio_material = custom_material # "my_object" is made of "my_material"
```

    
The material parameters can be assigned to TensorFlow variables or tensors, such as
the output of a Keras layer defining a neural network. This allows one to make materials
trainable:
```python
mat = RadioMaterial("my_mat",
                    relative_permittivity= tf.Variable(2.1, dtype=tf.float32))
mat.conductivity = tf.Variable(0.0, dtype=tf.float32)
```
