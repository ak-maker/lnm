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
## Scene<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#scene" title="Permalink to this headline"></a>
### preview<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#preview" title="Permalink to this headline"></a>
### render<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#render" title="Permalink to this headline"></a>
### render_to_file<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#render-to-file" title="Permalink to this headline"></a>
## Example Scenes<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#example-scenes" title="Permalink to this headline"></a>
### floor_wall<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#floor-wall" title="Permalink to this headline"></a>
### simple_street_canyon<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#simple-street-canyon" title="Permalink to this headline"></a>
### etoile<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#etoile" title="Permalink to this headline"></a>
### munich<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#munich" title="Permalink to this headline"></a>
### simple_wedge<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#simple-wedge" title="Permalink to this headline"></a>
### simple_reflector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#simple-reflector" title="Permalink to this headline"></a>
### double_reflector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#double-reflector" title="Permalink to this headline"></a>
### triple_reflector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#triple-reflector" title="Permalink to this headline"></a>
### Box<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#box" title="Permalink to this headline"></a>
## Paths<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#paths" title="Permalink to this headline"></a>
  
  

### preview<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#preview" title="Permalink to this headline"></a>

`sionna.rt.Scene.``preview`(<em class="sig-param">`paths``=``None`</em>, <em class="sig-param">`show_paths``=``True`</em>, <em class="sig-param">`show_devices``=``True`</em>, <em class="sig-param">`coverage_map``=``None`</em>, <em class="sig-param">`cm_tx``=``0`</em>, <em class="sig-param">`cm_vmin``=``None`</em>, <em class="sig-param">`cm_vmax``=``None`</em>, <em class="sig-param">`resolution``=``(655,` `500)`</em>, <em class="sig-param">`fov``=``45`</em>, <em class="sig-param">`background``=``'#ffffff'`</em>, <em class="sig-param">`clip_at``=``None`</em>, <em class="sig-param">`clip_plane_orientation``=``(0,` `0,` `-` `1)`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="Permalink to this definition"></a>
    
In an interactive notebook environment, opens an interactive 3D
viewer of the scene.
    
The returned value of this method must be the last line of
the cell so that it is displayed. For example:
```python
fig = scene.preview()
# ...
fig
```

    
Or simply:
```python
scene.preview()
```

    
Color coding:
 
- Green: Receiver
- Blue: Transmitter

    
Controls:
 
- Mouse left: Rotate
- Scroll wheel: Zoom
- Mouse right: Move

Input
 
- **paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> | <cite>None</cite>) – Simulated paths generated by
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a> or <cite>None</cite>.
If <cite>None</cite>, only the scene is rendered.
Defaults to <cite>None</cite>.
- **show_paths** (<em>bool</em>) – If <cite>paths</cite> is not <cite>None</cite>, shows the paths.
Defaults to <cite>True</cite>.
- **show_devices** (<em>bool</em>) – If set to <cite>True</cite>, shows the radio devices.
Defaults to <cite>True</cite>.
- **show_orientations** (<em>bool</em>) – If <cite>show_devices</cite> is <cite>True</cite>, shows the radio devices orientations.
Defaults to <cite>False</cite>.
- **coverage_map** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a> | <cite>None</cite>) – An optional coverage map to overlay in the scene for visualization.
Defaults to <cite>None</cite>.
- **cm_tx** (<em>int | str</em>) – When <cite>coverage_map</cite> is specified, controls which of the transmitters
to display the coverage map for. Either the transmitter’s name
or index can be given.
Defaults to <cite>0</cite>.
- **cm_db_scale** (<em>bool</em>) – Use logarithmic scale for coverage map visualization, i.e. the
coverage values are mapped with:
$y = 10 \cdot \log_{10}(x)$.
Defaults to <cite>True</cite>.
- **cm_vmin, cm_vmax** (<em>floot | None</em>) – For coverage map visualization, defines the range of path gains that
the colormap covers.
These parameters should be provided in dB if `cm_db_scale` is
set to <cite>True</cite>, or in linear scale otherwise.
If set to None, then covers the complete range.
Defaults to <cite>None</cite>.
- **resolution** (<em>[2], int</em>) – Size of the viewer figure.
Defaults to <cite>[655, 500]</cite>.
- **fov** (<em>float</em>) – Field of view, in degrees.
Defaults to 45°.
- **background** (<em>str</em>) – Background color in hex format prefixed by ‘#’.
Defaults to ‘#ffffff’ (white).
- **clip_at** (<em>float</em>) – If not <cite>None</cite>, the scene preview will be clipped (cut) by a plane
with normal orientation `clip_plane_orientation` and offset `clip_at`.
That means that everything <em>behind</em> the plane becomes invisible.
This allows visualizing the interior of meshes, such as buildings.
Defaults to <cite>None</cite>.
- **clip_plane_orientation** (<em>tuple[float, float, float]</em>) – Normal vector of the clipping plane.
Defaults to (0,0,-1).




### render<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#render" title="Permalink to this headline"></a>

`sionna.rt.Scene.``render`(<em class="sig-param">`camera`</em>, <em class="sig-param">`paths``=``None`</em>, <em class="sig-param">`show_paths``=``True`</em>, <em class="sig-param">`show_devices``=``True`</em>, <em class="sig-param">`coverage_map``=``None`</em>, <em class="sig-param">`cm_tx``=``0`</em>, <em class="sig-param">`cm_vmin``=``None`</em>, <em class="sig-param">`cm_vmax``=``None`</em>, <em class="sig-param">`cm_show_color_bar``=``True`</em>, <em class="sig-param">`num_samples``=``512`</em>, <em class="sig-param">`resolution``=``(655,` `500)`</em>, <em class="sig-param">`fov``=``45`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render" title="Permalink to this definition"></a>
    
Renders the scene from the viewpoint of a camera or the interactive
viewer
Input
 
- **camera** (str | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>) – The name or instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>.
If an interactive viewer was opened with
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>, set to <cite>“preview”</cite> to use its
viewpoint.
- **paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> | <cite>None</cite>) – Simulated paths generated by
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a> or <cite>None</cite>.
If <cite>None</cite>, only the scene is rendered.
Defaults to <cite>None</cite>.
- **show_paths** (<em>bool</em>) – If <cite>paths</cite> is not <cite>None</cite>, shows the paths.
Defaults to <cite>True</cite>.
- **show_devices** (<em>bool</em>) – If <cite>paths</cite> is not <cite>None</cite>, shows the radio devices.
Defaults to <cite>True</cite>.
- **coverage_map** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a> | <cite>None</cite>) – An optional coverage map to overlay in the scene for visualization.
Defaults to <cite>None</cite>.
- **cm_tx** (<em>int | str</em>) – When <cite>coverage_map</cite> is specified, controls which of the transmitters
to display the coverage map for. Either the transmitter’s name
or index can be given.
Defaults to <cite>0</cite>.
- **cm_db_scale** (<em>bool</em>) – Use logarithmic scale for coverage map visualization, i.e. the
coverage values are mapped with:
$y = 10 \cdot \log_{10}(x)$.
Defaults to <cite>True</cite>.
- **cm_vmin, cm_vmax** (<em>float | None</em>) – For coverage map visualization, defines the range of path gains that
the colormap covers.
These parameters should be provided in dB if `cm_db_scale` is
set to <cite>True</cite>, or in linear scale otherwise.
If set to None, then covers the complete range.
Defaults to <cite>None</cite>.
- **cm_show_color_bar** (<em>bool</em>) – For coverage map visualization, show the color bar describing the
color mapping used next to the rendering.
Defaults to <cite>True</cite>.
- **num_samples** (<em>int</em>) – Number of rays thrown per pixel.
Defaults to 512.
- **resolution** (<em>[2], int</em>) – Size of the rendered figure.
Defaults to <cite>[655, 500]</cite>.
- **fov** (<em>float</em>) – Field of view, in degrees.
Defaults to 45°.


Output
    
`Figure` – Rendered image



### render_to_file<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#render-to-file" title="Permalink to this headline"></a>

`sionna.rt.Scene.``render_to_file`(<em class="sig-param">`camera`</em>, <em class="sig-param">`filename`</em>, <em class="sig-param">`paths``=``None`</em>, <em class="sig-param">`show_paths``=``True`</em>, <em class="sig-param">`show_devices``=``True`</em>, <em class="sig-param">`coverage_map``=``None`</em>, <em class="sig-param">`cm_tx``=``0`</em>, <em class="sig-param">`cm_db_scale``=``True`</em>, <em class="sig-param">`cm_vmin``=``None`</em>, <em class="sig-param">`cm_vmax``=``None`</em>, <em class="sig-param">`num_samples``=``512`</em>, <em class="sig-param">`resolution``=``(655,` `500)`</em>, <em class="sig-param">`fov``=``45`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render_to_file" title="Permalink to this definition"></a>
    
Renders the scene from the viewpoint of a camera or the interactive
viewer, and saves the resulting image
Input
 
- **camera** (str | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>) – The name or instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>.
If an interactive viewer was opened with
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>, set to <cite>“preview”</cite> to use its
viewpoint.
- **filename** (<em>str</em>) – Filename for saving the rendered image, e.g., “my_scene.png”
- **paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> | <cite>None</cite>) – Simulated paths generated by
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a> or <cite>None</cite>.
If <cite>None</cite>, only the scene is rendered.
Defaults to <cite>None</cite>.
- **show_paths** (<em>bool</em>) – If <cite>paths</cite> is not <cite>None</cite>, shows the paths.
Defaults to <cite>True</cite>.
- **show_devices** (<em>bool</em>) – If <cite>paths</cite> is not <cite>None</cite>, shows the radio devices.
Defaults to <cite>True</cite>.
- **coverage_map** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a> | <cite>None</cite>) – An optional coverage map to overlay in the scene for visualization.
Defaults to <cite>None</cite>.
- **cm_tx** (<em>int | str</em>) – When <cite>coverage_map</cite> is specified, controls which of the transmitters
to display the coverage map for. Either the transmitter’s name
or index can be given.
Defaults to <cite>0</cite>.
- **cm_db_scale** (<em>bool</em>) – Use logarithmic scale for coverage map visualization, i.e. the
coverage values are mapped with:
$y = 10 \cdot \log_{10}(x)$.
Defaults to <cite>True</cite>.
- **cm_vmin, cm_vmax** (<em>float | None</em>) – For coverage map visualization, defines the range of path gains that
the colormap covers.
These parameters should be provided in dB if `cm_db_scale` is
set to <cite>True</cite>, or in linear scale otherwise.
If set to None, then covers the complete range.
Defaults to <cite>None</cite>.
- **num_samples** (<em>int</em>) – Number of rays thrown per pixel.
Defaults to 512.
- **resolution** (<em>[2], int</em>) – Size of the rendered figure.
Defaults to <cite>[655, 500]</cite>.
- **fov** (<em>float</em>) – Field of view, in degrees.
Defaults to 45°.




## Example Scenes<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#example-scenes" title="Permalink to this headline"></a>
    
Sionna has several integrated scenes that are listed below.
They can be loaded and used as follows:
```python
scene = load_scene(sionna.rt.scene.etoile)
scene.preview()
```
### floor_wall<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#floor-wall" title="Permalink to this headline"></a>

`sionna.rt.scene.``floor_wall`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.floor_wall" title="Permalink to this definition"></a>
    
Example scene containing a ground plane and a vertical wall

    
(<a class="reference external" href="https://drive.google.com/file/d/1djXBj3VYLT4_bQpmp4vR6o6agGmv_p1F/view?usp=share_link">Blender file</a>)

### simple_street_canyon<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#simple-street-canyon" title="Permalink to this headline"></a>

`sionna.rt.scene.``simple_street_canyon`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.simple_street_canyon" title="Permalink to this definition"></a>
    
Example scene containing a few rectangular building blocks and a ground plane

    
(<a class="reference external" href="https://drive.google.com/file/d/1_1nsLtSC8cy1QfRHAN_JetT3rPP21tNb/view?usp=share_link">Blender file</a>)

### etoile<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#etoile" title="Permalink to this headline"></a>

`sionna.rt.scene.``etoile`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.etoile" title="Permalink to this definition"></a>
    
Example scene containing the area around the Arc de Triomphe in Paris
The scene was created with data downloaded from <a class="reference external" href="https://www.openstreetmap.org">OpenStreetMap</a> and
the help of <a class="reference external" href="https://www.blender.org">Blender</a> and the <a class="reference external" href="https://github.com/vvoovv/blender-osm">Blender-OSM</a>
and <a class="reference external" href="https://github.com/mitsuba-renderer/mitsuba-blender">Mitsuba Blender</a> add-ons.
The data is licensed under the <a class="reference external" href="https://openstreetmap.org/copyright">Open Data Commons Open Database License (ODbL)</a>.

    
(<a class="reference external" href="https://drive.google.com/file/d/1bamQ67lLGZHTfNmcVajQDmq2oiSY8FEn/view?usp=share_link">Blender file</a>)

### munich<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#munich" title="Permalink to this headline"></a>

`sionna.rt.scene.``munich`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.munich" title="Permalink to this definition"></a>
    
Example scene containing the area around the Frauenkirche in Munich
The scene was created with data downloaded from <a class="reference external" href="https://www.openstreetmap.org">OpenStreetMap</a> and
the help of <a class="reference external" href="https://www.blender.org">Blender</a> and the <a class="reference external" href="https://github.com/vvoovv/blender-osm">Blender-OSM</a>
and <a class="reference external" href="https://github.com/mitsuba-renderer/mitsuba-blender">Mitsuba Blender</a> add-ons.
The data is licensed under the <a class="reference external" href="https://openstreetmap.org/copyright">Open Data Commons Open Database License (ODbL)</a>.

    
(<a class="reference external" href="https://drive.google.com/file/d/15WrvMGrPWsoVKYvDG6Ab7btq-ktTCGR1/view?usp=share_link">Blender file</a>)

### simple_wedge<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#simple-wedge" title="Permalink to this headline"></a>

`sionna.rt.scene.``simple_wedge`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.simple_wedge" title="Permalink to this definition"></a>
    
Example scene containing a wedge with a $90^{\circ}$ opening angle

    
(<a class="reference external" href="https://drive.google.com/file/d/1RnJoYzXKkILMEmf-UVSsyjq-EowU6JRA/view?usp=share_link">Blender file</a>)

### simple_reflector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#simple-reflector" title="Permalink to this headline"></a>

`sionna.rt.scene.``simple_reflector`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.simple_reflector" title="Permalink to this definition"></a>
    
Example scene containing a metallic square

    
(<a class="reference external" href="https://drive.google.com/file/d/1iYPD11zAAMj0gNUKv_nv6QdLhOJcPpIa/view?usp=share_link">Blender file</a>)

### double_reflector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#double-reflector" title="Permalink to this headline"></a>

`sionna.rt.scene.``double_reflector`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.double_reflector" title="Permalink to this definition"></a>
    
Example scene containing two metallic squares

    
(<a class="reference external" href="https://drive.google.com/file/d/1K2ZUYHPPkrq9iUauJtInRu7x2r16D1zN/view?usp=share_link">Blender file</a>)

### triple_reflector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#triple-reflector" title="Permalink to this headline"></a>

`sionna.rt.scene.``triple_reflector`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.triple_reflector" title="Permalink to this definition"></a>
    
Example scene containing three metallic rectangles

    
(<a class="reference external" href="https://drive.google.com/file/d/1l95_0U2b3cEVtz3G8mQxuLxy8xiPsVID/view?usp=share_link">Blender file</a>)

### Box<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#box" title="Permalink to this headline"></a>

`sionna.rt.scene.``box`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.box" title="Permalink to this definition"></a>
    
Example scene containing a metallic box

    
(<a class="reference external" href="https://drive.google.com/file/d/1pywetyKr0HBz3aSYpkmykGnjs_1JMsHY/view?usp=share_link">Blender file</a>)
## Paths<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#paths" title="Permalink to this headline"></a>
    
A propagation path $i$ starts at a transmit antenna and ends at a receive antenna. It is described by
its channel coefficient $a_i$ and delay $\tau_i$, as well as the
angles of departure $(\theta_{\text{T},i}, \varphi_{\text{T},i})$
and arrival $(\theta_{\text{R},i}, \varphi_{\text{R},i})$.
For more detail, see the <a class="reference external" href="../em_primer.html">Primer on Electromagnetics</a>.
    
In Sionna, paths are computed with the help of the function <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a> which returns an instance of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>. Paths can be visualized by providing them as arguments to the functions <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render" title="sionna.rt.Scene.render">`render()`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render_to_file" title="sionna.rt.Scene.render_to_file">`render_to_file()`</a>, or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>.
    
Channel impulse responses (CIRs) can be obtained with <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.cir" title="sionna.rt.Paths.cir">`cir()`</a> which can
then be used for link-level simulations. This is for example done in the <a class="reference external" href="../examples/Sionna_Ray_Tracing_Introduction.html">Sionna Ray Tracing Tutorial</a>.

