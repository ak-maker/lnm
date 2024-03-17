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
## Coverage Maps<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#coverage-maps" title="Permalink to this headline"></a>
### CoverageMap<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#coveragemap" title="Permalink to this headline"></a>


### CoverageMap<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#coveragemap" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``CoverageMap`<a class="reference internal" href="../_modules/sionna/rt/coverage_map.html#CoverageMap">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="Permalink to this definition"></a>
    
Stores the simulated coverage maps
    
A coverage map is generated for the loaded scene for every transmitter using
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.coverage_map" title="sionna.rt.Scene.coverage_map">`coverage_map()`</a>. Please refer to the documentation of this function
for further details.
    
An instance of this class can be indexed like a tensor of rank three with
shape `[num_tx,` `num_cells_y,` `num_cells_x]`, i.e.:
```python
cm = scene.coverage_map()
print(cm[0])      # prints the coverage map for transmitter 0
print(cm[0,1,2])  # prints the value of the cell (1,2) for transmitter 0
```

    
where `scene` is the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a> loaded using
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.load_scene" title="sionna.rt.load_scene">`load_scene()`</a>.
<p class="rubric">Example
```python
import sionna
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver
scene = load_scene(sionna.rt.scene.munich)
# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=8,
                          num_cols=2,
                          vertical_spacing=0.7,
                          horizontal_spacing=0.5,
                          pattern="tr38901",
                          polarization="VH")
# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                          num_cols=1,
                          vertical_spacing=0.5,
                          horizontal_spacing=0.5,
                          pattern="dipole",
                          polarization="cross")
# Add a transmitters
tx = Transmitter(name="tx",
              position=[8.5,21,30],
              orientation=[0,0,0])
scene.add(tx)
tx.look_at([40,80,1.5])
# Compute coverage map
cm = scene.coverage_map(max_depth=8)
# Show coverage map
cm.show()
```


`as_tensor`()<a class="reference internal" href="../_modules/sionna/rt/coverage_map.html#CoverageMap.as_tensor">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.as_tensor" title="Permalink to this definition"></a>
    
Returns the coverage map as a tensor
Output
    
<em>[num_tx, num_cells_y, num_cells_x], tf.float</em> – The coverage map as a tensor




<em class="property">`property` </em>`cell_centers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.cell_centers" title="Permalink to this definition"></a>
    
Get the positions of the
centers of the cells in the global coordinate system
Type
    
[num_cells_y, num_cells_x, 3], tf.float




<em class="property">`property` </em>`cell_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.cell_size" title="Permalink to this definition"></a>
    
Get the resolution of the coverage map, i.e., width
(in the local X direction) and height (in the local Y direction) in
of the cells of the coverage map
Type
    
[2], tf.float




<em class="property">`property` </em>`center`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.center" title="Permalink to this definition"></a>
    
Get the center of the coverage map
Type
    
[3], tf.float




<em class="property">`property` </em>`num_cells_x`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.num_cells_x" title="Permalink to this definition"></a>
    
Get the number of cells along the local X-axis
Type
    
int




<em class="property">`property` </em>`num_cells_y`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.num_cells_y" title="Permalink to this definition"></a>
    
Get the number of cells along the local Y-axis
Type
    
int




<em class="property">`property` </em>`num_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.num_tx" title="Permalink to this definition"></a>
    
Get the number of transmitters
Type
    
int




<em class="property">`property` </em>`orientation`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.orientation" title="Permalink to this definition"></a>
    
Get the orientation of the coverage map
Type
    
[3], tf.float




`sample_positions`(<em class="sig-param">`batch_size`</em>, <em class="sig-param">`tx``=``0`</em>, <em class="sig-param">`min_gain_db``=``None`</em>, <em class="sig-param">`max_gain_db``=``None`</em>, <em class="sig-param">`min_dist``=``None`</em>, <em class="sig-param">`max_dist``=``None`</em>, <em class="sig-param">`center_pos``=``False`</em>)<a class="reference internal" href="../_modules/sionna/rt/coverage_map.html#CoverageMap.sample_positions">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.sample_positions" title="Permalink to this definition"></a>
    
Sample random user positions from a coverage map
    
For a given coverage map, `batch_size` random positions are sampled
such that the <em>expected</em>  path gain of this position is larger
than a given threshold `min_gain_db` or smaller than `max_gain_db`,
respectively.
Similarly, `min_dist` and `max_dist` define the minimum and maximum
distance of the random positions to the transmitter `tx`.
    
Note that due to the quantization of the coverage map into cells it is
not guaranteed that all above parameters are exactly fulfilled for a
returned position. This stems from the fact that every
individual cell of the coverage map describes the expected <em>average</em>
behavior of the surface within this cell. For instance, it may happen
that half of the selected cell is shadowed and, thus, no path to the
transmitter exists but the average path gain is still larger than the
given threshold. Please use `center_pos` = <cite>True</cite> to sample only
positions from the cell centers.
    
The above figure shows an example for random positions between 220m and
250m from the transmitter and a `max_gain_db` of -100 dB.
Keep in mind that the transmitter can have a different height than the
coverage map which also contributes to this distance.
For example if the transmitter is located 20m above the surface of the
coverage map and a `min_dist` of 20m is selected, also positions
directly below the transmitter are sampled.
Input
 
- **batch_size** (<em>int</em>) – Number of returned random positions
- **min_gain_db** (<em>float | None</em>) – Minimum path gain [dB]. Positions are only sampled from cells where
the path gain is larger or equal to this value.
Ignored if <cite>None</cite>.
Defaults to <cite>None</cite>.
- **max_gain_db** (<em>float | None</em>) – Maximum path gain [dB]. Positions are only sampled from cells where
the path gain is smaller or equal to this value.
Ignored if <cite>None</cite>.
Defaults to <cite>None</cite>.
- **min_dist** (<em>float | None</em>) – Minimum distance [m] from transmitter for all random positions.
Ignored if <cite>None</cite>.
Defaults to <cite>None</cite>.
- **max_dist** (<em>float | None</em>) – Maximum distance [m] from transmitter for all random positions.
Ignored if <cite>None</cite>.
Defaults to <cite>None</cite>.
- **tx** (<em>int | str</em>) – Index or name of the transmitter from whose coverage map
positions are sampled
- **center_pos** (<em>bool</em>) – If <cite>True</cite>, all returned positions are sampled from the cell center
(i.e., the grid of the coverage map). Otherwise, the positions are
randomly drawn from the surface of the cell.
Defaults to <cite>False</cite>.


Output
    
<em>[batch_size, 3], tf.float</em> – Random positions $(x,y,z)$ [m] that are in cells fulfilling the
above constraints w.r.t. distance and path gain




`show`(<em class="sig-param">`tx``=``0`</em>, <em class="sig-param">`vmin``=``None`</em>, <em class="sig-param">`vmax``=``None`</em>, <em class="sig-param">`show_tx``=``True`</em>)<a class="reference internal" href="../_modules/sionna/rt/coverage_map.html#CoverageMap.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.show" title="Permalink to this definition"></a>
    
Visualizes a coverage map
    
The position of the transmitter is indicated by a red “+” marker.
Input
 
- **tx** (<em>int | str</em>) – Index or name of the transmitter for which to show the coverage map
Defaults to 0.
- **vmin,vmax** (float | <cite>None</cite>) – Define the range of path gains that the colormap covers.
If set to <cite>None</cite>, then covers the complete range.
Defaults to <cite>None</cite>.
- **show_tx** (<em>bool</em>) – If set to <cite>True</cite>, then the position of the transmitter is shown.
Defaults to <cite>True</cite>.


Output
    
`Figure` – Figure showing the coverage map




<em class="property">`property` </em>`size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.size" title="Permalink to this definition"></a>
    
Get the size of the coverage map
Type
    
[2], tf.float




