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
### compute_paths<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#compute-paths" title="Permalink to this headline"></a>
### trace_paths<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#trace-paths" title="Permalink to this headline"></a>
### compute_fields<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#compute-fields" title="Permalink to this headline"></a>
  
  

### compute_paths<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#compute-paths" title="Permalink to this headline"></a>

`sionna.rt.Scene.``compute_paths`(<em class="sig-param">`self`</em>, <em class="sig-param">`max_depth``=``3`</em>, <em class="sig-param">`method``=``'fibonacci'`</em>, <em class="sig-param">`num_samples``=``1000000`</em>, <em class="sig-param">`los``=``True`</em>, <em class="sig-param">`reflection``=``True`</em>, <em class="sig-param">`diffraction``=``False`</em>, <em class="sig-param">`scattering``=``False`</em>, <em class="sig-param">`scat_keep_prob``=``0.001`</em>, <em class="sig-param">`edge_diffraction``=``False`</em>, <em class="sig-param">`check_scene``=``True`</em>, <em class="sig-param">`scat_random_phases``=``True`</em>, <em class="sig-param">`testing``=``False`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="Permalink to this definition"></a>
    
Computes propagation paths
    
This function computes propagation paths between the antennas of
all transmitters and receivers in the current scene.
For each propagation path $i$, the corresponding channel coefficient
$a_i$ and delay $\tau_i$, as well as the
angles of departure $(\theta_{\text{T},i}, \varphi_{\text{T},i})$
and arrival $(\theta_{\text{R},i}, \varphi_{\text{R},i})$ are returned.
For more detail, see <a class="reference internal" href="../em_primer.html#equation-h-final">(26)</a>.
Different propagation phenomena, such as line-of-sight, reflection, diffraction,
and diffuse scattering can be individually enabled/disabled.
    
If the scene is configured to use synthetic arrays
(<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.synthetic_array" title="sionna.rt.Scene.synthetic_array">`synthetic_array`</a> is <cite>True</cite>), transmitters and receivers
are modelled as if they had a single antenna located at their
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter.position" title="sionna.rt.Transmitter.position">`position`</a>. The channel responses for each
individual antenna of the arrays are then computed “synthetically” by applying
appropriate phase shifts. This reduces the complexity significantly
for large arrays. Time evolution of the channel coefficients can be simulated with
the help of the function <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.apply_doppler" title="sionna.rt.Paths.apply_doppler">`apply_doppler()`</a> of the returned
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> object.
    
The path computation consists of two main steps as shown in the below figure.
    
For a configured <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a>, the function first traces geometric propagation paths
using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.trace_paths" title="sionna.rt.Scene.trace_paths">`trace_paths()`</a>. This step is independent of the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> of the scene objects as well as the transmitters’ and receivers’
antenna <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna.patterns" title="sionna.rt.Antenna.patterns">`patterns`</a> and  <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter.orientation" title="sionna.rt.Transmitter.orientation">`orientation`</a>,
but depends on the selected propagation
phenomena, such as reflection, scattering, and diffraction. The traced paths
are then converted to EM fields by the function <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_fields" title="sionna.rt.Scene.compute_fields">`compute_fields()`</a>.
The resulting <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> object can be used to compute channel
impulse responses via <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.cir" title="sionna.rt.Paths.cir">`cir()`</a>. The advantage of separating path tracing
and field computation is that one can study the impact of different radio materials
by executing <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_fields" title="sionna.rt.Scene.compute_fields">`compute_fields()`</a> multiple times without
re-tracing the propagation paths. This can for example speed-up the calibration of scene parameters
by several orders of magnitude.
<p class="rubric">Example
```python
import sionna
from sionna.rt import load_scene, Camera, Transmitter, Receiver, PlanarArray
# Load example scene
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
# Create transmitter
tx = Transmitter(name="tx",
              position=[8.5,21,27],
              orientation=[0,0,0])
scene.add(tx)
# Create a receiver
rx = Receiver(name="rx",
           position=[45,90,1.5],
           orientation=[0,0,0])
scene.add(rx)
# TX points towards RX
tx.look_at(rx)
# Compute paths
paths = scene.compute_paths()
# Open preview showing paths
scene.preview(paths=paths, resolution=[1000,600])
```


Input
 
- **max_depth** (<em>int</em>) – Maximum depth (i.e., number of bounces) allowed for tracing the
paths. Defaults to 3.
- **method** (<em>str (“exhaustive”|”fibonacci”)</em>) – Ray tracing method to be used.
The “exhaustive” method tests all possible combinations of primitives.
This method is not compatible with scattering.
The “fibonacci” method uses a shoot-and-bounce approach to find
candidate chains of primitives. Initial ray directions are chosen
according to a Fibonacci lattice on the unit sphere. This method can be
applied to very large scenes. However, there is no guarantee that
all possible paths are found.
Defaults to “fibonacci”.
- **num_samples** (<em>int</em>) – Number of rays to trace in order to generate candidates with
the “fibonacci” method.
This number is split equally among the different transmitters
(when using synthetic arrays) or transmit antennas (when not using
synthetic arrays).
This parameter is ignored when using the exhaustive method.
Tracing more rays can lead to better precision
at the cost of increased memory requirements.
Defaults to 1e6.
- **los** (<em>bool</em>) – If set to <cite>True</cite>, then the LoS paths are computed.
Defaults to <cite>True</cite>.
- **reflection** (<em>bool</em>) – If set to <cite>True</cite>, then the reflected paths are computed.
Defaults to <cite>True</cite>.
- **diffraction** (<em>bool</em>) – If set to <cite>True</cite>, then the diffracted paths are computed.
Defaults to <cite>False</cite>.
- **scattering** (<em>bool</em>) – if set to <cite>True</cite>, then the scattered paths are computed.
Only works with the Fibonacci method.
Defaults to <cite>False</cite>.
- **scat_keep_prob** (<em>float</em>) – Probability with which a scattered path is kept.
This is helpful to reduce the number of computed scattered
paths, which might be prohibitively high in some scenes.
Must be in the range (0,1). Defaults to 0.001.
- **edge_diffraction** (<em>bool</em>) – If set to <cite>False</cite>, only diffraction on wedges, i.e., edges that
connect two primitives, is considered.
Defaults to <cite>False</cite>.
- **check_scene** (<em>bool</em>) – If set to <cite>True</cite>, checks that the scene is well configured before
computing the paths. This can add a significant overhead.
Defaults to <cite>True</cite>.
- **scat_random_phases** (<em>bool</em>) – If set to <cite>True</cite> and if scattering is enabled, random uniform phase
shifts are added to the scattered paths.
Defaults to <cite>True</cite>.
- **testing** (<em>bool</em>) – If set to <cite>True</cite>, then additional data is returned for testing.
Defaults to <cite>False</cite>.


Output
    
paths : <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> – Simulated paths



### trace_paths<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#trace-paths" title="Permalink to this headline"></a>

`sionna.rt.Scene.``trace_paths`(<em class="sig-param">`self`</em>, <em class="sig-param">`max_depth``=``3`</em>, <em class="sig-param">`method``=``'fibonacci'`</em>, <em class="sig-param">`num_samples``=``1000000`</em>, <em class="sig-param">`los``=``True`</em>, <em class="sig-param">`reflection``=``True`</em>, <em class="sig-param">`diffraction``=``False`</em>, <em class="sig-param">`scattering``=``False`</em>, <em class="sig-param">`scat_keep_prob``=``0.001`</em>, <em class="sig-param">`edge_diffraction``=``False`</em>, <em class="sig-param">`check_scene``=``True`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.trace_paths" title="Permalink to this definition"></a>
    
Computes the trajectories of the paths by shooting rays
    
The EM fields corresponding to the traced paths are not computed.
They can be computed using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_fields" title="sionna.rt.Scene.compute_fields">`compute_fields()`</a>:
```python
traced_paths = scene.trace_paths()
paths = scene.compute_fields(*traced_paths)
```

    
Path tracing is independent of the radio materials, antenna patterns,
and radio device orientations.
Therefore, a set of traced paths could be reused for different values
of these quantities, e.g., to calibrate the ray tracer.
This can enable significant resource savings as path tracing is
typically significantly more resource-intensive than field computation.
    
Note that <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a> does both path tracing and
field computation.
Input
 
- **max_depth** (<em>int</em>) – Maximum depth (i.e., number of interaction with objects in the scene)
allowed for tracing the paths.
Defaults to 3.
- **method** (<em>str (“exhaustive”|”fibonacci”)</em>) – Method to be used to list candidate paths.
The “exhaustive” method tests all possible combination of primitives as
paths. This method is not compatible with scattering.
The “fibonacci” method uses a shoot-and-bounce approach to find
candidate chains of primitives. Initial ray directions are arranged
in a Fibonacci lattice on the unit sphere. This method can be
applied to very large scenes. However, there is no guarantee that
all possible paths are found.
Defaults to “fibonacci”.
- **num_samples** (<em>int</em>) – Number of random rays to trace in order to generate candidates.
A large sample count may exhaust GPU memory.
Defaults to 1e6. Only needed if `method` is “fibonacci”.
- **los** (<em>bool</em>) – If set to <cite>True</cite>, then the LoS paths are computed.
Defaults to <cite>True</cite>.
- **reflection** (<em>bool</em>) – If set to <cite>True</cite>, then the reflected paths are computed.
Defaults to <cite>True</cite>.
- **diffraction** (<em>bool</em>) – If set to <cite>True</cite>, then the diffracted paths are computed.
Defaults to <cite>False</cite>.
- **scattering** (<em>bool</em>) – If set to <cite>True</cite>, then the scattered paths are computed.
Only works with the Fibonacci method.
Defaults to <cite>False</cite>.
- **scat_keep_prob** (<em>float</em>) – Probability with which to keep scattered paths.
This is helpful to reduce the number of scattered paths computed,
which might be prohibitively high in some setup.
Must be in the range (0,1).
Defaults to 0.001.
- **edge_diffraction** (<em>bool</em>) – If set to <cite>False</cite>, only diffraction on wedges, i.e., edges that
connect two primitives, is considered.
Defaults to <cite>False</cite>.
- **check_scene** (<em>bool</em>) – If set to <cite>True</cite>, checks that the scene is well configured before
computing the paths. This can add a significant overhead.
Defaults to <cite>True</cite>.


Output
 
- **spec_paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>) – Computed specular paths
- **diff_paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>) – Computed diffracted paths
- **scat_paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>) – Computed scattered paths
- **spec_paths_tmp** (`PathsTmpData`) – Additional data required to compute the EM fields of the specular
paths
- **diff_paths_tmp** (`PathsTmpData`) – Additional data required to compute the EM fields of the diffracted
paths
- **scat_paths_tmp** (`PathsTmpData`) – Additional data required to compute the EM fields of the scattered
paths




### compute_fields<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#compute-fields" title="Permalink to this headline"></a>

`sionna.rt.Scene.``compute_fields`(<em class="sig-param">`self`</em>, <em class="sig-param">`spec_paths`</em>, <em class="sig-param">`diff_paths`</em>, <em class="sig-param">`scat_paths`</em>, <em class="sig-param">`spec_paths_tmp`</em>, <em class="sig-param">`diff_paths_tmp`</em>, <em class="sig-param">`scat_paths_tmp`</em>, <em class="sig-param">`check_scene``=``True`</em>, <em class="sig-param">`scat_random_phases``=``True`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_fields" title="Permalink to this definition"></a>
    
Computes the EM fields corresponding to traced paths
    
Paths can be traced using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.trace_paths" title="sionna.rt.Scene.trace_paths">`trace_paths()`</a>.
This method can then be used to finalize the paths calculation by
computing the corresponding fields:
```python
traced_paths = scene.trace_paths()
paths = scene.compute_fields(*traced_paths)
```

    
Paths tracing is independent from the radio materials, antenna patterns,
and radio devices orientations.
Therefore, a set of traced paths could be reused for different values
of these quantities, e.g., to calibrate the ray tracer.
This can enable significant resource savings as paths tracing is
typically significantly more resource-intensive than field computation.
    
Note that <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a> does both tracing and
field computation.
Input
 
- **spec_paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>) – Specular paths
- **diff_paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>) – Diffracted paths
- **scat_paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>) – Scattered paths
- **spec_paths_tmp** (`PathsTmpData`) – Additional data required to compute the EM fields of the specular
paths
- **diff_paths_tmp** (`PathsTmpData`) – Additional data required to compute the EM fields of the diffracted
paths
- **scat_paths_tmp** (`PathsTmpData`) – Additional data required to compute the EM fields of the scattered
paths
- **check_scene** (<em>bool</em>) – If set to <cite>True</cite>, checks that the scene is well configured before
computing the paths. This can add a significant overhead.
Defaults to <cite>True</cite>.
- **scat_random_phases** (<em>bool</em>) – If set to <cite>True</cite> and if scattering is enabled, random uniform phase
shifts are added to the scattered paths.
Defaults to <cite>True</cite>.


Output
    
**paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>) – Computed paths



