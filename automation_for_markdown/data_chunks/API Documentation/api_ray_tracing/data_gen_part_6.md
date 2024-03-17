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
## Paths<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#paths" title="Permalink to this headline"></a>
### Paths<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#id13" title="Permalink to this headline"></a>
## Coverage Maps<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#coverage-maps" title="Permalink to this headline"></a>
  
  

### Paths<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#id13" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``Paths`<a class="reference internal" href="../_modules/sionna/rt/paths.html#Paths">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="Permalink to this definition"></a>
    
Stores the simulated propagation paths
    
Paths are generated for the loaded scene using
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a>. Please refer to the
documentation of this function for further details.
These paths can then be used to compute channel impulse responses:
```python
paths = scene.compute_paths()
a, tau = paths.cir()
```

    
where `scene` is the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a> loaded using
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.load_scene" title="sionna.rt.load_scene">`load_scene()`</a>.

<em class="property">`property` </em>`a`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.a" title="Permalink to this definition"></a>
    
Passband channel coefficients $a_i$ of each path as defined in <a class="reference internal" href="../em_primer.html#equation-h-final">(26)</a>.
Type
    
[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], tf.complex




`apply_doppler`(<em class="sig-param">`sampling_frequency`</em>, <em class="sig-param">`num_time_steps`</em>, <em class="sig-param">`tx_velocities``=``(0.0,` `0.0,` `0.0)`</em>, <em class="sig-param">`rx_velocities``=``(0.0,` `0.0,` `0.0)`</em>)<a class="reference internal" href="../_modules/sionna/rt/paths.html#Paths.apply_doppler">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.apply_doppler" title="Permalink to this definition"></a>
    
Apply Doppler shifts corresponding to input transmitters and receivers
velocities.
    
This function replaces the last dimension of the tensor storing the
paths coefficients <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.a" title="sionna.rt.Paths.a">`a`</a>, which stores the the temporal evolution of
the channel, with a dimension of size `num_time_steps` computed
according to the input velocities.
    
Time evolution of the channel coefficients is simulated by computing the
Doppler shift due to movements of the transmitter and receiver. If we denote by
$\mathbf{v}_{\text{T}}\in\mathbb{R}^3$ and $\mathbf{v}_{\text{R}}\in\mathbb{R}^3$
the velocity vectors of the transmitter and receiver, respectively, the Doppler shifts are computed as

$$
\begin{split}f_{\text{T}, i} &= \frac{\hat{\mathbf{r}}(\theta_{\text{T},i}, \varphi_{\text{T},i})^\mathsf{T}\mathbf{v}_{\text{T}}}{\lambda}\qquad \text{[Hz]}\\
f_{\text{R}, i} &= \frac{\hat{\mathbf{r}}(\theta_{\text{R},i}, \varphi_{\text{R},i})^\mathsf{T}\mathbf{v}_{\text{R}}}{\lambda}\qquad \text{[Hz]}\end{split}
$$
    
for an arbitrary path $i$, where $(\theta_{\text{T},i}, \varphi_{\text{T},i})$ are the AoDs,
$(\theta_{\text{R},i}, \varphi_{\text{R},i})$ are the AoAs, and $\lambda$ is the wavelength.
This leads to the time-dependent path coefficient

$$
a_i(t) = a_i e^{j2\pi(f_{\text{T}, i}+f_{\text{R}, i})t}.
$$
    
Note that this model is only valid as long as the AoDs, AoAs, and path delay do not change.
    
When this function is called multiple times, it overwrites the previous
time steps dimension.
Input
 
- **sampling_frequency** (<em>float</em>) – Frequency [Hz] at which the channel impulse response is sampled
- **num_time_steps** (<em>int</em>) – Number of time steps.
- **tx_velocities** ([batch_size, num_tx, 3] or broadcastable, tf.float | <cite>None</cite>) – Velocity vectors $(v_\text{x}, v_\text{y}, v_\text{z})$ of all
transmitters [m/s].
Defaults to <cite>[0,0,0]</cite>.
- **rx_velocities** ([batch_size, num_tx, 3] or broadcastable, tf.float | <cite>None</cite>) – Velocity vectors $(v_\text{x}, v_\text{y}, v_\text{z})$ of all
receivers [m/s].
Defaults to <cite>[0,0,0]</cite>.





`cir`(<em class="sig-param">`los``=``True`</em>, <em class="sig-param">`reflection``=``True`</em>, <em class="sig-param">`diffraction``=``True`</em>, <em class="sig-param">`scattering``=``True`</em>, <em class="sig-param">`num_paths``=``None`</em>)<a class="reference internal" href="../_modules/sionna/rt/paths.html#Paths.cir">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.cir" title="Permalink to this definition"></a>
    
Returns the baseband equivalent channel impulse response <a class="reference internal" href="../em_primer.html#equation-h-b">(28)</a>
which can be used for link simulations by other Sionna components.
    
The baseband equivalent channel coefficients $a^{\text{b}}_{i}$
are computed as :

$$
a^{\text{b}}_{i} = a_{i} e^{-j2 \pi f \tau_{i}}
$$
    
where $i$ is the index of an arbitrary path, $a_{i}$
is the passband path coefficient (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.a" title="sionna.rt.Paths.a">`a`</a>),
$\tau_{i}$ is the path delay (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.tau" title="sionna.rt.Paths.tau">`tau`</a>),
and $f$ is the carrier frequency.
    
Note: For the paths of a given type to be returned (LoS, reflection, etc.), they
must have been previously computed by <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a>, i.e.,
the corresponding flags must have been set to <cite>True</cite>.
Input
 
- **los** (<em>bool</em>) – If set to <cite>False</cite>, LoS paths are not returned.
Defaults to <cite>True</cite>.
- **reflection** (<em>bool</em>) – If set to <cite>False</cite>, specular paths are not returned.
Defaults to <cite>True</cite>.
- **diffraction** (<em>bool</em>) – If set to <cite>False</cite>, diffracted paths are not returned.
Defaults to <cite>True</cite>.
- **scattering** (<em>bool</em>) – If set to <cite>False</cite>, scattered paths are not returned.
Defaults to <cite>True</cite>.
- **num_paths** (int or <cite>None</cite>) – All CIRs are either zero-padded or cropped to the largest
`num_paths` paths.
Defaults to <cite>None</cite> which means that no padding or cropping is done.


Output
 
- **a** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float</em>) – Path delays





`export`(<em class="sig-param">`filename`</em>)<a class="reference internal" href="../_modules/sionna/rt/paths.html#Paths.export">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.export" title="Permalink to this definition"></a>
    
Saves the paths as an OBJ file for visualisation, e.g., in Blender
Input
    
**filename** (<em>str</em>) – Path and name of the file




`from_dict`(<em class="sig-param">`data_dict`</em>)<a class="reference internal" href="../_modules/sionna/rt/paths.html#Paths.from_dict">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.from_dict" title="Permalink to this definition"></a>
    
Set the paths from a dictionnary which values are tensors
    
The format of the dictionnary is expected to be the same as the one
returned by <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.to_dict" title="sionna.rt.Paths.to_dict">`to_dict()`</a>.
Input
    
**data_dict** (<cite>dict</cite>)




<em class="property">`property` </em>`mask`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.mask" title="Permalink to this definition"></a>
    
Set to <cite>False</cite> for non-existent paths.
When there are multiple transmitters or receivers, path counts may vary between links. This is used to identify non-existent paths.
For such paths, the channel coefficient is set to <cite>0</cite> and the delay to <cite>-1</cite>.
Type
    
[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.bool




<em class="property">`property` </em>`normalize_delays`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.normalize_delays" title="Permalink to this definition"></a>
    
Set to <cite>True</cite> to normalize path delays such that the first path
between any pair of antennas of a transmitter and receiver arrives at
`tau` `=` `0`. Defaults to <cite>True</cite>.
Type
    
bool




<em class="property">`property` </em>`phi_r`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.phi_r" title="Permalink to this definition"></a>
    
Azimuth angles of arrival [rad]
Type
    
[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float




<em class="property">`property` </em>`phi_t`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.phi_t" title="Permalink to this definition"></a>
    
Azimuth angles of departure [rad]
Type
    
[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float




<em class="property">`property` </em>`reverse_direction`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.reverse_direction" title="Permalink to this definition"></a>
    
If set to <cite>True</cite>, swaps receivers and transmitters
Type
    
bool




<em class="property">`property` </em>`tau`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.tau" title="Permalink to this definition"></a>
    
Propagation delay $\tau_i$ [s] of each path as defined in <a class="reference internal" href="../em_primer.html#equation-h-final">(26)</a>.
Type
    
[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float




<em class="property">`property` </em>`theta_r`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.theta_r" title="Permalink to this definition"></a>
    
Zenith angles of arrival [rad]
Type
    
[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float




<em class="property">`property` </em>`theta_t`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.theta_t" title="Permalink to this definition"></a>
    
Zenith  angles of departure [rad]
Type
    
[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float




`to_dict`()<a class="reference internal" href="../_modules/sionna/rt/paths.html#Paths.to_dict">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.to_dict" title="Permalink to this definition"></a>
    
Returns the properties of the paths as a dictionnary which values are
tensors
Output
    
<cite>dict</cite>




<em class="property">`property` </em>`types`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.types" title="Permalink to this definition"></a>
    
Type of the paths:
 
- 0 : LoS
- 1 : Reflected
- 2 : Diffracted
- 3 : Scattered

Type
    
[batch_size, max_num_paths], tf.int




## Coverage Maps<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#coverage-maps" title="Permalink to this headline"></a>
    
A coverage map describes the received power from a specific transmitter at every point on a plane.
In other words, for a given transmitter, it associates every point on a surface  with the power that a receiver with
a specific orientation would observe at this point. A coverage map is not uniquely defined as it depends on
the transmit and receive arrays and their respective antenna patterns, the transmitter and receiver orientations, as well as
transmit precoding and receive combining vectors. Moreover, a coverage map is not continuous but discrete because the plane
needs to be quantized into small rectangular bins.
    
In Sionna, coverage maps are computed with the help of the function <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.coverage_map" title="sionna.rt.Scene.coverage_map">`coverage_map()`</a> which returns an instance of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a>. They can be visualized by providing them either as arguments to the functions <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render" title="sionna.rt.Scene.render">`render()`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render_to_file" title="sionna.rt.Scene.render_to_file">`render_to_file()`</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>, or by using the class method <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.show" title="sionna.rt.CoverageMap.show">`show()`</a>.
    
A very useful feature is <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.sample_positions" title="sionna.rt.CoverageMap.sample_positions">`sample_positions()`</a> which allows sampling
of random positions within the scene that have sufficient coverage from a specific transmitter.
This feature is used in the <a class="reference external" href="../examples/Sionna_Ray_Tracing_Introduction.html">Sionna Ray Tracing Tutorial</a> to generate a dataset of channel impulse responses
for link-level simulations.

