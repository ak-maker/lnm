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
## Antenna Arrays<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#antenna-arrays" title="Permalink to this headline"></a>
### PlanarArray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#planararray" title="Permalink to this headline"></a>
## Antennas<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#antennas" title="Permalink to this headline"></a>
### Antenna<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#antenna" title="Permalink to this headline"></a>
### compute_gain<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#compute-gain" title="Permalink to this headline"></a>
### visualize<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#visualize" title="Permalink to this headline"></a>
### dipole_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#dipole-pattern" title="Permalink to this headline"></a>
### hw_dipole_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#hw-dipole-pattern" title="Permalink to this headline"></a>
  
  

### PlanarArray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#planararray" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``PlanarArray`(<em class="sig-param">`num_rows`</em>, <em class="sig-param">`num_cols`</em>, <em class="sig-param">`vertical_spacing`</em>, <em class="sig-param">`horizontal_spacing`</em>, <em class="sig-param">`pattern`</em>, <em class="sig-param">`polarization``=``None`</em>, <em class="sig-param">`polarization_model``=``2`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna_array.html#PlanarArray">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray" title="Permalink to this definition"></a>
    
Class implementing a planar antenna array
    
The antennas are regularly spaced, located in the y-z plane, and
numbered column-first from the top-left to bottom-right corner.
Parameters
 
- **num_rows** (<em>int</em>) – Number of rows
- **num_cols** (<em>int</em>) – Number of columns
- **vertical_spacing** (<em>float</em>) – Vertical antenna spacing [multiples of wavelength].
- **horizontal_spacing** (<em>float</em>) – Horizontal antenna spacing [multiples of wavelength].
- **pattern** (<em>str</em><em>, </em><em>callable</em><em>, or </em><em>length-2 sequence of callables</em>) – Antenna pattern. Either one of
[“iso”, “dipole”, “hw_dipole”, “tr38901”],
or a callable, or a length-2 sequence of callables defining
antenna patterns. In the latter case, the antennas are dual
polarized and each callable defines the antenna pattern
in one of the two orthogonal polarization directions.
An antenna pattern is a callable that takes as inputs vectors of
zenith and azimuth angles of the same length and returns for each
pair the corresponding zenith and azimuth patterns. See <a class="reference internal" href="../em_primer.html#equation-c">(14)</a> for
more detail.
- **polarization** (<em>str</em><em> or </em><em>None</em>) – Type of polarization. For single polarization, must be “V” (vertical)
or “H” (horizontal). For dual polarization, must be “VH” or “cross”.
Only needed if `pattern` is a string.
- **polarization_model** (<em>int</em><em>, </em><em>one of</em><em> [</em><em>1</em><em>,</em><em>2</em><em>]</em>) – Polarization model to be used. Options <cite>1</cite> and <cite>2</cite>
refer to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_1" title="sionna.rt.antenna.polarization_model_1">`polarization_model_1()`</a>
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_2" title="sionna.rt.antenna.polarization_model_2">`polarization_model_2()`</a>,
respectively.
Defaults to <cite>2</cite>.
- **dtype** (<em>tf.complex64</em><em> or </em><em>tf.complex128</em>) – Datatype used for all computations.
Defaults to <cite>tf.complex64</cite>.



<p class="rubric">Example
```python
array = PlanarArray(8,4, 0.5, 0.5, "tr38901", "VH")
array.show()
```

<a class="reference internal image-reference" href="../_images/antenna_array.png"><img alt="../_images/antenna_array.png" src="https://nvlabs.github.io/sionna/_images/antenna_array.png" style="width: 640.0px; height: 480.0px;" /></a>

<em class="property">`property` </em>`antenna`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray.antenna" title="Permalink to this definition"></a>
    
Get/set the antenna
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna" title="sionna.rt.Antenna">`Antenna`</a>




<em class="property">`property` </em>`array_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray.array_size" title="Permalink to this definition"></a>
    
Number of antennas in the array.
Dual-polarized antennas are counted as a single antenna.
Type
    
int (read-only)




<em class="property">`property` </em>`num_ant`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray.num_ant" title="Permalink to this definition"></a>
    
Number of linearly polarized antennas in the array.
Dual-polarized antennas are counted as two linearly polarized
antennas.
Type
    
int (read-only)




<em class="property">`property` </em>`positions`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray.positions" title="Permalink to this definition"></a>
    
Get/set  array of relative positions
$(x,y,z)$ [m] of each antenna (dual-polarized antennas are
counted as a single antenna and share the same position).
Type
    
[array_size, 3], <cite>tf.float</cite>




`rotated_positions`(<em class="sig-param">`orientation`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray.rotated_positions" title="Permalink to this definition"></a>
    
Get the antenna positions rotated according to `orientation`
Input
    
**orientation** (<em>[3], tf.float</em>) – Orientation $(\alpha, \beta, \gamma)$ [rad] specified
through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.

Output
    
<em>[array_size, 3]</em> – Rotated positions




`show`()<a class="reference internal" href="../_modules/sionna/rt/antenna_array.html#PlanarArray.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray.show" title="Permalink to this definition"></a>
    
Visualizes the antenna array
    
Antennas are depicted by markers that are annotated with the antenna
number. The marker is not related to the polarization of an antenna.
Output
    
`matplotlib.pyplot.Figure` – Figure depicting the antenna array




## Antennas<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#antennas" title="Permalink to this headline"></a>
    
We refer the user to the section “<a class="reference internal" href="../em_primer.html#far-field">Far Field of a Transmitting Antenna</a>” for various useful definitions and background on antenna modeling.
An <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna" title="sionna.rt.Antenna">`Antenna`</a> can be single- or dual-polarized and has for each polarization direction a possibly different antenna pattern.
    
An antenna pattern is defined as a function $f:(\theta,\varphi)\mapsto (C_\theta(\theta, \varphi), C_\varphi(\theta, \varphi))$
that maps a pair of zenith and azimuth angles to zenith and azimuth pattern values.
You can easily define your own pattern or use one of the predefined <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#patterns">patterns</a> below.
    
Transmitters (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>) and receivers (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>) are not equipped with an <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna" title="sionna.rt.Antenna">`Antenna`</a> but an <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a> that is composed of one or more antennas. All transmitters in a scene share the same <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a> which can be set through the scene property <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.tx_array" title="sionna.rt.Scene.tx_array">`tx_array`</a>. The same holds for all receivers whose <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a> can be set through <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.rx_array" title="sionna.rt.Scene.rx_array">`rx_array`</a>.

### Antenna<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#antenna" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``Antenna`(<em class="sig-param">`pattern`</em>, <em class="sig-param">`polarization``=``None`</em>, <em class="sig-param">`polarization_model``=``2`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#Antenna">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna" title="Permalink to this definition"></a>
    
Class implementing an antenna
    
Creates an antenna object with an either predefined or custom antenna
pattern. Can be single or dual polarized.
Parameters
 
- **pattern** (<em>str</em><em>, </em><em>callable</em><em>, or </em><em>length-2 sequence of callables</em>) – Antenna pattern. Either one of
[“iso”, “dipole”, “hw_dipole”, “tr38901”],
or a callable, or a length-2 sequence of callables defining
antenna patterns. In the latter case, the antenna is dual
polarized and each callable defines the antenna pattern
in one of the two orthogonal polarization directions.
An antenna pattern is a callable that takes as inputs vectors of
zenith and azimuth angles of the same length and returns for each
pair the corresponding zenith and azimuth patterns.
- **polarization** (<em>str</em><em> or </em><em>None</em>) – Type of polarization. For single polarization, must be “V” (vertical)
or “H” (horizontal). For dual polarization, must be “VH” or “cross”.
Only needed if `pattern` is a string.
- **polarization_model** (<em>int</em><em>, </em><em>one of</em><em> [</em><em>1</em><em>,</em><em>2</em><em>]</em>) – Polarization model to be used. Options <cite>1</cite> and <cite>2</cite>
refer to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_1" title="sionna.rt.antenna.polarization_model_1">`polarization_model_1()`</a>
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_2" title="sionna.rt.antenna.polarization_model_2">`polarization_model_2()`</a>,
respectively.
Defaults to <cite>2</cite>.
- **dtype** (<em>tf.complex64</em><em> or </em><em>tf.complex128</em>) – Datatype used for all computations.
Defaults to <cite>tf.complex64</cite>.



<p class="rubric">Example
```python
>>> Antenna("tr38901", "VH")
```
<em class="property">`property` </em>`patterns`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna.patterns" title="Permalink to this definition"></a>
    
Antenna patterns for one or two
polarization directions
Type
    
<cite>list</cite>, <cite>callable</cite>




### compute_gain<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#compute-gain" title="Permalink to this headline"></a>

`sionna.rt.antenna.``compute_gain`(<em class="sig-param">`pattern`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#compute_gain">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.compute_gain" title="Permalink to this definition"></a>
    
Computes the directivity, gain, and radiation efficiency of an antenna pattern
    
Given a function $f:(\theta,\varphi)\mapsto (C_\theta(\theta, \varphi), C_\varphi(\theta, \varphi))$
describing an antenna pattern <a class="reference internal" href="../em_primer.html#equation-c">(14)</a>, this function computes the gain $G$,
directivity $D$, and radiation efficiency $\eta_\text{rad}=G/D$
(see <a class="reference internal" href="../em_primer.html#equation-g">(12)</a> and text below).
Input
    
**pattern** (<em>callable</em>) – A callable that takes as inputs vectors of zenith and azimuth angles of the same
length and returns for each pair the corresponding zenith and azimuth patterns.

Output
 
- **D** (<em>float</em>) – Directivity $D$
- **G** (<em>float</em>) – Gain $G$
- **eta_rad** (<em>float</em>) – Radiation efficiency $\eta_\text{rad}$



<p class="rubric">Examples
```python
>>> compute_gain(tr38901_pattern)
(<tf.Tensor: shape=(), dtype=float32, numpy=9.606758>,
 <tf.Tensor: shape=(), dtype=float32, numpy=6.3095527>,
 <tf.Tensor: shape=(), dtype=float32, numpy=0.65678275>)
```


### visualize<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#visualize" title="Permalink to this headline"></a>

`sionna.rt.antenna.``visualize`(<em class="sig-param">`pattern`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#visualize">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.visualize" title="Permalink to this definition"></a>
    
Visualizes an antenna pattern
    
This function visualizes an antenna pattern with the help of three
figures showing the vertical and horizontal cuts as well as a
three-dimensional visualization of the antenna gain.
Input
    
**pattern** (<em>callable</em>) – A callable that takes as inputs vectors of zenith and azimuth angles
of the same length and returns for each pair the corresponding zenith
and azimuth patterns.

Output
 
- `matplotlib.pyplot.Figure` – Vertical cut of the antenna gain
- `matplotlib.pyplot.Figure` – Horizontal cut of the antenna gain
- `matplotlib.pyplot.Figure` – 3D visualization of the antenna gain



<p class="rubric">Examples
```python
>>> fig_v, fig_h, fig_3d = visualize(hw_dipole_pattern)
```

<a class="reference internal image-reference" href="../_images/pattern_vertical.png"><img alt="../_images/pattern_vertical.png" src="https://nvlabs.github.io/sionna/_images/pattern_vertical.png" style="width: 512.0px; height: 384.0px;" /></a>

<a class="reference internal image-reference" href="../_images/pattern_horizontal.png"><img alt="../_images/pattern_horizontal.png" src="https://nvlabs.github.io/sionna/_images/pattern_horizontal.png" style="width: 512.0px; height: 384.0px;" /></a>

<a class="reference internal image-reference" href="../_images/pattern_3d.png"><img alt="../_images/pattern_3d.png" src="https://nvlabs.github.io/sionna/_images/pattern_3d.png" style="width: 512.0px; height: 384.0px;" /></a>

### dipole_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#dipole-pattern" title="Permalink to this headline"></a>

`sionna.rt.antenna.``dipole_pattern`(<em class="sig-param">`theta`</em>, <em class="sig-param">`phi`</em>, <em class="sig-param">`slant_angle``=``0.0`</em>, <em class="sig-param">`polarization_model``=``2`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#dipole_pattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.dipole_pattern" title="Permalink to this definition"></a>
    
Short dipole pattern with linear polarizarion (Eq. 4-26a) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#balanis97" id="id21">[Balanis97]</a>
Input
 
- **theta** (<em>array_like, float</em>) – Zenith angles wrapped within [0,pi] [rad]
- **phi** (<em>array_like, float</em>) – Azimuth angles wrapped within [-pi, pi) [rad]
- **slant_angle** (<em>float</em>) – Slant angle of the linear polarization [rad].
A slant angle of zero means vertical polarization.
- **polarization_model** (<em>int, one of [1,2]</em>) – Polarization model to be used. Options <cite>1</cite> and <cite>2</cite>
refer to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_1" title="sionna.rt.antenna.polarization_model_1">`polarization_model_1()`</a>
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_2" title="sionna.rt.antenna.polarization_model_2">`polarization_model_2()`</a>,
respectively.
Defaults to <cite>2</cite>.
- **dtype** (<em>tf.complex64 or tf.complex128</em>) – Datatype.
Defaults to <cite>tf.complex64</cite>.


Output
 
- **c_theta** (<em>array_like, complex</em>) – Zenith pattern
- **c_phi** (<em>array_like, complex</em>) – Azimuth pattern




### hw_dipole_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#hw-dipole-pattern" title="Permalink to this headline"></a>

`sionna.rt.antenna.``hw_dipole_pattern`(<em class="sig-param">`theta`</em>, <em class="sig-param">`phi`</em>, <em class="sig-param">`slant_angle``=``0.0`</em>, <em class="sig-param">`polarization_model``=``2`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#hw_dipole_pattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.hw_dipole_pattern" title="Permalink to this definition"></a>
    
Half-wavelength dipole pattern with linear polarizarion (Eq. 4-84) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#balanis97" id="id22">[Balanis97]</a>
Input
 
- **theta** (<em>array_like, float</em>) – Zenith angles wrapped within [0,pi] [rad]
- **phi** (<em>array_like, float</em>) – Azimuth angles wrapped within [-pi, pi) [rad]
- **slant_angle** (<em>float</em>) – Slant angle of the linear polarization [rad].
A slant angle of zero means vertical polarization.
- **polarization_model** (<em>int, one of [1,2]</em>) – Polarization model to be used. Options <cite>1</cite> and <cite>2</cite>
refer to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_1" title="sionna.rt.antenna.polarization_model_1">`polarization_model_1()`</a>
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_2" title="sionna.rt.antenna.polarization_model_2">`polarization_model_2()`</a>,
respectively.
Defaults to <cite>2</cite>.
- **dtype** (<em>tf.complex64 or tf.complex128</em>) – Datatype.
Defaults to <cite>tf.complex64</cite>.


Output
 
- **c_theta** (<em>array_like, complex</em>) – Zenith pattern
- **c_phi** (<em>array_like, complex</em>) – Azimuth pattern




