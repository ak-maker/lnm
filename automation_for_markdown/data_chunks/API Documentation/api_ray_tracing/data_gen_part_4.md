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
### coverage_map<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#coverage-map" title="Permalink to this headline"></a>
### load_scene<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#load-scene" title="Permalink to this headline"></a>
  
  

### coverage_map<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#coverage-map" title="Permalink to this headline"></a>

`sionna.rt.Scene.``coverage_map`(<em class="sig-param">`self`</em>, <em class="sig-param">`rx_orientation``=``(0.0,` `0.0,` `0.0)`</em>, <em class="sig-param">`max_depth``=``3`</em>, <em class="sig-param">`cm_center``=``None`</em>, <em class="sig-param">`cm_orientation``=``None`</em>, <em class="sig-param">`cm_size``=``None`</em>, <em class="sig-param">`cm_cell_size``=``(10.0,` `10.0)`</em>, <em class="sig-param">`combining_vec``=``None`</em>, <em class="sig-param">`precoding_vec``=``None`</em>, <em class="sig-param">`num_samples``=``2000000`</em>, <em class="sig-param">`los``=``True`</em>, <em class="sig-param">`reflection``=``True`</em>, <em class="sig-param">`diffraction``=``False`</em>, <em class="sig-param">`scattering``=``False`</em>, <em class="sig-param">`edge_diffraction``=``False`</em>, <em class="sig-param">`check_scene``=``True`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.coverage_map" title="Permalink to this definition"></a>
    
This function computes a coverage map for every transmitter in the scene.
    
For a given transmitter, a coverage map is a rectangular surface with
arbitrary orientation subdivded
into rectangular cells of size $\lvert C \rvert = \texttt{cm_cell_size[0]} \times  \texttt{cm_cell_size[1]}$.
The parameter `cm_cell_size` therefore controls the granularity of the map.
The coverage map associates with every cell $(i,j)$ the quantity

$$
b_{i,j} = \frac{1}{\lvert C \rvert} \int_{C_{i,j}} \lvert h(s) \rvert^2 ds
$$
    
where $\lvert h(s) \rvert^2$ is the squared amplitude
of the path coefficients $a_i$ at position $s=(x,y)$,
the integral is over the cell $C_{i,j}$, and
$ds$ is the infinitesimal small surface element
$ds=dx \cdot dy$.
The dimension indexed by $i$ ($j$) corresponds to the $y\, (x)$-axis of the
coverage map in its local coordinate system.
    
For specularly and diffusely reflected paths, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#equation-cm-def">(43)</a> can be rewritten as an integral over the directions
of departure of the rays from the transmitter, by substituting $s$
with the corresponding direction $\omega$:

$$
b_{i,j} = \frac{1}{\lvert C \rvert} \int_{\Omega} \lvert h\left(s(\omega) \right) \rvert^2 \frac{r(\omega)^2}{\lvert \cos{\alpha(\omega)} \rvert} \mathbb{1}_{\left\{ s(\omega) \in C_{i,j} \right\}} d\omega
$$
    
where the integration is over the unit sphere $\Omega$, $r(\omega)$ is the length of
the path with direction of departure $\omega$, $s(\omega)$ is the point
where the path with direction of departure $\omega$ intersects the coverage map,
$\alpha(\omega)$ is the angle between the coverage map normal and the direction of arrival
of the path with direction of departure $\omega$,
and $\mathbb{1}_{\left\{ s(\omega) \in C_{i,j} \right\}}$ is the function that takes as value
one if $s(\omega) \in C_{i,j}$ and zero otherwise.
Note that $ds = \frac{r(\omega)^2 d\omega}{\lvert \cos{\alpha(\omega)} \rvert}$.
    
The previous integral is approximated through Monte Carlo sampling by shooting $N$ rays
with directions $\omega_n$ arranged as a Fibonacci lattice on the unit sphere around the transmitter,
and bouncing the rays on the intersected objects until the maximum depth (`max_depth`) is reached or
the ray bounces out of the scene.
At every intersection with an object of the scene, a new ray is shot from the intersection which corresponds to either
specular reflection or diffuse scattering, following a Bernoulli distribution with parameter the
squared scattering coefficient.
When diffuse scattering is selected, the direction of the scattered ray is uniformly sampled on the half-sphere.
The resulting Monte Carlo estimate is:

$$
\hat{b}_{i,j}^{\text{(ref)}} = \frac{4\pi}{N\lvert C \rvert} \sum_{n=1}^N \lvert h\left(s(\omega_n)\right)  \rvert^2 \frac{r(\omega_n)^2}{\lvert \cos{\alpha(\omega_n)} \rvert} \mathbb{1}_{\left\{ s(\omega_n) \in C_{i,j} \right\}}.
$$
    
For the diffracted paths, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#equation-cm-def">(43)</a> can be rewritten for any wedge
with length $L$ and opening angle $\Phi$ as an integral over the wedge and its opening angle,
by substituting $s$ with the position on the wedge $\ell \in [1,L]$ and the angle $\phi \in [0, \Phi]$:

$$
b_{i,j} = \frac{1}{\lvert C \rvert} \int_{\ell} \int_{\phi} \lvert h\left(s(\ell,\phi) \right) \rvert^2 \mathbb{1}_{\left\{ s(\ell,\phi) \in C_{i,j} \right\}} \left\lVert \frac{\partial r}{\partial \ell} \times \frac{\partial r}{\partial \phi} \right\rVert d\ell d\phi
$$
    
where the integral is over the wedge length $L$ and opening angle $\Phi$, and
$r\left( \ell, \phi \right)$ is the reparametrization with respected to $(\ell, \phi)$ of the
intersection between the diffraction cone at $\ell$ and the rectangle defining the coverage map (see, e.g., <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#surfaceintegral" id="id2">[SurfaceIntegral]</a>).
The previous integral is approximated through Monte Carlo sampling by shooting $N'$ rays from equally spaced
locations $\ell_n$ along the wedge with directions $\phi_n$ sampled uniformly from $(0, \Phi)$:

$$
\hat{b}_{i,j}^{\text{(diff)}} = \frac{L\Phi}{N'\lvert C \rvert} \sum_{n=1}^{N'} \lvert h\left(s(\ell_n,\phi_n)\right) \rvert^2 \mathbb{1}_{\left\{ s(\ell_n,\phi_n) \in C_{i,j} \right\}} \left\lVert \left(\frac{\partial r}{\partial \ell}\right)_n \times \left(\frac{\partial r}{\partial \phi}\right)_n \right\rVert.
$$
    
The output of this function is therefore a real-valued matrix of size `[num_cells_y,` `num_cells_x]`,
for every transmitter, with elements equal to the sum of the contributions of the reflected and scattered paths
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#equation-cm-mc-ref">(44)</a> and diffracted paths <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#equation-cm-mc-diff">(45)</a> for all the wedges, and where

$$
\begin{split}\texttt{num_cells_x} = \bigg\lceil\frac{\texttt{cm_size[0]}}{\texttt{cm_cell_size[0]}} \bigg\rceil\\
\texttt{num_cells_y} = \bigg\lceil \frac{\texttt{cm_size[1]}}{\texttt{cm_cell_size[1]}} \bigg\rceil.\end{split}
$$
    
The surface defining the coverage map is a rectangle centered at
`cm_center`, with orientation `cm_orientation`, and with size
`cm_size`. An orientation of (0,0,0) corresponds to
a coverage map parallel to the XY plane, with surface normal pointing towards
the $+z$ axis. By default, the coverage map
is parallel to the XY plane, covers all of the scene, and has
an elevation of $z = 1.5\text{m}$.
The receiver is assumed to use the antenna array
`scene.rx_array`. If transmitter and/or receiver have multiple antennas, transmit precoding
and receive combining are applied which are defined by `precoding_vec` and
`combining_vec`, respectively.
    
The $(i,j)$ indices are omitted in the following for clarity.
For reflection and scattering, paths are generated by shooting `num_samples` rays from the
transmitters with directions arranged in a Fibonacci lattice on the unit
sphere and by simulating their propagation for up to `max_depth` interactions with
scene objects.
If `max_depth` is set to 0 and if `los` is set to <cite>True</cite>,
only the line-of-sight path is considered.
For diffraction, paths are generated by shooting `num_samples` rays from equally
spaced locations along the wedges in line-of-sight with the transmitter, with
directions uniformly sampled on the diffraction cone.
    
For every ray $n$ intersecting the coverage map cell $(i,j)$, the
channel coefficients, $a_n$, and the angles of departure (AoDs)
$(\theta_{\text{T},n}, \varphi_{\text{T},n})$
and arrival (AoAs) $(\theta_{\text{R},n}, \varphi_{\text{R},n})$
are computed. See the <a class="reference external" href="../em_primer.html">Primer on Electromagnetics</a> for more details.
    
A “synthetic” array is simulated by adding additional phase shifts that depend on the
antenna position relative to the position of the transmitter (receiver) as well as the AoDs (AoAs).
For the $k^\text{th}$ transmit antenna and $\ell^\text{th}$ receive antenna, let
us denote by $\mathbf{d}_{\text{T},k}$ and $\mathbf{d}_{\text{R},\ell}$ the relative positions (with respect to
the positions of the transmitter/receiver) of the pair of antennas
for which the channel impulse response shall be computed. These can be accessed through the antenna array’s property
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray.positions" title="sionna.rt.AntennaArray.positions">`positions`</a>. Using a plane-wave assumption, the resulting phase shifts
from these displacements can be computed as

$$
\begin{split}p_{\text{T}, n,k} &= \frac{2\pi}{\lambda}\hat{\mathbf{r}}(\theta_{\text{T},n}, \varphi_{\text{T},n})^\mathsf{T} \mathbf{d}_{\text{T},k}\\
p_{\text{R}, n,\ell} &= \frac{2\pi}{\lambda}\hat{\mathbf{r}}(\theta_{\text{R},n}, \varphi_{\text{R},n})^\mathsf{T} \mathbf{d}_{\text{R},\ell}.\end{split}
$$
    
The final expression for the path coefficient is

$$
h_{n,k,\ell} =  a_n e^{j(p_{\text{T}, i,k} + p_{\text{R}, i,\ell})}
$$
    
for every transmit antenna $k$ and receive antenna $\ell$.
These coefficients form the complex-valued channel matrix, $\mathbf{H}_n$,
of size $\texttt{num_rx_ant} \times \texttt{num_tx_ant}$.
    
Finally, the coefficient of the equivalent SISO channel is

$$
h_n =  \mathbf{c}^{\mathsf{H}} \mathbf{H}_n \mathbf{p}
$$
    
where $\mathbf{c}$ and $\mathbf{p}$ are the combining and
precoding vectors (`combining_vec` and `precoding_vec`),
respectively.
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
cm = scene.coverage_map(cm_cell_size=[1.,1.],
                    num_samples=int(10e6))
# Visualize coverage in preview
scene.preview(coverage_map=cm,
            resolution=[1000, 600])
```


Input
 
- **rx_orientation** (<em>[3], float</em>) – Orientation of the receiver $(\alpha, \beta, \gamma)$
specified through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>. Defaults to $(0,0,0)$.
- **max_depth** (<em>int</em>) – Maximum depth (i.e., number of bounces) allowed for tracing the
paths. Defaults to 3.
- **cm_center** ([3], float | <cite>None</cite>) – Center of the coverage map $(x,y,z)$ as three-dimensional
vector. If set to <cite>None</cite>, the coverage map is centered on the
center of the scene, except for the elevation $z$ that is set
to 1.5m. Otherwise, `cm_orientation` and `cm_scale` must also
not be <cite>None</cite>. Defaults to <cite>None</cite>.
- **cm_orientation** ([3], float | <cite>None</cite>) – Orientation of the coverage map $(\alpha, \beta, \gamma)$
specified through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.
An orientation of $(0,0,0)$ or <cite>None</cite> corresponds to a
coverage map that is parallel to the XY plane.
If not set to <cite>None</cite>, then `cm_center` and `cm_scale` must also
not be <cite>None</cite>.
Defaults to <cite>None</cite>.
- **cm_size** ([2], float | <cite>None</cite>) – Size of the coverage map [m].
If set to <cite>None</cite>, then the size of the coverage map is set such that
it covers the entire scene.
Otherwise, `cm_center` and `cm_orientation` must also not be
<cite>None</cite>. Defaults to <cite>None</cite>.
- **cm_cell_size** (<em>[2], float</em>) – Size of a cell of the coverage map [m].
Defaults to $(10,10)$.
- **combining_vec** (<em>[num_rx_ant], complex | None</em>) – Combining vector.
If set to <cite>None</cite>, then no combining is applied, and
the energy received by all antennas is summed.
- **precoding_vec** (<em>[num_tx_ant], complex | None</em>) – Precoding vector.
If set to <cite>None</cite>, then defaults to
$\frac{1}{\sqrt{\text{num_tx_ant}}} [1,\dots,1]^{\mathsf{T}}$.
- **num_samples** (<em>int</em>) – Number of random rays to trace.
For the reflected paths, this number is split equally over the different transmitters.
For the diffracted paths, it is split over the wedges in line-of-sight with the
transmitters such that the number of rays allocated
to a wedge is proportional to its length.
Defaults to 2e6.
- **los** (<em>bool</em>) – If set to <cite>True</cite>, then the LoS paths are computed.
Defaults to <cite>True</cite>.
- **reflection** (<em>bool</em>) – If set to <cite>True</cite>, then the reflected paths are computed.
Defaults to <cite>True</cite>.
- **diffraction** (<em>bool</em>) – If set to <cite>True</cite>, then the diffracted paths are computed.
Defaults to <cite>False</cite>.
- **scattering** (<em>bool</em>) – If set to <cite>True</cite>, then the scattered paths are computed.
Defaults to <cite>False</cite>.
- **edge_diffraction** (<em>bool</em>) – If set to <cite>False</cite>, only diffraction on wedges, i.e., edges that
connect two primitives, is considered.
Defaults to <cite>False</cite>.
- **check_scene** (<em>bool</em>) – If set to <cite>True</cite>, checks that the scene is well configured before
computing the coverage map. This can add a significant overhead.
Defaults to <cite>True</cite>.


Output
    
cm : <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a> – The coverage maps



### load_scene<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#load-scene" title="Permalink to this headline"></a>

`sionna.rt.``load_scene`(<em class="sig-param">`filename``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/scene.html#load_scene">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.load_scene" title="Permalink to this definition"></a>
    
Load a scene from file
    
Note that only one scene can be loaded at a time.
Input
 
- **filename** (<em>str</em>) – Name of a valid scene file. Sionna uses the simple XML-based format
from <a class="reference external" href="https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html">Mitsuba 3</a>.
Defaults to <cite>None</cite> for which an empty scene is created.
- **dtype** (<em>tf.complex</em>) – Dtype used for all internal computations and outputs.
Defaults to <cite>tf.complex64</cite>.


Output
    
**scene** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a>) – Reference to the current scene



