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
## Antennas<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#antennas" title="Permalink to this headline"></a>
### iso_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#iso-pattern" title="Permalink to this headline"></a>
### tr38901_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#tr38901-pattern" title="Permalink to this headline"></a>
### polarization_model_1<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#polarization-model-1" title="Permalink to this headline"></a>
### polarization_model_2<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#polarization-model-2" title="Permalink to this headline"></a>
## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#utility-functions" title="Permalink to this headline"></a>
### cross<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#cross" title="Permalink to this headline"></a>
### dot<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#dot" title="Permalink to this headline"></a>
### normalize<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#normalize" title="Permalink to this headline"></a>
### phi_hat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#phi-hat" title="Permalink to this headline"></a>
### rotate<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#rotate" title="Permalink to this headline"></a>
### rotation_matrix<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#rotation-matrix" title="Permalink to this headline"></a>
### rot_mat_from_unit_vecs<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#rot-mat-from-unit-vecs" title="Permalink to this headline"></a>
### r_hat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#r-hat" title="Permalink to this headline"></a>
  
  

### iso_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#iso-pattern" title="Permalink to this headline"></a>

`sionna.rt.antenna.``iso_pattern`(<em class="sig-param">`theta`</em>, <em class="sig-param">`phi`</em>, <em class="sig-param">`slant_angle``=``0.0`</em>, <em class="sig-param">`polarization_model``=``2`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#iso_pattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.iso_pattern" title="Permalink to this definition"></a>
    
Isotropic antenna pattern with linear polarizarion
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




### tr38901_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#tr38901-pattern" title="Permalink to this headline"></a>

`sionna.rt.antenna.``tr38901_pattern`(<em class="sig-param">`theta`</em>, <em class="sig-param">`phi`</em>, <em class="sig-param">`slant_angle``=``0.0`</em>, <em class="sig-param">`polarization_model``=``2`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#tr38901_pattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.tr38901_pattern" title="Permalink to this definition"></a>
    
Antenna pattern from 3GPP TR 38.901 (Table 7.3-1) <a class="reference internal" href="channel.wireless.html#tr38901" id="id23">[TR38901]</a>
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




### polarization_model_1<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#polarization-model-1" title="Permalink to this headline"></a>

`sionna.rt.antenna.``polarization_model_1`(<em class="sig-param">`c_theta`</em>, <em class="sig-param">`theta`</em>, <em class="sig-param">`phi`</em>, <em class="sig-param">`slant_angle`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#polarization_model_1">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_1" title="Permalink to this definition"></a>
    
Model-1 for polarized antennas from 3GPP TR 38.901
    
Transforms a vertically polarized antenna pattern $\tilde{C}_\theta(\theta, \varphi)$
into a linearly polarized pattern whose direction
is specified by a slant angle $\zeta$. For example,
$\zeta=0$ and $\zeta=\pi/2$ correspond
to vertical and horizontal polarization, respectively,
and $\zeta=\pm \pi/4$ to a pair of cross polarized
antenna elements.
    
The transformed antenna pattern is given by (7.3-3) <a class="reference internal" href="channel.wireless.html#tr38901" id="id24">[TR38901]</a>:

$$
\begin{split}\begin{align}
    \begin{bmatrix}
        C_\theta(\theta, \varphi) \\
        C_\varphi(\theta, \varphi)
    \end{bmatrix} &= \begin{bmatrix}
     \cos(\psi) \\
     \sin(\psi)
    \end{bmatrix} \tilde{C}_\theta(\theta, \varphi)\\
    \cos(\psi) &= \frac{\cos(\zeta)\sin(\theta)+\sin(\zeta)\sin(\varphi)\cos(\theta)}{\sqrt{1-\left(\cos(\zeta)\cos(\theta)-\sin(\zeta)\sin(\varphi)\sin(\theta)\right)^2}} \\
    \sin(\psi) &= \frac{\sin(\zeta)\cos(\varphi)}{\sqrt{1-\left(\cos(\zeta)\cos(\theta)-\sin(\zeta)\sin(\varphi)\sin(\theta)\right)^2}}
\end{align}\end{split}
$$

Input
 
- **c_tilde_theta** (<em>array_like, complex</em>) – Zenith pattern
- **theta** (<em>array_like, float</em>) – Zenith angles wrapped within [0,pi] [rad]
- **phi** (<em>array_like, float</em>) – Azimuth angles wrapped within [-pi, pi) [rad]
- **slant_angle** (<em>float</em>) – Slant angle of the linear polarization [rad].
A slant angle of zero means vertical polarization.


Output
 
- **c_theta** (<em>array_like, complex</em>) – Zenith pattern
- **c_phi** (<em>array_like, complex</em>) – Azimuth pattern




### polarization_model_2<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#polarization-model-2" title="Permalink to this headline"></a>

`sionna.rt.antenna.``polarization_model_2`(<em class="sig-param">`c`</em>, <em class="sig-param">`slant_angle`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#polarization_model_2">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_2" title="Permalink to this definition"></a>
    
Model-2 for polarized antennas from 3GPP TR 38.901
    
Transforms a vertically polarized antenna pattern $\tilde{C}_\theta(\theta, \varphi)$
into a linearly polarized pattern whose direction
is specified by a slant angle $\zeta$. For example,
$\zeta=0$ and $\zeta=\pi/2$ correspond
to vertical and horizontal polarization, respectively,
and $\zeta=\pm \pi/4$ to a pair of cross polarized
antenna elements.
    
The transformed antenna pattern is given by (7.3-4/5) <a class="reference internal" href="channel.wireless.html#tr38901" id="id25">[TR38901]</a>:

$$
\begin{split}\begin{align}
    \begin{bmatrix}
        C_\theta(\theta, \varphi) \\
        C_\varphi(\theta, \varphi)
    \end{bmatrix} &= \begin{bmatrix}
     \cos(\zeta) \\
     \sin(\zeta)
    \end{bmatrix} \tilde{C}_\theta(\theta, \varphi)
\end{align}\end{split}
$$

Input
 
- **c_tilde_theta** (<em>array_like, complex</em>) – Zenith pattern
- **slant_angle** (<em>float</em>) – Slant angle of the linear polarization [rad].
A slant angle of zero means vertical polarization.


Output
 
- **c_theta** (<em>array_like, complex</em>) – Zenith pattern
- **c_phi** (<em>array_like, complex</em>) – Azimuth pattern




## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#utility-functions" title="Permalink to this headline"></a>

### cross<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#cross" title="Permalink to this headline"></a>

`sionna.rt.``cross`(<em class="sig-param">`u`</em>, <em class="sig-param">`v`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#cross">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.cross" title="Permalink to this definition"></a>
    
Computes the cross (or vector) product between u and v
Input
 
- **u** (<em>[…,3]</em>) – First vector
- **v** (<em>[…,3]</em>) – Second vector


Output
    
<em>[…,3]</em> – Cross product between `u` and `v`



### dot<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#dot" title="Permalink to this headline"></a>

`sionna.rt.``dot`(<em class="sig-param">`u`</em>, <em class="sig-param">`v`</em>, <em class="sig-param">`keepdim``=``False`</em>, <em class="sig-param">`clip``=``False`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#dot">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.dot" title="Permalink to this definition"></a>
    
Computes and the dot (or scalar) product between u and v
Input
 
- **u** (<em>[…,3]</em>) – First vector
- **v** (<em>[…,3]</em>) – Second vector
- **keepdim** (<em>bool</em>) – If <cite>True</cite>, keep the last dimension.
Defaults to <cite>False</cite>.
- **clip** (<em>bool</em>) – If <cite>True</cite>, clip output to [-1,1].
Defaults to <cite>False</cite>.


Output
    
<em>[…,1] or […]</em> – Dot product between `u` and `v`.
The last dimension is removed if `keepdim`
is set to <cite>False</cite>.



### normalize<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#normalize" title="Permalink to this headline"></a>

`sionna.rt.``normalize`(<em class="sig-param">`v`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#normalize">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.normalize" title="Permalink to this definition"></a>
    
Normalizes `v` to unit norm
Input
    
**v** (<em>[…,3], tf.float</em>) – Vector

Output
 
- <em>[…,3], tf.float</em> – Normalized vector
- <em>[…], tf.float</em> – Norm of the unnormalized vector




### phi_hat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#phi-hat" title="Permalink to this headline"></a>

`sionna.rt.``phi_hat`(<em class="sig-param">`phi`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#phi_hat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.phi_hat" title="Permalink to this definition"></a>
    
Computes the spherical unit vector
$\hat{\boldsymbol{\varphi}}(\theta, \varphi)$
as defined in <a class="reference internal" href="../em_primer.html#equation-spherical-vecs">(1)</a>
Input
    
**phi** (same shape as `theta`, tf.float) – Azimuth angles $\varphi$ [rad]

Output
    
**theta_hat** (`phi.shape` + [3], tf.float) – Vector $\hat{\boldsymbol{\varphi}}(\theta, \varphi)$



### rotate<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#rotate" title="Permalink to this headline"></a>

`sionna.rt.``rotate`(<em class="sig-param">`p`</em>, <em class="sig-param">`angles`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#rotate">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.rotate" title="Permalink to this definition"></a>
    
Rotates points `p` by the `angles` according
to the 3D rotation defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>
Input
 
- **p** (<em>[…,3], tf.float</em>) – Points to rotate
- **angles** (<em>[…, 3]</em>) – Angles for the rotations [rad].
The last dimension corresponds to the angles
$(\alpha,\beta,\gamma)$ that define
rotations about the axes $(z, y, x)$,
respectively.


Output
    
<em>[…,3]</em> – Rotated points `p`



### rotation_matrix<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#rotation-matrix" title="Permalink to this headline"></a>

`sionna.rt.``rotation_matrix`(<em class="sig-param">`angles`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#rotation_matrix">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.rotation_matrix" title="Permalink to this definition"></a>
    
Computes rotation matrices as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>
    
The closed-form expression in (7.1-4) <a class="reference internal" href="channel.wireless.html#tr38901" id="id26">[TR38901]</a> is used.
Input
    
**angles** (<em>[…,3], tf.float</em>) – Angles for the rotations [rad].
The last dimension corresponds to the angles
$(\alpha,\beta,\gamma)$ that define
rotations about the axes $(z, y, x)$,
respectively.

Output
    
<em>[…,3,3], tf.float</em> – Rotation matrices



### rot_mat_from_unit_vecs<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#rot-mat-from-unit-vecs" title="Permalink to this headline"></a>

`sionna.rt.``rot_mat_from_unit_vecs`(<em class="sig-param">`a`</em>, <em class="sig-param">`b`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#rot_mat_from_unit_vecs">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.rot_mat_from_unit_vecs" title="Permalink to this definition"></a>
    
Computes Rodrigues` rotation formula <a class="reference internal" href="../em_primer.html#equation-rodrigues-matrix">(6)</a>
Input
 
- **a** (<em>[…,3], tf.float</em>) – First unit vector
- **b** (<em>[…,3], tf.float</em>) – Second unit vector


Output
    
<em>[…,3,3], tf.float</em> – Rodrigues’ rotation matrix



### r_hat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#r-hat" title="Permalink to this headline"></a>

`sionna.rt.``r_hat`(<em class="sig-param">`theta`</em>, <em class="sig-param">`phi`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#r_hat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.r_hat" title="Permalink to this definition"></a>
    
Computes the spherical unit vetor $\hat{\mathbf{r}}(\theta, \phi)$
as defined in <a class="reference internal" href="../em_primer.html#equation-spherical-vecs">(1)</a>
Input
 
- **theta** (<em>arbitrary shape, tf.float</em>) – Zenith angles $\theta$ [rad]
- **phi** (same shape as `theta`, tf.float) – Azimuth angles $\varphi$ [rad]


Output
    
**rho_hat** (`phi.shape` + [3], tf.float) – Vector $\hat{\mathbf{r}}(\theta, \phi)$  on unit sphere



