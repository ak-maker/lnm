# Primer on Electromagnetics<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#primer-on-electromagnetics" title="Permalink to this headline"></a>
    
This section provides useful background for the general understanding of ray tracing for wireless propagation modelling. In particular, our goal is to provide a concise definition of a <cite>channel impulse response</cite> between a transmitting and receiving antenna, as done in (Ch. 2 & 3) <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#wiesbeck" id="id1">[Wiesbeck]</a>. The notations and definitions will be used in the API documentation of Sionna’s <a class="reference internal" href="api/rt.html">Ray Tracing module</a>.

# Table of Content
## Modelling of a Receiving Antenna<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#modelling-of-a-receiving-antenna" title="Permalink to this headline"></a>
## General Propagation Path<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#general-propagation-path" title="Permalink to this headline"></a>
## Frequency & Impulse Response<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#frequency-impulse-response" title="Permalink to this headline"></a>
## Reflection and Refraction<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#reflection-and-refraction" title="Permalink to this headline"></a>
  
  

## Modelling of a Receiving Antenna<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#modelling-of-a-receiving-antenna" title="Permalink to this headline"></a>
    
Although the transmitting antenna radiates a spherical wave $\mathbf{E}_\text{T}(r,\theta_\text{T},\varphi_\text{T})$,
we assume that the receiving antenna observes a planar incoming wave $\mathbf{E}_\text{R}$ that arrives from the angles $\theta_\text{R}$ and $\varphi_\text{R}$
which are defined in the local spherical coordinates of the receiving antenna. The Poynting vector of the incoming wave $\mathbf{S}_\text{R}$ is hence <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-s-spherical">(11)</a>

$$
\mathbf{S}_\text{R} = -\frac{1}{2Z_0} \lVert \mathbf{E}_\text{R} \rVert^2 \hat{\mathbf{r}}(\theta_\text{R}, \varphi_\text{R})
$$
    
where $\hat{\mathbf{r}}(\theta_\text{R}, \varphi_\text{R})$ is the radial unit vector in the spherical coordinate system of the receiver.
    
The aperture or effective area $A_\text{R}$ of an antenna with gain $G_\text{R}$ is defined as the ratio of the available received power $P_\text{R}$ at the output of the antenna and the absolute value of the Poynting vector, i.e., the power density:

$$
A_\text{R} = \frac{P_\text{R}}{\lVert \mathbf{S}_\text{R}\rVert} = G_\text{R}\frac{\lambda^2}{4\pi}
$$
    
where $\frac{\lambda^2}{4\pi}$ is the aperture of an isotropic antenna. In the definition above, it is assumed that the antenna is ideally directed towards and polarization matched to the incoming wave.
For an arbitrary orientation of the antenna (but still assuming polarization matching), we can define a direction dependent effective area

$$
A_\text{R}(\theta_\text{R}, \varphi_\text{R}) = G_\text{R}(\theta_\text{R}, \varphi_\text{R})\frac{\lambda^2}{4\pi}.
$$
    
The available received power at the output of the antenna can be expressed as

$$
P_\text{R} = \frac{|V_\text{R}|^2}{8\Re\{Z_\text{R}\}}
$$
    
where $Z_\text{R}$ is the impedance of the receiving antenna and $V_\text{R}$ the open circuit voltage.
    
We can now combine <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-p-r">(20)</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-a-dir">(19)</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-a-r">(18)</a> to obtain the following expression for the absolute value of the voltage $|V_\text{R}|$
assuming matched polarization:

$$
\begin{split}\begin{align}
    |V_\text{R}| &= \sqrt{P_\text{R} 8\Re\{Z_\text{R}\}}\\
                 &= \sqrt{\frac{\lambda^2}{4\pi} G_\text{R}(\theta_\text{R}, \varphi_\text{R}) \frac{8\Re\{Z_\text{R}\}}{2 Z_0} \lVert \mathbf{E}_\text{R} \rVert^2}\\
                 &= \sqrt{\frac{\lambda^2}{4\pi} G_\text{R} \frac{4\Re\{Z_\text{R}\}}{Z_0}} \lVert \mathbf{F}_\text{R}(\theta_\text{R}, \varphi_\text{R})\rVert\lVert\mathbf{E}_\text{R}\rVert.
\end{align}\end{split}
$$
    
By extension of the previous equation, we can obtain an expression for $V_\text{R}$ which is valid for
arbitrary polarizations of the incoming wave and the receiving antenna:

$$
V_\text{R} = \sqrt{\frac{\lambda^2}{4\pi} G_\text{R} \frac{4\Re\{Z_\text{R}\}}{Z_0}} \mathbf{F}_\text{R}(\theta_\text{R}, \varphi_\text{R})^{\mathsf{H}}\mathbf{E}_\text{R}.
$$

**Example: Recovering Friis equation**
    
In the case of free space propagation, we have $\mathbf{E}_\text{R}=\mathbf{E}_\text{T}(r,\theta_\text{T},\varphi_\text{T})$.
Combining <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-v-r">(21)</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-p-r">(20)</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-e-t">(15)</a>, we obtain the following expression for the received power:

$$
P_\text{R} = \left(\frac{\lambda}{4\pi r}\right)^2 G_\text{R} G_\text{T} P_\text{T} \left|\mathbf{F}_\text{R}(\theta_\text{R}, \varphi_\text{R})^{\mathsf{H}} \mathbf{F}_\text{T}(\theta_\text{T}, \varphi_\text{T})\right|^2.
$$
    
It is important that $\mathbf{F}_\text{R}$ and $\mathbf{F}_\text{T}$ are expressed in the same coordinate system for the last equation to make sense.
For perfect orientation and polarization matching, we can recover the well-known Friis transmission equation:

$$
\frac{P_\text{R}}{P_\text{T}} = \left(\frac{\lambda}{4\pi r}\right)^2 G_\text{R} G_\text{T}.
$$
## General Propagation Path<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#general-propagation-path" title="Permalink to this headline"></a>
    
A single propagation path consists of a cascade of multiple scattering processes, where a scattering process can be anything that prevents the wave from propagating as in free space. This includes reflection, refraction, diffraction, and diffuse scattering. For each scattering process, one needs to compute a relationship between the incoming field at the scatter center and the created far field at the next scatter center or the receiving antenna.
We can represent this cascade of scattering processes by a single matrix $\widetilde{\mathbf{T}}$
that describes the transformation that the radiated field $\mathbf{E}_\text{T}(r, \theta_\text{T}, \varphi_\text{T})$ undergoes until it reaches the receiving antenna:

$$
\mathbf{E}_\text{R} = \sqrt{ \frac{P_\text{T} G_\text{T} Z_0}{2\pi}} \widetilde{\mathbf{T}} \mathbf{F}_\text{T}(\theta_\text{T}, \varphi_\text{T}).
$$
    
Note that we have obtained this expression by replacing the free space propagation term $\frac{e^{-jk_0r}}{r}$ in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-e-t">(15)</a> by the matrix $\widetilde{\mathbf{T}}$. This requires that all quantities are expressed in the same coordinate system which is also assumed in the following expressions. Further, it is assumed that the matrix $\widetilde{\mathbf{T}}$ includes the necessary coordinate transformations. In some cases, e.g., for diffuse scattering (see <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-scattered-field">(38)</a> in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#scattering">Scattering</a>), the matrix $\widetilde{\mathbf{T}}$ depends on the incoming field and is not a linear transformation.
    
Plugging <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-e-r">(22)</a> into <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-v-r">(21)</a>, we can obtain a general expression for the received voltage of a propagation path:

$$
V_\text{R} = \sqrt{\left(\frac{\lambda}{4\pi}\right)^2 G_\text{R}G_\text{T}P_\text{T} 8\Re\{Z_\text{R}\}} \,\mathbf{F}_\text{R}(\theta_\text{R}, \varphi_\text{R})^{\mathsf{H}}\widetilde{\mathbf{T}} \mathbf{F}_\text{T}(\theta_\text{T}, \varphi_\text{T}).
$$
    
If the electromagnetic wave arrives at the receiving antenna over $N$ propagation paths, we can simply add the received voltages
from all paths to obtain

$$
\begin{split}\begin{align}
V_\text{R} &= \sqrt{\left(\frac{\lambda}{4\pi}\right)^2 G_\text{R}G_\text{T}P_\text{T} 8\Re\{Z_\text{R}\}} \sum_{n=1}^N\mathbf{F}_\text{R}(\theta_{\text{R},i}, \varphi_{\text{R},i})^{\mathsf{H}}\widetilde{\mathbf{T}}_i \mathbf{F}_\text{T}(\theta_{\text{T},i}, \varphi_{\text{T},i})\\
&= \sqrt{\left(\frac{\lambda}{4\pi}\right)^2 P_\text{T} 8\Re\{Z_\text{R}\}} \sum_{n=1}^N\mathbf{C}_\text{R}(\theta_{\text{R},i}, \varphi_{\text{R},i})^{\mathsf{H}}\widetilde{\mathbf{T}}_i \mathbf{C}_\text{T}(\theta_{\text{T},i}, \varphi_{\text{T},i})
\end{align}\end{split}
$$
    
where all path-dependent quantities carry the subscript $i$. Note that the matrices $\widetilde{\mathbf{T}}_i$ also ensure appropriate scaling so that the total received power can never be larger than the transmit power.

## Frequency & Impulse Response<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#frequency-impulse-response" title="Permalink to this headline"></a>
    
The channel frequency response $H(f)$ at frequency $f=\frac{c}{\lambda}$ is defined as the ratio between the received voltage and the voltage at the input to the transmitting antenna:

$$
H(f) = \frac{V_\text{R}}{V_\text{T}} = \frac{V_\text{R}}{|V_\text{T}|}
$$
    
where it is assumed that the input voltage has zero phase.
    
It is useful to separate phase shifts due to wave propagation from the transfer matrices $\widetilde{\mathbf{T}}_i$. If we denote by $r_i$ the total length of path $i$ with average propagation speed $c_i$, the path delay is $\tau_i=r_i/c_i$. We can now define the new transfer matrix

$$
\mathbf{T}_i=\widetilde{\mathbf{T}}_ie^{j2\pi f \tau_i}.
$$
    
Using <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-p-t">(16)</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-t-tilde">(25)</a> in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-v-rmulti">(23)</a> while assuming equal real parts of both antenna impedances, i.e., $\Re\{Z_\text{T}\}=\Re\{Z_\text{R}\}$ (which is typically the case), we obtain the final expression for the channel frequency response:

$$
\boxed{H(f) = \sum_{i=1}^N \underbrace{\frac{\lambda}{4\pi} \mathbf{C}_\text{R}(\theta_{\text{R},i}, \varphi_{\text{R},i})^{\mathsf{H}}\mathbf{T}_i \mathbf{C}_\text{T}(\theta_{\text{T},i}, \varphi_{\text{T},i})}_{\triangleq a_i} e^{-j2\pi f\tau_i}}
$$
    
Taking the inverse Fourier transform, we finally obtain the channel impulse response

$$
\boxed{h(\tau) = \int_{-\infty}^{\infty} H(f) e^{j2\pi f \tau} df = \sum_{i=1}^N a_i \delta(\tau-\tau_i)}
$$
    
The baseband equivalent channel impulse reponse is then defined as (Eq. 2.28) <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#tse" id="id7">[Tse]</a>:

$$
h_\text{b}(\tau) = \sum_{i=1}^N \underbrace{a_i e^{-j2\pi f \tau_i}}_{\triangleq a^\text{b}_i} \delta(\tau-\tau_i).
$$
## Reflection and Refraction<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#reflection-and-refraction" title="Permalink to this headline"></a>
    
When a plane wave hits a plane interface which separates two materials, e.g., air and concrete, a part of the wave gets reflected and the other transmitted (or <em>refracted</em>), i.e., it propagates into the other material.  We assume in the following description that both materials are uniform non-magnetic dielectrics, i.e., $\mu_r=1$, and follow the definitions as in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#iturp20402" id="id8">[ITURP20402]</a>. The incoming wave phasor $\mathbf{E}_\text{i}$ is expressed by two arbitrary orthogonal polarization components, i.e.,

$$
\mathbf{E}_\text{i} = E_{\text{i},s} \hat{\mathbf{e}}_{\text{i},s} + E_{\text{i},p} \hat{\mathbf{e}}_{\text{i},p}
$$
    
which are both orthogonal to the incident wave vector, i.e., $\hat{\mathbf{e}}_{\text{i},s}^{\mathsf{T}} \hat{\mathbf{e}}_{\text{i},p}=\hat{\mathbf{e}}_{\text{i},s}^{\mathsf{T}} \hat{\mathbf{k}}_\text{i}=\hat{\mathbf{e}}_{\text{i},p}^{\mathsf{T}} \hat{\mathbf{k}}_\text{i} =0$.
<a class="reference internal image-reference" href="_images/reflection.svg"><img alt="_images/reflection.svg" src="_images/reflection.svg" width="90%" /></a>
<p class="caption">Fig. 1 Reflection and refraction of a plane wave at a plane interface between two materials.<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#id29" title="Permalink to this image"></a>
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#fig-reflection">Fig. 1</a> shows reflection and refraction of the incoming wave at the plane interface between two materials with relative permittivities $\eta_1$ and $\eta_2$. The coordinate system is chosen such that the wave vectors of the incoming, reflected, and transmitted waves lie within the plane of incidence, which is chosen to be the x-z plane. The normal vector of the interface $\hat{\mathbf{n}}$ is pointing toward the negative z axis.
The incoming wave is must be represented in a different basis, i.e., in the form two different orthogonal polarization components $E_{\text{i}, \perp}$ and $E_{\text{i}, \parallel}$, i.e.,

$$
\mathbf{E}_\text{i} = E_{\text{i},\perp} \hat{\mathbf{e}}_{\text{i},\perp} + E_{\text{i},\parallel} \hat{\mathbf{e}}_{\text{i},\parallel}
$$
    
where the former is orthogonal to the plane of incidence and called transverse electric (TE) polarization (left), and the latter is parallel to the plane of incidence and called transverse magnetic (TM) polarization (right). We adopt in the following the convention that all transverse components are coming out of the figure (indicated by the $\odot$ symbol). One can easily verify that the following relationships must hold:

$$
\begin{split}\begin{align}
    \hat{\mathbf{e}}_{\text{i},\perp} &= \frac{\hat{\mathbf{k}}_\text{i} \times \hat{\mathbf{n}}}{\lVert \hat{\mathbf{k}}_\text{i} \times \hat{\mathbf{n}} \rVert} \\
    \hat{\mathbf{e}}_{\text{i},\parallel} &= \hat{\mathbf{e}}_{\text{i},\perp} \times \hat{\mathbf{k}}_\text{i}
\end{align}\end{split}
$$

$$
\begin{split}\begin{align}
\begin{bmatrix}E_{\text{i},\perp} \\ E_{\text{i},\parallel} \end{bmatrix} &=
    \begin{bmatrix}
        \hat{\mathbf{e}}_{\text{i},\perp}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},s} & \hat{\mathbf{e}}_{\text{i},\perp}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},p}\\
        \hat{\mathbf{e}}_{\text{i},\parallel}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},s} & \hat{\mathbf{e}}_{\text{i},\parallel}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},p}
    \end{bmatrix}
 \begin{bmatrix}E_{\text{i},s} \\ E_{\text{i},p}\end{bmatrix} =
 \mathbf{W}\left(\hat{\mathbf{e}}_{\text{i},\perp}, \hat{\mathbf{e}}_{\text{i},\parallel}, \hat{\mathbf{e}}_{\text{i},s}, \hat{\mathbf{e}}_{\text{i},p}\right)
\end{align}\end{split}
$$
    
where we have defined the following matrix-valued function

$$
\begin{split}\begin{align}
\mathbf{W}\left(\hat{\mathbf{a}}, \hat{\mathbf{b}}, \hat{\mathbf{q}}, \hat{\mathbf{r}} \right) =
    \begin{bmatrix}
        \hat{\mathbf{a}}^\textsf{T} \hat{\mathbf{q}} & \hat{\mathbf{a}}^\textsf{T} \hat{\mathbf{r}} \\
        \hat{\mathbf{b}}^\textsf{T} \hat{\mathbf{q}} & \hat{\mathbf{b}}^\textsf{T} \hat{\mathbf{r}}
    \end{bmatrix}.
\end{align}\end{split}
$$
    
While the angles of incidence and reflection are both equal to $\theta_1$, the angle of the refracted wave $\theta_2$ is given by Snell’s law:

$$
\sin(\theta_2) = \sqrt{\frac{\eta_1}{\eta_2}} \sin(\theta_1)
$$
    
or, equivalently,

$$
\cos(\theta_2) = \sqrt{1 - \frac{\eta_1}{\eta_2} \sin^2(\theta_1)}.
$$
    
The reflected and transmitted wave phasors $\mathbf{E}_\text{r}$ and $\mathbf{E}_\text{t}$ are similarly represented as

$$
\begin{split}\begin{align}
    \mathbf{E}_\text{r} &= E_{\text{r},\perp} \hat{\mathbf{e}}_{\text{r},\perp} + E_{\text{r},\parallel} \hat{\mathbf{e}}_{\text{r},\parallel}\\
    \mathbf{E}_\text{t} &= E_{\text{t},\perp} \hat{\mathbf{e}}_{\text{t},\perp} + E_{\text{t},\parallel} \hat{\mathbf{e}}_{\text{t},\parallel}
\end{align}\end{split}
$$
    
where

$$
\begin{split}\begin{align}
    \hat{\mathbf{e}}_{\text{r},\perp} &= \hat{\mathbf{e}}_{\text{i},\perp}\\
    \hat{\mathbf{e}}_{\text{r},\parallel} &= \frac{\hat{\mathbf{e}}_{\text{r},\perp}\times\hat{\mathbf{k}}_\text{r}}{\lVert \hat{\mathbf{e}}_{\text{r},\perp}\times\hat{\mathbf{k}}_\text{r} \rVert}\\
    \hat{\mathbf{e}}_{\text{t},\perp} &= \hat{\mathbf{e}}_{\text{i},\perp}\\
    \hat{\mathbf{e}}_{\text{t},\parallel} &= \frac{\hat{\mathbf{e}}_{\text{t},\perp}\times\hat{\mathbf{k}}_\text{t}}{ \Vert \hat{\mathbf{e}}_{\text{t},\perp}\times\hat{\mathbf{k}}_\text{t} \rVert}
\end{align}\end{split}
$$
    
and

$$
\begin{split}\begin{align}
    \hat{\mathbf{k}}_\text{r} &= \hat{\mathbf{k}}_\text{i} - 2\left( \hat{\mathbf{k}}_\text{i}^\mathsf{T}\hat{\mathbf{n}} \right)\hat{\mathbf{n}}\\
    \hat{\mathbf{k}}_\text{t} &= \sqrt{\frac{\eta_1}{\eta_2}} \hat{\mathbf{k}}_\text{i} + \left(\sqrt{\frac{\eta_1}{\eta_2}}\cos(\theta_1) - \cos(\theta_2) \right)\hat{\mathbf{n}}.
\end{align}\end{split}
$$
    
The <em>Fresnel</em> equations provide relationships between the incident, reflected, and refracted field components for $\sqrt{\left| \eta_1/\eta_2 \right|}\sin(\theta_1)<1$:

$$
\begin{split}\begin{align}
    r_{\perp}     &= \frac{E_{\text{r}, \perp    }}{E_{\text{i}, \perp    }} = \frac{ \sqrt{\eta_1}\cos(\theta_1) - \sqrt{\eta_2}\cos(\theta_2) }{ \sqrt{\eta_1}\cos(\theta_1) + \sqrt{\eta_2}\cos(\theta_2) } \\
    r_{\parallel} &= \frac{E_{\text{r}, \parallel}}{E_{\text{i}, \parallel}} = \frac{ \sqrt{\eta_2}\cos(\theta_1) - \sqrt{\eta_1}\cos(\theta_2) }{ \sqrt{\eta_2}\cos(\theta_1) + \sqrt{\eta_1}\cos(\theta_2) } \\
    t_{\perp}     &= \frac{E_{\text{t}, \perp    }}{E_{\text{t}, \perp    }} = \frac{ 2\sqrt{\eta_1}\cos(\theta_1) }{ \sqrt{\eta_1}\cos(\theta_1) + \sqrt{\eta_2}\cos(\theta_2) } \\
    t_{\parallel} &= \frac{E_{\text{t}, \parallel}}{E_{\text{t}, \parallel}} = \frac{ 2\sqrt{\eta_1}\cos(\theta_1) }{ \sqrt{\eta_2}\cos(\theta_1) + \sqrt{\eta_1}\cos(\theta_2) }.
\end{align}\end{split}
$$
    
If $\sqrt{\left| \eta_1/\eta_2 \right|}\sin(\theta_1)\ge 1$, we have $r_{\perp}=r_{\parallel}=1$ and $t_{\perp}=t_{\parallel}=0$, i.e., total reflection.
    
For the case of an incident wave in vacuum, i.e., $\eta_1=1$, the Fresnel equations <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-fresnel">(33)</a> simplify to

$$
\begin{split}\begin{align}
    r_{\perp}     &= \frac{\cos(\theta_1) -\sqrt{\eta_2 -\sin^2(\theta_1)}}{\cos(\theta_1) +\sqrt{\eta_2 -\sin^2(\theta_1)}} \\
    r_{\parallel} &= \frac{\eta_2\cos(\theta_1) -\sqrt{\eta_2 -\sin^2(\theta_1)}}{\eta_2\cos(\theta_1) +\sqrt{\eta_2 -\sin^2(\theta_1)}} \\
    t_{\perp}     &= \frac{2\cos(\theta_1)}{\cos(\theta_1) + \sqrt{\eta_2-\sin^2(\theta_1)}}\\
    t_{\parallel} &= \frac{2\sqrt{\eta_2}\cos(\theta_1)}{\eta_2 \cos(\theta_1) + \sqrt{\eta_2-\sin^2(\theta_1)}}.
\end{align}\end{split}
$$
    
Putting everything together, we obtain the following relationships between incident, reflected, and transmitted waves:

$$
\begin{split}\begin{align}
    \begin{bmatrix}E_{\text{r},\perp} \\ E_{\text{r},\parallel} \end{bmatrix} &=
    \begin{bmatrix}
        r_{\perp} & 0 \\
        0         & r_{\parallel}
    \end{bmatrix}
    \mathbf{W}\left(\hat{\mathbf{e}}_{\text{i},\perp}, \hat{\mathbf{e}}_{\text{i},\parallel}, \hat{\mathbf{e}}_{\text{i},s}, \hat{\mathbf{e}}_{\text{i},p}\right)
 \begin{bmatrix}E_{\text{i},s} \\ E_{\text{i},p}\end{bmatrix} \\
 \begin{bmatrix}E_{\text{t},\perp} \\ E_{\text{t},\parallel} \end{bmatrix} &=
    \begin{bmatrix}
        t_{\perp} & 0 \\
        0         & t_{\parallel}
    \end{bmatrix}
    \mathbf{W}\left(\hat{\mathbf{e}}_{\text{i},\perp}, \hat{\mathbf{e}}_{\text{i},\parallel}, \hat{\mathbf{e}}_{\text{i},s}, \hat{\mathbf{e}}_{\text{i},p}\right)
 \begin{bmatrix}E_{\text{i},s} \\ E_{\text{i},p}\end{bmatrix}.
\end{align}\end{split}
$$
