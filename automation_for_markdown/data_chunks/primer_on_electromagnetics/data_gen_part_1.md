# Primer on Electromagnetics<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#primer-on-electromagnetics" title="Permalink to this headline"></a>
    
This section provides useful background for the general understanding of ray tracing for wireless propagation modelling. In particular, our goal is to provide a concise definition of a <cite>channel impulse response</cite> between a transmitting and receiving antenna, as done in (Ch. 2 & 3) <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#wiesbeck" id="id1">[Wiesbeck]</a>. The notations and definitions will be used in the API documentation of Sionna’s <a class="reference internal" href="api/rt.html">Ray Tracing module</a>.

# Table of Content
## Coordinate system, rotations, and vector fields<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#coordinate-system-rotations-and-vector-fields" title="Permalink to this headline"></a>
## Planar Time-Harmonic Waves<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#planar-time-harmonic-waves" title="Permalink to this headline"></a>
## Far Field of a Transmitting Antenna<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#far-field-of-a-transmitting-antenna" title="Permalink to this headline"></a>
  
  

## Coordinate system, rotations, and vector fields<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#coordinate-system-rotations-and-vector-fields" title="Permalink to this headline"></a>
    
We consider a global coordinate system (GCS) with Cartesian standard basis $\hat{\mathbf{x}}$, $\hat{\mathbf{y}}$, $\hat{\mathbf{z}}$.
The spherical unit vectors are defined as

$$
\begin{split}\begin{align}
    \hat{\mathbf{r}}          (\theta, \varphi) &= \sin(\theta)\cos(\varphi) \hat{\mathbf{x}} + \sin(\theta)\sin(\varphi) \hat{\mathbf{y}} + \cos(\theta)\hat{\mathbf{z}}\\
    \hat{\boldsymbol{\theta}} (\theta, \varphi) &= \cos(\theta)\cos(\varphi) \hat{\mathbf{x}} + \cos(\theta)\sin(\varphi) \hat{\mathbf{y}} - \sin(\theta)\hat{\mathbf{z}}\\
    \hat{\boldsymbol{\varphi}}(\theta, \varphi) &=            -\sin(\varphi) \hat{\mathbf{x}} +             \cos(\varphi) \hat{\mathbf{y}}.
\end{align}\end{split}
$$
    
For an arbitrary unit norm vector $\hat{\mathbf{v}} = (x, y, z)$, the elevation and azimuth angles $\theta$ and $\varphi$ can be computed as

$$
\begin{split}\theta  &= \cos^{-1}(z) \\
\varphi &= \mathop{\text{atan2}}(y, x)\end{split}
$$
    
where $\mathop{\text{atan2}}(y, x)$ is the two-argument inverse tangent function <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#atan2" id="id2">[atan2]</a>. As any vector uniquely determines $\theta$ and $\varphi$, we sometimes also
write $\hat{\boldsymbol{\theta}}(\hat{\mathbf{v}})$ and $\hat{\boldsymbol{\varphi}}(\hat{\mathbf{v}})$ instead of $\hat{\boldsymbol{\theta}} (\theta, \varphi)$ and $\hat{\boldsymbol{\varphi}}(\theta, \varphi)$.
    
A 3D rotation with yaw, pitch, and roll angles $\alpha$, $\beta$, and $\gamma$, respectively, is expressed by the matrix

$$
\begin{align}
    \mathbf{R}(\alpha, \beta, \gamma) = \mathbf{R}_z(\alpha)\mathbf{R}_y(\beta)\mathbf{R}_x(\gamma)
\end{align}
$$
    
where $\mathbf{R}_z(\alpha)$, $\mathbf{R}_y(\beta)$, and $\mathbf{R}_x(\gamma)$ are rotation matrices around the $z$, $y$, and $x$ axes, respectively, which are defined as

$$
\begin{split}\begin{align}
    \mathbf{R}_z(\alpha) &= \begin{pmatrix}
                    \cos(\alpha) & -\sin(\alpha) & 0\\
                    \sin(\alpha) & \cos(\alpha) & 0\\
                    0 & 0 & 1
                  \end{pmatrix}\\
    \mathbf{R}_y(\beta) &= \begin{pmatrix}
                    \cos(\beta) & 0 & \sin(\beta)\\
                    0 & 1 & 0\\
                    -\sin(\beta) & 0 & \cos(\beta)
                  \end{pmatrix}\\
    \mathbf{R}_x(\gamma) &= \begin{pmatrix}
                        1 & 0 & 0\\
                        0 & \cos(\gamma) & -\sin(\gamma)\\
                        0 & \sin(\gamma) & \cos(\gamma)
                  \end{pmatrix}.
\end{align}\end{split}
$$
    
A closed-form expression for $\mathbf{R}(\alpha, \beta, \gamma)$ can be found in (7.1-4) <a class="reference internal" href="api/channel.wireless.html#tr38901" id="id3">[TR38901]</a>.
The reverse rotation is simply defined by $\mathbf{R}^{-1}(\alpha, \beta, \gamma)=\mathbf{R}^\mathsf{T}(\alpha, \beta, \gamma)$.
A vector $\mathbf{x}$ defined in a first coordinate system is represented in a second coordinate system rotated by $\mathbf{R}(\alpha, \beta, \gamma)$ with respect to the first one as $\mathbf{x}'=\mathbf{R}^\mathsf{T}(\alpha, \beta, \gamma)\mathbf{x}$.
If a point in the first coordinate system has spherical angles $(\theta, \varphi)$, the corresponding angles $(\theta', \varphi')$ in the second coordinate system can be found to be

$$
\begin{split}\begin{align}
    \theta' &= \cos^{-1}\left( \mathbf{z}^\mathsf{T} \mathbf{R}^\mathsf{T}(\alpha, \beta, \gamma)\hat{\mathbf{r}}(\theta, \varphi)          \right)\\
    \varphi' &= \arg\left( \left( \mathbf{x} + j\mathbf{y}\right)^\mathsf{T} \mathbf{R}^\mathsf{T}(\alpha, \beta, \gamma)\hat{\mathbf{r}}(\theta, \varphi) \right).
\end{align}\end{split}
$$
    
For a vector field $\mathbf{F}'(\theta',\varphi')$ expressed in local spherical coordinates

$$
\mathbf{F}'(\theta',\varphi') = F_{\theta'}(\theta',\varphi')\hat{\boldsymbol{\theta}}'(\theta',\varphi') + F_{\varphi'}(\theta',\varphi')\hat{\boldsymbol{\varphi}}'(\theta',\varphi')
$$
    
that are rotated by $\mathbf{R}=\mathbf{R}(\alpha, \beta, \gamma)$ with respect to the GCS, the spherical field components in the GCS can be expressed as

$$
\begin{split}\begin{bmatrix}
    F_\theta(\theta, \varphi) \\
    F_\varphi(\theta, \varphi)
\end{bmatrix} =
\begin{bmatrix}
    \hat{\boldsymbol{\theta}}(\theta,\varphi)^\mathsf{T}\mathbf{R}\hat{\boldsymbol{\theta}}'(\theta',\varphi') & \hat{\boldsymbol{\theta}}(\theta,\varphi)^\mathsf{T}\mathbf{R}\hat{\boldsymbol{\varphi}}'(\theta',\varphi') \\
    \hat{\boldsymbol{\varphi}}(\theta,\varphi)^\mathsf{T}\mathbf{R}\hat{\boldsymbol{\theta}}'(\theta',\varphi') & \hat{\boldsymbol{\varphi}}(\theta,\varphi)^\mathsf{T}\mathbf{R}\hat{\boldsymbol{\varphi}}'(\theta',\varphi')
\end{bmatrix}
\begin{bmatrix}
    F_{\theta'}(\theta', \varphi') \\
    F_{\varphi'}(\theta', \varphi')
\end{bmatrix}\end{split}
$$
    
so that

$$
\mathbf{F}(\theta,\varphi) = F_{\theta}(\theta,\varphi)\hat{\boldsymbol{\theta}}(\theta,\varphi) + F_{\varphi}(\theta,\varphi)\hat{\boldsymbol{\varphi}}(\theta,\varphi).
$$
    
It sometimes also useful to find the rotation matrix that maps a unit vector $\hat{\mathbf{a}}$ to $\hat{\mathbf{b}}$. This can be achieved with the help of Rodrigues’ rotation formula <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#wikipedia-rodrigues" id="id4">[Wikipedia_Rodrigues]</a> which defines the matrix

$$
\mathbf{R}(\hat{\mathbf{a}}, \hat{\mathbf{b}}) = \mathbf{I} + \sin(\theta)\mathbf{K} + (1-\cos(\theta)) \mathbf{K}^2
$$
    
where

$$
\begin{split}\mathbf{K} &= \begin{bmatrix}
                        0 & -\hat{k}_z &  \hat{k}_y \\
                \hat{k}_z &          0 & -\hat{k}_x \\
               -\hat{k}_y &  \hat{k}_x &          0
             \end{bmatrix}\\
\hat{\mathbf{k}} &= \frac{\hat{\mathbf{a}} \times \hat{\mathbf{b}}}{\lVert \hat{\mathbf{a}} \times \hat{\mathbf{b}} \rVert}\\
\theta &=\hat{\mathbf{a}}^\mathsf{T}\hat{\mathbf{b}}\end{split}
$$
    
such that $\mathbf{R}(\hat{\mathbf{a}}, \hat{\mathbf{b}})\hat{\mathbf{a}}=\hat{\mathbf{b}}$.

## Planar Time-Harmonic Waves<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#planar-time-harmonic-waves" title="Permalink to this headline"></a>
    
A time-harmonic planar electric wave $\mathbf{E}(\mathbf{x}, t)\in\mathbb{C}^3$ travelling in a homogeneous medium with wave vector $\mathbf{k}\in\mathbb{C}^3$ can be described at position $\mathbf{x}\in\mathbb{R}^3$ and time $t$ as

$$
\begin{split}\begin{align}
    \mathbf{E}(\mathbf{x}, t) &= \mathbf{E}_0 e^{j(\omega t -\mathbf{k}^{\mathsf{H}}\mathbf{x})}\\
                              &= \mathbf{E}(\mathbf{x}) e^{j\omega t}
\end{align}\end{split}
$$
    
where $\mathbf{E}_0\in\mathbb{C}^3$ is the field phasor. The wave vector can be decomposed as $\mathbf{k}=k \hat{\mathbf{k}}$, where $\hat{\mathbf{k}}$ is a unit norm vector, $k=\omega\sqrt{\varepsilon\mu}$ is the wave number, and $\omega=2\pi f$ is the angular frequency. The permittivity $\varepsilon$ and permeability $\mu$ are defined as

$$
\varepsilon = \eta \varepsilon_0
$$

$$
\mu = \mu_r \mu_0
$$
    
where $\eta$ and $\varepsilon_0$ are the complex relative and vacuum permittivities, $\mu_r$ and $\mu_0$ are the relative and vacuum permeabilities, and $\sigma$ is the conductivity.
The complex relative permittivity $\eta$ is given as

$$
\eta = \varepsilon_r - j\frac{\sigma}{\varepsilon_0\omega}
$$
    
where $\varepsilon_r$ is the real relative permittivity of a non-conducting dielectric.
    
With these definitions, the speed of light is given as (Eq. 4-28d) <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#balanis" id="id5">[Balanis]</a>

$$
c=\frac{1}{\sqrt{\varepsilon_0\varepsilon_r\mu}}\left\{\frac12\left(\sqrt{1+\left(\frac{\sigma}{\omega\varepsilon_0\varepsilon_r}\right)^2}+1\right)\right\}^{-\frac{1}{2}}
$$
    
where the factor in curly brackets vanishes for non-conducting materials. The speed of light in vacuum is denoted $c_0=\frac{1}{\sqrt{\varepsilon_0 \mu_0}}$ and the vacuum wave number $k_0=\frac{\omega}{c_0}$. In conducting materials, the wave number is complex which translates to propagation losses.
    
The associated magnetic field $\mathbf{H}(\mathbf{x}, t)\in\mathbb{C}^3$ is

$$
\mathbf{H}(\mathbf{x}, t) = \frac{\hat{\mathbf{k}}\times  \mathbf{E}(\mathbf{x}, t)}{Z} = \mathbf{H}(\mathbf{x})e^{j\omega t}
$$
    
where $Z=\sqrt{\mu/\varepsilon}$ is the wave impedance. The vacuum impedance is denoted by $Z_0=\sqrt{\mu_0/\varepsilon_0}\approx 376.73\,\Omega$.
    
The time-averaged Poynting vector is defined as

$$
\mathbf{S}(\mathbf{x}) = \frac{1}{2} \Re\left\{\mathbf{E}(\mathbf{x})\times  \mathbf{H}(\mathbf{x})\right\}
                       = \frac{1}{2} \Re\left\{\frac{1}{Z} \right\} \lVert \mathbf{E}(\mathbf{x})  \rVert^2 \hat{\mathbf{k}}
$$
    
which describes the directional energy flux (W/m²), i.e., energy transfer per unit area per unit time.
    
Note that the actual electromagnetic waves are the real parts of $\mathbf{E}(\mathbf{x}, t)$ and $\mathbf{H}(\mathbf{x}, t)$.

## Far Field of a Transmitting Antenna<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#far-field-of-a-transmitting-antenna" title="Permalink to this headline"></a>
    
We assume that the electric far field of an antenna in free space can be described by a spherical wave originating from the center of the antenna:

$$
\mathbf{E}(r, \theta, \varphi, t) = \mathbf{E}(r,\theta, \varphi) e^{j\omega t} = \mathbf{E}_0(\theta, \varphi) \frac{e^{-jk_0r}}{r} e^{j\omega t}
$$
    
where $\mathbf{E}_0(\theta, \varphi)$ is the electric field phasor, $r$ is the distance (or radius), $\theta$ the zenith angle, and $\varphi$ the azimuth angle.
In contrast to a planar wave, the field strength decays as $1/r$.
    
The complex antenna field pattern $\mathbf{F}(\theta, \varphi)$ is defined as

$$
\begin{align}
    \mathbf{F}(\theta, \varphi) = \frac{ \mathbf{E}_0(\theta, \varphi)}{\max_{\theta,\varphi}\lVert  \mathbf{E}_0(\theta, \varphi) \rVert}.
\end{align}
$$
    
The time-averaged Poynting vector for such a spherical wave is

$$
\mathbf{S}(r, \theta, \varphi) = \frac{1}{2Z_0}\lVert \mathbf{E}(r, \theta, \varphi) \rVert^2 \hat{\mathbf{r}}
$$
    
where $\hat{\mathbf{r}}$ is the radial unit vector. It simplifies for an ideal isotropic antenna with input power $P_\text{T}$ to

$$
\mathbf{S}_\text{iso}(r, \theta, \varphi) = \frac{P_\text{T}}{4\pi r^2} \hat{\mathbf{r}}.
$$
    
The antenna gain $G$ is the ratio of the maximum radiation power density of the antenna in radial direction and that of an ideal isotropic radiating antenna:

$$
    G = \frac{\max_{\theta,\varphi}\lVert \mathbf{S}(r, \theta, \varphi)\rVert}{ \lVert\mathbf{S}_\text{iso}(r, \theta, \varphi)\rVert}
      = \frac{2\pi}{Z_0 P_\text{T}} \max_{\theta,\varphi}\lVert \mathbf{E}_0(\theta, \varphi) \rVert^2.
$$
    
One can similarly define a gain with directional dependency by ignoring the computation of the maximum the last equation:

$$
    G(\theta, \varphi) = \frac{2\pi}{Z_0 P_\text{T}} \lVert \mathbf{E}_0(\theta, \varphi) \rVert^2 = G \lVert \mathbf{F}(\theta, \varphi) \rVert^2.
$$
    
If one uses in the last equation the radiated power $P=\eta_\text{rad} P_\text{T}$, where $\eta_\text{rad}$ is the radiation efficiency, instead of the input power $P_\text{T}$, one obtains the directivity $D(\theta,\varphi)$. Both are related through $G(\theta, \varphi)=\eta_\text{rad} D(\theta, \varphi)$.

**Antenna pattern**
    
Since $\mathbf{F}(\theta, \varphi)$ contains no information about the maximum gain $G$ and $G(\theta, \varphi)$ does not carry any phase information, we define the <cite>antenna pattern</cite> $\mathbf{C}(\theta, \varphi)$ as

$$
\mathbf{C}(\theta, \varphi) = \sqrt{G}\mathbf{F}(\theta, \varphi)
$$
    
such that $G(\theta, \varphi)= \lVert\mathbf{C}(\theta, \varphi) \rVert^2$.
    
Using the spherical unit vectors $\hat{\boldsymbol{\theta}}\in\mathbb{R}^3$
and $\hat{\boldsymbol{\varphi}}\in\mathbb{R}^3$,
we can rewrite $\mathbf{C}(\theta, \varphi)$ as

$$
\mathbf{C}(\theta, \varphi) = C_\theta(\theta,\varphi) \hat{\boldsymbol{\theta}} + C_\varphi(\theta,\varphi) \hat{\boldsymbol{\varphi}}
$$
    
where $C_\theta(\theta,\varphi)\in\mathbb{C}$ and $C_\varphi(\theta,\varphi)\in\mathbb{C}$ are the
<cite>zenith pattern</cite> and <cite>azimuth pattern</cite>, respectively.
    
Combining <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-f">(10)</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-g">(12)</a>, we can obtain the following expression of the electric far field

$$
\mathbf{E}_\text{T}(r,\theta_\text{T},\varphi_\text{T}) = \sqrt{ \frac{P_\text{T} G_\text{T} Z_0}{2\pi}} \frac{e^{-jk_0 r}}{r} \mathbf{F}_\text{T}(\theta_\text{T}, \varphi_\text{T})
$$
    
where we have added the subscript $\text{T}$ to all quantities that are specific to the transmitting antenna.
    
The input power $P_\text{T}$ of an antenna with (conjugate matched) impedance $Z_\text{T}$, fed by a voltage source with complex amplitude $V_\text{T}$, is given by (see, e.g., <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#wikipedia" id="id6">[Wikipedia]</a>)

$$
P_\text{T} = \frac{|V_\text{T}|^2}{8\Re\{Z_\text{T}\}}.
$$

**Normalization of antenna patterns**
    
The radiated power $\eta_\text{rad} P_\text{T}$ of an antenna can be obtained by integrating the Poynting vector over the surface of a closed sphere of radius $r$ around the antenna:

$$
\begin{split}\begin{align}
    \eta_\text{rad} P_\text{T} &=  \int_0^{2\pi}\int_0^{\pi} \mathbf{S}(r, \theta, \varphi)^\mathsf{T} \hat{\mathbf{r}} r^2 \sin(\theta)d\theta d\varphi \\
                    &= \int_0^{2\pi}\int_0^{\pi} \frac{1}{2Z_0} \lVert \mathbf{E}(r, \theta, \varphi) \rVert^2 r^2\sin(\theta)d\theta d\varphi \\
                    &= \frac{P_\text{T}}{4 \pi} \int_0^{2\pi}\int_0^{\pi} G(\theta, \varphi) \sin(\theta)d\theta d\varphi.
\end{align}\end{split}
$$
    
We can see from the last equation that the directional gain of any antenna must satisfy

$$
\int_0^{2\pi}\int_0^{\pi} G(\theta, \varphi) \sin(\theta)d\theta d\varphi = 4 \pi \eta_\text{rad}.
$$
