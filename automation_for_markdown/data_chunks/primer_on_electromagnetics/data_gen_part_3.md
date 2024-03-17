# Primer on Electromagnetics<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#primer-on-electromagnetics" title="Permalink to this headline"></a>
    
This section provides useful background for the general understanding of ray tracing for wireless propagation modelling. In particular, our goal is to provide a concise definition of a <cite>channel impulse response</cite> between a transmitting and receiving antenna, as done in (Ch. 2 & 3) <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#wiesbeck" id="id1">[Wiesbeck]</a>. The notations and definitions will be used in the API documentation of Sionna’s <a class="reference internal" href="api/rt.html">Ray Tracing module</a>.

# Table of Content
## Diffraction<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#diffraction" title="Permalink to this headline"></a>
  
  

## Diffraction<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#diffraction" title="Permalink to this headline"></a>
    
While modern geometrical optics (GO) <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#kline" id="id9">[Kline]</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#luneberg" id="id10">[Luneberg]</a> can accurately describe phase and polarization properties of electromagnetic fields undergoing reflection and refraction (transmission) as described above, they fail to account for the phenomenon of diffraction, e.g., bending of waves around corners. This leads to the undesired and physically incorrect effect that the field abruptly falls to zero at geometrical shadow boundaries (for incident and reflected fields).
    
Joseph Keller presented in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#keller62" id="id11">[Keller62]</a> a method which allowed the incorporation of diffraction into GO which is known as the geometrical theory of diffraction (GTD). He introduced the notion of diffracted rays that follow the law of edge diffraction, i.e., the diffracted and incident rays make the same angle with the edge at the point of diffraction and lie on opposite sides of the plane normal to the edge. The GTD suffers, however from several shortcomings, most importantly the fact that the diffracted field is infinite at shadow boundaries.
    
The uniform theory of diffraction (UTD) <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#kouyoumjian74" id="id12">[Kouyoumjian74]</a> alleviates this problem and provides solutions that are uniformly valid, even at shadow boundaries. For a great introduction to the UTD, we refer to <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#mcnamara90" id="id13">[McNamara90]</a>. While <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#kouyoumjian74" id="id14">[Kouyoumjian74]</a> deals with diffraction at edges of perfectly conducting surfaces, it was heuristically extended to finitely conducting wedges in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#luebbers84" id="id15">[Luebbers84]</a>. This solution, which is also recomended by the ITU <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#iturp52615" id="id16">[ITURP52615]</a>, is implemented in Sionna. However, both <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#luebbers84" id="id17">[Luebbers84]</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#iturp52615" id="id18">[ITURP52615]</a> only deal with two-dimensional scenes where source and observation lie in the same plane, orthogonal to the edge. We will provide below the three-dimensional version of <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#luebbers84" id="id19">[Luebbers84]</a>, following the defintitions of (Ch. 6) <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#mcnamara90" id="id20">[McNamara90]</a>. A similar result can be found, e.g., in (Eq. 6-29—6-39) <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#metis" id="id21">[METIS]</a>.
<a class="reference internal image-reference" href="_images/kellers_cone.svg"><img alt="_images/kellers_cone.svg" src="_images/kellers_cone.svg" width="80%" /></a>
<p class="caption">Fig. 2 Incident and diffracted rays for an infinitely long wedge in an edge-fixed coordinate system.<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#id30" title="Permalink to this image"></a>
    
We consider an infinitely long wedge with unit norm edge vector $\hat{\mathbf{e}}$, as shown in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#fig-kellers-cone">Fig. 2</a>. An incident ray of a spherical wave with field phasor $\mathbf{E}_i(S')$ at point $S'$ propagates in the direction $\hat{\mathbf{s}}'$ and is diffracted at point $Q_d$ on the edge. The diffracted ray of interest (there are infinitely many on Keller’s cone) propagates
in the direction $\hat{\mathbf{s}}$ towards the point of observation $S$. We denote by $s'=\lVert S'-Q_d \rVert$ and $s=\lVert Q_d - S\rVert$ the lengths of the incident and diffracted path segments, respectively. By the law of edge diffraction, the angles $\beta_0'$ and $\beta_0$ between the edge and the incident and diffracted rays, respectively, satisfy:

$$
\begin{equation}
    \cos(\beta_0') = |\hat{\mathbf{s}}'^\textsf{T}\hat{\mathbf{e}}| = |\hat{\mathbf{s}}^\textsf{T}\hat{\mathbf{e}}| = \cos(\beta_0).
\end{equation}
$$
    
To be able to express the diffraction coefficients as a 2x2 matrix—similar to what is done for reflection and refraction—the incident field must be resolved into two components $E_{i,\phi'}$ and $E_{i,\beta_0'}$, the former orthogonal and the latter parallel to the edge-fixed plane of incidence, i.e., the plane containing $\hat{\mathbf{e}}$ and $\hat{\mathbf{s}}'$. The diffracted field is then represented by two components $E_{d,\phi}$ and $E_{d,\beta_0}$ that are respectively orthogonal and parallel to the edge-fixed plane of diffraction, i.e., the plane containing $\hat{\mathbf{e}}$ and $\hat{\mathbf{s}}$.
The corresponding component unit vectors are defined as

$$
\begin{split}\begin{align}
    \hat{\boldsymbol{\phi}}' &= \frac{\hat{\mathbf{s}}' \times \hat{\mathbf{e}}}{\lVert \hat{\mathbf{s}}' \times \hat{\mathbf{e}} \rVert }\\
    \hat{\boldsymbol{\beta}}_0' &=  \hat{\boldsymbol{\phi}}' \times \hat{\mathbf{s}}' \\
    \hat{\boldsymbol{\phi}} &= -\frac{\hat{\mathbf{s}} \times \hat{\mathbf{e}}}{\lVert \hat{\mathbf{s}} \times \hat{\mathbf{e}} \rVert }\\
    \hat{\boldsymbol{\beta}}_0 &=  \hat{\boldsymbol{\phi}} \times \hat{\mathbf{s}}.
\end{align}\end{split}
$$
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#fig-diffraction">Fig. 3</a> below shows the top view on the wedge that we need for some additional definitions.
<a class="reference internal image-reference" href="_images/diffraction.svg"><img alt="_images/diffraction.svg" src="_images/diffraction.svg" width="80%" /></a>
<p class="caption">Fig. 3 Top view on the wedge with edge vector pointing upwards.<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#id31" title="Permalink to this image"></a>
    
The wedge has two faces called <em>0-face</em> and <em>n-face</em>, respectively, with surface normal vectors $\hat{\mathbf{n}}_0$ and $\hat{\mathbf{n}}_n$. The exterior wedge angle is $n\pi$, with $1\le n \le 2$. Note that the surfaces are chosen such that $\hat{\mathbf{e}} = \hat{\mathbf{n}}_0 \times \hat{\mathbf{n}}_n$. For $n=2$, the wedge reduces to a screen and the choice of the <em>0-face</em> and <em>n-face</em> is arbitrary as they point in opposite directions.
    
The incident and diffracted rays have angles $\phi'$ and $\phi$ measured with respect to the <em>0-face</em> in the plane perpendicular to the edge.
They can be computed as follows:

$$
\begin{split}\begin{align}
    \phi' & = \pi - \left[\pi - \cos^{-1}\left( -\hat{\mathbf{s}}_t'^\textsf{T} \hat{\mathbf{t}}_0\right) \right] \mathop{\text{sgn}}\left(-\hat{\mathbf{s}}_t'^\textsf{T} \hat{\mathbf{n}}_0\right)\\
    \phi & = \pi - \left[\pi - \cos^{-1}\left( \hat{\mathbf{s}}_t^\textsf{T} \hat{\mathbf{t}}_0\right) \right] \mathop{\text{sgn}}\left(\hat{\mathbf{s}}_t^\textsf{T} \hat{\mathbf{n}}_0\right)
\end{align}\end{split}
$$
    
where

$$
\begin{split}\begin{align}
    \hat{\mathbf{t}}_0 &= \hat{\mathbf{n}}_0 \times \hat{\mathbf{e}}\\
    \hat{\mathbf{s}}_t' &= \frac{ \hat{\mathbf{s}}' - \left( \hat{\mathbf{s}}'^\textsf{T}\hat{\mathbf{e}} \right)\hat{\mathbf{e}} }{\lVert \hat{\mathbf{s}}' - \left( \hat{\mathbf{s}}'^\textsf{T}\hat{\mathbf{e}} \right)\hat{\mathbf{e}}  \rVert}\\
    \hat{\mathbf{s}}_t  &= \frac{ \hat{\mathbf{s}} - \left( \hat{\mathbf{s}}^\textsf{T}\hat{\mathbf{e}} \right)\hat{\mathbf{e}} }{\lVert \hat{\mathbf{s}} - \left( \hat{\mathbf{s}}^\textsf{T}\hat{\mathbf{e}} \right)\hat{\mathbf{e}}  \rVert}
\end{align}\end{split}
$$
    
are the unit vector tangential to the <em>0-face</em>, as well as the unit vectors pointing in the directions of $\hat{\mathbf{s}}'$ and $\hat{\mathbf{s}}$, projected on the plane perpendicular to the edge, respectively. The function $\mathop{\text{sgn}}(x)$ is defined in this context as

$$
\begin{split}\mathop{\text{sgn}}(x) = \begin{cases}
                         1  &, x \ge 0\\
                         -1 &, x< 0.
                         \end{cases}\end{split}
$$
    
With these definitions, the diffracted field at point $S$ can be computed from the incoming field at point $S'$ as follows:

$$
\begin{split}\begin{align}
    \begin{bmatrix}
        E_{d,\phi} \\
        E_{d,\beta_0}
    \end{bmatrix} (S) = - \left( \left(D_1 + D_2\right)\mathbf{I} - D_3 \mathbf{R}_n - D_4\mathbf{R}_0 \right)\begin{bmatrix}
        E_{i,\phi'} \\
        E_{i,\beta_0'}
    \end{bmatrix}(S') \sqrt{\frac{1}{s's(s'+s)}} e^{-jk(s'+s)}
\end{align}\end{split}
$$
    
where $k=2\pi/\lambda$ is the wave number and the matrices $\mathbf{R}_\nu,\, \nu \in [0,n]$, are given as

$$
\begin{split}\begin{align}
    \mathbf{R}_\nu = \mathbf{W}\left(\hat{\boldsymbol{\phi}}, \hat{\boldsymbol{\beta}}_0, \hat{\mathbf{e}}_{r, \perp, \nu}, \hat{\mathbf{e}}_{r, \parallel, \nu}  \right)
                    \begin{bmatrix}
                        r_{\perp}(\theta_{r,\nu}, \eta_{\nu}) & 0\\
                        0 & r_{\parallel}(\theta_{r,\nu}, \eta_{nu})
                    \end{bmatrix}
                     \mathbf{W}\left( \hat{\mathbf{e}}_{i, \perp, \nu}, \hat{\mathbf{e}}_{i, \parallel, \nu}, \hat{\boldsymbol{\phi}}', \hat{\boldsymbol{\beta}}_0' \right)
\end{align}\end{split}
$$
    
with $\mathbf{W}(\cdot)$ as defined in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-w">(30)</a>, where $r_{\perp}(\theta_{r,\nu}, \eta_{\nu})$ and $r_{\parallel}(\theta_{r,\nu}, \eta_{\nu})$ are the Fresnel reflection coefficents from <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-fresnel-vac">(34)</a>, evaluated for the complex relative permittivities $\eta_{\nu}$ and angles $\theta_{r_,\nu}$ with cosines

$$
\begin{split}\begin{align}
    \cos\left(\theta_{r,0}\right) &= \left|\sin(\phi') \right|\\
    \cos\left(\theta_{r,n}\right) &= \left|\sin(n\pi -\phi) \right|.
\end{align}\end{split}
$$
    
and where

$$
\begin{split}\begin{align}
    \hat{\mathbf{e}}_{i,\perp,\nu} &= \frac{ \hat{\mathbf{s}}' \times \hat{\mathbf{n}}_{\nu} }{\lVert \hat{\mathbf{s}}' \times \hat{\mathbf{n}}_{\nu} \rVert}\\
    \hat{\mathbf{e}}_{i,\parallel,\nu} &=  \hat{\mathbf{e}}_{i,\perp,\nu} \times \hat{\mathbf{s}}'\\
    \hat{\mathbf{e}}_{r,\perp,\nu} &=  \hat{\mathbf{e}}_{i,\perp,\nu}\\
    \hat{\mathbf{e}}_{r,\parallel,\nu} &=  \hat{\mathbf{e}}_{i,\perp,\nu} \times \hat{\mathbf{s}}
\end{align}\end{split}
$$
    
as already defined in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-fresnel-in-vectors">(29)</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-fresnel-out-vectors">(31)</a>, but made explicit here for the case of diffraction. The matrices $\mathbf{R}_\nu$ simply describe the reflected field from both surfaces in the basis used for the description of the diffraction process. Note that the absolute value is used in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-diffraction-cos">(36)</a> to account for virtual reflections from shadowed surfaces, see the discussion in (p.185) <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#mcnamara90" id="id22">[McNamara90]</a>.
The diffraction coefficients $D_1,\dots,D_4$ are computed as

$$
\begin{split}\begin{align}
    D_1 &= \frac{-e^{-\frac{j\pi}{4}}}{2n\sqrt{2\pi k} \sin(\beta_0)} \mathop{\text{cot}}\left( \frac{\pi+(\phi-\phi')}{2n}\right) F\left( k L a^+(\phi-\phi')\right)\\
    D_2 &= \frac{-e^{-\frac{j\pi}{4}}}{2n\sqrt{2\pi k} \sin(\beta_0)} \mathop{\text{cot}}\left( \frac{\pi-(\phi-\phi')}{2n}\right) F\left( k L a^-(\phi-\phi')\right)\\
    D_3 &= \frac{-e^{-\frac{j\pi}{4}}}{2n\sqrt{2\pi k} \sin(\beta_0)} \mathop{\text{cot}}\left( \frac{\pi+(\phi+\phi')}{2n}\right) F\left( k L a^+(\phi+\phi')\right)\\
    D_4 &= \frac{-e^{-\frac{j\pi}{4}}}{2n\sqrt{2\pi k} \sin(\beta_0)} \mathop{\text{cot}}\left( \frac{\pi-(\phi+\phi')}{2n}\right) F\left( k L a^-(\phi+\phi')\right)
\end{align}\end{split}
$$
    
where

$$
\begin{split}\begin{align}
    L &= \frac{ss'}{s+s'}\sin^2(\beta_0)\\
    a^{\pm}(\beta) &= 2\cos^2\left(\frac{2n\pi N^{\pm}-\beta}{2}\right)\\
    N^{\pm} &= \mathop{\text{round}}\left(\frac{\beta\pm\pi}{2n\pi}\right)\\
    F(x) &= 2j\sqrt{x}e^{jx}\int_{\sqrt{x}}^\infty e^{-jt^2}dt
\end{align}\end{split}
$$
    
and $\mathop{\text{round}}()$ is the function that rounds to the closest integer. The function $F(x)$ can be expressed with the help of the standard Fresnel integrals <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#fresnel" id="id23">[Fresnel]</a>

$$
\begin{split}\begin{align}
    S(x) &= \int_0^x \sin\left( \pi t^2/2 \right)dt \\
    C(x) &= \int_0^x \cos\left( \pi t^2/2 \right)dt
\end{align}\end{split}
$$
    
as

$$
\begin{align}
    F(x) = \sqrt{\frac{\pi x}{2}} e^{jx} \left[1+j-2\left( S\left(\sqrt{2x/\pi}\right) +jC\left(\sqrt{2x/\pi}\right) \right) \right].
\end{align}
$$
