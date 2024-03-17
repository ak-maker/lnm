# Primer on Electromagnetics<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#primer-on-electromagnetics" title="Permalink to this headline"></a>
    
This section provides useful background for the general understanding of ray tracing for wireless propagation modelling. In particular, our goal is to provide a concise definition of a <cite>channel impulse response</cite> between a transmitting and receiving antenna, as done in (Ch. 2 & 3) <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#wiesbeck" id="id1">[Wiesbeck]</a>. The notations and definitions will be used in the API documentation of Sionna’s <a class="reference internal" href="api/rt.html">Ray Tracing module</a>.

# Table of Content
## Scattering<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#scattering" title="Permalink to this headline"></a>
## References
  
  

## Scattering<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#scattering" title="Permalink to this headline"></a>
    
When an electromagnetic wave impinges on a surface, one part of the energy gets reflected while the other part gets refracted, i.e., it propagates into the surface.
We distinguish between two types of reflection, specular and diffuse. The former type is discussed in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#reflection-and-refraction">Reflection and Refraction</a> and we will focus now on the latter type which is also called diffuse scattering. When a rays hits a diffuse reflection surface, it is not reflected into a single (specular) direction but rather scattered toward many different directions. Since most surfaces give both specular and diffuse reflections, we denote by $S^2$ the fraction of the reflected energy that is diffusely scattered, where $S\in[0,1]$ is the so-called <em>scattering coefficient</em> <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#degli-esposti07" id="id24">[Degli-Esposti07]</a>. Similarly, $R^2$ is the specularly reflected fraction of the reflected energy, where $R\in[0,1]$ is the <em>reflection reduction factor</em>. The following relationship between $R$ and $S$ holds:

$$
R = \sqrt{1-S^2}.
$$
    
Whenever a material has a scattering coefficient $S>0$, the Fresnel reflection coefficents in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-fresnel">(33)</a> must be multiplied by $R$. These <em>reduced</em> coefficients must then be also used in the compuation of the diffraction coefficients <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-diff-mat">(35)</a>.
<a class="reference internal image-reference" href="_images/scattering.svg"><img alt="_images/scattering.svg" src="_images/scattering.svg" width="80%" /></a>
<p class="caption">Fig. 4 Diffuse and specular reflection of an incoming wave.<a class="headerlink" href="https://nvlabs.github.io/sionna/em_primer.html#id32" title="Permalink to this image"></a>
    
Let us consider an incoming locally planar linearly polarized wave with field phasor $\mathbf{E}_\text{i}(\mathbf{q})$ at the scattering point $\mathbf{q}$ on the surface, as shown in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#fig-scattering">Fig. 4</a>. We focus on the scattered field of and infinitesimally small surface element $dA$ in the direction $\hat{\mathbf{k}}_\text{s}$. Note that the surface normal $\hat{\mathbf{n}}$ has an arbitrary orientation with respect to the global coordinate system, whose $(x,y,z)$ axes are shown in green dotted lines.
The incoming field phasor can be represented by two arbitrary orthogonal polarization components (both orthogonal to the incoming wave vector $\hat{\mathbf{k}}_i$):

$$
\begin{split}\begin{align}
\mathbf{E}_\text{i} &= E_{\text{i},s} \hat{\mathbf{e}}_{\text{i},s} + E_{\text{i},p} \hat{\mathbf{e}}_{\text{i},p} \\
                    &= E_{\text{i},\perp} \hat{\mathbf{e}}_{\text{i},\perp} + E_{\text{i},\parallel} \hat{\mathbf{e}}_{\text{i},\parallel} \\
                    &= E_{\text{i},\text{pol}} \hat{\mathbf{e}}_{\text{i},\text{pol}} + E_{\text{i},\text{xpol}} \hat{\mathbf{e}}_{\text{i},\text{xpol}}
\end{align}\end{split}
$$
    
where me have omitted the dependence of the field strength on the position $\mathbf{q}$ for brevity.
The second representation via $(E_{\text{i},\perp}, E_{\text{i},\parallel})$ is used for the computation of the specularly reflected field as explained in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#reflection-and-refraction">Reflection and refraction</a>. The third representation via $(E_{\text{i},\text{pol}}, E_{\text{i},\text{xpol}})$ will be used to express the scattered field, where

$$
\begin{split}\begin{align}
\hat{\mathbf{e}}_{\text{i},\text{pol}} &= = \frac{\Re\left\{\mathbf{E}_\text{i}\right\}}{\lVert \Re\left\{\mathbf{E}_\text{i}\right\} \rVert} =  \frac{\Re\left\{E_{\text{i},s}\right\}}{ \lVert\Re\left\{\mathbf{E}_\text{i} \right\} \rVert} \hat{\mathbf{e}}_{\text{i},s} + \frac{\Re\left\{E_{\text{i},p}\right\}}{\lVert\Re\left\{\mathbf{E}_\text{i} \right\} \rVert} \hat{\mathbf{e}}_{\text{i},p}\\
\hat{\mathbf{e}}_{\text{i},\text{xpol}} &= \hat{\mathbf{e}}_\text{pol} \times \hat{\mathbf{k}}_\text{i}
\end{align}\end{split}
$$
    
such that $|E_{\text{i},\text{pol}}|=\lVert \mathbf{E}_\text{i} \rVert$ and $E_{\text{i},\text{xpol}}=0$. That means that $\hat{\mathbf{e}}_{\text{i},\text{pol}}$ points toward the polarization direction which carries all of the energy.
    
According to (Eq. 9) <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#degli-esposti11" id="id25">[Degli-Esposti11]</a>, the diffusely scattered field $\mathbf{E}_\text{s}(\mathbf{r})$ at the observation point $\mathbf{r}$ can be modeled as
$\mathbf{E}_\text{s}(\mathbf{r})=E_{\text{s}, \theta}\hat{\boldsymbol{\theta}}(\hat{\mathbf{k}}_\text{s}) + E_{\text{s}, \varphi}\hat{\boldsymbol{\varphi}}(\hat{\mathbf{k}}_\text{s})$, where
$\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\varphi}}$ are defined in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-spherical-vecs">(1)</a> and the orthogonal field components are computed as

$$
\begin{split}\begin{bmatrix}E_{\text{s}, \theta} \\ E_{\text{s}, \varphi} \end{bmatrix}(\mathbf{r}) &= \frac{\lVert \mathbf{E}_\text{s}(\mathbf{q}) \rVert}{\lVert \mathbf{r} - \mathbf{q} \rVert}
\mathbf{W}\left( \hat{\boldsymbol{\theta}}(-\hat{\mathbf{k}}_\text{i}), \hat{\boldsymbol{\varphi}}(-\hat{\mathbf{k}}_\text{i}), \hat{\mathbf{e}}_{\text{i},\text{pol}}, \hat{\mathbf{e}}_{\text{i},\text{xpol}} \right)
 \begin{bmatrix} \sqrt{1-K_x}e^{j\chi_1} \\ \sqrt{K_x}e^{j\chi_2}  \end{bmatrix}\end{split}
$$
    
where $\mathbf{W}(\cdot)$ as defined in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-w">(30)</a>, $\chi_1, \chi_2 \in [0,2\pi]$ are independent random phase shifts, and the quantity $K_x\in[0,1]$ is defined by the scattering cross-polarization discrimination

$$
\text{XPD}_\text{s} = 10\log_{10}\left(\frac{|E_{\text{s}, \text{pol}}|^2}{|E_{\text{s}, \text{xpol}}|^2} \right) = 10\log_{10}\left(\frac{1-K_x}{K_x} \right).
$$
    
This quantity determines how much energy gets transfered from $\hat{\mathbf{e}}_{\text{i},\text{pol}}$ into the orthogonal polarization direction $\hat{\mathbf{e}}_{\text{i},\text{xpol}}$ through the scattering process. The matrix $\mathbf{W}$ is used to represent the scattered electric field in the vertical ($\hat{\boldsymbol{\theta}}$) and horizontal ($\hat{\boldsymbol{\varphi}}$) polarization components according to the incoming ray direction $-\hat{\mathbf{k}}_\text{i}$. It is then assumed that the same polarization is kept for the outgoing ray in the $\hat{\mathbf{k}}_\text{s}$ direction.
    
The squared amplitude of the diffusely scattered field in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-scattered-field">(38)</a> can be expressed as (Eq. 8) <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#degli-esposti07" id="id26">[Degli-Esposti07]</a>:

$$
\lVert \mathbf{E}_\text{s}(\mathbf{q})) \rVert^2 = \underbrace{\lVert \mathbf{E}_\text{i}(\mathbf{q}) \rVert^2 \cos(\theta_i) dA}_{\sim \text{incoming power} } \cdot \underbrace{\left(S\Gamma\right)^2}_{\text{fraction of diffusely reflected power}} \cdot \underbrace{f_\text{s}\left(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s}, \hat{\mathbf{n}}\right)}_{\text{scattering pattern}}
$$
    
where $\Gamma^2$ is the percentage of the incoming power that is reflected (specularly and diffuse), which can be computed as

$$
\Gamma = \frac{\sqrt{ |r_{\perp} E_{\text{i},\perp} |^2 + |r_{\parallel} E_{\text{i},\parallel} |^2}}
          {\lVert \mathbf{E}_\text{i}(\mathbf{q}) \rVert}
$$
    
where $r_{\perp}, r_{\parallel}$ are defined in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-fresnel">(33)</a>, $dA$ is the size of the small area element on the reflecting surface under consideration, and $f_\text{s}\left(\hat{\mathbf{k}}_i, \hat{\mathbf{k}}_s, \hat{\mathbf{n}}\right)$ is the <em>scattering pattern</em>, which has similarities with the bidirectional reflectance distribution function (BRDF) in computer graphics (Ch. 5.6.1) <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#pharr" id="id27">[Pharr]</a>.
The scattering pattern must be normalized to satisfy the condition

$$
\int_{0}^{\pi/2}\int_0^{2\pi} f_\text{s}\left(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s}, \hat{\mathbf{n}}\right) \sin(\theta_s) d\phi_s d\theta_s = 1
$$
    
which ensures the power balance between the incoming, reflected, and refracted fields.

**Example scattering patterns**
    
The authors of <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#degli-esposti07" id="id28">[Degli-Esposti07]</a> derived several simple scattering patterns that were shown to achieve good agreement with measurements when correctly parametrized.
    
**Lambertian Model** (<a class="reference internal" href="api/rt.html#sionna.rt.LambertianPattern" title="sionna.rt.LambertianPattern">`LambertianPattern`</a>):
This model describes a perfectly diffuse scattering surface whose <em>scattering radiation lobe</em> has its maximum in the direction of the surface normal:

$$
f^\text{Lambert}_\text{s}\left(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s}, \hat{\mathbf{n}}\right) = \frac{\hat{\mathbf{n}}^\mathsf{T} \hat{\mathbf{k}}_\text{s} }{\pi} = \frac{\cos(\theta_s)}{\pi}
$$
    
**Directive Model** (<a class="reference internal" href="api/rt.html#sionna.rt.DirectivePattern" title="sionna.rt.DirectivePattern">`DirectivePattern`</a>):
This model assumes that the scattered field is concentrated around the direction of the specular reflection $\hat{\mathbf{k}}_\text{r}$ (defined in <a class="reference internal" href="https://nvlabs.github.io/sionna/em_primer.html#equation-reflected-refracted-vectors">(32)</a>). The width of the scattering lobe
can be controlled via the integer parameter $\alpha_\text{R}=1,2,\dots$:

$$
f^\text{directive}_\text{s}\left(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s}, \hat{\mathbf{n}}\right) = F_{\alpha_\text{R}}(\theta_i)^{-1} \left(\frac{ 1 + \hat{\mathbf{k}}_\text{r}^\mathsf{T} \hat{\mathbf{k}}_\text{s}}{2}\right)^{\alpha_\text{R}}
$$

$$
F_{\alpha}(\theta_i) = \frac{1}{2^\alpha} \sum_{k=0}^\alpha \binom{\alpha}{k} I_k,\qquad \theta_i =\cos^{-1}(-\hat{\mathbf{k}}_\text{i}^\mathsf{T}\hat{\mathbf{n}})
$$

$$
\begin{split}I_k = \frac{2\pi}{k+1} \begin{cases}
        1 & ,\quad k \text{ even} \\
        \cos(\theta_i) \sum_{w=0}^{(k-1)/2} \binom{2w}{w} \frac{\sin^{2w}(\theta_i)}{2^{2w}}  &,\quad k \text{ odd}
      \end{cases}\end{split}
$$
    
**Backscattering Lobe Model** (<a class="reference internal" href="api/rt.html#sionna.rt.BackscatteringPattern" title="sionna.rt.BackscatteringPattern">`BackscatteringPattern`</a>):
This model adds a scattering lobe to the directive model described above which points toward the direction from which the incident wave arrives (i.e., $-\hat{\mathbf{k}}_\text{i}$). The width of this lobe is controlled by the parameter $\alpha_\text{I}=1,2,\dots$. The parameter $\Lambda\in[0,1]$ determines the distribution of energy between both lobes. For $\Lambda=1$, this models reduces to the directive model.

$$
f^\text{bs}_\text{s}\left(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s}, \hat{\mathbf{n}}\right) = F_{\alpha_\text{R}, \alpha_\text{I}}(\theta_i)^{-1} \left[ \Lambda \left(\frac{ 1 + \hat{\mathbf{k}}_\text{r}^\mathsf{T} \hat{\mathbf{k}}_\text{s}}{2}\right)^{\alpha_\text{R}} + (1-\Lambda) \left(\frac{ 1 - \hat{\mathbf{k}}_\text{i}^\mathsf{T} \hat{\mathbf{k}}_\text{s}}{2}\right)^{\alpha_\text{I}}\right]
$$

$$
F_{\alpha, \beta}(\theta_i)^{-1} = \Lambda F_\alpha(\theta_i) + (1-\Lambda)F_\beta(\theta_i)
$$

## References
<a class="fn-backref" href="https://nvlabs.github.io/sionna/em_primer.html#id2">atan2</a>
    
Wikipedia, “<a class="reference external" href="https://en.wikipedia.org/wiki/Atan2">atan2</a>,” accessed 8 Feb. 2023.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/em_primer.html#id5">Balanis</a>
<ol class="upperalpha simple">
- Balanis, “Advanced Engineering Electromagnetics,” John Wiley & Sons, 2012.
</ol>

Degli-Esposti07(<a href="https://nvlabs.github.io/sionna/em_primer.html#id24">1</a>,<a href="https://nvlabs.github.io/sionna/em_primer.html#id26">2</a>,<a href="https://nvlabs.github.io/sionna/em_primer.html#id28">3</a>)
    
Vittorio Degli-Esposti et al., “<a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/4052607">Measurement and modelling of scattering from buildings</a>,” IEEE Trans. Antennas Propag, vol. 55, no. 1,  pp.143-153, Jan. 2007.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/em_primer.html#id25">Degli-Esposti11</a>
    
Vittorio Degli-Esposti et al., “<a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/5979177">Analysis and Modeling on co- and Cross-Polarized Urban Radio Propagation for Dual-Polarized MIMO Wireless Systems</a>”, IEEE Trans. Antennas Propag, vol. 59, no. 11,  pp.4247-4256, Nov. 2011.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/em_primer.html#id23">Fresnel</a>
    
Wikipedia, “<a class="reference external" href="https://en.wikipedia.org/wiki/Fresnel_integral">Fresnel integral</a>,” accessed 21 Apr. 2023.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/em_primer.html#id8">ITURP20402</a>
    
ITU, “<a class="reference external" href="https://www.itu.int/rec/R-REC-P.2040/en">Recommendation ITU-R P.2040-2: Effects of building materials and structures on radiowave propagation above about 100 MHz</a>”. Sep. 2021.

ITURP52615(<a href="https://nvlabs.github.io/sionna/em_primer.html#id16">1</a>,<a href="https://nvlabs.github.io/sionna/em_primer.html#id18">2</a>)
    
ITU, “<a class="reference external" href="https://www.itu.int/rec/R-REC-P.526/en">Recommendation ITU-R P.526-15: Propagation by diffraction</a>,” Oct. 2019.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/em_primer.html#id11">Keller62</a>
    
J.B. Keller, “<a class="reference external" href="https://opg.optica.org/josa/abstract.cfm?uri=josa-52-2-116">Geometrical Theory of Diffraction</a>,” Journal of the Optical Society of America, vol. 52, no. 2, Feb. 1962.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/em_primer.html#id9">Kline</a>
<ol class="upperalpha simple" start="13">
- Kline, “An Asymptotic Solution of Maxwell’s Equations,” Commun. Pure Appl. Math., vol. 4, 1951.
</ol>

Kouyoumjian74(<a href="https://nvlabs.github.io/sionna/em_primer.html#id12">1</a>,<a href="https://nvlabs.github.io/sionna/em_primer.html#id14">2</a>)
    
R.G. Kouyoumjian, “<a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/1451581/authors#authors">A uniform geometrical theory of diffraction for an edge in a perfectly conducting surface</a>,” Proc. of the IEEE, vol. 62, no. 11, Nov. 1974.

Luebbers84(<a href="https://nvlabs.github.io/sionna/em_primer.html#id15">1</a>,<a href="https://nvlabs.github.io/sionna/em_primer.html#id17">2</a>,<a href="https://nvlabs.github.io/sionna/em_primer.html#id19">3</a>)
<ol class="upperalpha simple" start="18">
- Luebbers, “<a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/1143189">Finite conductivity uniform GTD versus knife edge diffraction in prediction of propagation path loss</a>,” IEEE Trans. Antennas and Propagation, vol. 32, no. 1, Jan. 1984.
</ol>

<a class="fn-backref" href="https://nvlabs.github.io/sionna/em_primer.html#id10">Luneberg</a>
    
R.M. Luneberg, “Mathematical Theory of Optics,” Brown University Press, 1944.

McNamara90(<a href="https://nvlabs.github.io/sionna/em_primer.html#id13">1</a>,<a href="https://nvlabs.github.io/sionna/em_primer.html#id20">2</a>,<a href="https://nvlabs.github.io/sionna/em_primer.html#id22">3</a>)
    
D.A. McNamara, C.W.I. Pistorius, J.A.G. Malherbe, “<a class="reference external" href="https://us.artechhouse.com/Introduction-to-the-Uniform-Geometrical-Theory-of-Diffraction-P288.aspx">Introduction to the Uniform Geometrical Theory of Diffraction</a>,” Artech House, 1990.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/em_primer.html#id21">METIS</a>
    
METIS Deliverable D1.4, “<a class="reference external" href="https://metis2020.com/wp-content/uploads/deliverables/METIS_D1.4_v1.0.pdf">METIS Channel Models</a>”, Feb. 2015.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/em_primer.html#id7">Tse</a>
<ol class="upperalpha simple" start="4">
- Tse, P. Viswanath, “<a class="reference external" href="https://web.stanford.edu/~dntse/wireless_book.html">Fundamentals of Wireless Communication</a>”, Cambridge University Press, 2005.
</ol>

<a class="fn-backref" href="https://nvlabs.github.io/sionna/em_primer.html#id1">Wiesbeck</a>
<ol class="upperalpha simple" start="14">
- Geng and W. Wiesbeck, “Planungsmethoden für die Mobilkommunikation,” Springer, 1998.
</ol>

<a class="fn-backref" href="https://nvlabs.github.io/sionna/em_primer.html#id6">Wikipedia</a>
    
Wikipedia, “<a class="reference external" href="https://en.wikipedia.org/wiki/Maximum_power_transfer_theorem">Maximum power transfer theorem</a>,” accessed 7 Oct. 2022.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/em_primer.html#id4">Wikipedia_Rodrigues</a>
    
Wikipedia, “<a class="reference external" href="https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula">Rodrigues’ rotation formula</a>,” accessed 16 Jun. 2023.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/em_primer.html#id27">Pharr</a>
<ol class="upperalpha simple" start="13">
- Pharr, J. Wenzel, G. Humphreys, “<a class="reference external" href="https://www.pbr-book.org/3ed-2018/contents">Physically Based Rendering: From Theory to Implementation</a>,” MIT Press, 2023.
</ol>



