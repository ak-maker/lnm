# Multiple-Input Multiple-Output (MIMO)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#multiple-input-multiple-output-mimo" title="Permalink to this headline"></a>
    
This module provides layers and functions to support simulation of multicell
MIMO transmissions.

# Table of Content
## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#utility-functions" title="Permalink to this headline"></a>
### complex2real_matrix<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#complex2real-matrix" title="Permalink to this headline"></a>
### real2complex_matrix<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#real2complex-matrix" title="Permalink to this headline"></a>
### complex2real_covariance<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#complex2real-covariance" title="Permalink to this headline"></a>
### real2complex_covariance<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#real2complex-covariance" title="Permalink to this headline"></a>
### complex2real_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#complex2real-channel" title="Permalink to this headline"></a>
### real2complex_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#real2complex-channel" title="Permalink to this headline"></a>
### whiten_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#whiten-channel" title="Permalink to this headline"></a>
  
  

### complex2real_matrix<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#complex2real-matrix" title="Permalink to this headline"></a>

`sionna.mimo.``complex2real_matrix`(<em class="sig-param">`z`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#complex2real_matrix">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_matrix" title="Permalink to this definition"></a>
    
Transforms a complex-valued matrix into its real-valued equivalent.
    
Transforms the last two dimensions of a complex-valued tensor into
their real-valued matrix equivalent representation.
    
For a matrix $\mathbf{Z}\in \mathbb{C}^{M\times K}$ with real and imaginary
parts $\mathbf{X}\in \mathbb{R}^{M\times K}$ and
$\mathbf{Y}\in \mathbb{R}^{M\times K}$, respectively, this function returns
the matrix $\tilde{\mathbf{Z}}\in \mathbb{R}^{2M\times 2K}$, given as

$$
\begin{split}\tilde{\mathbf{Z}} = \begin{pmatrix}
                        \mathbf{X} & -\mathbf{Y}\\
                        \mathbf{Y} & \mathbf{X}
                     \end{pmatrix}.\end{split}
$$

Input
    
<em>[…,M,K], tf.complex</em>

Output
    
<em>[…,2M, 2K], tf.complex.real_dtype</em>



### real2complex_matrix<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#real2complex-matrix" title="Permalink to this headline"></a>

`sionna.mimo.``real2complex_matrix`(<em class="sig-param">`z`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#real2complex_matrix">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.real2complex_matrix" title="Permalink to this definition"></a>
    
Transforms a real-valued matrix into its complex-valued equivalent.
    
Transforms the last two dimensions of a real-valued tensor into
their complex-valued matrix equivalent representation.
    
For a matrix $\tilde{\mathbf{Z}}\in \mathbb{R}^{2M\times 2K}$,
satisfying

$$
\begin{split}\tilde{\mathbf{Z}} = \begin{pmatrix}
                        \mathbf{X} & -\mathbf{Y}\\
                        \mathbf{Y} & \mathbf{X}
                     \end{pmatrix}\end{split}
$$
    
with $\mathbf{X}\in \mathbb{R}^{M\times K}$ and
$\mathbf{Y}\in \mathbb{R}^{M\times K}$, this function returns
the matrix $\mathbf{Z}=\mathbf{X}+j\mathbf{Y}\in\mathbb{C}^{M\times K}$.
Input
    
<em>[…,2M,2K], tf.float</em>

Output
    
<em>[…,M, 2], tf.complex</em>



### complex2real_covariance<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#complex2real-covariance" title="Permalink to this headline"></a>

`sionna.mimo.``complex2real_covariance`(<em class="sig-param">`r`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#complex2real_covariance">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_covariance" title="Permalink to this definition"></a>
    
Transforms a complex-valued covariance matrix to its real-valued equivalent.
    
Assume a proper complex random variable $\mathbf{z}\in\mathbb{C}^M$ <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#properrv" id="id11">[ProperRV]</a>
with covariance matrix $\mathbf{R}= \in\mathbb{C}^{M\times M}$
and real and imaginary parts $\mathbf{x}\in \mathbb{R}^M$ and
$\mathbf{y}\in \mathbb{R}^M$, respectively.
This function transforms the given $\mathbf{R}$ into the covariance matrix of the real-valued equivalent
vector $\tilde{\mathbf{z}}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}$, which
is computed as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#covproperrv" id="id12">[CovProperRV]</a>

$$
\begin{split}\mathbb{E}\left[\tilde{\mathbf{z}}\tilde{\mathbf{z}}^{\mathsf{H}} \right] =
\begin{pmatrix}
    \frac12\Re\{\mathbf{R}\} & -\frac12\Im\{\mathbf{R}\}\\
    \frac12\Im\{\mathbf{R}\} & \frac12\Re\{\mathbf{R}\}
\end{pmatrix}.\end{split}
$$

Input
    
<em>[…,M,M], tf.complex</em>

Output
    
<em>[…,2M, 2M], tf.complex.real_dtype</em>



### real2complex_covariance<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#real2complex-covariance" title="Permalink to this headline"></a>

`sionna.mimo.``real2complex_covariance`(<em class="sig-param">`q`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#real2complex_covariance">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.real2complex_covariance" title="Permalink to this definition"></a>
    
Transforms a real-valued covariance matrix to its complex-valued equivalent.
    
Assume a proper complex random variable $\mathbf{z}\in\mathbb{C}^M$ <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#properrv" id="id13">[ProperRV]</a>
with covariance matrix $\mathbf{R}= \in\mathbb{C}^{M\times M}$
and real and imaginary parts $\mathbf{x}\in \mathbb{R}^M$ and
$\mathbf{y}\in \mathbb{R}^M$, respectively.
This function transforms the given covariance matrix of the real-valued equivalent
vector $\tilde{\mathbf{z}}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}$, which
is given as <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#covproperrv" id="id14">[CovProperRV]</a>

$$
\begin{split}\mathbb{E}\left[\tilde{\mathbf{z}}\tilde{\mathbf{z}}^{\mathsf{H}} \right] =
\begin{pmatrix}
    \frac12\Re\{\mathbf{R}\} & -\frac12\Im\{\mathbf{R}\}\\
    \frac12\Im\{\mathbf{R}\} & \frac12\Re\{\mathbf{R}\}
\end{pmatrix},\end{split}
$$
    
into is complex-valued equivalent $\mathbf{R}$.
Input
    
<em>[…,2M,2M], tf.float</em>

Output
    
<em>[…,M, M], tf.complex</em>



### complex2real_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#complex2real-channel" title="Permalink to this headline"></a>

`sionna.mimo.``complex2real_channel`(<em class="sig-param">`y`</em>, <em class="sig-param">`h`</em>, <em class="sig-param">`s`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#complex2real_channel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_channel" title="Permalink to this definition"></a>
    
Transforms a complex-valued MIMO channel into its real-valued equivalent.
    
Assume the canonical MIMO channel model

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$
    
where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathbb{C}^K$ is the vector of transmitted symbols,
$\mathbf{H}\in\mathbb{C}^{M\times K}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a noise vector with covariance
matrix $\mathbf{S}\in\mathbb{C}^{M\times M}$.
    
This function returns the real-valued equivalent representations of
$\mathbf{y}$, $\mathbf{H}$, and $\mathbf{S}$,
which are used by a wide variety of MIMO detection algorithms (Section VII) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#yh2015" id="id15">[YH2015]</a>.
These are obtained by applying <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_vector" title="sionna.mimo.complex2real_vector">`complex2real_vector()`</a> to $\mathbf{y}$,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_matrix" title="sionna.mimo.complex2real_matrix">`complex2real_matrix()`</a> to $\mathbf{H}$,
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_covariance" title="sionna.mimo.complex2real_covariance">`complex2real_covariance()`</a> to $\mathbf{S}$.
Input
 
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals.
- **h** (<em>[…,M,K], tf.complex</em>) – 2+D tensor containing the channel matrices.
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices.


Output
 
- <em>[…,2M], tf.complex.real_dtype</em> – 1+D tensor containing the real-valued equivalent received signals.
- <em>[…,2M,2K], tf.complex.real_dtype</em> – 2+D tensor containing the real-valued equivalent channel matrices.
- <em>[…,2M,2M], tf.complex.real_dtype</em> – 2+D tensor containing the real-valued equivalent noise covariance matrices.




### real2complex_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#real2complex-channel" title="Permalink to this headline"></a>

`sionna.mimo.``real2complex_channel`(<em class="sig-param">`y`</em>, <em class="sig-param">`h`</em>, <em class="sig-param">`s`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#real2complex_channel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.real2complex_channel" title="Permalink to this definition"></a>
    
Transforms a real-valued MIMO channel into its complex-valued equivalent.
    
Assume the canonical MIMO channel model

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$
    
where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathbb{C}^K$ is the vector of transmitted symbols,
$\mathbf{H}\in\mathbb{C}^{M\times K}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a noise vector with covariance
matrix $\mathbf{S}\in\mathbb{C}^{M\times M}$.
    
This function transforms the real-valued equivalent representations of
$\mathbf{y}$, $\mathbf{H}$, and $\mathbf{S}$, as, e.g.,
obtained with the function <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.complex2real_channel" title="sionna.mimo.complex2real_channel">`complex2real_channel()`</a>,
back to their complex-valued equivalents (Section VII) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#yh2015" id="id16">[YH2015]</a>.
Input
 
- **y** (<em>[…,2M], tf.float</em>) – 1+D tensor containing the real-valued received signals.
- **h** (<em>[…,2M,2K], tf.float</em>) – 2+D tensor containing the real-valued channel matrices.
- **s** (<em>[…,2M,2M], tf.float</em>) – 2+D tensor containing the real-valued noise covariance matrices.


Output
 
- <em>[…,M], tf.complex</em> – 1+D tensor containing the complex-valued equivalent received signals.
- <em>[…,M,K], tf.complex</em> – 2+D tensor containing the complex-valued equivalent channel matrices.
- <em>[…,M,M], tf.complex</em> – 2+D tensor containing the complex-valued equivalent noise covariance matrices.




### whiten_channel<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#whiten-channel" title="Permalink to this headline"></a>

`sionna.mimo.``whiten_channel`(<em class="sig-param">`y`</em>, <em class="sig-param">`h`</em>, <em class="sig-param">`s`</em>, <em class="sig-param">`return_s``=``True`</em>)<a class="reference internal" href="../_modules/sionna/mimo/utils.html#whiten_channel">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.whiten_channel" title="Permalink to this definition"></a>
    
Whitens a canonical MIMO channel.
    
Assume the canonical MIMO channel model

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$
    
where $\mathbf{y}\in\mathbb{C}^M(\mathbb{R}^M)$ is the received signal vector,
$\mathbf{x}\in\mathbb{C}^K(\mathbb{R}^K)$ is the vector of transmitted symbols,
$\mathbf{H}\in\mathbb{C}^{M\times K}(\mathbb{R}^{M\times K})$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M(\mathbb{R}^M)$ is a noise vector with covariance
matrix $\mathbf{S}\in\mathbb{C}^{M\times M}(\mathbb{R}^{M\times M})$.
    
This function whitens this channel by multiplying $\mathbf{y}$ and
$\mathbf{H}$ from the left by $\mathbf{S}^{-\frac{1}{2}}$.
Optionally, the whitened noise covariance matrix $\mathbf{I}_M$
can be returned.
Input
 
- **y** (<em>[…,M], tf.float or tf.complex</em>) – 1+D tensor containing the received signals.
- **h** (<em>[…,M,K], tf.float or tf.complex</em>) – 2+D tensor containing the  channel matrices.
- **s** (<em>[…,M,M], tf.float or complex</em>) – 2+D tensor containing the noise covariance matrices.
- **return_s** (<em>bool</em>) – If <cite>True</cite>, the whitened covariance matrix is returned.
Defaults to <cite>True</cite>.


Output
 
- <em>[…,M], tf.float or tf.complex</em> – 1+D tensor containing the whitened received signals.
- <em>[…,M,K], tf.float or tf.complex</em> – 2+D tensor containing the whitened channel matrices.
- <em>[…,M,M], tf.float or tf.complex</em> – 2+D tensor containing the whitened noise covariance matrices.
Only returned if `return_s` is <cite>True</cite>.





References:
ProperRV(<a href="https://nvlabs.github.io/sionna/api/mimo.html#id11">1</a>,<a href="https://nvlabs.github.io/sionna/api/mimo.html#id13">2</a>)
    
<a class="reference external" href="https://en.wikipedia.org/wiki/Complex_random_variable#Proper_complex_random_variables">Proper complex random variables</a>,
Wikipedia, accessed 11 September, 2022.

CovProperRV(<a href="https://nvlabs.github.io/sionna/api/mimo.html#id12">1</a>,<a href="https://nvlabs.github.io/sionna/api/mimo.html#id14">2</a>)
    
<a class="reference external" href="https://en.wikipedia.org/wiki/Complex_random_vector#Covariance_matrices_of_real_and_imaginary_parts">Covariance matrices of real and imaginary parts</a>,
Wikipedia, accessed 11 September, 2022.

YH2015(<a href="https://nvlabs.github.io/sionna/api/mimo.html#id15">1</a>,<a href="https://nvlabs.github.io/sionna/api/mimo.html#id16">2</a>)
    
S. Yang and L. Hanzo, <a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/7244171">“Fifty Years of MIMO Detection: The Road to Large-Scale MIMOs”</a>,
IEEE Communications Surveys & Tutorials, vol. 17, no. 4, pp. 1941-1988, 2015.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/mimo.html#id6">FT2015</a>
    
W. Fu and J. S. Thompson, <a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/7454351">“Performance analysis of K-best detection with adaptive modulation”</a>, IEEE Int. Symp. Wirel. Commun. Sys. (ISWCS), 2015.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/mimo.html#id5">EP2014</a>
    
J. Céspedes, P. M. Olmos, M. Sánchez-Fernández, and F. Perez-Cruz,
<a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/6841617">“Expectation Propagation Detection for High-Order High-Dimensional MIMO Systems”</a>,
IEEE Trans. Commun., vol. 62, no. 8, pp. 2840-2849, Aug. 2014.

CST2011(<a href="https://nvlabs.github.io/sionna/api/mimo.html#id7">1</a>,<a href="https://nvlabs.github.io/sionna/api/mimo.html#id8">2</a>,<a href="https://nvlabs.github.io/sionna/api/mimo.html#id9">3</a>,<a href="https://nvlabs.github.io/sionna/api/mimo.html#id10">4</a>)
    
C. Studer, S. Fateh, and D. Seethaler,
<a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/5779722">“ASIC Implementation of Soft-Input Soft-Output MIMO Detection Using MMSE Parallel Interference Cancellation”</a>,
IEEE Journal of Solid-State Circuits, vol. 46, no. 7, pp. 1754–1765, July 2011.



