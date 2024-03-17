# Multiple-Input Multiple-Output (MIMO)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#multiple-input-multiple-output-mimo" title="Permalink to this headline"></a>
    
This module provides layers and functions to support simulation of multicell
MIMO transmissions.

# Table of Content
## Equalization<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#equalization" title="Permalink to this headline"></a>
### lmmse_equalizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#lmmse-equalizer" title="Permalink to this headline"></a>
### mf_equalizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#mf-equalizer" title="Permalink to this headline"></a>
### zf_equalizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#zf-equalizer" title="Permalink to this headline"></a>
## Detection<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#detection" title="Permalink to this headline"></a>
  
  

### lmmse_equalizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#lmmse-equalizer" title="Permalink to this headline"></a>

`sionna.mimo.``lmmse_equalizer`(<em class="sig-param">`y`</em>, <em class="sig-param">`h`</em>, <em class="sig-param">`s`</em>, <em class="sig-param">`whiten_interference``=``True`</em>)<a class="reference internal" href="../_modules/sionna/mimo/equalization.html#lmmse_equalizer">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.lmmse_equalizer" title="Permalink to this definition"></a>
    
MIMO LMMSE Equalizer
    
This function implements LMMSE equalization for a MIMO link, assuming the
following model:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$
    
where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathbb{C}^K$ is the vector of transmitted symbols,
$\mathbf{H}\in\mathbb{C}^{M\times K}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a noise vector.
It is assumed that $\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}$,
$\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K$ and
$\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}$.
    
The estimated symbol vector $\hat{\mathbf{x}}\in\mathbb{C}^K$ is given as
(Lemma B.19) <a class="reference internal" href="channel.wireless.html#bhs2017" id="id2">[BHS2017]</a> :

$$
\hat{\mathbf{x}} = \mathop{\text{diag}}\left(\mathbf{G}\mathbf{H}\right)^{-1}\mathbf{G}\mathbf{y}
$$
    
where

$$
\mathbf{G} = \mathbf{H}^{\mathsf{H}} \left(\mathbf{H}\mathbf{H}^{\mathsf{H}} + \mathbf{S}\right)^{-1}.
$$
    
This leads to the post-equalized per-symbol model:

$$
\hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1
$$
    
where the variances $\sigma^2_k$ of the effective residual noise
terms $e_k$ are given by the diagonal elements of

$$
\mathop{\text{diag}}\left(\mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]\right)
= \mathop{\text{diag}}\left(\mathbf{G}\mathbf{H} \right)^{-1} - \mathbf{I}.
$$
    
Note that the scaling by $\mathop{\text{diag}}\left(\mathbf{G}\mathbf{H}\right)^{-1}$
is important for the <a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a> although it does
not change the signal-to-noise ratio.
    
The function returns $\hat{\mathbf{x}}$ and
$\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}$.
Input
 
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals.
- **h** (<em>[…,M,K], tf.complex</em>) – 2+D tensor containing the channel matrices.
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices.
- **whiten_interference** (<em>bool</em>) – If <cite>True</cite> (default), the interference is first whitened before equalization.
In this case, an alternative expression for the receive filter is used that
can be numerically more stable. Defaults to <cite>True</cite>.


Output
 
- **x_hat** (<em>[…,K], tf.complex</em>) – 1+D tensor representing the estimated symbol vectors.
- **no_eff** (<em>tf.float</em>) – Tensor of the same shape as `x_hat` containing the effective noise
variance estimates.




**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

### mf_equalizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#mf-equalizer" title="Permalink to this headline"></a>

`sionna.mimo.``mf_equalizer`(<em class="sig-param">`y`</em>, <em class="sig-param">`h`</em>, <em class="sig-param">`s`</em>)<a class="reference internal" href="../_modules/sionna/mimo/equalization.html#mf_equalizer">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.mf_equalizer" title="Permalink to this definition"></a>
    
MIMO MF Equalizer
    
This function implements matched filter (MF) equalization for a
MIMO link, assuming the following model:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$
    
where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathbb{C}^K$ is the vector of transmitted symbols,
$\mathbf{H}\in\mathbb{C}^{M\times K}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a noise vector.
It is assumed that $\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}$,
$\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K$ and
$\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}$.
    
The estimated symbol vector $\hat{\mathbf{x}}\in\mathbb{C}^K$ is given as
(Eq. 4.11) <a class="reference internal" href="channel.wireless.html#bhs2017" id="id3">[BHS2017]</a> :

$$
\hat{\mathbf{x}} = \mathbf{G}\mathbf{y}
$$
    
where

$$
\mathbf{G} = \mathop{\text{diag}}\left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}\mathbf{H}^{\mathsf{H}}.
$$
    
This leads to the post-equalized per-symbol model:

$$
\hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1
$$
    
where the variances $\sigma^2_k$ of the effective residual noise
terms $e_k$ are given by the diagonal elements of the matrix

$$
\mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]
= \left(\mathbf{I}-\mathbf{G}\mathbf{H} \right)\left(\mathbf{I}-\mathbf{G}\mathbf{H} \right)^{\mathsf{H}} + \mathbf{G}\mathbf{S}\mathbf{G}^{\mathsf{H}}.
$$
    
Note that the scaling by $\mathop{\text{diag}}\left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}$
in the definition of $\mathbf{G}$
is important for the <a class="reference internal" href="mapping.html#sionna.mapping.Demapper" title="sionna.mapping.Demapper">`Demapper`</a> although it does
not change the signal-to-noise ratio.
    
The function returns $\hat{\mathbf{x}}$ and
$\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}$.
Input
 
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals.
- **h** (<em>[…,M,K], tf.complex</em>) – 2+D tensor containing the channel matrices.
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices.


Output
 
- **x_hat** (<em>[…,K], tf.complex</em>) – 1+D tensor representing the estimated symbol vectors.
- **no_eff** (<em>tf.float</em>) – Tensor of the same shape as `x_hat` containing the effective noise
variance estimates.




### zf_equalizer<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#zf-equalizer" title="Permalink to this headline"></a>

`sionna.mimo.``zf_equalizer`(<em class="sig-param">`y`</em>, <em class="sig-param">`h`</em>, <em class="sig-param">`s`</em>)<a class="reference internal" href="../_modules/sionna/mimo/equalization.html#zf_equalizer">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.zf_equalizer" title="Permalink to this definition"></a>
    
MIMO ZF Equalizer
    
This function implements zero-forcing (ZF) equalization for a MIMO link, assuming the
following model:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$
    
where $\mathbf{y}\in\mathbb{C}^M$ is the received signal vector,
$\mathbf{x}\in\mathbb{C}^K$ is the vector of transmitted symbols,
$\mathbf{H}\in\mathbb{C}^{M\times K}$ is the known channel matrix,
and $\mathbf{n}\in\mathbb{C}^M$ is a noise vector.
It is assumed that $\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}$,
$\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K$ and
$\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}$.
    
The estimated symbol vector $\hat{\mathbf{x}}\in\mathbb{C}^K$ is given as
(Eq. 4.10) <a class="reference internal" href="channel.wireless.html#bhs2017" id="id4">[BHS2017]</a> :

$$
\hat{\mathbf{x}} = \mathbf{G}\mathbf{y}
$$
    
where

$$
\mathbf{G} = \left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}\mathbf{H}^{\mathsf{H}}.
$$
    
This leads to the post-equalized per-symbol model:

$$
\hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1
$$
    
where the variances $\sigma^2_k$ of the effective residual noise
terms $e_k$ are given by the diagonal elements of the matrix

$$
\mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]
= \mathbf{G}\mathbf{S}\mathbf{G}^{\mathsf{H}}.
$$
    
The function returns $\hat{\mathbf{x}}$ and
$\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}$.
Input
 
- **y** (<em>[…,M], tf.complex</em>) – 1+D tensor containing the received signals.
- **h** (<em>[…,M,K], tf.complex</em>) – 2+D tensor containing the channel matrices.
- **s** (<em>[…,M,M], tf.complex</em>) – 2+D tensor containing the noise covariance matrices.


Output
 
- **x_hat** (<em>[…,K], tf.complex</em>) – 1+D tensor representing the estimated symbol vectors.
- **no_eff** (<em>tf.float</em>) – Tensor of the same shape as `x_hat` containing the effective noise
variance estimates.




**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

## Detection<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#detection" title="Permalink to this headline"></a>

