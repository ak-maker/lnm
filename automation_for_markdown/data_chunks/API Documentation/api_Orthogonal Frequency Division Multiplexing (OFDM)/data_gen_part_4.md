# Orthogonal Frequency-Division Multiplexing (OFDM)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#orthogonal-frequency-division-multiplexing-ofdm" title="Permalink to this headline"></a>
    
This module provides layers and functions to support
simulation of OFDM-based systems. The key component is the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> that defines how data and pilot symbols
are mapped onto a sequence of OFDM symbols with a given FFT size. The resource
grid can also define guard and DC carriers which are nulled. In 4G/5G parlance,
a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> would be a slot.
Once a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> is defined, one can use the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGridMapper" title="sionna.ofdm.ResourceGridMapper">`ResourceGridMapper`</a> to map a tensor of complex-valued
data symbols onto the resource grid, prior to OFDM modulation using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.OFDMModulator" title="sionna.ofdm.OFDMModulator">`OFDMModulator`</a> or further processing in the
frequency domain.
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a> allows for a fine-grained configuration
of how transmitters send pilots for each of their streams or antennas. As the
management of pilots in multi-cell MIMO setups can quickly become complicated,
the module provides the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.KroneckerPilotPattern" title="sionna.ofdm.KroneckerPilotPattern">`KroneckerPilotPattern`</a> class
that automatically generates orthogonal pilot transmissions for all transmitters
and streams.
    
Additionally, the module contains layers for channel estimation, precoding,
equalization, and detection,
such as the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LSChannelEstimator" title="sionna.ofdm.LSChannelEstimator">`LSChannelEstimator`</a>, the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ZFPrecoder" title="sionna.ofdm.ZFPrecoder">`ZFPrecoder`</a>, and the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEEqualizer" title="sionna.ofdm.LMMSEEqualizer">`LMMSEEqualizer`</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LinearDetector" title="sionna.ofdm.LinearDetector">`LinearDetector`</a>.
These are good starting points for the development of more advanced algorithms
and provide robust baselines for benchmarking.

# Table of Content
## Channel Estimation
### LMMSEInterpolator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#lmmseinterpolator" title="Permalink to this headline"></a>
  
  

### LMMSEInterpolator<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#lmmseinterpolator" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``LMMSEInterpolator`(<em class="sig-param">`pilot_pattern`</em>, <em class="sig-param">`cov_mat_time`</em>, <em class="sig-param">`cov_mat_freq`</em>, <em class="sig-param">`cov_mat_space``=``None`</em>, <em class="sig-param">`order``=``'t-f'`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/channel_estimation.html#LMMSEInterpolator">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.LMMSEInterpolator" title="Permalink to this definition"></a>
    
LMMSE interpolation on a resource grid with optional spatial smoothing.
    
This class computes for each element of an OFDM resource grid
a channel estimate and error variance
through linear minimum mean square error (LMMSE) interpolation/smoothing.
It is assumed that the measurements were taken at the nonzero positions
of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>.
    
Depending on the value of `order`, the interpolation is carried out
accross time (t), i.e., OFDM symbols, frequency (f), i.e., subcarriers,
and optionally space (s), i.e., receive antennas, in any desired order.
    
For simplicity, we describe the underlying algorithm assuming that interpolation
across the sub-carriers is performed first, followed by interpolation across
OFDM symbols, and finally by spatial smoothing across receive
antennas.
The algorithm is similar if interpolation and/or smoothing are performed in
a different order.
For clarity, antenna indices are omitted when describing frequency and time
interpolation, as the same process is applied to all the antennas.
    
The input `h_hat` is first reshaped to a resource grid
$\hat{\mathbf{H}} \in \mathbb{C}^{N \times M}$, by scattering the channel
estimates at pilot locations according to the `pilot_pattern`. $N$
denotes the number of OFDM symbols and $M$ the number of sub-carriers.
    
The first pass consists in interpolating across the sub-carriers:

$$
\hat{\mathbf{h}}_n^{(1)} = \mathbf{A}_n \hat{\mathbf{h}}_n
$$
    
where $1 \leq n \leq N$ is the OFDM symbol index and $\hat{\mathbf{h}}_n$ is
the $n^{\text{th}}$ (transposed) row of $\hat{\mathbf{H}}$.
$\mathbf{A}_n$ is the $M \times M$ matrix such that:

$$
\mathbf{A}_n = \bar{\mathbf{A}}_n \mathbf{\Pi}_n^\intercal
$$
    
where

$$
\bar{\mathbf{A}}_n = \underset{\mathbf{Z} \in \mathbb{C}^{M \times K_n}}{\text{argmin}} \left\lVert \mathbf{Z}\left( \mathbf{\Pi}_n^\intercal \mathbf{R^{(f)}} \mathbf{\Pi}_n + \mathbf{\Sigma}_n \right) - \mathbf{R^{(f)}} \mathbf{\Pi}_n \right\rVert_{\text{F}}^2
$$
    
and $\mathbf{R^{(f)}}$ is the $M \times M$ channel frequency covariance matrix,
$\mathbf{\Pi}_n$ the $M \times K_n$ matrix that spreads $K_n$
values to a vector of size $M$ according to the `pilot_pattern` for the $n^{\text{th}}$ OFDM symbol,
and $\mathbf{\Sigma}_n \in \mathbb{R}^{K_n \times K_n}$ is the channel estimation error covariance built from
`err_var` and assumed to be diagonal.
Computation of $\bar{\mathbf{A}}_n$ is done using an algorithm based on complete orthogonal decomposition.
This is done to avoid matrix inversion for badly conditioned covariance matrices.
    
The channel estimation error variances after the first interpolation pass are computed as

$$
\mathbf{\Sigma}^{(1)}_n = \text{diag} \left( \mathbf{R^{(f)}} - \mathbf{A}_n \mathbf{\Xi}_n \mathbf{R^{(f)}} \right)
$$
    
where $\mathbf{\Xi}_n$ is the diagonal matrix of size $M \times M$ that zeros the
columns corresponding to sub-carriers not carrying any pilots.
Note that interpolation is not performed for OFDM symbols which do not carry pilots.
    
**Remark**: The interpolation matrix differs across OFDM symbols as different
OFDM symbols may carry pilots on different sub-carriers and/or have different
estimation error variances.
    
Scaling of the estimates is then performed to ensure that their
variances match the ones expected by the next interpolation step, and the error variances are updated accordingly:

$$
\begin{split}\begin{align}
    \left[\hat{\mathbf{h}}_n^{(2)}\right]_m &= s_{n,m} \left[\hat{\mathbf{h}}_n^{(1)}\right]_m\\
    \left[\mathbf{\Sigma}^{(2)}_n\right]_{m,m}  &= s_{n,m}\left( s_{n,m}-1 \right) \left[\hat{\mathbf{\Sigma}}^{(1)}_n\right]_{m,m} + \left( 1 - s_{n,m} \right) \left[\mathbf{R^{(f)}}\right]_{m,m} + s_{n,m} \left[\mathbf{\Sigma}^{(1)}_n\right]_{m,m}
\end{align}\end{split}
$$
    
where the scaling factor $s_{n,m}$ is such that:

$$
\mathbb{E} \left\{ \left\lvert s_{n,m} \left[\hat{\mathbf{h}}_n^{(1)}\right]_m \right\rvert^2 \right\} = \left[\mathbf{R^{(f)}}\right]_{m,m} +  \mathbb{E} \left\{ \left\lvert s_{n,m} \left[\hat{\mathbf{h}}^{(1)}_n\right]_m - \left[\mathbf{h}_n\right]_m \right\rvert^2 \right\}
$$
    
which leads to:

$$
\begin{split}\begin{align}
    s_{n,m} &= \frac{2 \left[\mathbf{R^{(f)}}\right]_{m,m}}{\left[\mathbf{R^{(f)}}\right]_{m,m} - \left[\mathbf{\Sigma}^{(1)}_n\right]_{m,m} + \left[\hat{\mathbf{\Sigma}}^{(1)}_n\right]_{m,m}}\\
    \hat{\mathbf{\Sigma}}^{(1)}_n &= \mathbf{A}_n \mathbf{R^{(f)}} \mathbf{A}_n^{\mathrm{H}}.
\end{align}\end{split}
$$
    
The second pass consists in interpolating across the OFDM symbols:

$$
\hat{\mathbf{h}}_m^{(3)} = \mathbf{B}_m \tilde{\mathbf{h}}^{(2)}_m
$$
    
where $1 \leq m \leq M$ is the sub-carrier index and $\tilde{\mathbf{h}}^{(2)}_m$ is
the $m^{\text{th}}$ column of

$$
\begin{split}\hat{\mathbf{H}}^{(2)} = \begin{bmatrix}
                            {\hat{\mathbf{h}}_1^{(2)}}^\intercal\\
                            \vdots\\
                            {\hat{\mathbf{h}}_N^{(2)}}^\intercal
                         \end{bmatrix}\end{split}
$$
    
and $\mathbf{B}_m$ is the $N \times N$ interpolation LMMSE matrix:

$$
\mathbf{B}_m = \bar{\mathbf{B}}_m \tilde{\mathbf{\Pi}}_m^\intercal
$$
    
where

$$
\bar{\mathbf{B}}_m = \underset{\mathbf{Z} \in \mathbb{C}^{N \times L_m}}{\text{argmin}} \left\lVert \mathbf{Z} \left( \tilde{\mathbf{\Pi}}_m^\intercal \mathbf{R^{(t)}}\tilde{\mathbf{\Pi}}_m + \tilde{\mathbf{\Sigma}}^{(2)}_m \right) -  \mathbf{R^{(t)}}\tilde{\mathbf{\Pi}}_m \right\rVert_{\text{F}}^2
$$
    
where $\mathbf{R^{(t)}}$ is the $N \times N$ channel time covariance matrix,
$\tilde{\mathbf{\Pi}}_m$ the $N \times L_m$ matrix that spreads $L_m$
values to a vector of size $N$ according to the `pilot_pattern` for the $m^{\text{th}}$ sub-carrier,
and $\tilde{\mathbf{\Sigma}}^{(2)}_m \in \mathbb{R}^{L_m \times L_m}$ is the diagonal matrix of channel estimation error variances
built by gathering the error variances from ($\mathbf{\Sigma}^{(2)}_1,\dots,\mathbf{\Sigma}^{(2)}_N$) corresponding
to resource elements carried by the $m^{\text{th}}$ sub-carrier.
Computation of $\bar{\mathbf{B}}_m$ is done using an algorithm based on complete orthogonal decomposition.
This is done to avoid matrix inversion for badly conditioned covariance matrices.
    
The resulting channel estimate for the resource grid is

$$
\hat{\mathbf{H}}^{(3)} = \left[ \hat{\mathbf{h}}_1^{(3)} \dots \hat{\mathbf{h}}_M^{(3)} \right]
$$
    
The resulting channel estimation error variances are the diagonal coefficients of the matrices

$$
\mathbf{\Sigma}^{(3)}_m = \mathbf{R^{(t)}} - \mathbf{B}_m \tilde{\mathbf{\Xi}}_m \mathbf{R^{(t)}}, 1 \leq m \leq M
$$
    
where $\tilde{\mathbf{\Xi}}_m$ is the diagonal matrix of size $N \times N$ that zeros the
columns corresponding to OFDM symbols not carrying any pilots.
    
**Remark**: The interpolation matrix differs across sub-carriers as different
sub-carriers may have different estimation error variances computed by the first
pass.
However, all sub-carriers carry at least one channel estimate as a result of
the first pass, ensuring that a channel estimate is computed for all the resource
elements after the second pass.
    
**Remark:** LMMSE interpolation requires knowledge of the time and frequency
covariance matrices of the channel. The notebook <a class="reference internal" href="../examples/OFDM_MIMO_Detection.html">OFDM MIMO Channel Estimation and Detection</a> shows how to estimate
such matrices for arbitrary channel models.
Moreover, the functions <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.tdl_time_cov_mat" title="sionna.ofdm.tdl_time_cov_mat">`tdl_time_cov_mat()`</a>
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.tdl_freq_cov_mat" title="sionna.ofdm.tdl_freq_cov_mat">`tdl_freq_cov_mat()`</a> compute the expected time and frequency
covariance matrices, respectively, for the <a class="reference internal" href="channel.wireless.html#sionna.channel.tr38901.TDL" title="sionna.channel.tr38901.TDL">`TDL`</a> channel models.
    
Scaling of the estimates is then performed to ensure that their
variances match the ones expected by the next smoothing step, and the
error variances are updated accordingly:

$$
\begin{split}\begin{align}
    \left[\hat{\mathbf{h}}_m^{(4)}\right]_n &= \gamma_{m,n} \left[\hat{\mathbf{h}}_m^{(3)}\right]_n\\
    \left[\mathbf{\Sigma}^{(4)}_m\right]_{n,n}  &= \gamma_{m,n}\left( \gamma_{m,n}-1 \right) \left[\hat{\mathbf{\Sigma}}^{(3)}_m\right]_{n,n} + \left( 1 - \gamma_{m,n} \right) \left[\mathbf{R^{(t)}}\right]_{n,n} + \gamma_{m,n} \left[\mathbf{\Sigma}^{(3)}_n\right]_{m,m}
\end{align}\end{split}
$$
    
where:

$$
\begin{split}\begin{align}
    \gamma_{m,n} &= \frac{2 \left[\mathbf{R^{(t)}}\right]_{n,n}}{\left[\mathbf{R^{(t)}}\right]_{n,n} - \left[\mathbf{\Sigma}^{(3)}_m\right]_{n,n} + \left[\hat{\mathbf{\Sigma}}^{(3)}_n\right]_{m,m}}\\
    \hat{\mathbf{\Sigma}}^{(3)}_m &= \mathbf{B}_m \mathbf{R^{(t)}} \mathbf{B}_m^{\mathrm{H}}
\end{align}\end{split}
$$
    
Finally, a spatial smoothing step is applied to every resource element carrying
a channel estimate.
For clarity, we drop the resource element indexing $(n,m)$.
We denote by $L$ the number of receive antennas, and by
$\mathbf{R^{(s)}}\in\mathbb{C}^{L \times L}$ the spatial covariance matrix.
    
LMMSE spatial smoothing consists in the following computations:

$$
\hat{\mathbf{h}}^{(5)} = \mathbf{C} \hat{\mathbf{h}}^{(4)}
$$
    
where

$$
\mathbf{C} = \mathbf{R^{(s)}} \left( \mathbf{R^{(s)}} + \mathbf{\Sigma}^{(4)} \right)^{-1}.
$$
    
The estimation error variances are the digonal coefficients of

$$
\mathbf{\Sigma}^{(5)} = \mathbf{R^{(s)}} - \mathbf{C}\mathbf{R^{(s)}}
$$
    
The smoothed channel estimate $\hat{\mathbf{h}}^{(5)}$ and corresponding
error variances $\text{diag}\left( \mathbf{\Sigma}^{(5)} \right)$ are
returned for every resource element $(m,n)$.
    
**Remark:** No scaling is performed after the last interpolation or smoothing
step.
    
**Remark:** All passes assume that the estimation error covariance matrix
($\mathbf{\Sigma}$, $\tilde{\mathbf{\Sigma}}^{(2)}$, or $\tilde{\mathbf{\Sigma}}^{(4)}$) is diagonal, which
may not be accurate. When this assumption does not hold, this interpolator is only
an approximation of LMMSE interpolation.
    
**Remark:** The order in which frequency interpolation, temporal
interpolation, and, optionally, spatial smoothing are applied, is controlled using the
`order` parameter.

**Note**
    
This layer does not support graph mode with XLA.

Parameters
 
- **pilot_pattern** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern"><em>PilotPattern</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>
- **cov_mat_time** (<em>[</em><em>num_ofdm_symbols</em><em>, </em><em>num_ofdm_symbols</em><em>]</em><em>, </em><em>tf.complex</em>) – Time covariance matrix of the channel
- **cov_mat_freq** (<em>[</em><em>fft_size</em><em>, </em><em>fft_size</em><em>]</em><em>, </em><em>tf.complex</em>) – Frequency covariance matrix of the channel
- **cov_time_space** (<em>[</em><em>num_rx_ant</em><em>, </em><em>num_rx_ant</em><em>]</em><em>, </em><em>tf.complex</em>) – Spatial covariance matrix of the channel.
Defaults to <cite>None</cite>.
Only required if spatial smoothing is requested (see `order`).
- **order** (<em>str</em>) – Order in which to perform interpolation and optional smoothing.
For example, `"t-f-s"` means that interpolation across the OFDM symbols
is performed first (`"t"`: time), followed by interpolation across the
sub-carriers (`"f"`: frequency), and finally smoothing across the
receive antennas (`"s"`: space).
Similarly, `"f-t"` means interpolation across the sub-carriers followed
by interpolation across the OFDM symbols and no spatial smoothing.
The spatial covariance matrix (`cov_time_space`) is only required when
spatial smoothing is requested.
Time and frequency interpolation are not optional to ensure that a channel
estimate is computed for all resource elements.


Input
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimates for the pilot-carrying resource elements
- **err_var** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex</em>) – Channel estimation error variances for the pilot-carrying resource elements


Output
 
- **h_hat** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex</em>) – Channel estimates accross the entire resource grid for all
transmitters and streams
- **err_var** (Same shape as `h_hat`, tf.float) – Channel estimation error variances accross the entire resource grid
for all transmitters and streams




