# Multiple-Input Multiple-Output (MIMO)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#multiple-input-multiple-output-mimo" title="Permalink to this headline"></a>
    
This module provides layers and functions to support simulation of multicell
MIMO transmissions.

# Table of Content
## Stream Management<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#stream-management" title="Permalink to this headline"></a>
## Precoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#precoding" title="Permalink to this headline"></a>
### zero_forcing_precoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#zero-forcing-precoder" title="Permalink to this headline"></a>
## Equalization<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#equalization" title="Permalink to this headline"></a>
  
  

## Stream Management<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#stream-management" title="Permalink to this headline"></a>
    
Stream management determines which transmitter is sending which stream to
which receiver. Transmitters and receivers can be user terminals or base
stations, depending on whether uplink or downlink transmissions are considered.
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> class has various properties that
are needed to recover desired or interfering channel coefficients for precoding
and equalization. In order to understand how the various properties of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> can be used, we recommend to have a look
at the source code of the <a class="reference internal" href="ofdm.html#sionna.ofdm.LMMSEEqualizer" title="sionna.ofdm.LMMSEEqualizer">`LMMSEEqualizer`</a> or
<a class="reference internal" href="ofdm.html#sionna.ofdm.ZFPrecoder" title="sionna.ofdm.ZFPrecoder">`ZFPrecoder`</a>.
    
The following code snippet shows how to configure
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> for a simple uplink scenario, where
four transmitters send each one stream to a receiver. Note that
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a> is independent of the actual number of
antennas at the transmitters and receivers.
```python
num_tx = 4
num_rx = 1
num_streams_per_tx = 1
# Indicate which transmitter is associated with which receiver
# rx_tx_association[i,j] = 1 means that transmitter j sends one
# or mutiple streams to receiver i.
rx_tx_association = np.zeros([num_rx, num_tx])
rx_tx_association[0,0] = 1
rx_tx_association[0,1] = 1
rx_tx_association[0,2] = 1
rx_tx_association[0,3] = 1
sm = StreamManagement(rx_tx_association, num_streams_per_tx)
```
<em class="property">`class` </em>`sionna.mimo.``StreamManagement`(<em class="sig-param">`rx_tx_association`</em>, <em class="sig-param">`num_streams_per_tx`</em>)<a class="reference internal" href="../_modules/sionna/mimo/stream_management.html#StreamManagement">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement" title="Permalink to this definition"></a>
    
Class for management of streams in multi-cell MIMO networks.
Parameters
 
- **rx_tx_association** (<em>[</em><em>num_rx</em><em>, </em><em>num_tx</em><em>]</em><em>, </em><em>np.int</em>) – A binary NumPy array where `rx_tx_association[i,j]=1` means
that receiver <cite>i</cite> gets one or multiple streams from
transmitter <cite>j</cite>.
- **num_streams_per_tx** (<em>int</em>) – Indicates the number of streams that are transmitted by each
transmitter.




**Note**
    
Several symmetry constraints on `rx_tx_association` are imposed
to ensure efficient processing. All row sums and all column sums
must be equal, i.e., all receivers have the same number of associated
transmitters and all transmitters have the same number of associated
receivers. It is also assumed that all transmitters send the same
number of streams `num_streams_per_tx`.

<em class="property">`property` </em>`detection_desired_ind`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.detection_desired_ind" title="Permalink to this definition"></a>
    
Indices needed to gather desired channels for receive processing.
    
A NumPy array of shape <cite>[num_rx*num_streams_per_rx]</cite> that
can be used to gather desired channels from the flattened
channel tensor of shape
<cite>[…,num_rx, num_tx, num_streams_per_tx,…]</cite>.
The result of the gather operation can be reshaped to
<cite>[…,num_rx, num_streams_per_rx,…]</cite>.


<em class="property">`property` </em>`detection_undesired_ind`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.detection_undesired_ind" title="Permalink to this definition"></a>
    
Indices needed to gather undesired channels for receive processing.
    
A NumPy array of shape <cite>[num_rx*num_streams_per_rx]</cite> that
can be used to gather undesired channels from the flattened
channel tensor of shape <cite>[…,num_rx, num_tx, num_streams_per_tx,…]</cite>.
The result of the gather operation can be reshaped to
<cite>[…,num_rx, num_interfering_streams_per_rx,…]</cite>.


<em class="property">`property` </em>`num_interfering_streams_per_rx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.num_interfering_streams_per_rx" title="Permalink to this definition"></a>
    
Number of interfering streams received at each eceiver.


<em class="property">`property` </em>`num_rx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.num_rx" title="Permalink to this definition"></a>
    
Number of receivers.


<em class="property">`property` </em>`num_rx_per_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.num_rx_per_tx" title="Permalink to this definition"></a>
    
Number of receivers communicating with a transmitter.


<em class="property">`property` </em>`num_streams_per_rx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.num_streams_per_rx" title="Permalink to this definition"></a>
    
Number of streams transmitted to each receiver.


<em class="property">`property` </em>`num_streams_per_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.num_streams_per_tx" title="Permalink to this definition"></a>
    
Number of streams per transmitter.


<em class="property">`property` </em>`num_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.num_tx" title="Permalink to this definition"></a>
    
Number of transmitters.


<em class="property">`property` </em>`num_tx_per_rx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.num_tx_per_rx" title="Permalink to this definition"></a>
    
Number of transmitters communicating with a receiver.


<em class="property">`property` </em>`precoding_ind`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.precoding_ind" title="Permalink to this definition"></a>
    
Indices needed to gather channels for precoding.
    
A NumPy array of shape <cite>[num_tx, num_rx_per_tx]</cite>,
where `precoding_ind[i,:]` contains the indices of the
receivers to which transmitter <cite>i</cite> is sending streams.


<em class="property">`property` </em>`rx_stream_ids`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.rx_stream_ids" title="Permalink to this definition"></a>
    
Mapping of streams to receivers.
    
A Numpy array of shape <cite>[num_rx, num_streams_per_rx]</cite>.
This array is obtained from `tx_stream_ids` together with
the `rx_tx_association`. `rx_stream_ids[i,:]` contains
the indices of streams that are supposed to be decoded by receiver <cite>i</cite>.


<em class="property">`property` </em>`rx_tx_association`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.rx_tx_association" title="Permalink to this definition"></a>
    
Association between receivers and transmitters.
    
A binary NumPy array of shape <cite>[num_rx, num_tx]</cite>,
where `rx_tx_association[i,j]=1` means that receiver <cite>i</cite>
gets one ore multiple streams from transmitter <cite>j</cite>.


<em class="property">`property` </em>`stream_association`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.stream_association" title="Permalink to this definition"></a>
    
Association between receivers, transmitters, and streams.
    
A binary NumPy array of shape
<cite>[num_rx, num_tx, num_streams_per_tx]</cite>, where
`stream_association[i,j,k]=1` means that receiver <cite>i</cite> gets
the <cite>k</cite> th stream from transmitter <cite>j</cite>.


<em class="property">`property` </em>`stream_ind`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.stream_ind" title="Permalink to this definition"></a>
    
Indices needed to gather received streams in the correct order.
    
A NumPy array of shape <cite>[num_rx*num_streams_per_rx]</cite> that can be
used to gather streams from the flattened tensor of received streams
of shape <cite>[…,num_rx, num_streams_per_rx,…]</cite>. The result of the
gather operation is then reshaped to
<cite>[…,num_tx, num_streams_per_tx,…]</cite>.


<em class="property">`property` </em>`tx_stream_ids`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.StreamManagement.tx_stream_ids" title="Permalink to this definition"></a>
    
Mapping of streams to transmitters.
    
A NumPy array of shape <cite>[num_tx, num_streams_per_tx]</cite>.
Streams are numbered from 0,1,… and assiged to transmitters in
increasing order, i.e., transmitter 0 gets the first
<cite>num_streams_per_tx</cite> and so on.


## Precoding<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#precoding" title="Permalink to this headline"></a>

### zero_forcing_precoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#zero-forcing-precoder" title="Permalink to this headline"></a>

`sionna.mimo.``zero_forcing_precoder`(<em class="sig-param">`x`</em>, <em class="sig-param">`h`</em>, <em class="sig-param">`return_precoding_matrix``=``False`</em>)<a class="reference internal" href="../_modules/sionna/mimo/precoding.html#zero_forcing_precoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#sionna.mimo.zero_forcing_precoder" title="Permalink to this definition"></a>
    
Zero-Forcing (ZF) Precoder
    
This function implements ZF precoding for a MIMO link, assuming the
following model:

$$
\mathbf{y} = \mathbf{H}\mathbf{G}\mathbf{x} + \mathbf{n}
$$
    
where $\mathbf{y}\in\mathbb{C}^K$ is the received signal vector,
$\mathbf{H}\in\mathbb{C}^{K\times M}$ is the known channel matrix,
$\mathbf{G}\in\mathbb{C}^{M\times K}$ is the precoding matrix,
$\mathbf{x}\in\mathbb{C}^K$ is the symbol vector to be precoded,
and $\mathbf{n}\in\mathbb{C}^K$ is a noise vector. It is assumed that
$K\le M$.
    
The precoding matrix $\mathbf{G}$ is defined as (Eq. 4.37) <a class="reference internal" href="channel.wireless.html#bhs2017" id="id1">[BHS2017]</a> :

$$
\mathbf{G} = \mathbf{V}\mathbf{D}
$$
    
where

$$
\begin{split}\mathbf{V} &= \mathbf{H}^{\mathsf{H}}\left(\mathbf{H} \mathbf{H}^{\mathsf{H}}\right)^{-1}\\
\mathbf{D} &= \mathop{\text{diag}}\left( \lVert \mathbf{v}_{k} \rVert_2^{-1}, k=0,\dots,K-1 \right).\end{split}
$$
    
This ensures that each stream is precoded with a unit-norm vector,
i.e., $\mathop{\text{tr}}\left(\mathbf{G}\mathbf{G}^{\mathsf{H}}\right)=K$.
The function returns the precoded vector $\mathbf{G}\mathbf{x}$.
Input
 
- **x** (<em>[…,K], tf.complex</em>) – 1+D tensor containing the symbol vectors to be precoded.
- **h** (<em>[…,K,M], tf.complex</em>) – 2+D tensor containing the channel matrices
- **return_precoding_matrices** (<em>bool</em>) – Indicates if the precoding matrices should be returned or not.
Defaults to False.


Output
 
- **x_precoded** (<em>[…,M], tf.complex</em>) – Tensor of the same shape and dtype as `x` apart from the last
dimensions that has changed from <cite>K</cite> to <cite>M</cite>. It contains the
precoded symbol vectors.
- **g** (<em>[…,M,K], tf.complex</em>) – 2+D tensor containing the precoding matrices. It is only returned
if `return_precoding_matrices=True`.




**Note**
    
If you want to use this function in Graph mode with XLA, i.e., within
a function that is decorated with `@tf.function(jit_compile=True)`,
you must set `sionna.Config.xla_compat=true`.
See <a class="reference internal" href="config.html#sionna.Config.xla_compat" title="sionna.Config.xla_compat">`xla_compat`</a>.

## Equalization<a class="headerlink" href="https://nvlabs.github.io/sionna/api/mimo.html#equalization" title="Permalink to this headline"></a>

