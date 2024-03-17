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
## Resource Grid<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#resource-grid" title="Permalink to this headline"></a>
### ResourceGrid<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#resourcegrid" title="Permalink to this headline"></a>
### ResourceGridMapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#resourcegridmapper" title="Permalink to this headline"></a>
### ResourceGridDemapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#resourcegriddemapper" title="Permalink to this headline"></a>
### RemoveNulledSubcarriers<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#removenulledsubcarriers" title="Permalink to this headline"></a>
## Modulation & Demodulation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#modulation-demodulation" title="Permalink to this headline"></a>
  
  

## Resource Grid<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#resource-grid" title="Permalink to this headline"></a>
    
The following code snippet shows how to setup and visualize an instance of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>:
```python
rg = ResourceGrid(num_ofdm_symbols = 14,
                  fft_size = 64,
                  subcarrier_spacing = 30e3,
                  num_tx = 1,
                  num_streams_per_tx = 1,
                  num_guard_carriers = [5, 6],
                  dc_null = True,
                  pilot_pattern = "kronecker",
                  pilot_ofdm_symbol_indices = [2, 11])
rg.show();
```

    
This code creates a resource grid consisting of 14 OFDM symbols with 64
subcarriers. The first five and last six subcarriers as well as the DC
subcarriers are nulled. The second and eleventh OFDM symbol are reserved
for pilot transmissions.
    
Subcarriers are numbered from $0$ to $N-1$, where $N$
is the FTT size. The index $0$ corresponds to the lowest frequency,
which is $-\frac{N}{2}\Delta_f$ (for $N$ even) or
$-\frac{N-1}{2}\Delta_f$ (for $N$ odd), where $\Delta_f$
is the subcarrier spacing which is irrelevant for the resource grid.
The index $N-1$ corresponds to the highest frequency,
which is $(\frac{N}{2}-1)\Delta_f$ (for $N$ even) or
$\frac{N-1}{2}\Delta_f$ (for $N$ odd).

### ResourceGrid<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#resourcegrid" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``ResourceGrid`(<em class="sig-param">`num_ofdm_symbols`</em>, <em class="sig-param">`fft_size`</em>, <em class="sig-param">`subcarrier_spacing`</em>, <em class="sig-param">`num_tx``=``1`</em>, <em class="sig-param">`num_streams_per_tx``=``1`</em>, <em class="sig-param">`cyclic_prefix_length``=``0`</em>, <em class="sig-param">`num_guard_carriers``=``(0,` `0)`</em>, <em class="sig-param">`dc_null``=``False`</em>, <em class="sig-param">`pilot_pattern``=``None`</em>, <em class="sig-param">`pilot_ofdm_symbol_indices``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/resource_grid.html#ResourceGrid">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="Permalink to this definition"></a>
    
Defines a <cite>ResourceGrid</cite> spanning multiple OFDM symbols and subcarriers.
Parameters
 
- **num_ofdm_symbols** (<em>int</em>) – Number of OFDM symbols.
- **fft_size** (<em>int</em>) – FFT size (, i.e., the number of subcarriers).
- **subcarrier_spacing** (<em>float</em>) – The subcarrier spacing in Hz.
- **num_tx** (<em>int</em>) – Number of transmitters.
- **num_streams_per_tx** (<em>int</em>) – Number of streams per transmitter.
- **cyclic_prefix_length** (<em>int</em>) – Length of the cyclic prefix.
- **num_guard_carriers** (<em>int</em>) – List of two integers defining the number of guardcarriers at the
left and right side of the resource grid.
- **dc_null** (<em>bool</em>) – Indicates if the DC carrier is nulled or not.
- **pilot_pattern** (<em>One of</em><em> [</em><em>None</em><em>, </em><em>"kronecker"</em><em>, </em><em>"empty"</em><em>, </em><a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern"><em>PilotPattern</em></a><em>]</em>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.PilotPattern" title="sionna.ofdm.PilotPattern">`PilotPattern`</a>, a string
shorthand for the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.KroneckerPilotPattern" title="sionna.ofdm.KroneckerPilotPattern">`KroneckerPilotPattern`</a>
or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.EmptyPilotPattern" title="sionna.ofdm.EmptyPilotPattern">`EmptyPilotPattern`</a>, or <cite>None</cite>.
Defaults to <cite>None</cite> which is equivalent to <cite>“empty”</cite>.
- **pilot_ofdm_symbol_indices** (<em>List</em><em>, </em><em>int</em>) – List of indices of OFDM symbols reserved for pilot transmissions.
Only needed if `pilot_pattern="kronecker"`. Defaults to <cite>None</cite>.
- **dtype** (<em>tf.Dtype</em>) – Defines the datatype for internal calculations and the output
dtype. Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`bandwidth`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.bandwidth" title="Permalink to this definition"></a>
    
`fft_size*subcarrier_spacing`.
Type
    
The occupied bandwidth [Hz]




`build_type_grid`()<a class="reference internal" href="../_modules/sionna/ofdm/resource_grid.html#ResourceGrid.build_type_grid">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.build_type_grid" title="Permalink to this definition"></a>
    
Returns a tensor indicating the type of each resource element.
    
Resource elements can be one of
 
- 0 : Data symbol
- 1 : Pilot symbol
- 2 : Guard carrier symbol
- 3 : DC carrier symbol

Output
    
<em>[num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.int32</em> – Tensor indicating for each transmitter and stream the type of
the resource elements of the corresponding resource grid.
The type can be one of [0,1,2,3] as explained above.




<em class="property">`property` </em>`cyclic_prefix_length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.cyclic_prefix_length" title="Permalink to this definition"></a>
    
Length of the cyclic prefix.


<em class="property">`property` </em>`dc_ind`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.dc_ind" title="Permalink to this definition"></a>
    
Index of the DC subcarrier.
    
If `fft_size` is odd, the index is (`fft_size`-1)/2.
If `fft_size` is even, the index is `fft_size`/2.


<em class="property">`property` </em>`dc_null`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.dc_null" title="Permalink to this definition"></a>
    
Indicates if the DC carriers is nulled or not.


<em class="property">`property` </em>`effective_subcarrier_ind`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.effective_subcarrier_ind" title="Permalink to this definition"></a>
    
Returns the indices of the effective subcarriers.


<em class="property">`property` </em>`fft_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.fft_size" title="Permalink to this definition"></a>
    
The FFT size.


<em class="property">`property` </em>`num_data_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_data_symbols" title="Permalink to this definition"></a>
    
Number of resource elements used for data transmissions.


<em class="property">`property` </em>`num_effective_subcarriers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_effective_subcarriers" title="Permalink to this definition"></a>
    
Number of subcarriers used for data and pilot transmissions.


<em class="property">`property` </em>`num_guard_carriers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_guard_carriers" title="Permalink to this definition"></a>
    
Number of left and right guard carriers.


<em class="property">`property` </em>`num_ofdm_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_ofdm_symbols" title="Permalink to this definition"></a>
    
The number of OFDM symbols of the resource grid.


<em class="property">`property` </em>`num_pilot_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_pilot_symbols" title="Permalink to this definition"></a>
    
Number of resource elements used for pilot symbols.


<em class="property">`property` </em>`num_resource_elements`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_resource_elements" title="Permalink to this definition"></a>
    
Number of resource elements.


<em class="property">`property` </em>`num_streams_per_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_streams_per_tx" title="Permalink to this definition"></a>
    
Number of streams  per transmitter.


<em class="property">`property` </em>`num_time_samples`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_time_samples" title="Permalink to this definition"></a>
    
The number of time-domain samples occupied by the resource grid.


<em class="property">`property` </em>`num_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_tx" title="Permalink to this definition"></a>
    
Number of transmitters.


<em class="property">`property` </em>`num_zero_symbols`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.num_zero_symbols" title="Permalink to this definition"></a>
    
Number of empty resource elements.


<em class="property">`property` </em>`ofdm_symbol_duration`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.ofdm_symbol_duration" title="Permalink to this definition"></a>
    
Duration of an OFDM symbol with cyclic prefix [s].


<em class="property">`property` </em>`pilot_pattern`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.pilot_pattern" title="Permalink to this definition"></a>
    
The used PilotPattern.


`show`(<em class="sig-param">`tx_ind``=``0`</em>, <em class="sig-param">`tx_stream_ind``=``0`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/resource_grid.html#ResourceGrid.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.show" title="Permalink to this definition"></a>
    
Visualizes the resource grid for a specific transmitter and stream.
Input
 
- **tx_ind** (<em>int</em>) – Indicates the transmitter index.
- **tx_stream_ind** (<em>int</em>) – Indicates the index of the stream.


Output
    
<cite>matplotlib.figure</cite> – A handle to a matplot figure object.




<em class="property">`property` </em>`subcarrier_spacing`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid.subcarrier_spacing" title="Permalink to this definition"></a>
    
The subcarrier spacing [Hz].


### ResourceGridMapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#resourcegridmapper" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``ResourceGridMapper`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/resource_grid.html#ResourceGridMapper">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGridMapper" title="Permalink to this definition"></a>
    
Maps a tensor of modulated data symbols to a ResourceGrid.
    
This layer takes as input a tensor of modulated data symbols
and maps them together with pilot symbols onto an
OFDM <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>. The output can be
converted to a time-domain signal with the
`Modulator` or further processed in the
frequency domain.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
    
<em>[batch_size, num_tx, num_streams_per_tx, num_data_symbols], tf.complex</em> – The modulated data symbols to be mapped onto the resource grid.

Output
    
<em>[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex</em> – The full OFDM resource grid in the frequency domain.



### ResourceGridDemapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#resourcegriddemapper" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``ResourceGridDemapper`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`stream_management`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/resource_grid.html#ResourceGridDemapper">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGridDemapper" title="Permalink to this definition"></a>
    
Extracts data-carrying resource elements from a resource grid.
    
This layer takes as input an OFDM <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a> and
extracts the data-carrying resource elements. In other words, it implements
the reverse operation of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGridMapper" title="sionna.ofdm.ResourceGridMapper">`ResourceGridMapper`</a>.
Parameters
 
- **resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.
- **stream_management** (<a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement"><em>StreamManagement</em></a>) – An instance of <a class="reference internal" href="mimo.html#sionna.mimo.StreamManagement" title="sionna.mimo.StreamManagement">`StreamManagement`</a>.
- **dtype** (<em>tf.Dtype</em>) – Datatype for internal calculations and the output dtype.
Defaults to <cite>tf.complex64</cite>.


Input
    
<em>[batch_size, num_rx, num_streams_per_rx, num_ofdm_symbols, fft_size, data_dim]</em> – The full OFDM resource grid in the frequency domain.
The last dimension <cite>data_dim</cite> is optional. If <cite>data_dim</cite>
is used, it refers to the dimensionality of the data that should be
demapped to individual streams. An example would be LLRs.

Output
    
<em>[batch_size, num_rx, num_streams_per_rx, num_data_symbols, data_dim]</em> – The data that were mapped into the resource grid.
The last dimension <cite>data_dim</cite> is only returned if it was used for the
input.



### RemoveNulledSubcarriers<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#removenulledsubcarriers" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.ofdm.``RemoveNulledSubcarriers`(<em class="sig-param">`resource_grid`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/ofdm/resource_grid.html#RemoveNulledSubcarriers">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.RemoveNulledSubcarriers" title="Permalink to this definition"></a>
    
Removes nulled guard and/or DC subcarriers from a resource grid.
Parameters
    
**resource_grid** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid"><em>ResourceGrid</em></a>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>.

Input
    
<em>[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex64</em> – Full resource grid.

Output
    
<em>[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex64</em> – Resource grid without nulled subcarriers.



## Modulation & Demodulation<a class="headerlink" href="https://nvlabs.github.io/sionna/api/ofdm.html#modulation-demodulation" title="Permalink to this headline"></a>

