# 5G NR<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#g-nr" title="Permalink to this headline"></a>
    
This module provides layers and functions to support simulations of
5G NR compliant features, in particular, the physical uplink shared channel (PUSCH). It provides implementations of a subset of the physical layer functionalities as described in the 3GPP specifications <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id1">[3GPP38211]</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38212" id="id2">[3GPP38212]</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id3">[3GPP38214]</a>.
    
The best way to discover this module’s components is by having a look at the <a class="reference external" href="../examples/5G_NR_PUSCH.html">5G NR PUSCH Tutorial</a>.
    
The following code snippet shows how you can make standard-compliant
simulations of the 5G NR PUSCH with a few lines of code:
```python
# Create a PUSCH configuration with default settings
pusch_config = PUSCHConfig()
# Instantiate a PUSCHTransmitter from the PUSCHConfig
pusch_transmitter = PUSCHTransmitter(pusch_config)
# Create a PUSCHReceiver using the PUSCHTransmitter
pusch_receiver = PUSCHReceiver(pusch_transmitter)
# AWGN channel
channel = AWGN()
# Simulate transmissions over the AWGN channel
batch_size = 16
no = 0.1 # Noise variance
x, b = pusch_transmitter(batch_size) # Generate transmit signal and info bits
y = channel([x, no]) # Simulate channel output
b_hat = pusch_receiver([x, no]) # Recover the info bits
# Compute BER
print("BER:", compute_ber(b, b_hat).numpy())
```

    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter" title="sionna.nr.PUSCHTransmitter">`PUSCHTransmitter`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHReceiver" title="sionna.nr.PUSCHReceiver">`PUSCHReceiver`</a> provide high-level abstractions of all required processing blocks. You can easily modify them according to your needs.

# Table of Content
## Carrier<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#carrier" title="Permalink to this headline"></a>
### CarrierConfig<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#carrierconfig" title="Permalink to this headline"></a>
## Layer Mapping<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#layer-mapping" title="Permalink to this headline"></a>
### LayerMapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#layermapper" title="Permalink to this headline"></a>
  
  

## Carrier<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#carrier" title="Permalink to this headline"></a>

### CarrierConfig<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#carrierconfig" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``CarrierConfig`(<em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/carrier_config.html#CarrierConfig">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig" title="Permalink to this definition"></a>
    
The CarrierConfig objects sets parameters for a specific OFDM numerology,
as described in Section 4 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id4">[3GPP38211]</a>.
    
All configurable properties can be provided as keyword arguments during the
initialization or changed later.
<p class="rubric">Example
```python
>>> carrier_config = CarrierConfig(n_cell_id=41)
>>> carrier_config.subcarrier_spacing = 30
```
<em class="property">`property` </em>`cyclic_prefix`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.cyclic_prefix" title="Permalink to this definition"></a>
    
Cyclic prefix length
    
The option “normal” corresponds to 14 OFDM symbols per slot, while
“extended” corresponds to 12 OFDM symbols. The latter option is
only possible with a <cite>subcarrier_spacing</cite> of 60 kHz.
Type
    
str, “normal” (default) | “extended”




<em class="property">`property` </em>`cyclic_prefix_length`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.cyclic_prefix_length" title="Permalink to this definition"></a>
    
Cyclic prefix length
$N_{\text{CP},l}^{\mu} \cdot T_{\text{c}}$ [s]
Type
    
float, read-only




<em class="property">`property` </em>`frame_duration`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.frame_duration" title="Permalink to this definition"></a>
    
Duration of a frame
$T_\text{f}$ [s]
Type
    
float, 10e-3 (default), read-only




<em class="property">`property` </em>`frame_number`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.frame_number" title="Permalink to this definition"></a>
    
System frame number $n_\text{f}$
Type
    
int, 0 (default), [0,…,1023]




<em class="property">`property` </em>`kappa`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.kappa" title="Permalink to this definition"></a>
    
The constant
$\kappa = T_\text{s}/T_\text{c}$
Type
    
float, 64, read-only




<em class="property">`property` </em>`mu`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.mu" title="Permalink to this definition"></a>
    
Subcarrier
spacing configuration, $\Delta f = 2^\mu 15$ kHz
Type
    
int, 0 (default) | 1 | 2 | 3 | 4 | 5 | 6, read-only




<em class="property">`property` </em>`n_cell_id`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.n_cell_id" title="Permalink to this definition"></a>
    
Physical layer cell identity
$N_\text{ID}^\text{cell}$
Type
    
int, 1 (default) | [0,…,1007]




<em class="property">`property` </em>`n_size_grid`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.n_size_grid" title="Permalink to this definition"></a>
    
Number of resource blocks in the
carrier resource grid $N^{\text{size},\mu}_{\text{grid},x}$
Type
    
int, 4 (default) | [1,…,275]




<em class="property">`property` </em>`n_start_grid`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.n_start_grid" title="Permalink to this definition"></a>
    
Start of resource grid relative to
common resource block (CRB) 0
$N^{\text{start},\mu}_{\text{grid},x}$
Type
    
int, 0 (default) | [0,…,2199]




<em class="property">`property` </em>`num_slots_per_frame`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.num_slots_per_frame" title="Permalink to this definition"></a>
    
Number
of slots per frame $N_\text{slot}^{\text{frame},\mu}$
    
Depends on the <cite>subcarrier_spacing</cite>.
Type
    
int, 10 (default) | 20 | 40 | 80 | 160 | 320 | 640, read-only




<em class="property">`property` </em>`num_slots_per_subframe`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.num_slots_per_subframe" title="Permalink to this definition"></a>
    
Number of
slots per subframe $N_\text{slot}^{\text{subframe},\mu}$
    
Depends on the <cite>subcarrier_spacing</cite>.
Type
    
int, 1 (default) | 2 | 4 | 8 | 16 | 32 | 64, read-only




<em class="property">`property` </em>`num_symbols_per_slot`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.num_symbols_per_slot" title="Permalink to this definition"></a>
    
Number of OFDM symbols per slot
$N_\text{symb}^\text{slot}$
    
Configured through the <cite>cyclic_prefix</cite>.
Type
    
int, 14 (default) | 12, read-only




<em class="property">`property` </em>`slot_number`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.slot_number" title="Permalink to this definition"></a>
    
Slot number within a frame
$n^\mu_{s,f}$
Type
    
int, 0 (default), [0,…,num_slots_per_frame]




<em class="property">`property` </em>`sub_frame_duration`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.sub_frame_duration" title="Permalink to this definition"></a>
    
Duration of a subframe
$T_\text{sf}$ [s]
Type
    
float, 1e-3 (default), read-only




<em class="property">`property` </em>`subcarrier_spacing`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.subcarrier_spacing" title="Permalink to this definition"></a>
    
Subcarrier
spacing $\Delta f$ [kHz]
Type
    
float, 15 (default) | 30 | 60 | 120 | 240 | 480 | 960




<em class="property">`property` </em>`t_c`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.t_c" title="Permalink to this definition"></a>
    
Sampling time $T_\text{c}$ for
subcarrier spacing 480kHz.
Type
    
float, 0.509e-9 [s], read-only




<em class="property">`property` </em>`t_s`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.t_s" title="Permalink to this definition"></a>
    
Sampling time $T_\text{s}$ for
subcarrier spacing 15kHz.
Type
    
float, 32.552e-9 [s], read-only




## Layer Mapping<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#layer-mapping" title="Permalink to this headline"></a>

### LayerMapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#layermapper" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``LayerMapper`(<em class="sig-param">`num_layers``=``1`</em>, <em class="sig-param">`verbose``=``False`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/layer_mapping.html#LayerMapper">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper" title="Permalink to this definition"></a>
    
Performs MIMO layer mapping of modulated symbols to layers as defined in
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id5">[3GPP38211]</a>.
    
The LayerMapper supports PUSCH and PDSCH channels and follows the procedure
as defined in Sec. 6.3.1.3 and Sec. 7.3.1.3 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id6">[3GPP38211]</a>, respectively.
    
As specified in Tab. 7.3.1.3.-1 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id7">[3GPP38211]</a>, the LayerMapper expects two
input streams for multiplexing if more than 4 layers are active (only
relevant for PDSCH).
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **num_layers** (<em>int</em><em>, </em><em>1</em><em> (</em><em>default</em><em>) </em><em>|</em><em> [</em><em>1</em><em>,</em><em>...</em><em>,</em><em>8</em><em>]</em>) – Number of MIMO layers. If
`num_layers` >=4, a list of two inputs is expected.
- **verbose** (<em>bool</em><em>, </em><em>False</em><em> (</em><em>default</em><em>)</em>) – If True, additional parameters are printed.


Input
    
**inputs** (<em>[…,n], or [[…,n1], […,n2]], tf.complex</em>) – 2+D tensor containing the sequence of symbols to be mapped. If
`num_layers` >=4, a list of two inputs is expected and <cite>n1</cite>/<cite>n2</cite>
must be chosen as defined in Tab. 7.3.1.3.-1 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id8">[3GPP38211]</a>.

Output
    
<em>[…,num_layers, n/num_layers], tf.complex</em> – 2+D tensor containing the sequence of symbols mapped to the MIMO
layers.



<em class="property">`property` </em>`num_codewords`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper.num_codewords" title="Permalink to this definition"></a>
    
Number of input codewords for layer mapping. Can be either 1 or 2.


<em class="property">`property` </em>`num_layers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper.num_layers" title="Permalink to this definition"></a>
    
Number of MIMO layers


<em class="property">`property` </em>`num_layers0`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper.num_layers0" title="Permalink to this definition"></a>
    
Number of layers for first codeword (only relevant for
<cite>num_codewords</cite> =2)


<em class="property">`property` </em>`num_layers1`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper.num_layers1" title="Permalink to this definition"></a>
    
Number of layers for second codeword (only relevant for
<cite>num_codewords</cite> =2)


