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
## PUSCH
### PUSCHConfig<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschconfig" title="Permalink to this headline"></a>
  
  

### PUSCHConfig<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#puschconfig" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``PUSCHConfig`(<em class="sig-param">`carrier_config``=``None`</em>, <em class="sig-param">`pusch_dmrs_config``=``None`</em>, <em class="sig-param">`tb_config``=``None`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_config.html#PUSCHConfig">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="Permalink to this definition"></a>
    
The PUSCHConfig objects sets parameters for a physical uplink shared
channel (PUSCH), as described in Sections 6.3 and 6.4 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id10">[3GPP38211]</a>.
    
All configurable properties can be provided as keyword arguments during the
initialization or changed later.
Parameters
 
- **carrier_config** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig" title="sionna.nr.CarrierConfig">`CarrierConfig`</a> or <cite>None</cite>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig" title="sionna.nr.CarrierConfig">`CarrierConfig`</a>. If <cite>None</cite>, a
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig" title="sionna.nr.CarrierConfig">`CarrierConfig`</a> instance with default settings
will be created.
- **pusch_dmrs_config** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a> or <cite>None</cite>) – An instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a>. If <cite>None</cite>, a
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a> instance with default settings
will be created.



<p class="rubric">Example
```python
>>> pusch_config = PUSCHConfig(mapping_type="B")
>>> pusch_config.dmrs.config_type = 2
>>> pusch_config.carrier.subcarrier_spacing = 30
```
`c_init`(<em class="sig-param">`l`</em>)<a class="reference internal" href="../_modules/sionna/nr/pusch_config.html#PUSCHConfig.c_init">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.c_init" title="Permalink to this definition"></a>
    
Compute RNG initialization $c_\text{init}$ as in Section 6.4.1.1.1.1 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id11">[3GPP38211]</a>
Input
    
**l** (<em>int</em>) – OFDM symbol index relative to a reference $l$

Output
    
**c_init** (<em>int</em>) – RNG initialization value




<em class="property">`property` </em>`carrier`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.carrier" title="Permalink to this definition"></a>
    
Carrier configuration
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig" title="sionna.nr.CarrierConfig">`CarrierConfig`</a>




<em class="property">`property` </em>`dmrs`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.dmrs" title="Permalink to this definition"></a>
    
PUSCH DMRS configuration
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHDMRSConfig" title="sionna.nr.PUSCHDMRSConfig">`PUSCHDMRSConfig`</a>




<em class="property">`property` </em>`dmrs_grid`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.dmrs_grid" title="Permalink to this definition"></a>
    
Empty
resource grid for each DMRS port, filled with DMRS signals
    
This property returns for each configured DMRS port an empty
resource grid filled with DMRS signals as defined in
Section 6.4.1.1 [3GPP38211]. Not all possible options are implemented,
e.g., frequency hopping and transform precoding are not available.
    
This property provides the <em>unprecoded</em> DMRS for each configured DMRS port.
Precoding might be applied to map the DMRS to the antenna ports. However,
in this case, the number of DMRS ports cannot be larger than the number of
layers.
Type
    
complex, [num_dmrs_ports, num_subcarriers, num_symbols_per_slot], read-only




<em class="property">`property` </em>`dmrs_mask`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.dmrs_mask" title="Permalink to this definition"></a>
    
Masked
resource elements in the resource grid. <cite>True</cite> corresponds to
resource elements on which no data is transmitted.
Type
    
bool, [num_subcarriers, num_symbols_per_slot], read-only




<em class="property">`property` </em>`dmrs_symbol_indices`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.dmrs_symbol_indices" title="Permalink to this definition"></a>
    
Indices of DMRS symbols within a slot
Type
    
list, int, read-only




<em class="property">`property` </em>`frequency_hopping`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.frequency_hopping" title="Permalink to this definition"></a>
    
Frequency hopping configuration
Type
    
str, “neither” (default), read-only




<em class="property">`property` </em>`l_bar`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.l_bar" title="Permalink to this definition"></a>
    
List of possible values of
$\bar{l}$ used for DMRS generation
    
Defined in Tables 6.4.1.1.3-3 and 6.4.1.1.3-4 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id12">[3GPP38211]</a>.
Type
    
list, elements in [0,…,11], read-only




<em class="property">`property` </em>`mapping_type`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.mapping_type" title="Permalink to this definition"></a>
    
Mapping type
Type
    
string, “A” (default) | “B”




<em class="property">`property` </em>`n_rnti`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.n_rnti" title="Permalink to this definition"></a>
    
Radio network temporary identifier
$n_\text{RNTI}$
Type
    
int, 1 (default), [0,…,65535]




<em class="property">`property` </em>`n_size_bwp`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.n_size_bwp" title="Permalink to this definition"></a>
    
Number of resource blocks in the
bandwidth part (BWP) $N^{\text{size},\mu}_{\text{BWP},i}$
    
If set to <cite>None</cite>, the property
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.CarrierConfig.n_size_grid" title="sionna.nr.CarrierConfig.n_size_grid">`n_size_grid`</a> of
<cite>carrier</cite> will be used.
Type
    
int, None (default), [1,…,275]




<em class="property">`property` </em>`n_start_bwp`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.n_start_bwp" title="Permalink to this definition"></a>
    
Start of BWP relative to
common resource block (CRB) 0
$N^{\text{start},\mu}_{\text{BWP},i}$
Type
    
int, 0 (default) | [0,…,2199]




<em class="property">`property` </em>`num_antenna_ports`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_antenna_ports" title="Permalink to this definition"></a>
    
Number of antenna ports
    
Must be larger than or equal to
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_layers" title="sionna.nr.PUSCHConfig.num_layers">`num_layers`</a>.
Type
    
int, 1 (default) | 2 | 4




<em class="property">`property` </em>`num_coded_bits`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_coded_bits" title="Permalink to this definition"></a>
    
Number of coded bits that fit into one PUSCH slot.
Type
    
int, read-only




<em class="property">`property` </em>`num_layers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_layers" title="Permalink to this definition"></a>
    
Number of transmission layers
$\nu$
    
Must be smaller than or equal to
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_antenna_ports" title="sionna.nr.PUSCHConfig.num_antenna_ports">`num_antenna_ports`</a>.
Type
    
int, 1 (default) | 2 | 3 | 4




<em class="property">`property` </em>`num_ov`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_ov" title="Permalink to this definition"></a>
    
Number of unused resource elements due to additional overhead as specified by higher layer.
Type
    
int, 0 (default), read-only




<em class="property">`property` </em>`num_res_per_prb`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_res_per_prb" title="Permalink to this definition"></a>
    
Number of resource elements per PRB
available for data
Type
    
int, read-only




<em class="property">`property` </em>`num_resource_blocks`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_resource_blocks" title="Permalink to this definition"></a>
    
Number of allocated resource blocks for the
PUSCH transmissions.
Type
    
int, read-only




<em class="property">`property` </em>`num_subcarriers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.num_subcarriers" title="Permalink to this definition"></a>
    
Number of allocated subcarriers for the
PUSCH transmissions
Type
    
int, read-only




<em class="property">`property` </em>`precoding`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.precoding" title="Permalink to this definition"></a>
    
PUSCH
transmission scheme
Type
    
str, “non-codebook” (default), “codebook”




<em class="property">`property` </em>`precoding_matrix`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.precoding_matrix" title="Permalink to this definition"></a>
    
Precoding matrix
$\mathbf{W}$ as defined in
Tables 6.3.1.5-1 to 6.3.1.5-7 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id13">[3GPP38211]</a>.
    
Only relevant if `precoding`
is “codebook”.
Type
    
nd_array, complex, [num_antenna_ports, numLayers]




`show`()<a class="reference internal" href="../_modules/sionna/nr/pusch_config.html#PUSCHConfig.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.show" title="Permalink to this definition"></a>
    
Print all properties of the PUSCHConfig and children


<em class="property">`property` </em>`symbol_allocation`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.symbol_allocation" title="Permalink to this definition"></a>
    
PUSCH symbol allocation
    
The first elements denotes the start of the symbol allocation.
The second denotes the positive number of allocated OFDM symbols.
For <cite>mapping_type</cite> “A”, the first element must be zero.
For <cite>mapping_type</cite> “B”, the first element must be in
[0,…,13]. The second element must be such that the index
of the last allocated OFDM symbol is not larger than 13
(for “normal” cyclic prefix) or 11 (for “extended” cyclic prefix).
Type
    
2-tuple, int, [0, 14] (default)




<em class="property">`property` </em>`tb`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.tb" title="Permalink to this definition"></a>
    
Transport block configuration
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig" title="sionna.nr.TBConfig">`TBConfig`</a>




<em class="property">`property` </em>`tb_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.tb_size" title="Permalink to this definition"></a>
    
Transport block size, i.e., how many information bits can be encoded into a slot for the given slot configuration.
Type
    
int, read-only




<em class="property">`property` </em>`tpmi`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.tpmi" title="Permalink to this definition"></a>
    
Transmit precoding matrix indicator
    
The allowed value depends on the number of layers and
the number of antenna ports according to Table 6.3.1.5-1
until Table 6.3.1.5-7 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id14">[3GPP38211]</a>.
Type
    
int,  0 (default) | [0,…,27]




<em class="property">`property` </em>`transform_precoding`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig.transform_precoding" title="Permalink to this definition"></a>
    
Use transform precoding
Type
    
bool, False (default)




