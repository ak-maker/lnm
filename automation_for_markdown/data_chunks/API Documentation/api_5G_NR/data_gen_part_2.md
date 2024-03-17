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
## Layer Mapping<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#layer-mapping" title="Permalink to this headline"></a>
### LayerDemapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#layerdemapper" title="Permalink to this headline"></a>
## PUSCH<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#pusch" title="Permalink to this headline"></a>
  
  

### LayerDemapper<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#layerdemapper" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``LayerDemapper`(<em class="sig-param">`layer_mapper`</em>, <em class="sig-param">`num_bits_per_symbol``=``1`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/layer_mapping.html#LayerDemapper">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerDemapper" title="Permalink to this definition"></a>
    
Demaps MIMO layers to coded transport block(s) by following Sec. 6.3.1.3
and Sec. 7.3.1.3 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id9">[3GPP38211]</a>.
    
This layer must be associated to a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper" title="sionna.nr.LayerMapper">`LayerMapper`</a> and
performs the inverse operation.
    
It is assumed that `num_bits_per_symbol` consecutive LLRs belong to
a single symbol position. This allows to apply the LayerDemapper after
demapping symbols to LLR values.
    
If the layer mapper is configured for dual codeword transmission, a list of
both transport block streams is returned.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **layer_mapper** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.LayerMapper" title="sionna.nr.LayerMapper">`LayerMapper`</a>) – Associated LayerMapper.
- **num_bits_per_symbol** (<em>int</em><em>, </em><em>1</em><em> (</em><em>default</em><em>)</em>) – Modulation order. Defines how many consecutive LLRs are associated
to the same symbol position.


Input
    
**inputs** (<em>[…,num_layers, n/num_layers], tf.float</em>) – 2+D tensor containing MIMO layer data sequences.

Output
    
<em>[…,n], or [[…,n1], […,n2]], tf.float</em> – 2+D tensor containing the sequence of bits after layer demapping.
If `num_codewords` =2, a list of two transport blocks is returned.



**Note**
    
As it is more convenient to apply the layer demapper after demapping
symbols to LLRs, this layer groups the input sequence into groups of
`num_bits_per_symbol` LLRs before restoring the original symbol sequence.
This behavior can be deactivated by setting `num_bits_per_symbol` =1.

## PUSCH<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#pusch" title="Permalink to this headline"></a>

