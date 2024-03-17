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
### select_mcs<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#select-mcs" title="Permalink to this headline"></a>
  
  

### select_mcs<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#select-mcs" title="Permalink to this headline"></a>

`sionna.nr.utils.``select_mcs`(<em class="sig-param">`mcs_index`</em>, <em class="sig-param">`table_index``=``1`</em>, <em class="sig-param">`channel_type``=``'PUSCH'`</em>, <em class="sig-param">`transform_precoding``=``False`</em>, <em class="sig-param">`pi2bpsk``=``False`</em>, <em class="sig-param">`verbose``=``False`</em>)<a class="reference internal" href="../_modules/sionna/nr/utils.html#select_mcs">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.utils.select_mcs" title="Permalink to this definition"></a>
    
Selects modulation and coding scheme (MCS) as specified in TS 38.214 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id40">[3GPP38214]</a>.
    
Implements MCS tables as defined in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id41">[3GPP38214]</a> for PUSCH and PDSCH.
Parameters
 
- **mcs_index** (<em>int|</em><em> [</em><em>0</em><em>,</em><em>...</em><em>,</em><em>28</em><em>]</em>) – MCS index (denoted as $I_{MCS}$ in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id42">[3GPP38214]</a>).
- **table_index** (<em>int</em><em>, </em><em>1</em><em> (</em><em>default</em><em>) </em><em>| 2 | 3 | 4</em>) – Indicates which MCS table from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id43">[3GPP38214]</a> to use. Starts with index “1”.
- **channel_type** (<em>str</em><em>, </em><em>"PUSCH"</em><em> (</em><em>default</em><em>) </em><em>| "PDSCH"</em>) – 5G NR physical channel type. Valid choices are “PDSCH” and “PUSCH”.
- **transform_precoding** (<em>bool</em><em>, </em><em>False</em><em> (</em><em>default</em><em>)</em>) – If True, the MCS tables as described in Sec. 6.1.4.1
in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id44">[3GPP38214]</a> are applied. Only relevant for “PUSCH”.
- **pi2bpsk** (<em>bool</em><em>, </em><em>False</em><em> (</em><em>default</em><em>)</em>) – If True, the higher-layer parameter <cite>tp-pi2BPSK</cite> as
described in Sec. 6.1.4.1 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id45">[3GPP38214]</a> is applied. Only relevant
for “PUSCH”.
- **verbose** (<em>bool</em><em>, </em><em>False</em><em> (</em><em>default</em><em>)</em>) – If True, additional information will be printed.


Returns
    
 
- <em>(modulation_order, target_rate)</em> – Tuple:
- **modulation_order** (<em>int</em>) – Modulation order, i.e., number of bits per symbol.
- **target_rate** (<em>float</em>) – Target coderate.





References:
3GPP38211(<a href="https://nvlabs.github.io/sionna/api/nr.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id4">2</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id5">3</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id6">4</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id7">5</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id8">6</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id9">7</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id10">8</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id11">9</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id12">10</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id13">11</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id14">12</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id15">13</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id17">14</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id18">15</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id19">16</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id20">17</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id30">18</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id32">19</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id35">20</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id39">21</a>)
    
3GPP TS 38.211. “NR; Physical channels and modulation.”

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/nr.html#id2">3GPP38212</a>
    
3GPP TS 38.212. “NR; Multiplexing and channel coding”

3GPP38214(<a href="https://nvlabs.github.io/sionna/api/nr.html#id3">1</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id16">2</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id21">3</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id22">4</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id23">5</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id24">6</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id25">7</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id26">8</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id27">9</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id28">10</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id29">11</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id31">12</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id33">13</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id34">14</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id38">15</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id40">16</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id41">17</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id42">18</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id43">19</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id44">20</a>,<a href="https://nvlabs.github.io/sionna/api/nr.html#id45">21</a>)
    
3GPP TS 38.214. “NR; Physical layer procedures for data.”



