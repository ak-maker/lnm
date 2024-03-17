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
## Transport Block<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#transport-block" title="Permalink to this headline"></a>
### TBEncoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#tbencoder" title="Permalink to this headline"></a>
### TBDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#tbdecoder" title="Permalink to this headline"></a>
## Utils<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#utils" title="Permalink to this headline"></a>
### calculate_tb_size<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#calculate-tb-size" title="Permalink to this headline"></a>
### generate_prng_seq<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#generate-prng-seq" title="Permalink to this headline"></a>
  
  

### TBEncoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#tbencoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``TBEncoder`(<em class="sig-param">`target_tb_size`</em>, <em class="sig-param">`num_coded_bits`</em>, <em class="sig-param">`target_coderate`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`num_layers=1`</em>, <em class="sig-param">`n_rnti=1`</em>, <em class="sig-param">`n_id=1`</em>, <em class="sig-param">`channel_type="PUSCH"`</em>, <em class="sig-param">`codeword_index=0`</em>, <em class="sig-param">`use_scrambler=True`</em>, <em class="sig-param">`verbose=False`</em>, <em class="sig-param">`output_dtype=tf.float32`</em>, <em class="sig-param">`**kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/tb_encoder.html#TBEncoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder" title="Permalink to this definition"></a>
    
5G NR transport block (TB) encoder as defined in TS 38.214
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id29">[3GPP38214]</a> and TS 38.211 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id30">[3GPP38211]</a>
    
The transport block (TB) encoder takes as input a <cite>transport block</cite> of
information bits and generates a sequence of codewords for transmission.
For this, the information bit sequence is segmented into multiple codewords,
protected by additional CRC checks and FEC encoded. Further, interleaving
and scrambling is applied before a codeword concatenation generates the
final bit sequence. Fig. 1 provides an overview of the TB encoding
procedure and we refer the interested reader to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id31">[3GPP38214]</a> and
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id32">[3GPP38211]</a> for further details.
<p class="caption">Fig. 10 Fig. 1: Overview TB encoding (CB CRC does not always apply).<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#id50" title="Permalink to this image"></a>
    
If `n_rnti` and `n_id` are given as list, the TBEncoder encodes
<cite>num_tx = len(</cite> `n_rnti` <cite>)</cite> parallel input streams with different
scrambling sequences per user.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **target_tb_size** (<em>int</em>) – Target transport block size, i.e., how many information bits are
encoded into the TB. Note that the effective TB size can be
slightly different due to quantization. If required, zero padding
is internally applied.
- **num_coded_bits** (<em>int</em>) – Number of coded bits after TB encoding.
- **target_coderate** (<em>float</em>) – Target coderate.
- **num_bits_per_symbol** (<em>int</em>) – Modulation order, i.e., number of bits per QAM symbol.
- **num_layers** (<em>int</em><em>, </em><em>1</em><em> (</em><em>default</em><em>) </em><em>|</em><em> [</em><em>1</em><em>,</em><em>...</em><em>,</em><em>8</em><em>]</em>) – Number of transmission layers.
- **n_rnti** (<em>int</em><em> or </em><em>list of ints</em><em>, </em><em>1</em><em> (</em><em>default</em><em>) </em><em>|</em><em> [</em><em>0</em><em>,</em><em>...</em><em>,</em><em>65335</em><em>]</em>) – RNTI identifier provided by higher layer. Defaults to 1 and must be
in range <cite>[0, 65335]</cite>. Defines a part of the random seed of the
scrambler. If provided as list, every list entry defines the RNTI
of an independent input stream.
- **n_id** (<em>int</em><em> or </em><em>list of ints</em><em>, </em><em>1</em><em> (</em><em>default</em><em>) </em><em>|</em><em> [</em><em>0</em><em>,</em><em>...</em><em>,</em><em>1023</em><em>]</em>) – Data scrambling ID $n_\text{ID}$ related to cell id and
provided by higher layer.
Defaults to 1 and must be in range <cite>[0, 1023]</cite>. If provided as
list, every list entry defines the scrambling id of an independent
input stream.
- **channel_type** (<em>str</em><em>, </em><em>"PUSCH"</em><em> (</em><em>default</em><em>) </em><em>| "PDSCH"</em>) – Can be either “PUSCH” or “PDSCH”.
- **codeword_index** (<em>int</em><em>, </em><em>0</em><em> (</em><em>default</em><em>) </em><em>| 1</em>) – Scrambler can be configured for two codeword transmission.
`codeword_index` can be either 0 or 1. Must be 0 for
`channel_type` = “PUSCH”.
- **use_scrambler** (<em>bool</em><em>, </em><em>True</em><em> (</em><em>default</em><em>)</em>) – If False, no data scrambling is applied (non standard-compliant).
- **verbose** (<em>bool</em><em>, </em><em>False</em><em> (</em><em>default</em><em>)</em>) – If <cite>True</cite>, additional parameters are printed during initialization.
- **dtype** (<em>tf.float32</em><em> (</em><em>default</em><em>)</em>) – Defines the datatype for internal calculations and the output dtype.


Input
    
**inputs** (<em>[…,target_tb_size] or […,num_tx,target_tb_size], tf.float</em>) – 2+D tensor containing the information bits to be encoded. If
`n_rnti` and `n_id` are a list of size <cite>num_tx</cite>, the input must
be of shape <cite>[…,num_tx,target_tb_size]</cite>.

Output
    
<em>[…,num_coded_bits], tf.float</em> – 2+D tensor containing the sequence of the encoded codeword bits of
the transport block.



**Note**
    
The parameters `tb_size` and `num_coded_bits` can be derived by the
`calculate_tb_size()` function or
by accessing the corresponding <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a> attributes.

<em class="property">`property` </em>`cb_crc_encoder`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.cb_crc_encoder" title="Permalink to this definition"></a>
    
CB CRC encoder. <cite>None</cite> if no CB CRC is applied.


<em class="property">`property` </em>`coderate`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.coderate" title="Permalink to this definition"></a>
    
Effective coderate of the TB after rate-matching including overhead
for the CRC.


<em class="property">`property` </em>`cw_lengths`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.cw_lengths" title="Permalink to this definition"></a>
    
Each list element defines the codeword length of each of the
codewords after LDPC encoding and rate-matching. The total number of
coded bits is $\sum$ <cite>cw_lengths</cite>.


<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.k" title="Permalink to this definition"></a>
    
Number of input information bits. Equals <cite>tb_size</cite> except for zero
padding of the last positions if the `target_tb_size` is quantized.


<em class="property">`property` </em>`k_padding`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.k_padding" title="Permalink to this definition"></a>
    
Number of zero padded bits at the end of the TB.


<em class="property">`property` </em>`ldpc_encoder`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.ldpc_encoder" title="Permalink to this definition"></a>
    
LDPC encoder used for TB encoding.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.n" title="Permalink to this definition"></a>
    
Total number of output bits.


<em class="property">`property` </em>`num_cbs`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.num_cbs" title="Permalink to this definition"></a>
    
Number code blocks.


<em class="property">`property` </em>`num_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.num_tx" title="Permalink to this definition"></a>
    
Number of independent streams


<em class="property">`property` </em>`output_perm_inv`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.output_perm_inv" title="Permalink to this definition"></a>
    
Inverse interleaver pattern for output bit interleaver.


<em class="property">`property` </em>`scrambler`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.scrambler" title="Permalink to this definition"></a>
    
Scrambler used for TB scrambling. <cite>None</cite> if no scrambler is used.


<em class="property">`property` </em>`tb_crc_encoder`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.tb_crc_encoder" title="Permalink to this definition"></a>
    
TB CRC encoder


<em class="property">`property` </em>`tb_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder.tb_size" title="Permalink to this definition"></a>
    
Effective number of information bits per TB.
Note that (if required) internal zero padding can be applied to match
the request exact `target_tb_size`.


### TBDecoder<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#tbdecoder" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``TBDecoder`(<em class="sig-param">`encoder`</em>, <em class="sig-param">`num_bp_iter``=``20`</em>, <em class="sig-param">`cn_type``=``'boxplus-phi'`</em>, <em class="sig-param">`output_dtype``=``tf.float32`</em>, <em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/tb_decoder.html#TBDecoder">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder" title="Permalink to this definition"></a>
    
5G NR transport block (TB) decoder as defined in TS 38.214
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id33">[3GPP38214]</a>.
    
The transport block decoder takes as input a sequence of noisy channel
observations and reconstructs the corresponding <cite>transport block</cite> of
information bits. The detailed procedure is described in TS 38.214
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id34">[3GPP38214]</a> and TS 38.211 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id35">[3GPP38211]</a>.
    
The class inherits from the Keras layer class and can be used as layer in a
Keras model.
Parameters
 
- **encoder** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBEncoder" title="sionna.nr.TBEncoder">`TBEncoder`</a>) – Associated transport block encoder used for encoding of the signal.
- **num_bp_iter** (<em>int</em><em>, </em><em>20</em><em> (</em><em>default</em><em>)</em>) – Number of BP decoder iterations
- **cn_type** (<em>str</em><em>, </em><em>"boxplus-phi"</em><em> (</em><em>default</em><em>) </em><em>| "boxplus" | "minsum"</em>) – The check node processing function of the LDPC BP decoder.
One of {<cite>“boxplus”</cite>, <cite>“boxplus-phi”</cite>, <cite>“minsum”</cite>} where
‘“boxplus”’ implements the single-parity-check APP decoding rule.
‘“boxplus-phi”’ implements the numerical more stable version of
boxplus <a class="reference internal" href="fec.ldpc.html#ryan" id="id36">[Ryan]</a>.
‘“minsum”’ implements the min-approximation of the CN update rule
<a class="reference internal" href="fec.ldpc.html#ryan" id="id37">[Ryan]</a>.
- **output_dtype** (<em>tf.float32</em><em> (</em><em>default</em><em>)</em>) – Defines the datatype for internal calculations and the output dtype.


Input
    
**inputs** (<em>[…,num_coded_bits], tf.float</em>) – 2+D tensor containing channel logits/llr values of the (noisy)
channel observations.

Output
 
- **b_hat** (<em>[…,target_tb_size], tf.float</em>) – 2+D tensor containing hard decided bit estimates of all information
bits of the transport block.
- **tb_crc_status** (<em>[…], tf.bool</em>) – Transport block CRC status indicating if a transport block was
(most likely) correctly recovered. Note that false positives are
possible.




<em class="property">`property` </em>`k`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder.k" title="Permalink to this definition"></a>
    
Number of input information bits. Equals TB size.


<em class="property">`property` </em>`n`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder.n" title="Permalink to this definition"></a>
    
Total number of output codeword bits.


<em class="property">`property` </em>`tb_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBDecoder.tb_size" title="Permalink to this definition"></a>
    
Number of information bits per TB.


## Utils<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#utils" title="Permalink to this headline"></a>

### calculate_tb_size<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#calculate-tb-size" title="Permalink to this headline"></a>

`sionna.nr.utils.``calculate_tb_size`(<em class="sig-param">`modulation_order`</em>, <em class="sig-param">`target_coderate`</em>, <em class="sig-param">`target_tb_size``=``None`</em>, <em class="sig-param">`num_coded_bits``=``None`</em>, <em class="sig-param">`num_prbs``=``None`</em>, <em class="sig-param">`num_ofdm_symbols``=``None`</em>, <em class="sig-param">`num_dmrs_per_prb``=``None`</em>, <em class="sig-param">`num_layers``=``1`</em>, <em class="sig-param">`num_ov``=``0`</em>, <em class="sig-param">`tb_scaling``=``1.0`</em>, <em class="sig-param">`verbose``=``True`</em>)<a class="reference internal" href="../_modules/sionna/nr/utils.html#calculate_tb_size">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.utils.calculate_tb_size" title="Permalink to this definition"></a>
    
Calculates transport block (TB) size for given system parameters.
    
This function follows the basic procedure as defined in TS 38.214 Sec.
5.1.3.2 and Sec. 6.1.4.2 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id38">[3GPP38214]</a>.
Parameters
 
- **modulation_order** (<em>int</em>) – Modulation order, i.e., number of bits per QAM symbol.
- **target_coderate** (<em>float</em>) – Target coderate.
- **target_tb_size** (<em>None</em><em> (</em><em>default</em><em>) </em><em>| int</em>) – Target transport block size, i.e., how many information bits can be
encoded into a slot for the given slot configuration. If provided,
`num_prbs`, `num_ofdm_symbols` and `num_dmrs_per_prb` will be
ignored.
- **num_coded_bits** (<em>None</em><em> (</em><em>default</em><em>) </em><em>| int</em>) – How many coded bits can be fit into a given slot. If provided,
`num_prbs`, `num_ofdm_symbols` and `num_dmrs_per_prb` will be
ignored.
- **num_prbs** (<em>None</em><em> (</em><em>default</em><em>) </em><em>| int</em>) – Total number of allocated PRBs per OFDM symbol where 1 PRB equals 12
subcarriers.
- **num_ofdm_symbols** (<em>None</em><em> (</em><em>default</em><em>) </em><em>| int</em>) – Number of OFDM symbols allocated for transmission. Cannot be larger
than 14.
- **num_dmrs_per_prb** (<em>None</em><em> (</em><em>default</em><em>) </em><em>| int</em>) – Number of DMRS (i.e., pilot) symbols per PRB that are NOT used for data
transmission. Sum over all `num_ofdm_symbols` OFDM symbols.
- **num_layers** (<em>int</em><em>, </em><em>1</em><em> (</em><em>default</em><em>)</em>) – Number of MIMO layers.
- **num_ov** (<em>int</em><em>, </em><em>0</em><em> (</em><em>default</em><em>)</em>) – Number of unused resource elements due to additional
overhead as specified by higher layer.
- **tb_scaling** (<em>float</em><em>, </em><em>0.25 | 0.5 | 1</em><em> (</em><em>default</em><em>)</em>) – TB scaling factor for PDSCH as defined in TS 38.214 Tab. 5.1.3.2-2.
Valid choices are 0.25, 0.5 and 1.0.
- **verbose** (<em>bool</em><em>, </em><em>False</em><em> (</em><em>default</em><em>)</em>) – If True, additional information will be printed.


Returns
    
 
- <em>(tb_size, cb_size, num_cbs, cw_length, tb_crc_length, cb_crc_length, cw_lengths)</em> – Tuple:
- **tb_size** (<em>int</em>) – Transport block size, i.e., how many information bits can be encoded
into a slot for the given slot configuration.
- **cb_size** (<em>int</em>) – Code block (CB) size. Determines the number of
information bits (including TB/CB CRC parity bits) per codeword.
- **num_cbs** (<em>int</em>) – Number of code blocks. Determines into how many CBs the TB is segmented.
- **cw_lengths** (<em>list of ints</em>) – Each list element defines the codeword length of each of the `num_cbs`
codewords after LDPC encoding and rate-matching. The total number of
coded bits is $\sum$ `cw_lengths`.
- **tb_crc_length** (<em>int</em>) – Length of the TB CRC.
- **cb_crc_length** (<em>int</em>) – Length of each CB CRC.




**Note**
    
Due to rounding, `cw_lengths` (=length of each codeword after encoding),
can be slightly different within a transport block. Thus,
`cw_lengths` is given as a list of ints where each list elements denotes
the number of codeword bits of the corresponding codeword after
rate-matching.

### generate_prng_seq<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#generate-prng-seq" title="Permalink to this headline"></a>

`sionna.nr.utils.``generate_prng_seq`(<em class="sig-param">`length`</em>, <em class="sig-param">`c_init`</em>)<a class="reference internal" href="../_modules/sionna/nr/utils.html#generate_prng_seq">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.utils.generate_prng_seq" title="Permalink to this definition"></a>
    
Implements pseudo-random sequence generator as defined in Sec. 5.2.1
in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id39">[3GPP38211]</a> based on a length-31 Gold sequence.
Parameters
 
- **length** (<em>int</em>) – Desired output sequence length.
- **c_init** (<em>int</em>) – Initialization sequence of the PRNG. Must be in the range of 0 to
$2^{32}-1$.


Output
    
[`length`], ndarray of 0s and 1s – Containing the scrambling sequence.



**Note**
    
The initialization sequence `c_init` is application specific and is
usually provided be higher layer protocols.

