
# Forward Error Correction (FEC)<a class="headerlink" href="https://nvlabs.github.io/sionna/#forward-error-correction-fec" title="Permalink to this headline"></a>

The forward error correction (FEC) package provides encoding and decoding
algorithms for several coding schemes such as low-density parity-check (LDPC),
Polar, Turbo, and convolutional codes as well as cyclic redundancy checks (CRC).

Although LDPC and Polar codes are 5G compliant, the decoding
algorithms are mostly general and can be used in combination with other
code designs.

Besides the encoding/decoding algorithms, this package also provides
interleavers, scramblers, and rate-matching for seamless integration of the FEC
package into the remaining physical layer processing chain.

The following figure shows the evolution of FEC codes from GSM (2G) up to the
5G NR wireless communication standard. The different codes are simulated with
the Sionna FEC package for two different codeword length of $n=1024$
(coderate $r=1/2$) and $n=6156$ (coderate $r=1/3$),
respectively.

<em>Remark</em>: The performance of different coding scheme varies significantly with
the choice of the exact code and decoding parameters which can be
found in the notebook <a class="reference external" href="https://nvlabs.github.io/sionna/../examples/Evolution_of_FEC.html">From GSM to 5G - The Evolution of Forward Error Correction</a>. Further, the situation also changes for short length codes and results can be found in <a class="reference external" href="https://nvlabs.github.io/sionna/../examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html">5G Channel Coding: Polar vs. LDPC Codes</a>.
<img alt="../_images/FEC_evolution.png" src="https://nvlabs.github.io/sionna/_images/FEC_evolution.png" />

Please note that the <em>best</em> choice of a coding scheme for a specific application
depends on many other criteria than just its error rate performance:

- 
Decoding complexity, latency, and scalability
- 
Level of parallelism of the decoding algorithm and memory access patterns
- 
Error-floor behavior
- 
Rate adaptivity and flexibility
All this–and much more–can be explored within the Sionna FEC module.

**Table of Contents**

- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.linear.html">Linear Codes</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.linear.html#encoder">Encoder</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.linear.html#linearencoder">LinearEncoder</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.linear.html#allzeroencoder">AllZeroEncoder</a>


- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.linear.html#decoder">Decoder</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.linear.html#osdecoder">OSDecoder</a>




- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.ldpc.html">Low-Density Parity-Check (LDPC)</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.ldpc.html#ldpc-encoder">LDPC Encoder</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.ldpc.html#ldpc5gencoder">LDPC5GEncoder</a>


- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.ldpc.html#ldpc-decoder">LDPC Decoder</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.ldpc.html#ldpcbpdecoder">LDPCBPDecoder</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.ldpc.html#ldpc5gdecoder">LDPC5GDecoder</a>




- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.polar.html">Polar Codes</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.polar.html#polar-encoding">Polar Encoding</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.polar.html#polar5gencoder">Polar5GEncoder</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.polar.html#polarencoder">PolarEncoder</a>


- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.polar.html#polar-decoding">Polar Decoding</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.polar.html#polar5gdecoder">Polar5GDecoder</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.polar.html#polarscdecoder">PolarSCDecoder</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.polar.html#polarscldecoder">PolarSCLDecoder</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.polar.html#polarbpdecoder">PolarBPDecoder</a>


- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.polar.html#polar-utility-functions">Polar Utility Functions</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.polar.html#generate-5g-ranking">generate_5g_ranking</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.polar.html#generate-polar-transform-mat">generate_polar_transform_mat</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.polar.html#generate-rm-code">generate_rm_code</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.polar.html#generate-dense-polar">generate_dense_polar</a>




- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.conv.html">Convolutional Codes</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.conv.html#convolutional-encoding">Convolutional Encoding</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.conv.html#viterbi-decoding">Viterbi Decoding</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.conv.html#bcjr-decoding">BCJR Decoding</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.conv.html#convolutional-code-utility-functions">Convolutional Code Utility Functions</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.conv.html#trellis">Trellis</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.conv.html#polynomial-selector">polynomial_selector</a>




- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.turbo.html">Turbo Codes</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.turbo.html#turbo-encoding">Turbo Encoding</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.turbo.html#turbo-decoding">Turbo Decoding</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.turbo.html#turbo-utility-functions">Turbo Utility Functions</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.turbo.html#turbotermination">TurboTermination</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.turbo.html#polynomial-selector">polynomial_selector</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.turbo.html#puncture-pattern">puncture_pattern</a>




- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.crc.html">Cyclic Redundancy Check (CRC)</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.crc.html#crcencoder">CRCEncoder</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.crc.html#crcdecoder">CRCDecoder</a>


- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.interleaving.html">Interleaving</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.interleaving.html#interleaver">Interleaver</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.interleaving.html#rowcolumninterleaver">RowColumnInterleaver</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.interleaving.html#randominterleaver">RandomInterleaver</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.interleaving.html#turbo3gppinterleaver">Turbo3GPPInterleaver</a>


- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.interleaving.html#deinterleaver">Deinterleaver</a>


- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.scrambling.html">Scrambling</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.scrambling.html#scrambler">Scrambler</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.scrambling.html#tb5gscrambler">TB5GScrambler</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.scrambling.html#descrambler">Descrambler</a>


- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html">Utility Functions</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#binary-linear-codes">(Binary) Linear Codes</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#load-parity-check-examples">load_parity_check_examples</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#alist2mat">alist2mat</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#load-alist">load_alist</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#generate-reg-ldpc">generate_reg_ldpc</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#make-systematic">make_systematic</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#gm2pcm">gm2pcm</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#pcm2gm">pcm2gm</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#verify-gm-pcm">verify_gm_pcm</a>


- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#exit-analysis">EXIT Analysis</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#plot-exit-chart">plot_exit_chart</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#get-exit-analytic">get_exit_analytic</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#plot-trajectory">plot_trajectory</a>


- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#miscellaneous">Miscellaneous</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#gaussianpriorsource">GaussianPriorSource</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#bin2int">bin2int</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#int2bin">int2bin</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#bin2int-tf">bin2int_tf</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#int2bin-tf">int2bin_tf</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#int-mod-2">int_mod_2</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#llr2mi">llr2mi</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#j-fun">j_fun</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#j-fun-inv">j_fun_inv</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#j-fun-tf">j_fun_tf</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/fec.utils.html#j-fun-inv-tf">j_fun_inv_tf</a>



