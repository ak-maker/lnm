# 5G Channel Coding and Rate-Matching: Polar vs. LDPC Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html#5G-Channel-Coding-and-Rate-Matching:-Polar-vs. LDPC-Codes" title="Permalink to this headline"></a>
    
<em>“For block lengths of about 500, an IBM 7090 computer requires about 0.1 seconds per iteration to decode a block by probabilistic decoding scheme. Consequently, many hours of computation time are necessary to evaluate even a</em> $P(e)$ <em>in the order of</em> ${10^{-4}}$ <em>.”</em> Robert G. Gallager, 1963 [7]
    
In this notebook, you will learn about the different coding schemes in 5G NR and how rate-matching works (cf. 3GPP TS 38.212 [3]). The coding schemes are compared under different length/rate settings and for different decoders.
    
You will learn about the following components:
 
- 5G low-density parity-checks (LDPC) codes [7]. These codes support - without further segmentation - up to <em>k=8448</em> information bits per codeword [3] for a wide range of coderates.
- Polar codes [1] including CRC concatenation and rate-matching for 5G compliant en-/decoding is implemented for the Polar uplink control channel (UCI) [3]. Besides Polar codes, Reed-Muller (RM) codes and several decoders are available:
 
- Successive cancellation (SC) decoding [1]
- Successive cancellation list (SCL) decoding [2]
- Hybrid SC / SCL decoding for enhanced throughput
- Iterative belief propagation (BP) decoding [6]



    
Further, we will demonstrate the basic functionality of the Sionna forward error correction (FEC) module which also includes support for:
 
- Convolutional codes with non-recursive encoding and Viterbi/BCJR decoding
- Turbo codes and iterative BCJR decoding
- Ordered statistics decoding (OSD) for any binary, linear code
- Interleaving and scrambling

    
For additional technical background we refer the interested reader to [4,5,8].
    
Please note that block segmentation is not implemented as it only concatenates multiple code blocks without increasing the effective codewords length (from decoder’s perspective).
    
Some simulations in this notebook require severe simulation time, in particular if parameter sweeps are involved (e.g., different length comparisons). Please keep in mind that each cell in this notebook already contains the pre-computed outputs and no new execution is required to understand the examples.

# Table of Content
## References

## References<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html#References" title="Permalink to this headline"></a>
    
[1] E. Arikan, “Channel polarization: A method for constructing capacity-achieving codes for symmetric binary-input memoryless channels,” IEEE Transactions on Information Theory, 2009.
    
[2] Ido Tal and Alexander Vardy, “List Decoding of Polar Codes.” IEEE Transactions on Information Theory, 2015.
    
[3] ETSI 3GPP TS 38.212 “5G NR Multiplexing and channel coding”, v.16.5.0, 2021-03.
    
[4] V. Bioglio, C. Condo, I. Land, “Design of Polar Codes in 5G New Radio.” IEEE Communications Surveys & Tutorials, 2020.
    
[5] D. Hui, S. Sandberg, Y. Blankenship, M. Andersson, L. Grosjean “Channel coding in 5G new radio: A Tutorial Overview and Performance Comparison with 4G LTE.” IEEE Vehicular Technology Magazine, 2018.
    
[6] E. Arikan, “A Performance Comparison of Polar Codes and Reed-Muller Codes,” IEEE Commun. Lett., vol. 12, no. 6, pp. 447–449, Jun. 2008.
    
[7] R. G. Gallager, Low-Density Parity-Check Codes, M.I.T. Press Classic Series, Cambridge MA, 1963.
    
[8] T. Richardson and S. Kudekar. “Design of low-density parity check codes for 5G new radio,” IEEE Communications Magazine 56.3, 2018.
    
[9] G. Liva, L. Gaudio, T. Ninacs, T. Jerkovits, “Code design for short blocks: A survey,” arXiv preprint arXiv:1610.00873, 2016.
    
[10] S. Cammerer, B. Leible, M. Stahl, J. Hoydis, and S ten Brink, “Combining Belief Propagation and Successive Cancellation List Decoding of Polar Codes on a GPU Platform,” IEEE ICASSP, 2017.
    
[11] V. Bioglio, F. Gabry, I. Land, “Low-complexity puncturing and shortening of polar codes,” IEEE Wireless Communications and Networking Conference Workshops (WCNCW), 2017.
    
[12] M. Fossorier, S. Lin, “Soft-Decision Decoding of Linear Block Codes Based on Ordered Statistics”, IEEE Transactions on Information Theory, vol. 41, no. 5, 1995.