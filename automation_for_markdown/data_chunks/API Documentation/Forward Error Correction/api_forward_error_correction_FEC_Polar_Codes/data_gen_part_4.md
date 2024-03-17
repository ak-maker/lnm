# Polar Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar-codes" title="Permalink to this headline"></a>
    
The Polar code module supports 5G-compliant Polar codes and includes successive cancellation (SC), successive cancellation list (SCL), and belief propagation (BP) decoding.
    
The module supports rate-matching and CRC-aided decoding.
Further, Reed-Muller (RM) code design is available and can be used in combination with the Polar encoding/decoding algorithms.
    
The following code snippets show how to setup and run a rate-matched 5G compliant Polar encoder and a corresponding successive cancellation list (SCL) decoder.
    
First, we need to create instances of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.encoding.Polar5GEncoder" title="sionna.fec.polar.encoding.Polar5GEncoder">`Polar5GEncoder`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.Polar5GDecoder" title="sionna.fec.polar.decoding.Polar5GDecoder">`Polar5GDecoder`</a>:
```python
encoder = Polar5GEncoder(k          = 100, # number of information bits (input)
                         n          = 200) # number of codeword bits (output)

decoder = Polar5GDecoder(encoder    = encoder, # connect the Polar decoder to the encoder
                         dec_type   = "SCL", # can be also "SC" or "BP"
                         list_size  = 8)
```

    
Now, the encoder and decoder can be used by:
```python
# --- encoder ---
# u contains the information bits to be encoded and has shape [...,k].
# c contains the polar encoded codewords and has shape [...,n].
c = encoder(u)
# --- decoder ---
# llr contains the log-likelihood ratios from the demapper and has shape [...,n].
# u_hat contains the estimated information bits and has shape [...,k].
u_hat = decoder(llr)
```
# Table of Content
## Polar Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#polar-utility-functions" title="Permalink to this headline"></a>
### generate_rm_code<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#generate-rm-code" title="Permalink to this headline"></a>
### generate_dense_polar<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#generate-dense-polar" title="Permalink to this headline"></a>
  
  

### generate_rm_code<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#generate-rm-code" title="Permalink to this headline"></a>

`sionna.fec.polar.utils.``generate_rm_code`(<em class="sig-param">`r`</em>, <em class="sig-param">`m`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/utils.html#generate_rm_code">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.utils.generate_rm_code" title="Permalink to this definition"></a>
    
Generate frozen positions of the (r, m) Reed Muller (RM) code.
Input
 
- **r** (<em>int</em>) – The order of the RM code.
- **m** (<em>int</em>) – <cite>log2</cite> of the desired codeword length.


Output
 
- **[frozen_pos, info_pos, n, k, d_min]** – List:
- **frozen_pos** (<em>ndarray</em>) – An array of ints of shape <cite>[n-k]</cite> containing the frozen
position indices.
- **info_pos** (<em>ndarray</em>) – An array of ints of shape <cite>[k]</cite> containing the information
position indices.
- **n** (<em>int</em>) – Resulting codeword length
- **k** (<em>int</em>) – Number of information bits
- **d_min** (<em>int</em>) – Minimum distance of the code.


Raises
 
- **AssertionError** – If `r` is larger than `m`.
- **AssertionError** – If `r` or `m` are not positive ints.




### generate_dense_polar<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#generate-dense-polar" title="Permalink to this headline"></a>

`sionna.fec.polar.utils.``generate_dense_polar`(<em class="sig-param">`frozen_pos`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`verbose``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/polar/utils.html#generate_dense_polar">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.utils.generate_dense_polar" title="Permalink to this definition"></a>
    
Generate <em>naive</em> (dense) Polar parity-check and generator matrix.
    
This function follows Lemma 1 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.polar.html#goala-lp" id="id34">[Goala_LP]</a> and returns a parity-check
matrix for Polar codes.

**Note**
    
The resulting matrix can be used for decoding with the
`LDPCBPDecoder` class. However, the resulting
parity-check matrix is (usually) not sparse and, thus, not suitable for
belief propagation decoding as the graph has many short cycles.
Please consider `PolarBPDecoder` for iterative
decoding over the encoding graph.

Input
 
- **frozen_pos** (<em>ndarray</em>) – Array of <cite>int</cite> defining the `n-k` indices of the frozen positions.
- **n** (<em>int</em>) – The codeword length.
- **verbose** (<em>bool</em>) – Defaults to True. If True, the code properties are printed.


Output
 
- **pcm** (ndarray of <cite>zeros</cite> and <cite>ones</cite> of shape [n-k, n]) – The parity-check matrix.
- **gm** (ndarray of <cite>zeros</cite> and <cite>ones</cite> of shape [k, n]) – The generator matrix.





References:
3GPPTS38212(<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id2">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id5">3</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id8">4</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id9">5</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id10">6</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id11">7</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id12">8</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id13">9</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id14">10</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id15">11</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id16">12</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id17">13</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id33">14</a>)
    
ETSI 3GPP TS 38.212 “5G NR Multiplexing and channel
coding”, v.16.5.0, 2021-03.

Bioglio_Design(<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id3">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id4">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id6">3</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id18">4</a>)
    
V. Bioglio, C. Condo, I. Land, “Design of
Polar Codes in 5G New Radio,” IEEE Communications Surveys &
Tutorials, 2020. Online availabe <a class="reference external" href="https://arxiv.org/pdf/1804.04389.pdf">https://arxiv.org/pdf/1804.04389.pdf</a>

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id7">Hui_ChannelCoding</a>
    
D. Hui, S. Sandberg, Y. Blankenship, M.
Andersson, L. Grosjean “Channel coding in 5G new radio: A
Tutorial Overview and Performance Comparison with 4G LTE,” IEEE
Vehicular Technology Magazine, 2018.

Arikan_Polar(<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id19">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id20">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id29">3</a>)
    
E. Arikan, “Channel polarization: A method for
constructing capacity-achieving codes for symmetric
binary-input memoryless channels,” IEEE Trans. on Information
Theory, 2009.

Gross_Fast_SCL(<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id21">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id25">2</a>)
    
Seyyed Ali Hashemi, Carlo Condo, and Warren J.
Gross, “Fast and Flexible Successive-cancellation List Decoders
for Polar Codes.” IEEE Trans. on Signal Processing, 2017.

Tal_SCL(<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id22">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id23">2</a>)
    
Ido Tal and Alexander Vardy, “List Decoding of Polar
Codes.” IEEE Trans Inf Theory, 2015.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id24">Stimming_LLR</a>
    
Alexios Balatsoukas-Stimming, Mani Bastani Parizi,
Andreas Burg, “LLR-Based Successive Cancellation List Decoding
of Polar Codes.” IEEE Trans Signal Processing, 2015.

Hashemi_SSCL(<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id26">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.polar.html#id27">2</a>)
    
Seyyed Ali Hashemi, Carlo Condo, and Warren J.
Gross, “Simplified Successive-Cancellation List Decoding
of Polar Codes.” IEEE ISIT, 2016.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id28">Cammerer_Hybrid_SCL</a>
    
Sebastian Cammerer, Benedikt Leible, Matthias
Stahl, Jakob Hoydis, and Stephan ten Brink, “Combining Belief
Propagation and Successive Cancellation List Decoding of Polar
Codes on a GPU Platform,” IEEE ICASSP, 2017.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id30">Arikan_BP</a>
    
E. Arikan, “A Performance Comparison of Polar Codes and
Reed-Muller Codes,” IEEE Commun. Lett., vol. 12, no. 6, pp.
447-449, Jun. 2008.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id31">Forney_Graphs</a>
    
G. D. Forney, “Codes on graphs: normal realizations,”
IEEE Trans. Inform. Theory, vol. 47, no. 2, pp. 520-548, Feb. 2001.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id32">Ebada_Design</a>
    
M. Ebada, S. Cammerer, A. Elkelesh and S. ten Brink,
“Deep Learning-based Polar Code Design”, Annual Allerton
Conference on Communication, Control, and Computing, 2019.

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/fec.polar.html#id34">Goala_LP</a>
    
N. Goela, S. Korada, M. Gastpar, “On LP decoding of Polar
Codes,” IEEE ITW 2010.



