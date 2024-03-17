# Turbo Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#turbo-codes" title="Permalink to this headline"></a>
    
This module supports encoding and decoding of Turbo codes <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#berrou" id="id1">[Berrou]</a>, e.g., as
used in the LTE wireless standard. The convolutional component encoders and
decoders are composed of the <a class="reference internal" href="fec.conv.html#sionna.fec.conv.ConvEncoder" title="sionna.fec.conv.encoding.ConvEncoder">`ConvEncoder`</a> and
<a class="reference internal" href="fec.conv.html#sionna.fec.conv.BCJRDecoder" title="sionna.fec.conv.decoding.BCJRDecoder">`BCJRDecoder`</a> layers, respectively.
    
Please note that various notations are used in literature to represent the
generator polynomials for the underlying convolutional codes. For simplicity,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder" title="sionna.fec.turbo.encoding.TurboEncoder">`TurboEncoder`</a> only accepts the binary
format, i.e., <cite>10011</cite>, for the generator polynomial which corresponds to the
polynomial $1 + D^3 + D^4$.
    
The following code snippet shows how to set-up a rate-1/3, constraint-length-4 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboEncoder" title="sionna.fec.turbo.encoding.TurboEncoder">`TurboEncoder`</a> and the corresponding <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboDecoder" title="sionna.fec.turbo.decoding.TurboDecoder">`TurboDecoder`</a>.
You can find further examples in the <a class="reference external" href="../examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html">Channel Coding Tutorial Notebook</a>.
    
Setting-up:
```python
encoder = TurboEncoder(constraint_length=4, # Desired constraint length of the polynomials
                       rate=1/3,  # Desired rate of Turbo code
                       terminate=True) # Terminate the constituent convolutional encoders to all-zero state
# or
encoder = TurboEncoder(gen_poly=gen_poly, # Generator polynomials to use in the underlying convolutional encoders
                       rate=1/3, # Rate of the desired Turbo code
                       terminate=False) # Do not terminate the constituent convolutional encoders
# the decoder can be initialized with a reference to the encoder
decoder = TurboDecoder(encoder,
                       num_iter=6, # Number of iterations between component BCJR decoders
                       algorithm="map", # can be also "maxlog"
                       hard_out=True) # hard_decide output
```

    
Running the encoder / decoder:
```python
# --- encoder ---
# u contains the information bits to be encoded and has shape [...,k].
# c contains the turbo encoded codewords and has shape [...,n], where n=k/rate when terminate is False.
c = encoder(u)
# --- decoder ---
# llr contains the log-likelihood ratio values from the de-mapper and has shape [...,n].
# u_hat contains the estimated information bits and has shape [...,k].
u_hat = decoder(llr)
```
# Table of Content
## Turbo Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#turbo-utility-functions" title="Permalink to this headline"></a>
### TurboTermination<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#turbotermination" title="Permalink to this headline"></a>
### polynomial_selector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#polynomial-selector" title="Permalink to this headline"></a>
### puncture_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#puncture-pattern" title="Permalink to this headline"></a>
  
  

### TurboTermination<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#turbotermination" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.fec.turbo.``TurboTermination`(<em class="sig-param">`constraint_length`</em>, <em class="sig-param">`conv_n``=``2`</em>, <em class="sig-param">`num_conv_encs``=``2`</em>, <em class="sig-param">`num_bit_streams``=``3`</em>)<a class="reference internal" href="../_modules/sionna/fec/turbo/utils.html#TurboTermination">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboTermination" title="Permalink to this definition"></a>
    
Termination object, handles the transformation of termination bits from
the convolutional encoders to a Turbo codeword. Similarly, it handles the
transformation of channel symbols corresponding to the termination of a
Turbo codeword to the underlying convolutional codewords.
Parameters
 
- **constraint_length** (<em>int</em>) – Constraint length of the convolutional encoder used in the Turbo code.
Note that the memory of the encoder is `constraint_length` - 1.
- **conv_n** (<em>int</em>) – Number of output bits for one state transition in the underlying
convolutional encoder
- **num_conv_encs** (<em>int</em>) – Number of parallel convolutional encoders used in the Turbo code
- **num_bit_streams** (<em>int</em>) – Number of output bit streams from Turbo code




`get_num_term_syms`()<a class="reference internal" href="../_modules/sionna/fec/turbo/utils.html#TurboTermination.get_num_term_syms">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboTermination.get_num_term_syms" title="Permalink to this definition"></a>
    
Computes the number of termination symbols for the Turbo
code based on the underlying convolutional code parameters,
primarily the memory $\mu$.
Note that it is assumed that one Turbo symbol implies
`num_bitstreams` bits.
Input
    
**None**

Output
    
**turbo_term_syms** (<em>int</em>) – Total number of termination symbols for the Turbo Code. One
symbol equals `num_bitstreams` bits.




`term_bits_turbo2conv`(<em class="sig-param">`term_bits`</em>)<a class="reference internal" href="../_modules/sionna/fec/turbo/utils.html#TurboTermination.term_bits_turbo2conv">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboTermination.term_bits_turbo2conv" title="Permalink to this definition"></a>
    
This method splits the termination symbols from a Turbo codeword
to the termination symbols corresponding to the two convolutional
encoders, respectively.
    
Let’s assume $\mu=4$ and the underlying convolutional encoders
are systematic and rate-1/2, for demonstration purposes.
    
Let `term_bits` tensor, corresponding to the termination symbols of
the Turbo codeword be as following:
    
$y = [x_1(K), z_1(K), x_1(K+1), z_1(K+1), x_1(K+2), z_1(K+2)$,
$x_1(K+3), z_1(K+3), x_2(K), z_2(K), x_2(K+1), z_2(K+1),$
$x_2(K+2), z_2(K+2), x_2(K+3), z_2(K+3), 0, 0]$
    
The two termination tensors corresponding to the convolutional encoders
are:
$y[0,..., 2\mu]$, $y[2\mu,..., 4\mu]$. The output from this method is a tuple of two tensors, each of
size $2\mu$ and shape $[\mu,2]$.
    
$[[x_1(K), z_1(K)]$,
    
$[x_1(K+1), z_1(K+1)]$,
    
$[x_1(K+2, z_1(K+2)]$,
    
$[x_1(K+3), z_1(K+3)]]$
    
and
    
$[[x_2(K), z_2(K)],$
    
$[x_2(K+1), z_2(K+1)]$,
    
$[x_2(K+2), z_2(K+2)]$,
    
$[x_2(K+3), z_2(K+3)]]$
Input
    
**term_bits** (<em>tf.float32</em>) – Channel output of the Turbo codeword, corresponding to the
termination part

Output
    
<em>tf.float32</em> – Two tensors of channel outputs, corresponding to encoders 1 and 2,
respectively




`termbits_conv2turbo`(<em class="sig-param">`term_bits1`</em>, <em class="sig-param">`term_bits2`</em>)<a class="reference internal" href="../_modules/sionna/fec/turbo/utils.html#TurboTermination.termbits_conv2turbo">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.TurboTermination.termbits_conv2turbo" title="Permalink to this definition"></a>
    
This method merges `term_bits1` and `term_bits2`, termination
bit streams from the two convolutional encoders, to a bit stream
corresponding to the Turbo codeword.
    
Let `term_bits1` and `term_bits2` be:
    
$[x_1(K), z_1(K), x_1(K+1), z_1(K+1),..., x_1(K+\mu-1),z_1(K+\mu-1)]$
    
$[x_2(K), z_2(K), x_2(K+1), z_2(K+1),..., x_2(K+\mu-1), z_2(K+\mu-1)]$
    
where $x_i, z_i$ are the systematic and parity bit streams
respectively for a rate-1/2 convolutional encoder i, for i = 1, 2.
    
In the example output below, we assume $\mu=4$ to demonstrate zero
padding at the end. Zero padding is done such that the total length is
divisible by `num_bitstreams` (defaults to  3) which is the number of
Turbo bit streams.
    
Assume `num_bitstreams` = 3. Then number of termination symbols for
the TurboEncoder is $\lceil \frac{2*conv\_n*\mu}{3} \rceil$:
    
$[x_1(K), z_1(K), x_1(K+1)]$
    
$[z_1(K+1), x_1(K+2, z_1(K+2)]$
    
$[x_1(K+3), z_1(K+3), x_2(K)]$
    
$[z_2(K), x_2(K+1), z_2(K+1)]$
    
$[x_2(K+2), z_2(K+2), x_2(K+3)]$
    
$[z_2(K+3), 0, 0]$
    
Therefore, the output from this method is a single dimension vector
where all Turbo symbols are concatenated together.
    
$[x_1(K), z_1(K), x_1(K+1), z_1(K+1), x_1(K+2, z_1(K+2), x_1(K+3),$
    
$z_1(K+3), x_2(K),z_2(K), x_2(K+1), z_2(K+1), x_2(K+2), z_2(K+2),$
    
$x_2(K+3), z_2(K+3), 0, 0]$
Input
 
- **term_bits1** (<em>tf.int32</em>) – 2+D Tensor containing termination bits from convolutional encoder 1
- **term_bits2** (<em>tf.int32</em>) – 2+D Tensor containing termination bits from convolutional encoder 2


Output
    
<em>tf.int32</em> – 1+D tensor of termination bits. The output is obtained by
concatenating the inputs and then adding right zero-padding if
needed.




### polynomial_selector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#polynomial-selector" title="Permalink to this headline"></a>

`sionna.fec.turbo.utils.``polynomial_selector`(<em class="sig-param">`constraint_length`</em>)<a class="reference internal" href="../_modules/sionna/fec/turbo/utils.html#polynomial_selector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.utils.polynomial_selector" title="Permalink to this definition"></a>
    
Returns the generator polynomials for rate-1/2 convolutional codes
for a given `constraint_length`.
Input
    
**constraint_length** (<em>int</em>) – An integer defining the desired constraint length of the encoder.
The memory of the encoder is `constraint_length` - 1.

Output
    
**gen_poly** (<em>tuple</em>) – Tuple of strings with each string being a 0,1 sequence where
each polynomial is represented in binary form.



**Note**
    
Please note that the polynomials are optimized for rsc codes and are
not necessarily the same as used in the polynomial selector
<a class="reference internal" href="fec.conv.html#sionna.fec.conv.utils.polynomial_selector" title="sionna.fec.conv.utils.polynomial_selector">`polynomial_selector`</a> of the
convolutional codes.

### puncture_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#puncture-pattern" title="Permalink to this headline"></a>

`sionna.fec.turbo.utils.``puncture_pattern`(<em class="sig-param">`turbo_coderate`</em>, <em class="sig-param">`conv_coderate`</em>)<a class="reference internal" href="../_modules/sionna/fec/turbo/utils.html#puncture_pattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.turbo.html#sionna.fec.turbo.utils.puncture_pattern" title="Permalink to this definition"></a>
    
This method returns puncturing pattern such that the
Turbo code has rate `turbo_coderate` given the underlying
convolutional encoder is of rate `conv_coderate`.
Input
 
- **turbo_coderate** (<em>float</em>) – Desired coderate of the Turbo code
- **conv_coderate** (<em>float</em>) – Coderate of the underlying convolutional encoder


Output
    
<em>tf.bool</em> – 2D tensor indicating the positions to be punctured.




References:
Berrou(<a href="https://nvlabs.github.io/sionna/api/fec.turbo.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.turbo.html#id2">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.turbo.html#id3">3</a>,<a href="https://nvlabs.github.io/sionna/api/fec.turbo.html#id5">4</a>)
<ol class="upperalpha simple" start="3">
- Berrou, A. Glavieux, P. Thitimajshima, “Near Shannon limit error-correcting coding and decoding: Turbo-codes”, IEEE ICC, 1993.
</ol>

3GPPTS36212_Turbo(<a href="https://nvlabs.github.io/sionna/api/fec.turbo.html#id4">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.turbo.html#id6">2</a>)
    
ETSI 3GPP TS 36.212 “Evolved Universal Terrestrial
Radio Access (EUTRA); Multiplexing and channel coding”, v.15.3.0, 2018-09.



