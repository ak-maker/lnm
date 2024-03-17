# Convolutional Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.conv.html#convolutional-codes" title="Permalink to this headline"></a>
    
This module supports encoding of convolutional codes and provides layers for Viterbi <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.conv.html#viterbi" id="id1">[Viterbi]</a> and BCJR <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.conv.html#bcjr" id="id2">[BCJR]</a> decoding.
    
While the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.ViterbiDecoder" title="sionna.fec.conv.decoding.ViterbiDecoder">`ViterbiDecoder`</a> decoding algorithm produces maximum likelihood <em>sequence</em> estimates, the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.BCJRDecoder" title="sionna.fec.conv.decoding.BCJRDecoder">`BCJRDecoder`</a> produces the maximum a posterior (MAP) bit-estimates.
    
The following code snippet shows how to set up a rate-1/2, constraint-length-3 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.ConvEncoder" title="sionna.fec.conv.encoding.ConvEncoder">`ConvEncoder`</a> in two alternate ways and a corresponding <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.ViterbiDecoder" title="sionna.fec.conv.decoding.ViterbiDecoder">`ViterbiDecoder`</a> or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.BCJRDecoder" title="sionna.fec.conv.decoding.BCJRDecoder">`BCJRDecoder`</a>. You can find further examples in the <a class="reference external" href="../examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html">Channel Coding Tutorial Notebook</a>.
    
Setting-up:
```python
encoder = ConvEncoder(rate=1/2, # rate of the desired code
                      constraint_length=3) # constraint length of the code
# or
encoder = ConvEncoder(gen_poly=['101', '111']) # or polynomial can be used as input directly
# --- Viterbi decoding ---
decoder = ViterbiDecoder(gen_poly=encoder.gen_poly) # polynomial used in encoder
# or just reference to the encoder
decoder = ViterbiDecoder(encoder=encoder) # the code parameters are infered from the encoder
# --- or BCJR decoding ---
decoder = BCJRDecoder(gen_poly=encoder.gen_poly, algorithm="map") # polynomial used in encoder
# or just reference to the encoder
decoder = BCJRDecoder(encoder=encoder, algorithm="map") # the code parameters are infered from the encoder
```

    
Running the encoder / decoder:
```python
# --- encoder ---
# u contains the information bits to be encoded and has shape [...,k].
# c contains the convolutional encoded codewords and has shape [...,n].
c = encoder(u)
# --- decoder ---
# y contains the de-mapped received codeword from channel and has shape [...,n].
# u_hat contains the estimated information bits and has shape [...,k].
u_hat = decoder(y)
```
# Table of Content
## Convolutional Code Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.conv.html#convolutional-code-utility-functions" title="Permalink to this headline"></a>
### Trellis<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.conv.html#trellis" title="Permalink to this headline"></a>
### polynomial_selector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.conv.html#polynomial-selector" title="Permalink to this headline"></a>
  
  

### Trellis<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.conv.html#trellis" title="Permalink to this headline"></a>

`sionna.fec.conv.utils.``Trellis`(<em class="sig-param">`gen_poly`</em>, <em class="sig-param">`rsc``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/conv/utils.html#Trellis">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.utils.Trellis" title="Permalink to this definition"></a>
    
Trellis structure for a given generator polynomial. Defines
state transitions and output symbols (and bits) for each current
state and input.
Parameters
 
- **gen_poly** (<em>tuple</em>) – Sequence of strings with each string being a 0,1 sequence.
If <cite>None</cite>, `rate` and `constraint_length` must be provided. If
<cite>rsc</cite> is True, then first polynomial will act as denominator for
the remaining generator polynomials. For e.g., `rsc` = <cite>True</cite> and
`gen_poly` = (<cite>111</cite>, <cite>101</cite>, <cite>011</cite>) implies generator matrix equals
$G(D)=[\frac{1+D^2}{1+D+D^2}, \frac{D+D^2}{1+D+D^2}]$.
Currently Trellis is only implemented for generator matrices of
size $\frac{1}{n}$.
- **rsc** (<em>boolean</em>) – Boolean flag indicating whether the Trellis is recursive systematic
or not. If <cite>True</cite>, the encoder is recursive systematic in which
case first polynomial in `gen_poly` is used as the feedback
polynomial. Default is <cite>True</cite>.




### polynomial_selector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.conv.html#polynomial-selector" title="Permalink to this headline"></a>

`sionna.fec.conv.utils.``polynomial_selector`(<em class="sig-param">`rate`</em>, <em class="sig-param">`constraint_length`</em>)<a class="reference internal" href="../_modules/sionna/fec/conv/utils.html#polynomial_selector">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.conv.html#sionna.fec.conv.utils.polynomial_selector" title="Permalink to this definition"></a>
    
Returns generator polynomials for given code parameters. The
polynomials are chosen from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.conv.html#moon" id="id7">[Moon]</a> which are tabulated by searching
for polynomials with best free distances for a given rate and
constraint length.
Input
 
- **rate** (<em>float</em>) – Desired rate of the code.
Currently, only r=1/3 and r=1/2 are supported.
- **constraint_length** (<em>int</em>) – Desired constraint length of the encoder


Output
    
<em>tuple</em> – Tuple of strings with each string being a 0,1 sequence where
each polynomial is represented in binary form.




References:
Viterbi(<a href="https://nvlabs.github.io/sionna/api/fec.conv.html#id1">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.conv.html#id5">2</a>)
    
A. Viterbi, “Error bounds for convolutional codes and an
asymptotically optimum decoding algorithm”, IEEE Trans. Inf. Theory, 1967.

BCJR(<a href="https://nvlabs.github.io/sionna/api/fec.conv.html#id2">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.conv.html#id6">2</a>)
    
L. Bahl, J. Cocke, F. Jelinek, und J. Raviv, “Optimal Decoding
of Linear Codes for Minimizing Symbol Error Rate”, IEEE Trans. Inf.
Theory, March 1974.

Moon(<a href="https://nvlabs.github.io/sionna/api/fec.conv.html#id3">1</a>,<a href="https://nvlabs.github.io/sionna/api/fec.conv.html#id4">2</a>,<a href="https://nvlabs.github.io/sionna/api/fec.conv.html#id7">3</a>)
    
Todd. K. Moon, “Error Correction Coding: Mathematical
Methods and Algorithms”, John Wiley & Sons, 2020.



