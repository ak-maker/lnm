# Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#utility-functions" title="Permalink to this headline"></a>
    
This module provides utility functions for the FEC package. It also provides serval functions to simplify EXIT analysis of iterative receivers.

# Table of Content
## (Binary) Linear Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#binary-linear-codes" title="Permalink to this headline"></a>
### load_parity_check_examples<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#load-parity-check-examples" title="Permalink to this headline"></a>
### alist2mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#alist2mat" title="Permalink to this headline"></a>
### load_alist<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#load-alist" title="Permalink to this headline"></a>
### generate_reg_ldpc<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#generate-reg-ldpc" title="Permalink to this headline"></a>
### make_systematic<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#make-systematic" title="Permalink to this headline"></a>
### gm2pcm<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#gm2pcm" title="Permalink to this headline"></a>
  
  

## (Binary) Linear Codes<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#binary-linear-codes" title="Permalink to this headline"></a>
    
Several functions are provided to convert parity-check matrices into generator matrices and vice versa. Please note that currently only binary codes are supported.
```python
# load example parity-check matrix
pcm, k, n, coderate = load_parity_check_examples(pcm_id=3)
```

    
Note that many research projects provide their parity-check matrices in the  <cite>alist</cite> format <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#mackay" id="id1">[MacKay]</a> (e.g., see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#unikl" id="id2">[UniKL]</a>). The follwing code snippet provides an example of how to import an external LDPC parity-check matrix from an <cite>alist</cite> file and how to set-up an encoder/decoder.
```python
# load external example parity-check matrix in alist format
al = load_alist(path=filename)
pcm, k, n, coderate = alist2mat(al)
# the linear encoder can be directly initialized with a parity-check matrix
encoder = LinearEncoder(pcm, is_pcm=True)
# initalize BP decoder for the given parity-check matrix
decoder = LDPCBPDecoder(pcm, num_iter=20)
# and run simulation with random information bits
no = 1.
batch_size = 10
num_bits_per_symbol = 2
source = BinarySource()
mapper = Mapper("qam", num_bits_per_symbol)
channel = AWGN()
demapper = Demapper("app", "qam", num_bits_per_symbol)
u = source([batch_size, k])
c = encoder(u)
x = mapper(c)
y = channel([x, no])
llr = demapper([y, no])
c_hat = decoder(llr)
```
### load_parity_check_examples<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#load-parity-check-examples" title="Permalink to this headline"></a>

`sionna.fec.utils.``load_parity_check_examples`(<em class="sig-param">`pcm_id`</em>, <em class="sig-param">`verbose``=``False`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#load_parity_check_examples">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.load_parity_check_examples" title="Permalink to this definition"></a>
    
Utility function to load example codes stored in sub-folder LDPC/codes.
    
The following codes are available
 
- 0 : <cite>(7,4)</cite>-Hamming code of length <cite>k=4</cite> information bits and codeword    length <cite>n=7</cite>.
- 1 : <cite>(63,45)</cite>-BCH code of length <cite>k=45</cite> information bits and codeword    length <cite>n=63</cite>.
- 2 : (127,106)-BCH code of length <cite>k=106</cite> information bits and codeword    length <cite>n=127</cite>.
- 3 : Random LDPC code with regular variable node degree 3 and check node degree 6 of length <cite>k=50</cite> information bits and codeword length         <cite>n=100</cite>.
- 4 : 802.11n LDPC code of length of length <cite>k=324</cite> information bits and    codeword length <cite>n=648</cite>.

Input
 
- **pcm_id** (<em>int</em>) – An integer defining which matrix id to load.
- **verbose** (<em>bool</em>) – Defaults to False. If True, the code parameters are
printed.


Output
 
- **pcm** (ndarray of <cite>zeros</cite> and <cite>ones</cite>) – An ndarray containing the parity check matrix.
- **k** (<em>int</em>) – An integer defining the number of information bits.
- **n** (<em>int</em>) – An integer defining the number of codeword bits.
- **coderate** (<em>float</em>) – A float defining the coderate (assuming full rank of
parity-check matrix).




### alist2mat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#alist2mat" title="Permalink to this headline"></a>

`sionna.fec.utils.``alist2mat`(<em class="sig-param">`alist`</em>, <em class="sig-param">`verbose``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#alist2mat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.alist2mat" title="Permalink to this definition"></a>
    
Convert <cite>alist</cite> <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#mackay" id="id3">[MacKay]</a> code definition to <cite>full</cite> parity-check matrix.
    
Many code examples can be found in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#unikl" id="id4">[UniKL]</a>.
    
About <cite>alist</cite> (see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#mackay" id="id5">[MacKay]</a> for details):
<blockquote>
<div> 
- <cite>1.</cite> Row defines parity-check matrix dimension <cite>m x n</cite>
- <cite>2.</cite> Row defines int with <cite>max_CN_degree</cite>, <cite>max_VN_degree</cite>
- <cite>3.</cite> Row defines VN degree of all <cite>n</cite> column
- <cite>4.</cite> Row defines CN degree of all <cite>m</cite> rows
- Next <cite>n</cite> rows contain non-zero entries of each column (can be zero padded at the end)
- Next <cite>m</cite> rows contain non-zero entries of each row.

</blockquote>
Input
 
- **alist** (<em>list</em>) – Nested list in <cite>alist</cite>-format <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#mackay" id="id6">[MacKay]</a>.
- **verbose** (<em>bool</em>) – Defaults to True. If True, the code parameters are printed.


Output
 
- **(pcm, k, n, coderate)** – Tuple:
- **pcm** (<em>ndarray</em>) – NumPy array of shape <cite>[n-k, n]</cite> containing the parity-check matrix.
- **k** (<em>int</em>) – Number of information bits.
- **n** (<em>int</em>) – Number of codewords bits.
- **coderate** (<em>float</em>) – Coderate of the code.




**Note**
    
Use <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.load_alist" title="sionna.fec.utils.load_alist">`load_alist`</a> to import alist from a
textfile.
    
For example, the following code snippet will import an alist from a file called `filename`:
```python
al = load_alist(path = filename)
pcm, k, n, coderate = alist2mat(al)
```


### load_alist<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#load-alist" title="Permalink to this headline"></a>

`sionna.fec.utils.``load_alist`(<em class="sig-param">`path`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#load_alist">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.load_alist" title="Permalink to this definition"></a>
    
Read <cite>alist</cite>-file <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#mackay" id="id7">[MacKay]</a> and return nested list describing the
parity-check matrix of a code.
    
Many code examples can be found in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#unikl" id="id8">[UniKL]</a>.
Input
    
**path** (<em>str</em>) – Path to file to be loaded.

Output
    
**alist** (<em>list</em>) – A nested list containing the imported alist data.



### generate_reg_ldpc<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#generate-reg-ldpc" title="Permalink to this headline"></a>

`sionna.fec.utils.``generate_reg_ldpc`(<em class="sig-param">`v`</em>, <em class="sig-param">`c`</em>, <em class="sig-param">`n`</em>, <em class="sig-param">`allow_flex_len``=``True`</em>, <em class="sig-param">`verbose``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#generate_reg_ldpc">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.generate_reg_ldpc" title="Permalink to this definition"></a>
    
Generate random regular (v,c) LDPC codes.
    
This functions generates a random LDPC parity-check matrix of length `n`
where each variable node (VN) has degree `v` and each check node (CN) has
degree `c`. Please note that the LDPC code is not optimized to avoid
short cycles and the resulting codes may show a non-negligible error-floor.
For encoding, the `LinearEncoder` layer can be
used, however, the construction does not guarantee that the pcm has full
rank.
Input
 
- **v** (<em>int</em>) – Desired variable node (VN) degree.
- **c** (<em>int</em>) – Desired check node (CN) degree.
- **n** (<em>int</em>) – Desired codeword length.
- **allow_flex_len** (<em>bool</em>) – Defaults to True. If True, the resulting codeword length can be
(slightly) increased.
- **verbose** (<em>bool</em>) – Defaults to True. If True, code parameters are printed.


Output
 
- **(pcm, k, n, coderate)** – Tuple:
- **pcm** (<em>ndarray</em>) – NumPy array of shape <cite>[n-k, n]</cite> containing the parity-check matrix.
- **k** (<em>int</em>) – Number of information bits per codeword.
- **n** (<em>int</em>) – Number of codewords bits.
- **coderate** (<em>float</em>) – Coderate of the code.




**Note**
    
This algorithm works only for regular node degrees. For state-of-the-art
bit-error-rate performance, usually one needs to optimize irregular degree
profiles (see <a class="reference internal" href="https://nvlabs.github.io/sionna/api/fec.utils.html#tenbrink" id="id9">[tenBrink]</a>).

### make_systematic<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#make-systematic" title="Permalink to this headline"></a>

`sionna.fec.utils.``make_systematic`(<em class="sig-param">`mat`</em>, <em class="sig-param">`is_pcm``=``False`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#make_systematic">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.make_systematic" title="Permalink to this definition"></a>
    
Bring binary matrix in its systematic form.
Input
 
- **mat** (<em>ndarray</em>) – Binary matrix to be transformed to systematic form of shape <cite>[k, n]</cite>.
- **is_pcm** (<em>bool</em>) – Defaults to False. If true, `mat` is interpreted as parity-check
matrix and, thus, the last k columns will be the identity part.


Output
 
- **mat_sys** (<em>ndarray</em>) – Binary matrix in systematic form, i.e., the first <cite>k</cite> columns equal the
identity matrix (or last <cite>k</cite> if `is_pcm` is True).
- **column_swaps** (<em>list of int tuples</em>) – A list of integer tuples that describes the swapped columns (in the
order of execution).




**Note**
    
This algorithm (potentially) swaps columns of the input matrix. Thus, the
resulting systematic matrix (potentially) relates to a permuted version of
the code, this is defined by the returned list `column_swap`.
Note that, the inverse permutation must be applied in the inverse list
order (in case specific columns are swapped multiple times).
    
If a parity-check matrix is passed as input (i.e., `is_pcm` is True), the
identity part will be re-arranged to the last columns.

### gm2pcm<a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#gm2pcm" title="Permalink to this headline"></a>

`sionna.fec.utils.``gm2pcm`(<em class="sig-param">`gm`</em>, <em class="sig-param">`verify_results``=``True`</em>)<a class="reference internal" href="../_modules/sionna/fec/utils.html#gm2pcm">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/fec.utils.html#sionna.fec.utils.gm2pcm" title="Permalink to this definition"></a>
    
Generate the parity-check matrix for a given generator matrix.
    
This function brings `gm` $\mathbf{G}$ in its systematic form and
uses the following relation to find the parity-check matrix
$\mathbf{H}$ in GF(2)

$$
\mathbf{G} = [\mathbf{I} |  \mathbf{M}]
\Leftrightarrow \mathbf{H} = [\mathbf{M} ^t | \mathbf{I}]. \tag{1}
$$
    
This follows from the fact that for an all-zero syndrome, it must hold that

$$
\mathbf{H} \mathbf{c}^t = \mathbf{H} * (\mathbf{u} * \mathbf{G})^t =
\mathbf{H} * \mathbf{G} ^t * \mathbf{u}^t =: \mathbf{0}
$$
    
where $\mathbf{c}$ denotes an arbitrary codeword and
$\mathbf{u}$ the corresponding information bits.
    
This leads to

$$
\mathbf{G} * \mathbf{H} ^t =: \mathbf{0}. \tag{2}
$$
    
It can be seen that (1) fulfills (2), as it holds in GF(2) that

$$
[\mathbf{I} |  \mathbf{M}] * [\mathbf{M} ^t | \mathbf{I}]^t
 = \mathbf{M} + \mathbf{M} = \mathbf{0}.
$$

Input
 
- **gm** (<em>ndarray</em>) – Binary generator matrix of shape <cite>[k, n]</cite>.
- **verify_results** (<em>bool</em>) – Defaults to True. If True, it is verified that the generated
parity-check matrix is orthogonal to the generator matrix in GF(2).


Output
    
<em>ndarray</em> – Binary parity-check matrix of shape <cite>[n-k, n]</cite>.



**Note**
    
This algorithm only works if `gm` has full rank. Otherwise an error is
raised.

