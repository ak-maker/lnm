# Bit-Interleaved Coded Modulation (BICM)<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#Bit-Interleaved-Coded-Modulation-(BICM)" title="Permalink to this headline"></a>
    
In this notebook you will learn about the principles of bit interleaved coded modulation (BICM) and focus on the interface between LDPC decoding and demapping for higher order modulation. Further, we will discuss the idea of <em>all-zero codeword</em> simulations that enable bit-error rate simulations without having an explicit LDPC encoder available. In the last part, we analyze what happens for mismatched demapping, e.g., if the SNR is unknown and show how min-sum decoding can have practical
advantages in such cases.
    
<em>“From the coding viewpoint, the modulator, waveform channel, and demodulator together constitute a discrete channel with</em> $q$ <em>input letters and</em> $q'$ <em>output letters. […] the real goal of the modulation system is to create the “best” discrete memoryless channel (DMC) as seen by the coding system.”</em> James L. Massey, 1974 [4, cf. preface in 5].
    
The fact that we usually separate modulation and coding into two individual tasks is strongly connected to the concept of bit-interleaved coded modulation (BICM) [1,2,5]. However, the joint optimization of coding and modulation has a long history, for example by Gottfried Ungerböck’s <em>Trellis coded modulation</em> (TCM) [3] and we refer the interested reader to [1,2,5,6] for these <em>principles of coded modulation</em> [5]. Nonetheless, BICM has become the <em>de facto</em> standard in virtually any modern
communication system due to its engineering simplicity.
    
In this notebook, you will use the following components:
 
- Mapper / demapper and the constellation class
- LDPC5GEncoder / LDPC5GDecoder
- AWGN channel
- BinarySource and GaussianPriorSource
- Interleaver / deinterleaver
- Scrambler / descrambler
# Table of Content
## GPU Configuration and Imports
## EXIT Charts
## Mismatched Demapping and the Advantages of Min-sum Decoding
## References
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

```python
[1]:
```

```python
import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna
# Load the required Sionna components
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder, LDPCBPDecoder
from sionna.fec.interleaving import RandomInterleaver, Deinterleaver
from sionna.fec.scrambling import Scrambler, Descrambler
from sionna.fec.utils import GaussianPriorSource, load_parity_check_examples, get_exit_analytic, plot_exit_chart, plot_trajectory
from sionna.utils import BinarySource, ebnodb2no, hard_decisions
from sionna.utils.plotting import PlotBER
from sionna.channel import AWGN
```
```python
[2]:
```

```python
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```

## EXIT Charts<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#EXIT-Charts" title="Permalink to this headline"></a>
    
You now learn about how the convergence behavior of iterative receivers can be visualized.
    
Extrinsic Information Transfer (EXIT) charts [7] are a widely adopted tool to analyze the convergence behavior of iterative receiver algorithms. The principle idea is to treat each component decoder (or demapper etc.) as individual entity with its own EXIT characteristic. EXIT charts not only allow to predict the decoding behavior (<em>open decoding tunnel</em>) but also enable LDPC code design (cf. [8]). However, this is beyond the scope of this notebook.
    
We can analytically derive the EXIT characteristic for check node (CN) and variable node (VN) decoder for a given code with `get_exit_analytic`. Further, if the `LDPCBPDecoder` is initialized with option `track_exit`=True, it internally stores the average extrinsic mutual information after each iteration at the output of the VN/CN decoder.
    
Please note that this is only an approximation for the AWGN channel and assumes infinite code length. However, it turns out that the results are often accurate enough and

```python
[18]:
```

```python
# parameters
ebno_db = 2.3
batch_size = 10000
num_bits_per_symbol = 2
pcm_id = 4 # decide which parity check matrix should be used (0-2: BCH; 3: (3,6)-LDPC 4: LDPC 802.11n
pcm, k_exit, n_exit, coderate = load_parity_check_examples(pcm_id, verbose=True)
# init components
decoder_exit = LDPCBPDecoder(pcm,
                             hard_out=False,
                             cn_type="boxplus",
                             trainable=False,
                             track_exit=True,
                             num_iter=20)
# generates fake llrs as if the all-zero codeword was transmitted over an AWNG channel with BPSK modulation (see early sections)
llr_source = GaussianPriorSource()
noise_var = ebnodb2no(ebno_db=ebno_db,
                      num_bits_per_symbol=num_bits_per_symbol,
                      coderate=coderate)
# use fake llrs from GA
llr = llr_source([[batch_size, n_exit], noise_var])
# simulate free runing trajectory
decoder_exit(llr)
# calculate analytical EXIT characteristics
# Hint: these curves assume asymptotic code length, i.e., may become inaccurate in the short length regime
Ia, Iev, Iec = get_exit_analytic(pcm, ebno_db)
# and plot the EXIT curves
plt = plot_exit_chart(Ia, Iev, Iec)
# however, as track_exit=True, the decoder logs the actual exit trajectory during decoding. This can be accessed by decoder.ie_v/decoder.ie_c after the simulation
# and add simulated trajectory to plot
plot_trajectory(plt, decoder_exit.ie_v, decoder_exit.ie_c, ebno_db)
```


```python

n: 648, k: 324, coderate: 0.500
```


    
As can be seen, the simulated trajectory of the decoder matches (relatively) well with the predicted EXIT functions of the VN and CN decoder, respectively.
    
A few things to try:
 
- Change the SNR; which curves change? Why is one curve constant? Hint: does every component directly <em>see</em> the channel?
- What happens for other codes?
- Can you predict the <em>threshold</em> of this curve (i.e., the minimum SNR required for successful decoding)
- Verify the correctness of this threshold via BER simulations (hint: the codes are relatively short, thus the prediction is less accurate)
## Mismatched Demapping and the Advantages of Min-sum Decoding<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#Mismatched-Demapping-and-the-Advantages-of-Min-sum-Decoding" title="Permalink to this headline"></a>
    
So far, we have demapped with exact knowledge of the underlying noise distribution (including the exact SNR). However, in practice estimating the SNR can be a complicated task and, as such, the estimated SNR used for demapping can be inaccurate.
    
In this part, you will learn about the advantages of min-sum decoding and we will see that it is more robust against mismatched demapping.

```python
[19]:
```

```python
# let us first remove the non-scrambled result from the previous experiment
ber_plot_allzero16qam.remove(idx=1) # remove curve with index 1
```
```python
[20]:
```

```python
# simulate with mismatched noise estimation
model_allzero_16_no = LDPC_QAM_AWGN(k,
                                    n,
                                    num_bits_per_symbol=4,
                                    use_allzero=False, # full simulation
                                    no_est_mismatch=0.15) # noise variance estimation mismatch (no scaled by 0.15 )
ber_plot_allzero16qam.simulate(model_allzero_16_no,
                               ebno_dbs=np.arange(0, 7, 0.5),
                               legend="Mismatched Demapping / 16-QAM",
                               max_mc_iter=50,
                               num_target_bit_errors=1000,
                               batch_size=1000,
                               soft_estimates=False,
                               show_fig=True,
                               forward_keyboard_interrupt=False);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 2.9167e-01 | 1.0000e+00 |      175004 |      600000 |         1000 |        1000 |         0.6 |reached target bit errors
      0.5 | 2.7983e-01 | 1.0000e+00 |      167896 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      1.0 | 2.6990e-01 | 1.0000e+00 |      161938 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      1.5 | 2.5992e-01 | 1.0000e+00 |      155954 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      2.0 | 2.4621e-01 | 1.0000e+00 |      147729 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      2.5 | 2.3175e-01 | 1.0000e+00 |      139048 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
      3.0 | 2.0946e-01 | 1.0000e+00 |      125674 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
      3.5 | 1.6710e-01 | 9.7600e-01 |      100260 |      600000 |          976 |        1000 |         0.0 |reached target bit errors
      4.0 | 7.9163e-02 | 7.3300e-01 |       47498 |      600000 |          733 |        1000 |         0.0 |reached target bit errors
      4.5 | 1.5238e-02 | 3.1200e-01 |        9143 |      600000 |          312 |        1000 |         0.1 |reached target bit errors
      5.0 | 1.2142e-03 | 1.1600e-01 |        1457 |     1200000 |          232 |        2000 |         0.1 |reached target bit errors
      5.5 | 2.6595e-04 | 8.9714e-02 |        1117 |     4200000 |          628 |        7000 |         0.4 |reached target bit errors
      6.0 | 1.9722e-04 | 7.7889e-02 |        1065 |     5400000 |          701 |        9000 |         0.4 |reached target bit errors
      6.5 | 1.6750e-04 | 7.0700e-02 |        1005 |     6000000 |          707 |       10000 |         0.5 |reached target bit errors
```

```python
[21]:
```

```python
# simulate with mismatched noise estimation
model_allzero_16_ms = LDPC_QAM_AWGN(k,
                                    n,
                                    num_bits_per_symbol=4,
                                    use_allzero=False, # full simulation
                                    decoder_type="minsum", # activate min-sum decoding
                                    no_est_mismatch=1.) # no mismatch
ber_plot_allzero16qam.simulate(model_allzero_16_ms,
                               ebno_dbs=np.arange(0, 7, 0.5),
                               legend="Min-sum decoding / 16-QAM (no mismatch)",
                               max_mc_iter=50,
                               num_target_bit_errors=1000,
                               batch_size=1000,
                               soft_estimates=False,
                               show_fig=True,
                               forward_keyboard_interrupt=False);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 2.9673e-01 | 1.0000e+00 |      178038 |      600000 |         1000 |        1000 |         1.6 |reached target bit errors
      0.5 | 2.8642e-01 | 1.0000e+00 |      171853 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
      1.0 | 2.7497e-01 | 1.0000e+00 |      164979 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
      1.5 | 2.6341e-01 | 1.0000e+00 |      158046 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
      2.0 | 2.5386e-01 | 1.0000e+00 |      152316 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
      2.5 | 2.3969e-01 | 1.0000e+00 |      143816 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
      3.0 | 2.2282e-01 | 9.9800e-01 |      133695 |      600000 |          998 |        1000 |         0.1 |reached target bit errors
      3.5 | 1.8001e-01 | 9.4100e-01 |      108007 |      600000 |          941 |        1000 |         0.1 |reached target bit errors
      4.0 | 9.4245e-02 | 6.2100e-01 |       56547 |      600000 |          621 |        1000 |         0.1 |reached target bit errors
      4.5 | 1.8808e-02 | 1.6300e-01 |       11285 |      600000 |          163 |        1000 |         0.1 |reached target bit errors
      5.0 | 7.1944e-04 | 8.0000e-03 |        1295 |     1800000 |           24 |        3000 |         0.4 |reached target bit errors
      5.5 | 5.7667e-06 | 1.0000e-04 |         173 |    30000000 |            5 |       50000 |         6.3 |reached max iter
      6.0 | 0.0000e+00 | 0.0000e+00 |           0 |    30000000 |            0 |       50000 |         6.3 |reached max iter
Simulation stopped as no error occurred @ EbNo = 6.0 dB.

```

```python
[22]:
```

```python
# simulate with mismatched noise estimation
model_allzero_16_ms = LDPC_QAM_AWGN(k,
                                    n,
                                    num_bits_per_symbol=4,
                                    use_allzero=False, # full simulation
                                    decoder_type="minsum", # activate min-sum decoding
                                    no_est_mismatch=0.15) # noise_var mismatch at demapper
ber_plot_allzero16qam.simulate(model_allzero_16_ms,
                            ebno_dbs=np.arange(0, 7, 0.5),
                            legend="Min-sum decoding / 16-QAM (with mismatch)",
                            max_mc_iter=50,
                            num_target_bit_errors=1000,
                            batch_size=1000,
                            soft_estimates=False,
                            show_fig=True,
                            forward_keyboard_interrupt=False);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 2.9721e-01 | 1.0000e+00 |      178324 |      600000 |         1000 |        1000 |         1.4 |reached target bit errors
      0.5 | 2.8528e-01 | 1.0000e+00 |      171168 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
      1.0 | 2.7617e-01 | 1.0000e+00 |      165701 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
      1.5 | 2.6409e-01 | 1.0000e+00 |      158451 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
      2.0 | 2.5094e-01 | 1.0000e+00 |      150564 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
      2.5 | 2.3911e-01 | 1.0000e+00 |      143464 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
      3.0 | 2.1863e-01 | 9.9900e-01 |      131179 |      600000 |          999 |        1000 |         0.1 |reached target bit errors
      3.5 | 1.7542e-01 | 9.6100e-01 |      105252 |      600000 |          961 |        1000 |         0.1 |reached target bit errors
      4.0 | 9.2570e-02 | 7.2900e-01 |       55542 |      600000 |          729 |        1000 |         0.1 |reached target bit errors
      4.5 | 1.9367e-02 | 2.5800e-01 |       11620 |      600000 |          258 |        1000 |         0.1 |reached target bit errors
      5.0 | 1.5808e-03 | 4.9000e-02 |        1897 |     1200000 |           98 |        2000 |         0.3 |reached target bit errors
      5.5 | 9.1930e-05 | 2.6316e-02 |        1048 |    11400000 |          500 |       19000 |         2.4 |reached target bit errors
      6.0 | 5.5215e-05 | 2.2935e-02 |        1027 |    18600000 |          711 |       31000 |         3.8 |reached target bit errors
      6.5 | 4.8619e-05 | 2.2029e-02 |        1021 |    21000000 |          771 |       35000 |         4.4 |reached target bit errors
```


    
Interestingly, <em>min-sum</em> decoding is more robust w.r.t. inaccurate LLR estimations. It is worth mentioning that <em>min-sum</em> decoding itself causes a performance loss. However, more advanced min-sum-based decoding approaches (offset-corrected min-sum) can operate close to <em>full BP</em> decoding.
    
You can also try:
 
- What happens with max-log demapping?
- Implement offset corrected min-sum decoding
- Have a closer look at the error-floor behavior
- Apply the concept of <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html">Weighted BP</a> to mismatched demapping
## References<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#References" title="Permalink to this headline"></a>
    
[1] E. Zehavi, “8-PSK Trellis Codes for a Rayleigh Channel,” IEEE Transactions on Communications, vol. 40, no. 5, 1992.
    
[2] G. Caire, G. Taricco and E. Biglieri, “Bit-interleaved Coded Modulation,” IEEE Transactions on Information Theory, vol. 44, no. 3, 1998.
    
[3] G. Ungerböck, “Channel Coding with Multilevel/Phase Signals.”IEEE Transactions on Information Theory, vol. 28, no. 1, 1982.
    
[4] J. L. Massey, “Coding and modulation in digital communications,” in Proc. Int. Zurich Seminar Commun., 1974.
    
[5] G. Böcherer, “Principles of Coded Modulation,” Habilitation thesis, Tech. Univ. Munich, Munich, Germany, 2018.
    
[6] F. Schreckenbach, “Iterative Decoding of Bit-Interleaved Coded Modulation”, PhD thesis, Tech. Univ. Munich, Munich, Germany, 2007.
    
[7] S. ten Brink, “Convergence Behavior of Iteratively Decoded Parallel Concatenated Codes,” IEEE Transactions on Communications, vol. 49, no. 10, pp. 1727-1737, 2001.
    
[8] S. ten Brink, G. Kramer, and A. Ashikhmin, “Design of low-density parity-check codes for modulation and detection,” IEEE Trans. Commun., vol. 52, no. 4, pp. 670–678, Apr. 2004.
    
[9] J. Hou, P. H. Siegel, L. B. Milstein, and H. D. Pfister, “Capacity-approaching bandwidth-efficient coded modulation schemes based on low-density parity-check codes,” IEEE Trans. Inform. Theory, vol. 49, no. 9, pp. 2141–2155, 2003.
    
[10] A. Alvarado, L. Szczecinski, R. Feick, and L. Ahumada, “Distribution of L-values in Gray-mapped M 2-QAM: Closed-form approximations and applications,” IEEE Transactions on Communications, vol. 57, no. 7, pp. 2071-2079, 2009.
    
[11] ETSI 3GPP TS 38.212 “5G NR Multiplexing and channel coding”, v.16.5.0, 2021-03.