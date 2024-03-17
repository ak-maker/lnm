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
## A Simple BICM System
  
  

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

## A Simple BICM System<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#A-Simple-BICM-System" title="Permalink to this headline"></a>
    
The principle idea of higher order modulation is to map <em>m</em> bits to one (complex-valued) symbol <em>x</em>. As each received symbol now contains information about <em>m</em> transmitted bits, the demapper produces <em>m</em> bit-wise LLR estimates (one per transmitted bit) where each LLR contains information about an individual bit. This scheme allows a simple binary interface between demapper and decoder.
    
From a decoder’s perspective, the transmission of all <em>m</em> bits - mapped onto one symbol - could be modeled as if they have been transmitted over <em>m</em> different <em>surrogate</em> channels with certain properties as shown in the figure below.
    
    
In the following, we are now interested in the LLR distribution at the decoder input (= demapper output) for each of these <em>surrogate</em> channels (denoted as <em>bit-channels</em> in the following). Please note that in some scenario these surrogate channels can share the same statistical properties, e.g., for QPSK, both bit-channels behave equally due to symmetry.
    
Advanced note: the <em>m</em> binary LLR values are treated as independent estimates which is not exactly true for higher order modulation. As a result, the sum of the <em>bitwise</em> mutual information of all <em>m</em> transmitted bits does not exactly coincide with the <em>symbol-wise</em> mutual information describing the relation between channel input / output from a symbol perspective. However, in practice the (small) losses are usually neglected if a QAM with a rectangular grid and Gray labeling is used.

### Constellations and Bit-Channels<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#Constellations-and-Bit-Channels" title="Permalink to this headline"></a>
    
Let us first look at some higher order constellations.

```python
[3]:
```

```python
# show QPSK constellation
constellation = Constellation("qam", num_bits_per_symbol=2)
constellation.show();
```


    
Assuming an AWGN channel and QPSK modulation all symbols behave equally due to the symmetry (all constellation points are located on a circle). However, for higher order modulation such as 16-QAM the situation changes and the LLRs after demapping are not equally distributed anymore.

```python
[4]:
```

```python
# generate 16QAM with Gray labeling
constellation = Constellation("qam", num_bits_per_symbol=4)
constellation.show();
```


    
We can visualize this by applying <em>a posteriori propability</em> (APP) demapping and plotting of the corresponding LLR distributions for each of the <em>m</em> transmitted bits per symbol individually. As each bit could be either <em>0</em> or <em>1</em>, we flip the signs of the LLRs <em>after</em> demapping accordingly. Otherwise, we would observe two symmetric distributions per bit <em>b_i</em> for <em>b_i=0</em> and <em>b_i=1</em>, respectively. See [10] for a closed-form approximation and further details.

```python
[5]:
```

```python
# simulation parameters
batch_size = int(1e6) # number of symbols to be analyzed
num_bits_per_symbol = 4 # bits per modulated symbol, i.e., 2^4 = 16-QAM
ebno_db = 4 # simulation SNR
# init system components
source = BinarySource() # generates random info bits
# we use a simple AWGN channel
channel = AWGN()
# calculate noise var for given Eb/No (no code used at the moment)
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate=1)
# and generate bins for the histogram
llr_bins = np.arange(-20,20,0.1)
# initialize mapper and demapper for constellation object
constellation = Constellation("qam", num_bits_per_symbol=num_bits_per_symbol)
mapper = Mapper(constellation=constellation)
# APP demapper
demapper = Demapper("app", constellation=constellation)
# Binary source that generates random 0s/1s
b = source([batch_size, num_bits_per_symbol])
# init mapper, channel and demapper
x = mapper(b)
y = channel([x, no])
llr = demapper([y, no])
# we flip the sign of all LLRs where b_i=0
# this ensures that all positive LLRs mark correct decisions
# all negative LLR values would lead to erroneous decisions
llr_b = tf.multiply(llr, (2.*b-1.))
# calculate LLR distribution for all bit-channels individually
llr_dist = []
for i in range(num_bits_per_symbol):
    llr_np = tf.reshape(llr_b[:,i],[-1]).numpy()
    t, _ = np.histogram(llr_np, bins=llr_bins, density=True);
    llr_dist.append(t)
# and plot the results
plt.figure(figsize=(20,8))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(which="both")
plt.xlabel("LLR value", fontsize=25)
plt.ylabel("Probability density", fontsize=25)
for idx, llr_hist in enumerate(llr_dist):
    leg_str = f"Demapper output for bit_channel {idx} (sign corrected)".format()
    plt.plot(llr_bins[:-1], llr_hist, label=leg_str)
plt.title("LLR distribution after demapping (16-QAM / AWGN)", fontsize=25)
plt.legend(fontsize=20);

```


    
This also shows up in the bit-wise BER without any forward-error correction (FEC).

```python
[6]:
```

```python
# calculate bitwise BERs
b_hat = hard_decisions(llr) # hard decide the LLRs
# each bit where b != b_hat is defines a decision error
# cast to tf.float32 to allow tf.reduce_mean operation
errors = tf.cast(tf.not_equal(b, b_hat), tf.float32)
# calculate ber PER bit_channel
# axis = 0 is the batch-dimension, i.e. contains individual estimates
# axis = 1 contains the m individual bit channels
ber_per_bit = tf.reduce_mean(errors, axis=0)
print("BER per bit-channel: ", ber_per_bit.numpy())
```


```python
BER per bit-channel:  [0.039274 0.039197 0.078234 0.077881]
```

    
So far, we have not applied any outer channel coding. However, from the previous histograms it is obvious that the quality of the received LLRs depends bit index within a symbol. Further, LLRs may become correlated and each symbol error may lead to multiple erroneous received bits (mapped to the same symbol). The principle idea of BICM is to <em>break</em> the local dependencies by adding an interleaver between channel coding and mapper (or demapper and decoder, respectively).
    
For sufficiently long codes (and well-suited interleavers), the channel decoder effectively <em>sees</em> one channel. This separation enables the - from engineering’s perspective - simplified and elegant design of channel coding schemes based on binary bit-metric decoding while following Massey’s original spirit that <em>the real goal of the modulation system is to create the “best” discrete memoryless channel (DMC) as seen by the coding system”</em> [1].

### Simple BER Simulations<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#Simple-BER-Simulations" title="Permalink to this headline"></a>
    
We are now interested to simulate the BER of the BICM system including LDPC codes. For this, we use the class `PlotBER` which essentially provides convenience functions for BER simulations. It internally calls `sim_ber()` to simulate each SNR point until reaching a pre-defined target number of errors.
    
**Note**: a custom BER simulation is always possible. However, without early stopping the simulations can take significantly more simulation time and `PlotBER` directly stores the results internally for later comparison.

```python
[7]:
```

```python
# generate new figure
ber_plot_allzero = PlotBER("BER Performance of All-zero Codeword Simulations")
# and define baseline
num_bits_per_symbol = 2 # QPSK
num_bp_iter = 20 # number of decoder iterations
# LDPC code parameters
k = 600 # number of information bits per codeword
n = 1200 # number of codeword bits
# and the initialize the LDPC encoder / decoder
encoder = LDPC5GEncoder(k, n)
decoder = LDPC5GDecoder(encoder, # connect encoder (for shared code parameters)
                        cn_type="boxplus-phi", # use the exact boxplus function
                        num_iter=num_bp_iter)
# initialize a random interleaver and corresponding deinterleaver
interleaver = RandomInterleaver()
deinterleaver = Deinterleaver(interleaver)
# mapper and demapper
constellation = Constellation("qam", num_bits_per_symbol=num_bits_per_symbol)
mapper = Mapper(constellation=constellation)
demapper = Demapper("app", constellation=constellation) # APP demapper
# define system
@tf.function() # we enable graph mode for faster simulations
def run_ber(batch_size, ebno_db):
    # calculate noise variance
    no = ebnodb2no(ebno_db,
                   num_bits_per_symbol=num_bits_per_symbol,
                   coderate=k/n)
    u = source([batch_size, k]) # generate random bit sequence to transmit
    c = encoder(u) # LDPC encode (incl. rate-matching and CRC concatenation)
    c_int = interleaver(c)
    x = mapper(c_int) # map to symbol (QPSK)
    y = channel([x, no]) # transmit over AWGN channel
    llr_ch = demapper([y, no]) # demapp
    llr_deint = deinterleaver(llr_ch)
    u_hat = decoder(llr_deint) # run LDPC decoder (incl. de-rate-matching)
    return u, u_hat

```

    
We simulate the BER at each SNR point in `ebno_db` for a given `batch_size` of samples. In total, per SNR point `max_mc_iter` batches are simulated.
    
To improve the simulation throughput, several optimizations are available:
<ol class="arabic simple">
- ) Continue with next SNR point if `num_target_bit_errors` is reached (or `num_target_block_errors`).
- ) Stop simulation if current SNR point returned no error (usually the BER is monotonic w.r.t. the SNR, i.e., a higher SNR point will also return BER=0)
</ol>
    
**Note**: by setting `forward_keyboard_interrupt`=False, the simulation can be interrupted at any time and returns the intermediate results.

```python
[8]:
```

```python
 # the first argument must be a callable (function) that yields u and u_hat for batch_size and ebno
ber_plot_allzero.simulate(run_ber, # the function have defined previously
                          ebno_dbs=np.arange(0, 5, 0.25), # sim SNR range
                          legend="Baseline (with encoder)",
                          max_mc_iter=50,
                          num_target_bit_errors=1000,
                          batch_size=1000,
                          soft_estimates=False,
                          early_stop=True,
                          show_fig=True,
                          forward_keyboard_interrupt=False);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6451e-01 | 1.0000e+00 |       98705 |      600000 |         1000 |        1000 |         2.8 |reached target bit errors
     0.25 | 1.3982e-01 | 9.8800e-01 |       83894 |      600000 |          988 |        1000 |         0.1 |reached target bit errors
      0.5 | 1.0626e-01 | 9.2300e-01 |       63753 |      600000 |          923 |        1000 |         0.1 |reached target bit errors
     0.75 | 6.5253e-02 | 7.5400e-01 |       39152 |      600000 |          754 |        1000 |         0.1 |reached target bit errors
      1.0 | 2.9843e-02 | 4.6000e-01 |       17906 |      600000 |          460 |        1000 |         0.1 |reached target bit errors
     1.25 | 1.0292e-02 | 2.0900e-01 |        6175 |      600000 |          209 |        1000 |         0.1 |reached target bit errors
      1.5 | 2.8617e-03 | 7.1000e-02 |        1717 |      600000 |           71 |        1000 |         0.1 |reached target bit errors
     1.75 | 6.5556e-04 | 1.5000e-02 |        1180 |     1800000 |           45 |        3000 |         0.2 |reached target bit errors
      2.0 | 7.7955e-05 | 1.9545e-03 |        1029 |    13200000 |           43 |       22000 |         1.8 |reached target bit errors
     2.25 | 5.4000e-06 | 3.6000e-04 |         162 |    30000000 |           18 |       50000 |         4.1 |reached max iter
      2.5 | 5.6667e-07 | 1.2000e-04 |          17 |    30000000 |            6 |       50000 |         4.1 |reached max iter
     2.75 | 2.0000e-07 | 2.0000e-05 |           6 |    30000000 |            1 |       50000 |         4.1 |reached max iter
      3.0 | 0.0000e+00 | 0.0000e+00 |           0 |    30000000 |            0 |       50000 |         4.1 |reached max iter
Simulation stopped as no error occurred @ EbNo = 3.0 dB.

```


