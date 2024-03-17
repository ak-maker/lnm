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
## All-zero Codeword Simulations
  
  

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

## All-zero Codeword Simulations<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#All-zero-Codeword-Simulations" title="Permalink to this headline"></a>
    
In this section you will learn about how to simulate accurate BER curves without the need for having an actual encoder in-place. We compare each step with the ground truth from the Sionna encoder:
<ol class="arabic simple">
- ) Simulate baseline with encoder as done above.
- ) Remove encoder: Simulate QPSK with all-zero codeword transmission.
- ) Gaussian approximation (for BPSK/QPSK): Remove (de-)mapping and mimic the LLR distribution for the all-zero codeword.
- ) Learn that a scrambler is required for higher order modulation schemes.
</ol>
    
An important property of linear codes is that each codewords has - in average - the same behavior. Thus, for BER simulations the all-zero codeword is sufficient.
    
**Note**: strictly speaking, this requires <em>symmetric</em> decoders in a sense that the decoder is not biased towards positive or negative LLRs (e.g., by interpreting $\ell_\text{ch}=0$ as positive value). However, in practice this can be either avoided or is often neglected.
    
Recall that we have simulated the following setup as baseline. Note: for simplicity and readability, the interleaver is omitted in the following.
    
    
Let us implement a Keras model that can be re-used and configured for the later experiments.

```python
[9]:
```

```python
class LDPC_QAM_AWGN(tf.keras.Model):
    """System model for channel coding BER simulations.
    This model allows to simulate BERs over an AWGN channel with
    QAM modulation. It can enable/disable multiple options to analyse all-zero codeword simulations.
    If active, the system uses the 5G LDPC encoder/decoder module.
    Parameters
    ----------
        k: int
            number of information bits per codeword.
        n: int
            codeword length.
        num_bits_per_symbol: int
            number of bits per QAM symbol.
        demapping_method: str
            A string defining the demapping method. Can be either "app" or "maxlog".
        decoder_type: str
            A string defining the check node update function type of the LDPC decoder.
        use_allzero: bool
            A boolean defaults to False. If True, no encoder is used and all-zero codewords are sent.
        use_scrambler: bool
            A boolean defaults to False. If True, a scrambler after the encoder and a descrambler before the decoder
            is used, respectively.
        use_ldpc_output_interleaver: bool
            A boolean defaults to False. If True, the output interleaver as
            defined in 3GPP 38.212 is applied after rate-matching.
        no_est_mismatch: float
            A float defaults to 1.0. Defines the SNR estimation mismatch of the demapper such that the effective demapping
            noise variance estimate is the scaled by ``no_est_mismatch`` version of the true noise_variance
    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.
        ebno_db: float or tf.float
            A float defining the simulation SNR.
    Output
    ------
        (u, u_hat):
            Tuple:
        u: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the transmitted information bits.
        u_hat: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the estimated information bits.
    """
    def __init__(self,
                 k,
                 n,
                 num_bits_per_symbol,
                 demapping_method="app",
                 decoder_type="boxplus",
                 use_allzero=False,
                 use_scrambler=False,
                 use_ldpc_output_interleaver=False,
                 no_est_mismatch=1.):
        super().__init__()
        self.k = k
        self.n = n
        self.num_bits_per_symbol = num_bits_per_symbol
        self.use_allzero = use_allzero
        self.use_scrambler = use_scrambler
        # adds noise to SNR estimation at demapper
        # see last section "mismatched demapping"
        self.no_est_mismatch = no_est_mismatch
        # init components
        self.source = BinarySource()
        # initialize mapper and demapper with constellation object
        self.constellation = Constellation("qam",
                                num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(demapping_method,
                                 constellation=self.constellation)
        self.channel = AWGN()
        # LDPC encoder / decoder
        if use_ldpc_output_interleaver:
            # the output interleaver needs knowledge of the modulation order
            self.encoder = LDPC5GEncoder(self.k, self.n, num_bits_per_symbol)
        else:
            self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(self.encoder, cn_type=decoder_type)
        self.scrambler = Scrambler()
        # connect descrambler to scrambler
        self.descrambler = Descrambler(self.scrambler, binary=False)
    @tf.function() # enable graph mode for higher throughputs
    def call(self, batch_size, ebno_db):
        # calculate noise variance
        no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=self.num_bits_per_symbol,
                       coderate=self.k/self.n)
        if self.use_allzero:
            u = tf.zeros([batch_size, self.k]) # only needed for
            c = tf.zeros([batch_size, self.n]) # replace enc. with all-zero codeword
        else:
            u = self.source([batch_size, self.k])
            c = self.encoder(u) # explicitly encode
        # scramble codeword if actively required
        if self.use_scrambler:
            c = self.scrambler(c)
        x = self.mapper(c) # map c to symbols
        y = self.channel([x, no]) # transmit over AWGN channel
        # add noise estimation mismatch for demapper (see last section)
        # set to 1 per default -> no mismatch
        no_est = no * self.no_est_mismatch
        llr_ch = self.demapper([y, no_est]) # demapp
        if self.use_scrambler:
            llr_ch = self.descrambler(llr_ch)
        u_hat = self.decoder(llr_ch) # run LDPC decoder (incl. de-rate-matching)
        return u, u_hat
```
### Remove Encoder: Simulate QPSK with All-zero Codeword Transmission<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#Remove-Encoder:-Simulate-QPSK-with-All-zero-Codeword-Transmission" title="Permalink to this headline"></a>
    
We now simulate the same system <em>without</em> encoder and transmit constant <em>0</em>s.
    
Due to the symmetry of the QPSK, no scrambler is required. You will learn about the effect of the scrambler in the last section.
    

```python
[10]:
```

```python
model_allzero = LDPC_QAM_AWGN(k,
                              n,
                              num_bits_per_symbol=2,
                              use_allzero=True, # disable encoder
                              use_scrambler=False) # we do not use a scrambler for the moment (QPSK!)
# and simulate the new curve
# Hint: as the model is callable, we can directly pass it to the
# Monte Carlo simulation
ber_plot_allzero.simulate(model_allzero,
                          ebno_dbs=np.arange(0, 5, 0.25),
                          legend="All-zero / QPSK (no encoder)",
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
      0.0 | 1.6225e-01 | 1.0000e+00 |       97348 |      600000 |         1000 |        1000 |         0.7 |reached target bit errors
     0.25 | 1.3823e-01 | 9.9000e-01 |       82941 |      600000 |          990 |        1000 |         0.0 |reached target bit errors
      0.5 | 1.0630e-01 | 9.3400e-01 |       63783 |      600000 |          934 |        1000 |         0.0 |reached target bit errors
     0.75 | 6.3673e-02 | 7.5400e-01 |       38204 |      600000 |          754 |        1000 |         0.0 |reached target bit errors
      1.0 | 2.8445e-02 | 4.4100e-01 |       17067 |      600000 |          441 |        1000 |         0.0 |reached target bit errors
     1.25 | 1.0038e-02 | 2.1400e-01 |        6023 |      600000 |          214 |        1000 |         0.0 |reached target bit errors
      1.5 | 3.2500e-03 | 5.7000e-02 |        1950 |      600000 |           57 |        1000 |         0.0 |reached target bit errors
     1.75 | 3.9667e-04 | 1.4600e-02 |        1190 |     3000000 |           73 |        5000 |         0.2 |reached target bit errors
      2.0 | 5.0960e-05 | 1.7273e-03 |        1009 |    19800000 |           57 |       33000 |         1.5 |reached target bit errors
     2.25 | 9.2333e-06 | 1.8000e-04 |         277 |    30000000 |            9 |       50000 |         2.3 |reached max iter
      2.5 | 2.0667e-06 | 2.0000e-05 |          62 |    30000000 |            1 |       50000 |         2.4 |reached max iter
     2.75 | 0.0000e+00 | 0.0000e+00 |           0 |    30000000 |            0 |       50000 |         2.4 |reached max iter
Simulation stopped as no error occurred @ EbNo = 2.8 dB.

```


    
As expected, the BER curves are identical (within the accuracy of the Monte Carlo simulation).

### Remove (De-)Mapping: Approximate the LLR Distribution of the All-zero Codeword (and BPSK/QPSK)<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#Remove-(De-)Mapping:-Approximate-the-LLR-Distribution-of-the-All-zero-Codeword-(and-BPSK/QPSK)" title="Permalink to this headline"></a>
    
For the all-zero codeword, the BPSK mapper generates the <em>all-one</em> signal (as each <em>0</em> is mapped to a <em>1</em>).
    
Assuming an AWGN channel with noise variance $\sigma_\text{ch}^2$, it holds that the output of the channel $y$ is Gaussian distributed with mean $\mu=1$ and noise variance $\sigma_\text{ch}^2$. Demapping of the BPSK symbols is given as $\ell_\text{ch} = -\frac{2}{\sigma_\text{ch}^2}y$.
    
This leads to the effective LLR distribution of $\ell_\text{ch} \sim \mathcal{N}(-\frac{2}{\sigma_\text{ch}^2},\frac{4}{\sigma_\text{ch}^2})$ and, thereby, allows to mimic the mapper, AWGN channel and demapper by a Gaussian distribution. The layer `GaussianPriorSource` provides such a source for arbitrary shapes.
    
The same derivation holds for QPSK. Let us quickly verify the correctness of these results by a Monte Carlo simulation.
    
**Note**: the negative sign for the BPSK demapping rule comes from the (in communications) unusual definition of logits $\ell = \operatorname{log} \frac{p(x=1)}{p(x=0)}$.
    

```python
[11]:
```

```python
num_bits_per_symbol = 2 # we use QPSK
ebno_db = 4 # choose any SNR
batch_size = 100000 # we only simulate 1 symbol per batch
# calculate noise variance
no = ebnodb2no(ebno_db,
               num_bits_per_symbol=num_bits_per_symbol,
               coderate=k/n)
# generate bins for the histogram
llr_bins = np.arange(-20, 20, 0.2)
c = tf.zeros([batch_size, num_bits_per_symbol]) # all-zero codeword
x = mapper(c) # mapped to constant symbol
y = channel([x, no])
llr = demapper([y, no]) # and generate LLRs
llr_dist, _ = np.histogram(llr.numpy() , bins=llr_bins, density=True);
# negative mean value due to different logit/llr definition
# llr = log[(x=1)/p(x=0)]
mu_llr = -2 / no
no_llr = 4 / no
# generate Gaussian pdf
llr_pred = 1/np.sqrt(2*np.pi*no_llr) * np.exp(-(llr_bins-mu_llr)**2/(2*no_llr))
# and compare the results
plt.figure(figsize=(20,8))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(which="both")
plt.xlabel("LLR value", fontsize=25)
plt.ylabel("Probability density", fontsize=25)
plt.plot(llr_bins[:-1], llr_dist, label="Measured LLR distribution")
plt.plot(llr_bins, llr_pred, label="Analytical LLR distribution (GA)")
plt.title("LLR distribution after demapping", fontsize=25)
plt.legend(fontsize=20);
```

```python
[12]:
```

```python
num_bits_per_symbol = 2 # QPSK
# initialize LLR source
ga_source = GaussianPriorSource()
@tf.function() # enable graph mode
def run_ber_ga(batch_size, ebno_db):
    # calculate noise variance
    no = ebnodb2no(ebno_db,
                   num_bits_per_symbol=num_bits_per_symbol,
                   coderate=k/n)
    u = tf.zeros([batch_size, k]) # only needed for ber calculations
    llr_ch = ga_source([[batch_size, n], no]) # generate LLRs directly
    u_hat = decoder(llr_ch) # run LDPC decoder (incl. de-rate-matching)
    return u, u_hat
# and simulate the new curve
ber_plot_allzero.simulate(run_ber_ga,
                          ebno_dbs=np.arange(0, 5, 0.25), # simulation SNR,
                          max_mc_iter=50,
                          num_target_bit_errors=1000,
                          legend="Gaussian Approximation of LLRs",
                          batch_size = 10000,
                          soft_estimates=False,
                          show_fig=True,
                          forward_keyboard_interrupt=False);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 1.6506e-01 | 9.9930e-01 |      990347 |     6000000 |         9993 |       10000 |         1.3 |reached target bit errors
     0.25 | 1.3984e-01 | 9.9030e-01 |      839044 |     6000000 |         9903 |       10000 |         0.4 |reached target bit errors
      0.5 | 1.0517e-01 | 9.3200e-01 |      631049 |     6000000 |         9320 |       10000 |         0.4 |reached target bit errors
     0.75 | 6.5437e-02 | 7.6580e-01 |      392623 |     6000000 |         7658 |       10000 |         0.4 |reached target bit errors
      1.0 | 3.1496e-02 | 4.7400e-01 |      188979 |     6000000 |         4740 |       10000 |         0.4 |reached target bit errors
     1.25 | 1.0650e-02 | 2.0850e-01 |       63902 |     6000000 |         2085 |       10000 |         0.4 |reached target bit errors
      1.5 | 2.5667e-03 | 6.3700e-02 |       15400 |     6000000 |          637 |       10000 |         0.4 |reached target bit errors
     1.75 | 3.8483e-04 | 1.1200e-02 |        2309 |     6000000 |          112 |       10000 |         0.4 |reached target bit errors
      2.0 | 4.6333e-05 | 1.9500e-03 |        1112 |    24000000 |           78 |       40000 |         1.5 |reached target bit errors
     2.25 | 6.7133e-06 | 2.5200e-04 |        1007 |   150000000 |           63 |      250000 |         9.6 |reached target bit errors
      2.5 | 1.1267e-06 | 4.4000e-05 |         338 |   300000000 |           22 |      500000 |        19.2 |reached max iter
     2.75 | 5.0000e-08 | 6.0000e-06 |          15 |   300000000 |            3 |      500000 |        19.2 |reached max iter
      3.0 | 0.0000e+00 | 0.0000e+00 |           0 |   300000000 |            0 |      500000 |        19.2 |reached max iter
Simulation stopped as no error occurred @ EbNo = 3.0 dB.

```

### The Role of the Scrambler<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#The-Role-of-the-Scrambler" title="Permalink to this headline"></a>
    
So far, we have seen that the all-zero codeword yields the same error-rates as any other sequence. Intuitively, the <em>all-zero codeword trick</em> generates a constant stream of <em>0</em>s at the input of the mapper. However, if the channel is not symmetric we need to ensure that we capture the <em>average</em> behavior of all possible symbols equally. Mathematically this symmetry condition can be expressed as $p(Y=y|c=0)=p(Y=-y|c=1)$
    
As shown in the previous experiments, for QPSK both <em>bit-channels</em> have the same behavior but for 16-QAM systems this does not hold anymore and our simulated BER does not represent the average decoding performance of the original system.
    
One possible solution is to scramble the all-zero codeword before transmission and descramble the received LLRs before decoding (i.e., flip the sign accordingly). This ensures that the mapper/demapper (+channel) operate on (pseudo-)random data, but from decoder’s perspective the the all-zero codeword assumption is still valid. This avoids the need for an actual encoder. For further details, we refer to <em>i.i.d. channel adapters</em> in [9].
    
**Note**: another example is that the recorded LLRs can be even used to evaluate different codes as the all-zero codeword is a valid codeword for all linear codes. Going one step further, one can even simulate codes of different rates with the same pre-recorded LLRs.
    

```python
[13]:
```

```python
# we generate a new plot
ber_plot_allzero16qam = PlotBER("BER Performance for 64-QAM")
```
```python
[14]:
```

```python
# simulate a new baseline for 16-QAM
model_baseline_16 = LDPC_QAM_AWGN(k,
                                  n,
                                  num_bits_per_symbol=4,
                                  use_allzero=False, # baseline without all-zero
                                  use_scrambler=False)

# and simulate the new curve
# Hint: as the model is callable, we can directly pass it to the
# Monte Carlo simulation
ber_plot_allzero16qam.simulate(model_baseline_16,
                               ebno_dbs=np.arange(0, 5, 0.25),
                               legend="Baseline 16-QAM",
                               max_mc_iter=50,
                               num_target_bit_errors=2000,
                               batch_size=1000,
                               soft_estimates=False,
                               show_fig=True,
                               forward_keyboard_interrupt=False);
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 2.4823e-01 | 1.0000e+00 |      148935 |      600000 |         1000 |        1000 |         0.6 |reached target bit errors
     0.25 | 2.4005e-01 | 1.0000e+00 |      144029 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      0.5 | 2.3212e-01 | 1.0000e+00 |      139273 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     0.75 | 2.2350e-01 | 1.0000e+00 |      134100 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      1.0 | 2.1467e-01 | 1.0000e+00 |      128802 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     1.25 | 2.0421e-01 | 1.0000e+00 |      122527 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      1.5 | 1.9363e-01 | 1.0000e+00 |      116180 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
     1.75 | 1.8410e-01 | 1.0000e+00 |      110457 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      2.0 | 1.6776e-01 | 1.0000e+00 |      100654 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     2.25 | 1.5180e-01 | 9.9800e-01 |       91079 |      600000 |          998 |        1000 |         0.0 |reached target bit errors
      2.5 | 1.2620e-01 | 9.8100e-01 |       75718 |      600000 |          981 |        1000 |         0.0 |reached target bit errors
     2.75 | 9.9148e-02 | 9.2800e-01 |       59489 |      600000 |          928 |        1000 |         0.0 |reached target bit errors
      3.0 | 6.3128e-02 | 7.5700e-01 |       37877 |      600000 |          757 |        1000 |         0.0 |reached target bit errors
     3.25 | 3.3372e-02 | 5.3100e-01 |       20023 |      600000 |          531 |        1000 |         0.0 |reached target bit errors
      3.5 | 1.3293e-02 | 2.5000e-01 |        7976 |      600000 |          250 |        1000 |         0.0 |reached target bit errors
     3.75 | 4.1550e-03 | 9.6000e-02 |        2493 |      600000 |           96 |        1000 |         0.1 |reached target bit errors
      4.0 | 8.7625e-04 | 2.6750e-02 |        2103 |     2400000 |          107 |        4000 |         0.2 |reached target bit errors
     4.25 | 2.2256e-04 | 7.4667e-03 |        2003 |     9000000 |          112 |       15000 |         0.7 |reached target bit errors
      4.5 | 2.8000e-05 | 1.3800e-03 |         840 |    30000000 |           69 |       50000 |         2.5 |reached max iter
     4.75 | 9.8333e-06 | 2.6000e-04 |         295 |    30000000 |           13 |       50000 |         2.5 |reached max iter
```


    
We now apply the <em>all-zero trick</em> as above and simulate ther BER performance without scrambling.

```python
[15]:
```

```python
# and repeat the experiment for a 16QAM WITHOUT scrambler
model_allzero_16_no_sc = LDPC_QAM_AWGN(k,
                                       n,
                                       num_bits_per_symbol=4,
                                       use_allzero=True, # all-zero codeword
                                       use_scrambler=False) # no scrambler used

# and simulate the new curve
# Hint: as the model is callable, we can directly pass it to the
# Monte Carlo simulation
ber_plot_allzero16qam.simulate(model_allzero_16_no_sc,
                               ebno_dbs=np.arange(0, 5, 0.25),
                               legend="All-zero / 16-QAM (no scrambler!)",
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
      0.0 | 3.0219e-01 | 1.0000e+00 |      181313 |      600000 |         1000 |        1000 |         0.6 |reached target bit errors
     0.25 | 2.9814e-01 | 1.0000e+00 |      178883 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      0.5 | 2.9287e-01 | 1.0000e+00 |      175723 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     0.75 | 2.8685e-01 | 1.0000e+00 |      172110 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      1.0 | 2.8033e-01 | 1.0000e+00 |      168200 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     1.25 | 2.7465e-01 | 1.0000e+00 |      164788 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      1.5 | 2.6829e-01 | 1.0000e+00 |      160973 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     1.75 | 2.6168e-01 | 1.0000e+00 |      157010 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      2.0 | 2.5387e-01 | 1.0000e+00 |      152321 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     2.25 | 2.4709e-01 | 1.0000e+00 |      148254 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      2.5 | 2.3761e-01 | 1.0000e+00 |      142565 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     2.75 | 2.2822e-01 | 1.0000e+00 |      136932 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      3.0 | 2.1826e-01 | 1.0000e+00 |      130957 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     3.25 | 2.0818e-01 | 1.0000e+00 |      124910 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      3.5 | 1.9476e-01 | 1.0000e+00 |      116858 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     3.75 | 1.8143e-01 | 1.0000e+00 |      108859 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      4.0 | 1.6426e-01 | 9.9900e-01 |       98558 |      600000 |          999 |        1000 |         0.0 |reached target bit errors
     4.25 | 1.4198e-01 | 9.9600e-01 |       85186 |      600000 |          996 |        1000 |         0.0 |reached target bit errors
      4.5 | 1.0259e-01 | 9.4800e-01 |       61554 |      600000 |          948 |        1000 |         0.0 |reached target bit errors
     4.75 | 6.0183e-02 | 7.6300e-01 |       36110 |      600000 |          763 |        1000 |         0.0 |reached target bit errors
```


    
As expected the results are wrong as we have transmitted all bits over the <em>less reliable</em> channel (cf <a class="reference external" href="https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html#Constellations-and-Bit-Channels">BER per bit-channel</a>).
    
Let us repeat this experiment with scrambler and descrambler at the correct position.

```python
[16]:
```

```python
# and repeat the experiment for a 16QAM WITHOUT scrambler
model_allzero_16_sc = LDPC_QAM_AWGN(k,
                                    n,
                                    num_bits_per_symbol=4,
                                    use_allzero=True, # all-zero codeword
                                    use_scrambler=True) # activate scrambler

# and simulate the new curve
# Hint: as the model is callable, we can directly pass it to the
# Monte Carlo simulation
ber_plot_allzero16qam.simulate(model_allzero_16_sc,
                               ebno_dbs=np.arange(0, 5, 0.25),
                               legend="All-zero / 16-QAM (with scrambler)",
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
      0.0 | 2.4735e-01 | 1.0000e+00 |      148407 |      600000 |         1000 |        1000 |         0.9 |reached target bit errors
     0.25 | 2.4123e-01 | 1.0000e+00 |      144737 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      0.5 | 2.3287e-01 | 1.0000e+00 |      139721 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     0.75 | 2.2510e-01 | 1.0000e+00 |      135062 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      1.0 | 2.1514e-01 | 1.0000e+00 |      129083 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     1.25 | 2.0605e-01 | 1.0000e+00 |      123628 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      1.5 | 1.9327e-01 | 1.0000e+00 |      115961 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     1.75 | 1.8327e-01 | 1.0000e+00 |      109961 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      2.0 | 1.6901e-01 | 1.0000e+00 |      101406 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     2.25 | 1.5102e-01 | 9.9900e-01 |       90611 |      600000 |          999 |        1000 |         0.0 |reached target bit errors
      2.5 | 1.2854e-01 | 9.7500e-01 |       77123 |      600000 |          975 |        1000 |         0.0 |reached target bit errors
     2.75 | 9.5707e-02 | 9.1200e-01 |       57424 |      600000 |          912 |        1000 |         0.0 |reached target bit errors
      3.0 | 6.2627e-02 | 7.5700e-01 |       37576 |      600000 |          757 |        1000 |         0.0 |reached target bit errors
     3.25 | 3.5712e-02 | 5.2800e-01 |       21427 |      600000 |          528 |        1000 |         0.0 |reached target bit errors
      3.5 | 1.4118e-02 | 2.6100e-01 |        8471 |      600000 |          261 |        1000 |         0.0 |reached target bit errors
     3.75 | 4.9117e-03 | 1.0900e-01 |        2947 |      600000 |          109 |        1000 |         0.0 |reached target bit errors
      4.0 | 9.9583e-04 | 2.9000e-02 |        1195 |     1200000 |           58 |        2000 |         0.1 |reached target bit errors
     4.25 | 2.8905e-04 | 8.5714e-03 |        1214 |     4200000 |           60 |        7000 |         0.3 |reached target bit errors
      4.5 | 3.4320e-05 | 1.2857e-03 |        1009 |    29400000 |           63 |       49000 |         2.3 |reached target bit errors
     4.75 | 4.1000e-06 | 2.6000e-04 |         123 |    30000000 |           13 |       50000 |         2.4 |reached max iter
```


    
The 5G standard defines an additional output interleaver after the rate-matching (see Sec. 5.4.2.2 in [11]).
    
We now activate this additional interleaver to enable additional BER gains.

```python
[17]:
```

```python
# activate output interleaver
model_output_interleaver= LDPC_QAM_AWGN(k,
                                        n,
                                        num_bits_per_symbol=4,
                                        use_ldpc_output_interleaver=True,
                                        use_allzero=False,
                                        use_scrambler=False)

# and simulate the new curve
# Hint: as the model is callable, we can directly pass it to the
# Monte Carlo simulation
ber_plot_allzero16qam.simulate(model_output_interleaver,
                               ebno_dbs=np.arange(0, 5, 0.25),
                               legend="16-QAM with 5G FEC interleaver",
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
      0.0 | 1.9663e-01 | 1.0000e+00 |      117980 |      600000 |         1000 |        1000 |         0.7 |reached target bit errors
     0.25 | 1.9138e-01 | 1.0000e+00 |      114830 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      0.5 | 1.8279e-01 | 1.0000e+00 |      109673 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     0.75 | 1.7542e-01 | 1.0000e+00 |      105255 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
      1.0 | 1.6715e-01 | 1.0000e+00 |      100289 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     1.25 | 1.5677e-01 | 1.0000e+00 |       94061 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
      1.5 | 1.4749e-01 | 1.0000e+00 |       88497 |      600000 |         1000 |        1000 |         0.0 |reached target bit errors
     1.75 | 1.3550e-01 | 1.0000e+00 |       81298 |      600000 |         1000 |        1000 |         0.1 |reached target bit errors
      2.0 | 1.1884e-01 | 9.9900e-01 |       71302 |      600000 |          999 |        1000 |         0.0 |reached target bit errors
     2.25 | 1.0189e-01 | 9.8200e-01 |       61135 |      600000 |          982 |        1000 |         0.1 |reached target bit errors
      2.5 | 7.6883e-02 | 9.5000e-01 |       46130 |      600000 |          950 |        1000 |         0.0 |reached target bit errors
     2.75 | 5.0982e-02 | 8.0000e-01 |       30589 |      600000 |          800 |        1000 |         0.0 |reached target bit errors
      3.0 | 2.9085e-02 | 5.7600e-01 |       17451 |      600000 |          576 |        1000 |         0.0 |reached target bit errors
     3.25 | 1.0312e-02 | 2.9500e-01 |        6187 |      600000 |          295 |        1000 |         0.0 |reached target bit errors
      3.5 | 3.6233e-03 | 1.3000e-01 |        2174 |      600000 |          130 |        1000 |         0.1 |reached target bit errors
     3.75 | 1.0858e-03 | 4.7500e-02 |        1303 |     1200000 |           95 |        2000 |         0.1 |reached target bit errors
      4.0 | 2.3857e-04 | 1.4286e-02 |        1002 |     4200000 |          100 |        7000 |         0.4 |reached target bit errors
     4.25 | 3.6146e-05 | 2.4792e-03 |        1041 |    28800000 |          119 |       48000 |         2.4 |reached target bit errors
      4.5 | 1.2333e-05 | 9.6000e-04 |         370 |    30000000 |           48 |       50000 |         2.5 |reached max iter
     4.75 | 1.9333e-06 | 1.8000e-04 |          58 |    30000000 |            9 |       50000 |         2.5 |reached max iter
```


