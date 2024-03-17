INSTRUCTION: Explain the difference between Polar and LDPC codes as described in the Sionna package documentation and which one is better for longer codeword lengths.
ANSWER:Polar codes and LDPC (Low-Density Parity-Check) codes are two types of error-correcting codes that are used in 5G NR (New Radio) to ensure that data is transmitted reliably over communication channels prone to errors. The Sionna package, which you are referencing, provides an implementation of these codes for simulation purposes.

**Difference between Polar and LDPC codes based on the Sionna package documentation:**

1. **Polar Codes:**
   - Polar codes are based on the concept of channel polarization, which involves combining two identical but independent copies of a channel into a synthesized pair of channels, one better and one worse than the original.
   - In the 5G context, Polar codes are mainly used for uplink control information.
   - In the Sionna package, Polar codes have associated decoding methods, such as Successive Cancellation (SC), Successive Cancellation List (SCL), and Hybrid SC/SCL.

2. **LDPC Codes:**
   - LDPC codes are linear error-correcting codes that are constructed using a sparse bipartite graph.
   - LDPC codes tend to have good performance near the Shannon limit and are well-suited for hardware implementation due to their iterative decoding algorithms and parallelizable structure.
   - The Sionna package supports 5G LDPC codes that can handle up to k=8448 information bits per codeword for a wide range of coderates.
   - Decoding of LDPC codes is typically performed using the Belief Propagation (BP) algorithm.

**Comparison for Longer Codeword Lengths:**

- **LDPC Codes for Long Codewords:** LDPC codes are generally preferred for longer codeword lengths. They have been shown to perform well with long block lengths, and they are part of the 5G NR standard for data channels.
  
- **Polar Codes for Short to Moderate Codewords:** Polar codes are often used for shorter to moderate codeword lengths, particularly for control information in 5G NR. They are known for having theoretical guarantees of achieving capacity on symmetric binary-input discrete memoryless channels as the block length goes to infinity.

**Which One is Better for Longer Codeword Lengths?**

According to the Sionna documentation, LDPC codes support up to k=8448 information bits per codeword, suggesting that they are designed to perform efficiently at longer codeword lengths. This, combined with their iterative decoding methods, which are suited for longer codes, makes LDPC codes the preferred choice for longer codeword lengths in the context of the Sionna package and in practical implementations of 5G standards.

It is important to note that the "better" choice can depend on various factors, including the specific application, decoding complexity, and performance trade-offs. However, based on the information provided in the context, LDPC codes are better suited for longer codeword lengths in the 5G NR framework.

INSTRUCTION: Write a Python function using Sionna to create a 5G NR compatible system model for BER simulations, utilizing the System_Model class defined in the notebook.
ANSWER:To create a 5G NR compatible system model for BER simulations using Sionna, we first need to ensure Sionna is installed and all the required components are imported. From the given context, it looks like the `System_Model` class has already been defined in the provided code, which will be leveraged to create instances for BER simulations.

Below is a Python function that does this utilizing the `System_Model` class. The function initializes the model with a specified encoder and decoder for a given modulation and code length. This function can be adjusted to simulate other channel conditions, modulation schemes, or encoder/decoder pairs.

```python
import tensorflow as tf
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.utils import BinarySource, ebnodb2no
from sionna.channel import AWGN

# Assuming the System_Model class is already defined as per the context provided

def create_5g_nr_ber_simulation_model(k, n, num_bits_per_symbol, encoder, decoder, demapping_method="app", sim_esno=False):
    """
    Create a 5G NR compatible system model for BER simulations using Sionna's System_Model class.

    Parameters:
    k (int): Number of information bits per codeword
    n (int): Codeword length
    num_bits_per_symbol (int): Number of bits per QAM symbol
    encoder (tf.keras.layers.Layer): Encoder layer from Sionna
    decoder (tf.keras.layers.Layer): Decoder layer from Sionna
    demapping_method (str): Demapping method, either 'app' or 'maxlog'
    sim_esno (bool): If True, simulate ES/NO instead of EB/NO

    Returns:
    System_Model: An instance of the System_Model class for BER simulations
    """

    # Create the 5G NR compatible system model
    model = System_Model(k=k,
                         n=n,
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=encoder,
                         decoder=decoder,
                         demapping_method=demapping_method,
                         sim_esno=sim_esno)
    return model

# Example usage:
k = 64  # Number of information bits
n = 128  # Codeword length
num_bits_per_symbol = 2  # Bits per symbol for QPSK

# Create encoder and decoder instances for a given code
encoder_instance = LDPC5GEncoder(k=k, n=n)
decoder_instance = LDPC5GDecoder(encoder_instance, num_iter=20)

# Create the model for BER simulations
model_for_ber_simulation = create_5g_nr_ber_simulation_model(k, n, num_bits_per_symbol, encoder_instance, decoder_instance)

# Now, you can use 'model_for_ber_simulation' to run BER simulations
```

This function creates a `System_Model` instance that can be used for Bit Error Rate (BER) simulations. To actually perform a simulation, you would pass this model along with other parameters (such as a range of SNR values, batch sizes, etc.) to a simulation function that could look like `ber_plot.simulate()` in the given context. Since we do not have the entire notebook's context, we assumed from the previous cells that the method `ber_plot.simulate()` would be available and provided the arguments as per the Sionna examples referenced in the context.

INSTRUCTION: Use the Sionna package in Python to simulate the BER performance of various 5G coding schemes such as LDPC and Polar codes and compare the results.
ANSWER:To simulate the Bit Error Rate (BER) performance of various 5G coding schemes such as LDPC and Polar codes using the Sionna package in Python and compare the results, you can follow these instructions:

1. Install and Import Necessary Libraries:
Make sure that the `sionna` package is installed. If it's not installed, install it using pip:

```python
!pip install sionna
```

Then, import the necessary modules from Sionna:

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.fec.polar import Polar5GEncoder, Polar5GDecoder
from sionna.utils import ebnodb2no, BinarySource
from sionna.channel import AWGN
```

2. Define Simulation Parameters:
Decide on the parameters for your simulation, such as the codeword length, number of information bits, modulation scheme, and range of Eb/N0 values.

```python
k = 64  # Number of information bits
n = 128  # Codeword length
num_bits_per_symbol = 2  # for QPSK
ebno_dbs = np.arange(0, 5, 0.5)  # Eb/N0 values in dB
batch_size = 1000  # Size of each batch of simulations
num_iter_ldpc = 20  # Number of iterations for LDPC decoding
```

3. Define the System Model:
This involves creating the encoder and decoder objects for the codes you want to simulate and defining the communication channel model.

Refer to the context [4] block which contains the `System_Model` definition to define the BER simulation system since it encapsulates the entire process from bit generation to channel impairment and decoding.
```python
[4]:
```

```python
class System_Model(tf.keras.Model):
    """System model for channel coding BER simulations.
    This model allows to simulate BERs over an AWGN channel with
    QAM modulation. Arbitrary FEC encoder/decoder layers can be used to
    initialize the model.
    Parameters
    ----------
        k: int
            number of information bits per codeword.
        n: int
            codeword length.
        num_bits_per_symbol: int
            number of bits per QAM symbol.
        encoder: Keras layer
            A Keras layer that encodes information bit tensors.
        decoder: Keras layer
            A Keras layer that decodes llr tensors.
        demapping_method: str
            A string denoting the demapping method. Can be either "app" or "maxlog".
        sim_esno: bool
            A boolean defaults to False. If true, no rate-adjustment is done for the SNR calculation.
         cw_estiamtes: bool
            A boolean defaults to False. If true, codewords instead of information estimates are returned.
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
                 encoder,
                 decoder,
                 demapping_method="app",
                 sim_esno=False,
                 cw_estimates=False):
        super().__init__()
        # store values internally
        self.k = k
        self.n = n
        self.sim_esno = sim_esno # disable rate-adjustment for SNR calc
        self.cw_estimates=cw_estimates # if true codewords instead of info bits are returned
        # number of bit per QAM symbol
        self.num_bits_per_symbol = num_bits_per_symbol
        # init components
        self.source = BinarySource()
        # initialize mapper and demapper for constellation object
        self.constellation = Constellation("qam",
                                num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(demapping_method,
                                 constellation=self.constellation)
        # the channel can be replaced by more sophisticated models
        self.channel = AWGN()
        # FEC encoder / decoder
        self.encoder = encoder
        self.decoder = decoder
    @tf.function() # enable graph mode for increased throughputs
    def call(self, batch_size, ebno_db):
        # calculate noise variance
        if self.sim_esno:
                no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=1,
                       coderate=1)
        else:
            no = ebnodb2no(ebno_db,
                           num_bits_per_symbol=self.num_bits_per_symbol,
                           coderate=self.k/self.n)
        u = self.source([batch_size, self.k]) # generate random data
        c = self.encoder(u) # explicitly encode
        x = self.mapper(c) # map c to symbols x
        y = self.channel([x, no]) # transmit over AWGN channel
        llr_ch = self.demapper([y, no]) # demap y to LLRs
        u_hat = self.decoder(llr_ch) # run FEC decoder (incl. rate-recovery)
        if self.cw_estimates:
            return c, u_hat
        return u, u_hat
```

4. Simulate LDPC and Polar Codes:
You will run the BER simulations using the defined system model for each code scheme. This involves creating a loop over the range of signal-to-noise ratio (SNR) values, simulating the communication process, and collecting the BER results.

5. Collect and Plot Results:
After running the simulations, you'll collect the BER data for each coding scheme and Eb/N0 value. You can then plot these results using matplotlib to visualize the BER performance and compare the different coding schemes.

Refer to the context [7] and [8] blocks as a guide for the simulation and plotting process, but be sure to adjust the code to your specific requirements for the simulation, such as information length and codeword length, decoder types, and iteration counts.

```python
[7]:
```

```python
num_bits_per_symbol = 2 # QPSK
ebno_db = np.arange(0, 5, 0.5) # sim SNR range
# run ber simulations for each code we have added to the list
for code in codes_under_test:
    print("\nRunning: " + code[2])
    # generate a new model with the given encoder/decoder
    model = System_Model(k=k,
                         n=n,
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=code[0],
                         decoder=code[1])
    # the first argument must be a callable (function) that yields u and u_hat for batch_size and ebno
    ber_plot128.simulate(model, # the function have defined previously
                         ebno_dbs=ebno_db, # SNR to simulate
                         legend=code[2], # legend string for plotting
                         max_mc_iter=100, # run 100 Monte Carlo runs per SNR point
                         num_target_block_errors=1000, # continue with next SNR point after 1000 bit errors
                         batch_size=10000, # batch-size per Monte Carlo run
                         soft_estimates=False, # the model returns hard-estimates
                         early_stop=True, # stop simulation if no error has been detected at current SNR point
                         show_fig=False, # we show the figure after all results are simulated
                         add_bler=True, # in case BLER is also interesting
                         forward_keyboard_interrupt=True); # should be True in a loop
# and show the figure
ber_plot128(ylim=(1e-5, 1), show_bler=False) # we set the ylim to 1e-5 as otherwise more extensive simulations would be required for accurate curves.

```

```python
[8]:
```

```python
ber_plot128(ylim=(1e-5, 1), show_ber=False)
```

6. Analyze Results:
Analyze the plotted results to compare the performance of LDPC and Polar codes. Look for error floor behaviors, waterfall regions, and the overall shapes of the BER curves.

This step involves scientific and critical thinking to draw conclusions from the data obtained.

Please note that the entire simulation process is computationally intensive and may take considerable time, depending on the simulation parameters and the hardware available. Moreover, the specific implementation details and optimization of code depend on the goals of your simulation study and may require fine-tuning various parameters within the Sionna package

INSTRUCTION: Generate a BER plot for short length codes comparing the performance of 5G coding schemes using the PlotBER class in Sionna.
ANSWER:To generate a BER (Bit Error Rate) plot for comparing the performance of 5G coding schemes with the `PlotBER` class in Sionna, you will need to follow a multi-step process, which involves setting up the simulation environment, defining the system model, instantiating the encoders and decoders for the coding schemes you wish to compare, conducting the simulations over a range of Signal-to-Noise Ratios (Eb/No), and finally plotting the results using `PlotBER`.

Here is a high-level overview of the steps you would need to follow, based on the provided context:

1. **Set Up the Simulation Environment:**
   - Import the necessary modules and classes from Sionna.
   - Configure the GPU and TensorFlow settings if necessary.

2. **Define the System Model:**
   - Create a `System_Model` class that centralizes the encoding, modulation, channel simulation, demodulation, and decoding processes.

3. **Instantiate Coding Schemes:**
   - Define the parameters for your codes, for example, the length of the codewords (`k` and `n`) and the type of modulation (e.g., QPSK).
   - Instantiate encoder and decoder objects for each 5G coding scheme you wish to compare (e.g., Polar codes with SC and SCL decoding, LDPC codes with belief propagation decoding).

4. **Set Up Simulation Parameters:**
   - Define an array of Eb/No values over which you will perform the simulation.
   - Decide on the batch size for each Monte Carlo run.
   - Define the stopping criteria, such as the maximum number of Monte Carlo iterations or the target number of bit errors.

5. **Run Simulations:**
   - Use a loop to iterate over each coding scheme and simulate the BER performance over the defined range of Eb/No values.
   - Accumulate the results needed for the BER plot.

6. **Plot BER Performance:**
   - Initialize the `PlotBER` class instance.
   - Add the simulation results for each coding scheme to the `PlotBER` instance using the provided methods.
   - Call the plot function to generate the BER plot, setting appropriate limits for the y-axis (BER values) and any other plot configurations needed to clearly visualize the performance comparison.

7. **Analyze Results:**
   - Analyze the BER plots to compare the performance of the different 5G coding schemes.

Please note that the actual Python code for these steps has been excluded due to instruction, but you can find the relevant code and more detailed instructions on how to implement the steps in the provided context or the Sionna documentation and examples. If you need the detailed Python implementation of these steps, please let me know, and I can provide the code snippets based on the context.

INSTRUCTION: Discuss the decoding complexity of the different coding schemes in Sionna and mention how it affects their practical performance in a 5G context.
ANSWER:In the context of Sionna, a library for simulating communication systems, different coding schemes utilized in 5G New Radio (NR) have varying decoding complexities that can impact their practical performance. Two notable coding schemes are Low-Density Parity-Check (LDPC) codes and Polar codes, both of which are used in 5G NR for error correction.

**LDPC Codes:**
LDPC codes in 5G are designed to support a wide range of code rates and block sizes, up to 8448 bits. The decoding of LDPC codes commonly uses the belief propagation (BP) algorithm, which can be computationally intensive as it involves iterative probabilistic message passing. The complexity of BP decoding scales with the number of iterations and the size of the code's parity-check matrix. 

In the context of Sionna, LDPC codes are simulated using `LDPC5GEncoder` and `LDPC5GDecoder`, with the decoder typically set for a predetermined number of iterations (e.g., 20 iterations in the provided examples). The computational burden can be substantial, especially as the code block size increases, potentially resulting in longer decoding times that may not be suitable for time-critical applications like Ultra-Reliable Low-Latency Communications (URLLC) in 5G. However, LDPC codes are known for excellent performance at long block lengths, which can be a deciding factor when high reliability is critical.

**Polar Codes:**
Polar codes are the other primary coding scheme in 5G, specifically selected for the control channel and for smaller data payloads. The core decoding algorithms for Polar codes include Successive Cancellation (SC) and Successive Cancellation List (SCL) decoding. SC decoding is less complex than SCL but provides inferior error-correction performance. On the other hand, SCL decoding incorporates a list of candidate codewords and typically includes a Cyclic Redundancy Check (CRC) to improve performance, at the expense of higher decoding complexity due to list management and CRC processing.

In the provided Sionna code, Polar codes are handled by `Polar5GEncoder` and `Polar5GDecoder` classes, with `Polar5GDecoder` supporting multiple decoding types such as "SC" and "SCL" with varying list sizes. Higher list sizes lead to better performance but increase the computational load, which may affect the practicality of Polar codes in latency-sensitive scenarios.

**Practical Performance Implications:**
Decoding complexity directly affects the throughput and latency of communications systems. Especially for 5G applications, where high data rates and low response times are paramount, the choice of coding scheme and decoder must balance performance with computational efficiency. For instance, while BP decoding for LDPC codes is iterative and can be more computationally taxing, it can provide higher throughput at larger code lengths, benefiting applications like Enhanced Mobile Broadband (eMBB). Meanwhile, SCL decoding for Polar codes is beneficial for URLLC due to its strong error correction at small to moderate block sizes, in spite of its higher complexity than SC decoding.

Sionna provides tools to simulate the decoding performance and complexity, giving insights into the trade-offs between different coding schemes and their implementations. Users can run extensive simulations, even including the ordered statistics decoding (OSD) approximations for maximum likelihood performance, to fully understand these trade-offs in the context of their specific 5G use cases. However, it's important to note that simulation times can be lengthy, particularly when simulating across various block lengths and SNR points, hence a thorough evaluation requires careful planning and resource allocation.

INSTRUCTION: Perform simulations in Sionna to compare the BER/BLER performance of LDPC codes under different codeword lengths while keeping the code rate fixed.
ANSWER:To perform simulations in Sionna to compare the Bit Error Rate (BER) / Block Error Rate (BLER) performance of Low-Density Parity-Check (LDPC) codes under different codeword lengths while keeping the code rate fixed, you would typically follow these steps in a Python environment where Sionna is available:

1. **Set Up the Simulation Parameters**: Decide on the range of codeword lengths you want to test and the fixed code rate. For instance, you may choose codeword lengths of 128, 256, 512, etc., and a fixed code rate such as 1/2.

2. **Code Definition**: For each codeword length, define the LDPC encoder and decoder with the fixed code rate. This involves initializing `LDPC5GEncoder` and `LDPC5GDecoder` instances with the corresponding lengths and rate.

3. **Simulation Environment Setup**: Create the system simulation environment, which includes defining the modulation scheme (e.g., QPSK, 16-QAM), the AWGN channel, and possibly the rate-matching scheme if used.

4. **Running Simulations**: Perform the BER/BLER simulations for each set of encoder and decoder pairs over a range of E_b/N_0 (energy per bit to noise power spectral density ratio) values. You will need to set a batch size for the number of simulations and the number of iterations for decoding.

5. **Data Collection and Analysis**: Collect BER/BLER results for different codeword lengths and compare them. Plot the results to visualize the performance differences.

Below is a pseudo-code outline based on the given context, which details these steps:

```python
import sionna
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
import numpy as np
import matplotlib.pyplot as plt

# Fixed code rate
fixed_rate = 1/2  # You can adjust this to your target fixed rate.

# Codeword lengths to be tested
codeword_lengths = [128, 256, 512, 1024, 2048]  # Example lengths

# Range of E_b/N_0 values to simulate
ebno_range = np.linspace(0, 10, num=21)  # From 0 to 10 dB in 0.5 dB steps

# Initialize BER/BLER storage
ber_results = {}
bler_results = {}

# Simulation for each codeword length
for n in codeword_lengths:
    k = int(n*fixed_rate)  # Calculate the number of information bits
    
    # Define LDPC Encoder and Decoder for the given length and rate
    enc = LDPC5GEncoder(k=k, n=n)
    dec = LDPC5GDecoder(enc, num_iter=20)  # Number of iterations can be adjusted
    
    # System model definition goes here (not included in context)
    # ...
    
    # Simulation across the E_b/N_0 range
    for ebno_db in ebno_range:
        # Perform simulation here (not included in context)
        # ...
        
        # Store BER/BLER results
        ber_results[(n, ebno_db)] = computed_ber
        bler_results[(n, ebno_db)] = computed_bler

# Plotting the results
plt.figure(figsize=(10, 6))
for n in codeword_lengths:
    ber_for_length = [ber_results[(n, ebno_db)] for ebno_db in ebno_range]
    plt.semilogy(ebno_range, ber_for_length, label=f"Length={n}")

plt.title("BER Performance of LDPC Codes with Fixed Code Rate")
plt.xlabel("$E_b/N_0$ (dB)")
plt.ylabel("BER")
plt.legend()
plt.grid()
plt.show()
```

Please note that the code above is a pseudo-code outline to illustrate the process. Some portions like actual system model definition and simulation details are not included because they were not provided in the context, and they are nontrivial to implement. In practice, the simulation involves generating input data, encoding it, modulating it, passing it through a channel, demodulating it, and decoding it to estimate the output data. The BER/BLER metrics are then computed by comparing the input and output data. The code assumes that these steps have been implemented in Sionna and are called in placeholders marked with comments.

INSTRUCTION: Configure a simulation in Sionna to investigate the BER performance gains via longer codewords for LDPC codes and interpret the resulting plots.
ANSWER:To configure a simulation in Sionna to investigate the BER (Bit Error Rate) performance gains via longer codewords for LDPC (Low-Density Parity-Check) codes, follow these steps:

1. Import necessary libraries and set up the GPU configuration, if applicable:

```python
# Import libraries and modules
import tensorflow as tf 
import numpy as np
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.utils import BinarySource, ebnodb2no
from sionna.channel import AWGN
from sionna.utils.metrics import count_block_errors
from sionna.utils.plotting import PlotBER
# ... other necessary imports based on context ...

# Configure GPU settings
# ... Apply GPU configuration as needed ...
```

2. Define the system model for simulations (use the provided `System_Model` class from the context):

```python
# Use the provided System_Model class from the context
# ...
```

3. Set up LDPC code parameters and instantiate encoder-decoder pairs for the various codeword lengths you wish to test:

```python
# Code parameters
coderates = [1/2] # Example to use a code rate of 1/2
codeword_lengths = [128, 256, 512, 1024, 2048] # Codeword lengths to test

# Create encoder-decoder pairs with different codeword lengths
codes_under_test = []
for n in codeword_lengths:
    k = int(coderates[0] * n) # Calculate the number of information bits
    enc = LDPC5GEncoder(k=k, n=n)
    dec = LDPC5GDecoder(enc)
    codes_under_test.append((enc, dec, f"LDPC n={n}"))
```

4. Configure the BER simulation parameters:

```python
# Simulation parameters
ebno_dbs = np.arange(0, 10, 0.5) # Eb/N0 values to simulate
max_mc_iters = 1000 # Maximum number of Monte Carlo iterations
num_target_block_errors = 100 # Target number of block errors
batch_size = 1000 # Number of codewords per batch
```

5. Run the BER simulations for each LDPC code configuration and collect the results:

```python
# Run BER simulations
ber_plot = PlotBER("BER Performance for Different Codeword Lengths")
for encoder, decoder, name in codes_under_test:
    model = System_Model(k=encoder.k,
                         n=encoder.n,
                         num_bits_per_symbol=2, # Assuming QPSK
                         encoder=encoder,
                         decoder=decoder)
    ber_plot.simulate(model.call, 
                      ebno_dbs=ebno_dbs, 
                      max_mc_iters=max_mc_iters, 
                      num_target_block_errors=num_target_block_errors, 
                      batch_size=batch_size, 
                      legend=name)
```

6. Display the simulation results as BER vs. Eb/N0 plots:

```python
# Display the BER plots
ber_plot.show()
```

**Interpreting the Results:**

After running the simulations, you should interpret the resulting plots by comparing the BER curves for different codeword lengths at the same code rate. 

- Longer codewords typically offer better BER performance thanks to their increased error-correction capability. This is due to the benefits of longer block lengths in error correction codes, such as LDPC, which leverage larger block sizes to distribute parity information more effectively and provide stronger protection against errors.

- Look for areas where the BER curves for longer codewords start to diverge from those of shorter codewords. A steeper "waterfall" region in the BER curve indicates a more significant performance gain when crossing a certain Eb/N0 threshold.

- Consider the decoding complexity: longer codewords usually require more computational resources and time to decode. Although not explicitly considered in the code above, a realistic evaluation should also weigh the trade-offs between decoding performance and decoding complexity.

Please note that the code snippets provided here are illustrative and based on interpreting the provided context. However, they should be adapted and combined with the complete `System_Model` and other necessary code details from the context to create a fully functional simulation.

