### 1. INSTRUCTION:
Explain how to create a custom constellation in Sionna with trainable points and normalization enabled.

**ANSWER:**

To create a custom constellation in Sionna with trainable points and normalization enabled, we can use the `sionna.mapping.Constellation` class. Here is a step-by-step guide based on the provided context:

1. **Import necessary modules**:
   Import TensorFlow and Sionna's mapping module, since Sionna uses TensorFlow as its backend:
   ```python
   import tensorflow as tf
   import sionna.mapping
   ```

2. **Determine the size of your constellation**:
   Choose how many bits each symbol in your constellation will represent. For example, `num_bits_per_symbol=2` would typically correspond to a QPSK constellation.

3. **Create initial constellation values (optional)**:
   You can provide initial complex values for the constellation points. For a constellation with `num_bits_per_symbol`, you'll need an array of size `2^num_bits_per_symbol`. This step is optional because if you do not provide an initial value, the constellation points will be randomly initialized.
   ```python
   # Example for a constellation with 2 bits per symbol (QPSK-like)
   initial_constellation_points = tf.constant([1+1j, 1-1j, -1+1j, -1-1j], dtype=tf.complex64)
   ```

4. **Create the custom, trainable constellation**:
   Use the `sionna.mapping.Constellation` class to define your constellation. Make sure to set `trainable=True` to make constellation points trainable, and `normalize=True` if you want the constellation normalized to unit power.
   ```python
   custom_constellation = sionna.mapping.Constellation(
       constellation_type="custom",
       num_bits_per_symbol=num_bits_per_symbol,
       initial_value=initial_constellation_points,  # Omit this if you want random initial points
       normalize=True,   # This will normalize the constellation to have unit average power
       trainable=True    # This enables training of the constellation points
   )
   ```

By following these steps, you have created a custom trainable constellation in Sionna with normalization enabled. You can integrate this constellation into a Mapper or Demapper as required for your communication system model.

Keep in mind that during the training process, you'll need to include an appropriate learning algorithm and loss function to update the constellation points. Since they are trainable variables, they will be adjusted to minimize the loss function during training.

### 2. INSTRUCTION:
Generate a 16-QAM constellation using Sionna and visualize it, including the bit labels.

**ANSWER:**

Based on the context provided, here are the steps to generate a 16-QAM constellation using Sionna and visualize it, including the bit labels:

```python
# Import necessary modules from Sionna
import sionna
import numpy as np
import matplotlib.pyplot as plt

# Set the number of bits per symbol for 16-QAM
num_bits_per_symbol = 4 

# Create a 16-QAM constellation
qam_constellation = sionna.mapping.qam(num_bits_per_symbol)

# Visualize the constellation with labels
# The Constellation class has a method called show, which can be used 
# to plot the constellation with bit labels
# However, as per the context, we directly used the qam function to generate the constellation
# So we will first need to create a Constellation object, which allows us to use the show method

# Creating a Constellation object
constellation = sionna.mapping.Constellation(constellation_type="qam", 
                                             num_bits_per_symbol=num_bits_per_symbol, 
                                             initial_value=qam_constellation, 
                                             normalize=True, 
                                             dtype=sionna.utils.tf.complex64)

# Now, use the show method to visualize the constellation with bit labels
constellation.show(labels=True)
plt.show()  # This is often not necessary in Jupyter, but can be required in other environments
```
In the context you've provided, the Sionna package appears to be used for the mapping of bits to constellation symbols, a common operation in digital communications. Specifically, it is being used to create and visualize a 16-QAM (Quadrature Amplitude Modulation) constellation.

What this script does is initialize a 16-QAM constellation using the `sionna.mapping.Constellation` class, with parameters to normalize the constellation to have unit power and to plot using its `show` method. Normally, in a Python environment, you can execute the provided code to display the visualization. If you're running this in a notebook environment like Jupyter, the visualization should appear within the notebook. Please make sure to have Sionna and its dependencies properly installed in your Python environment to execute the code successfully.

### 3. INSTRUCTION:
Create a Mapper in Sionna using an existing PAM constellation and ensure it returns the symbol indices.

**ANSWER:**

To create a Mapper in Sionna that uses an existing PAM constellation and ensures it returns the symbol indices, you can utilize the `sionna.mapping.Mapper` class and the `sionna.mapping.pam` function. Here are the steps for doing this:

1. Import the necessary modules.
2. Define a variable for the number of bits per PAM symbol based on the desired PAM constellation order.
3. Use the `sionna.mapping.pam` function to generate the PAM constellation.
4. Create an instance of the `Mapper` class with the `return_indices` set to `True`.

Here is some sample code to illustrate this:

```python
import sionna
import tensorflow as tf

# Step 1: Import necessary modules already done above

# Step 2: Define the number of bits per PAM symbol
num_bits_per_symbol = 2  # Example for 4-PAM (2^2)
# Step 3: Generate PAM constellation
pam_constellation = sionna.mapping.pam(num_bits_per_symbol)

# Step 4: Create a Mapper instance with return_indices=True
mapper = sionna.mapping.Mapper(constellation_type="pam",
                               num_bits_per_symbol=num_bits_per_symbol,
                               constellation=pam_constellation,
                               return_indices=True)

# Example: Map binary input tensor to constellation points
# Assuming input_tensor is a predefined binary input tensor:
# input_tensor = tf.constant([...], dtype=tf.int32)
# mapped_symbols, symbol_indices = mapper(input_tensor)

# mapped_symbols would contain the mapped constellation symbols 
# and symbol_indices would contain the corresponding symbol indices
```

Please replace the `input_tensor` with an actual input tensor of binary values that you would like to map to constellation points. The tensor should have its last dimension be an integer multiple of `num_bits_per_symbol`.

Remember to change the `num_bits_per_symbol` based on the order of PAM you intend to use. For example, `num_bits_per_symbol = 3` for 8-PAM, and `num_bits_per_symbol = 4` for 16-PAM.

### 4. INSTRUCTION:
Demonstrate how to create a `SymbolLogits2LLRs` instance in the `sionna.mapping` module using the "maxlog" method for QAM16 constellation points with input logits, without prior knowledge, and with default data type settings.

**ANSWER:**

To create a `SymbolLogits2LLRs` instance using the "maxlog" method for QAM16 constellation points with input logits, without prior knowledge, and with default data type settings in the `sionna.mapping` module, follow this Python code snippet:

```python
import sionna
from sionna.mapping import SymbolLogits2LLRs

# Number of bits per symbol for QAM16 is 4
num_bits_per_symbol = 4

# Create a SymbolLogits2LLRs instance with "maxlog" method, no prior, and default dtype
symbol_logits_to_llrs = SymbolLogits2LLRs(method="maxlog", num_bits_per_symbol=num_bits_per_symbol, with_prior=False)
```

Here we specify the `method` parameter as "maxlog," the `num_bits_per_symbol` parameter as 4 (since QAM16 has 4 bits per symbol), and `with_prior` as False (indicating that we do not have prior knowledge). We are using default data type settings, so we don't need to specify `dtype`.

### 5. INSTRUCTION:
Explain the process of computing hard-decided bits from logits using the `SymbolLogits2LLRs` class with the "app" method and by assuming prior knowledge of bits is given as an input to the layer.

**ANSWER:**

To compute hard-decided bits from logits using the `SymbolLogits2LLRs` class with the "app" method and assuming prior knowledge of bits is given as an input to the layer, you should follow these steps:

1. **Initialization**: Instantiate an object of the `SymbolLogits2LLRs` class using appropriate parameters. Ensure that the `method` is set to "app", the `num_bits_per_symbol` matches your use case (for example, 4 if you are using QAM16), `hard_out` is set to `True` to obtain hard-decided bits, and `with_prior` is set to `True` since we're considering prior knowledge.

    ```python
    sll = SymbolLogits2LLRs(method="app", num_bits_per_symbol=4, hard_out=True, with_prior=True)
    ```

2. **Prepare Inputs**: You need two sets of inputs:
   - Logits (unnormalized log-probabilities) tensor of shape `[…, n, num_points]`.
   - Prior knowledge tensor as LLRs, which can be a single-dimensional tensor with shape `[num_bits_per_symbol]` applicable to all in the batch or a tensor broadcastable to the shape `[…, n, num_bits_per_symbol]`.

3. **Execute the Method**: Call the `app` method of the instantiated `SymbolLogits2LLRs` class with the logits and prior LLRs as input arguments. The class will compute the probability ratio for each bit considering both the logits and the prior information. Here's pseudo-code representation of how this could look:
    
    ```python
    logits = # your logits tensor
    prior_llrs = # your prior knowledge tensor as LLRs
    hard_decisions = sll((logits, prior_llrs))
    ```
   
   The result, `hard_decisions`, will be a tensor of the same batch dimensionality as `logits`, containing the hard-decided bits derived from the logits using prior knowledge.

In summary, the `SymbolLogits2LLRs` class facilitates the computation of hard decisions from logits through a method that combines the intrinsic likelihoods of certain symbol representations (logits) with extrinsic knowledge (prior LLRs), and ultimately provides the bit decisions. Make sure you provide logits and prior information in the correct format and shape expected by the function for accurate results.

### 6. INSTRUCTION:
Illustrate the use of the `LLRs2SymbolLogits` class to compute unnormalized log-probabilities from given LLRs corresponding to bits in a constellation symbol, and discuss the output format when the `hard_out` parameter is set to `True`.

**ANSWER:**

The `LLRs2SymbolLogits` class in the sionna.mapping module is used to calculate the logits, which are unnormalized log-probabilities, on constellation points from a tensor of log-likelihood ratios (LLRs) for each bit. This process is typically part of a soft-demapping step in communication systems where estimated probabilities (or soft decisions) concerning the transmitted constellation symbols are required for further processing, like decoding.

To use the `LLRs2SymbolLogits` class to compute logits from given LLRs, you can follow these steps. Note that the code will be hypothetical and conceptual since no actual API implementation is provided in the context:

```python
import tensorflow as tf
from sionna.mapping import LLRs2SymbolLogits

# Define the number of bits per symbol for your constellation
num_bits_per_symbol = 4  # Example for QAM16

# Instantiate the LLRs2SymbolLogits class
llrs2symbol_logits = LLRs2SymbolLogits(num_bits_per_symbol=num_bits_per_symbol, 
                                       hard_out=False, # Set to False for soft-output logits
                                       dtype=tf.float32)

# Define a tensor of LLRs
# The shape [..., n, num_bits_per_symbol] must conform to the input specification
llrs = tf.constant([[...]], dtype=tf.float32)

# Compute the logits
logits = llrs2symbol_logits(llrs)
```

If the `hard_out` parameter is set to `True`, instead of returning soft logits, the output will be hard decisions on constellation points. This means the class will produce estimated symbol indices directly, corresponding to the most probable constellation symbol for each group of bits. The format of this output will be an integer tensor with the shape `[..., n]`, where `n` is the last dimension of the LLR input tensor, with each integer representing a constellation point (a symbol index):

```python
# Instantiate the LLRs2SymbolLogits class with hard_out set to True
llrs2symbol_logits_hard = LLRs2SymbolLogits(num_bits_per_symbol=num_bits_per_symbol, 
                                            hard_out=True, # Set to True for hard-output decisions
                                            dtype=tf.float32)

# Compute hard decisions (symbol indices)
hard_decision_output = llrs2symbol_logits_hard(llrs)
```

The resulting `hard_decision_output` tensor will thus contain the estimated indices of the constellation symbols, which can subsequently be mapped back to the actual constellation points using another utility, if necessary. The values would be within the range `[0, 2^num_bits_per_symbol - 1]`, representing all possible symbols in the constellation.

### 7. INSTRUCTION:
Describe the steps to calculate the mean and variance of a constellation using the `SymbolLogits2Moments` class for a custom constellation provided as an instance of Constellation.

**ANSWER:**

To calculate the mean and variance of a custom constellation using the `SymbolLogits2Moments` class, you would need to follow these steps:

1. **Create an instance of your custom constellation** if it's not already available:
   Suppose you have a custom constellation class `MyCustomConstellation` which follows the interface of sionna's `Constellation` class. Create an instance of this class.

```python
custom_constellation = MyCustomConstellation()
```

2. **Create an instance of `SymbolLogits2Moments`** with your custom constellation:
   According to the context provided, you must supply an instance of a Constellation to the `SymbolLogits2Moments` class when you are working with a custom constellation.

```python
from sionna.mapping import SymbolLogits2Moments

# You are using a custom constellation, so you must instantiate SymbolLogits2Moments with `constellation=custom_constellation`
symbol_logits_2_moments = SymbolLogits2Moments(constellation=custom_constellation)
```

3. **Prepare the logits**:
   The logits are unnormalized log-probabilities associated with the constellation points. Prepare an input tensor containing these logits.

```python
import tensorflow as tf

# Example logits tensor (assuming the constellation has N points)
logits = tf.constant([[0.1, 0.2, 0.3, ..., 0.]], dtype=tf.float32) # replace with actual logits
```

4. **Calculate mean and variance**:
   Call the instance of `SymbolLogits2Moments` with the logits tensor to compute the mean and variance of the constellation points based on the provided logits.

```python
mean, variance = symbol_logits_2_moments(logits)
```

5. **Use the output**:
   The `mean` and `variance` which you get from the `SymbolLogits2Moments` instance are TensorFlow tensors. You can use them as needed in your application. If you're running this in an interactive environment or want to evaluate the values, you need to run them within a TensorFlow session (for TensorFlow 1.x) or directly (in TensorFlow 2.x using eager execution).

Example in eager execution (TensorFlow 2.x):
```python
print("Mean of the constellation: ", mean.numpy())
print("Variance of the constellation: ", variance.numpy())
```

Please replace the placeholders (like logits, custom constellation instance) with the actual data and instances respectively according to your project requirements. The detailed functionality of the `SymbolLogits2Moments` class suggests that it relies on internally computing probabilities from logits using the softmax function before computing mean and variance.

### 8. INSTRUCTION:
Guide on transforming symbol indices into their binary representations using the `SymbolInds2Bits` class and discuss how the output tensor's shape is determined based on the input tensor shape and the number of bits per symbol.

**ANSWER:**

The `SymbolInds2Bits` class in the Sionna mapping module is designed to convert symbol indices into their binary representations. This is a crucial step when performing the mapping between bits and constellation symbols in digital communication systems. To use `SymbolInds2Bits`, we need the `num_bits_per_symbol` parameter, which specifies how many binary bits are used to represent each symbol in the constellation.

Here is a step-by-step guide on how to use the `SymbolInds2Bits` class:

1. **Initialize the Class:**
   You need to create an instance of the `SymbolInds2Bits` class with the required parameter `num_bits_per_symbol`. This parameter should equal the number of binary bits that corresponds to each constellation symbol. For example, for a QAM16 (16-QAM) constellation, we need 4 bits to represent each symbol (since 2^4 = 16).

    ```python
    import tensorflow as tf
    from sionna.mapping import SymbolInds2Bits

    # Initialize the SymbolInds2Bits for a QAM16 constellation
    num_bits_per_symbol = 4
    symbol_inds2bits = SymbolInds2Bits(num_bits_per_symbol)
    ```

2. **Prepare the Input Tensor:**
   The input to the `SymbolInds2Bits` class should be a tensor containing the symbol indices you want to convert to binary format. The shape of the input tensor can vary depending on your specific context, for example, `[batch_size, num_symbols]`.

    ```python
    # Example input: a batch of symbol indices for QAM16
    symbol_indices = tf.constant([[0, 1, 15], [3, 7, 4]])
    ```

3. **Transform Symbol Indices to Binary:**
   Call the instance of `SymbolInds2Bits` you initialized with the input tensor.

    ```python
    binary_representation = symbol_inds2bits(symbol_indices)
    ```

4. **Output Tensor Shape:**
   The shape of the output tensor depends on the shape of your input tensor and the `num_bits_per_symbol` parameter. For each symbol in the input tensor, the output will have `num_bits_per_symbol` bits. Thus, for an input tensor with shape `[batch_size, num_symbols]`, the output tensor will have shape `[batch_size, num_symbols, num_bits_per_symbol]`.

    ```python
    # Assuming `symbol_indices` has shape [2, 3], and `num_bits_per_symbol` is 4
    # The shape of `binary_representation` will be [2, 3, 4]
    print(binary_representation.shape)
    ```

And that is how you would transform symbol indices into their binary representations using the `SymbolInds2Bits` class in Sionna. The output tensor contains the binary representation of each symbol index present in the input tensor, expanded along the last dimension.

### 9. INSTRUCTION:
Provide an example of converting PAM symbol indices to corresponding QAM symbol indices using the `PAM2QAM` class, including an explanation of how to handle both indices and logits based on the `hard_in_out` parameter.

**ANSWER:**

Based on the provided context, the `PAM2QAM` class in the Sionna library converts PAM (Pulse Amplitude Modulation) symbol indices or logits to QAM (Quadrature Amplitude Modulation) symbol indices or logits. Here's how you would typically perform the conversion with a given number of bits per QAM symbol and how the `hard_in_out` parameter affects the process:

```python
import tensorflow as tf
from sionna.mapping import PAM2QAM

# Define the number of bits per QAM symbol (e.g., 4 for 16-QAM)
num_bits_per_qam_symbol = 4

# Initialize the PAM2QAM class
pam2qam = PAM2QAM(num_bits_per_qam_symbol=num_bits_per_qam_symbol, hard_in_out=True)

# Example PAM symbol indices for hard_in_out=True
# These could be the indices of the real and imaginary parts respectively
pam1_indices = tf.constant([1, 2, 0, 3])
pam2_indices = tf.constant([2, 1, 3, 0])

# Convert PAM indices to QAM indices
qam_indices = pam2qam(pam1_indices, pam2_indices)
print("QAM indices:", qam_indices)

# Assuming logits rather than hard indices, set hard_in_out=False
pam2qam_logits = PAM2QAM(num_bits_per_qam_symbol=num_bits_per_qam_symbol, hard_in_out=False)

# Example logits for PAM constellations for real and imaginary parts respectively
# Assume 2 bits per PAM symbol for a 16-QAM (hence 2**2 entries)
pam1_logits = tf.constant([[0.1, 0.2, 0.3, 0.4], 
                           [0.4, 0.3, 0.2, 0.1],
                           [0.2, 0.1, 0.4, 0.3],
                           [0.3, 0.4, 0.1, 0.2]])

pam2_logits = tf.constant([[0.2, 0.1, 0.4, 0.3],
                           [0.3, 0.4, 0.1, 0.2],
                           [0.4, 0.3, 0.2, 0.1],
                           [0.1, 0.2, 0.3, 0.4]])

# Convert PAM logits to QAM logits
qam_logits = pam2qam_logits(pam1_logits, pam2_logits)
print("QAM logits:", qam_logits)
```

In this example:

- If `hard_in_out=True`, the inputs (`pam1_indices` and `pam2_indices`) and the output (`qam_indices`) are tensor indices that represent the symbols in their respective PAM and QAM constellations. So the method expects and returns hard indices.

- If `hard_in_out=False`, the method expects and returns logits, which are the unnormalized log-probabilities of the symbols. The inputs (`pam1_logits` and `pam2_logits`) are tensors with logits for each symbol in the PAM constellations for the real and imaginary components, and the output (`qam_logits`) is a tensor with logits for the corresponding QAM symbols.

Please note that the actual conversion, its mathematical basis, and how the logits are computed or used, would normally be detailed in the class's implementation, which is not provided in the context. The example given is an approximate illustration based on typical usage patterns within the context of machine learning or signal processing libraries.

### 10. INSTRUCTION:
Explain how to map QAM symbol indices back to two sets of PAM symbol indices corresponding to their real and imaginary components using the `QAM2PAM` class, and discuss how the input and output tensor shapes correlate.

**ANSWER:**

To map QAM symbol indices back to two sets of PAM symbol indices pertaining to their real and imaginary components using the `QAM2PAM` class from the Sionna library, follow these steps considering the given context:

1. First, ensure you have the required number of bits per QAM symbol (`num_bits_per_symbol`). Typically, for a QAM constellation like 16-QAM, `num_bits_per_symbol` would be 4.

2. Instantiate the `QAM2PAM` object with the number of bits per symbol:

```python
from sionna.mapping import QAM2PAM

# Example for 16-QAM
num_bits_per_symbol = 4
qam2pam = QAM2PAM(num_bits_per_symbol)
```

3. Input your QAM symbol indices as a TensorFlow integer tensor (`tf.int`). The shape of this input tensor `ind_qam` should correspond to the number of QAM symbols you want to map back to PAM indices.

```python
import tensorflow as tf

# Example QAM indices tensor
# It could be a 1D tensor for a single sequence of symbols, or a higher-dimensional tensor
# representing batches or multiple sequences
ind_qam = tf.constant([...], dtype=tf.int32)  # Replace [...] with the actual QAM symbol indices
```

4. Apply the `qam2pam` object on the input QAM symbol indices tensor to get two sets of PAM symbol indices as output:

```python
ind_pam1, ind_pam2 = qam2pam(ind_qam)
```

5. `ind_pam1` and `ind_pam2` are TensorFlow integer tensors containing the PAM indices for the real and imaginary components, respectively, of the original QAM symbols. The shape of these tensors will be the same as the shape of the `ind_qam` input tensor.

For example, if you had a 1D tensor of shape `[n]` for `ind_qam`, the output tensors `ind_pam1` and `ind_pam2` would also have the shape `[n]`.

In summary, you use the `QAM2PAM` class to map a tensor of QAM indices (`ind_qam`) to two separate tensors (`ind_pam1` and `ind_pam2`) which represent the PAM indices of the real and imaginary components. The input shape directly corresponds to the output shapes, which effectively means there is no change in tensor dimensions throughout this operation. Each QAM symbol index is decomposed into two PAM component indices without any expansion or contraction of tensor dimensions.

### 11. INSTRUCTION:
Summarize key features of the `sionna.mapping` classes and functions by correlating them with relevant parts of the 3GPP TS 38.211 specification, and outline a use case where such correlation is particularly beneficial.

**ANSWER:**

The `sionna.mapping` classes and functions provide utilities for bit-to-symbol mapping and symbol-to-bit demapping in compliance with digital communication standards like the 3GPP TS 38.211 for 5G.

### Correlation with 3GPP TS 38.211
The 3GPP TS 38.211 specification outlines various modulation schemes and the corresponding symbol mapping to be utilized in 5G NR systems. The relevant parts of the specification that `sionna.mapping` aligns with include:

- **Modulation schemes**: The `sionna.mapping` classes support the modulation schemes, including Quadrature Amplitude Modulation (QAM) and Pulse Amplitude Modulation (PAM), as defined in the specification. For instance, the `QAM2PAM` and `PAM2QAM` classes provide functionality specific to the conversion between QAM and PAM constellations.
- **Symbol mapping**: The bits-to-symbol mapping and symbol-to-bits demapping are vital in Section 5 of 3GPP TS 38.211 that describes physical channels and modulation. Classes like `SymbolInds2Bits` and `LLRs2SymbolLogits` help translate between symbol indices and their binary or log likelihood representations.
- **Soft bit information**: The usage of LLRs in representing soft bit information aligns with the channel coding techniques outlined in 3GPP TS 38.211. The classes `SymbolLogits2LLRs` and `SymbolLogits2LLRsWithPrior` compute LLRs from unnormalized log probabilities (logits), which are crucial for soft decision decoding and error correction performance.

### Key Features
- **`SymbolLogits2LLRs`**: This class handles the demapping of constellation points (provided as logits) back to LLRs or hard decisions on bits, which is a critical step for channel decoding in receiver implementations.
- **`LLRs2SymbolLogits`**: It performs the inverse of `SymbolLogits2LLRs`, mapping LLRs to logits or hard decisions on constellation points, useful for encoding purposes.
- **`SymbolLogits2LLRsWithPrior`**: Although deprecated, this class emphasizes the importance of incorporating prior information in the LLR calculations, revealing the interconnectedness with iterative decoding schemes.
- **`SymbolLogits2Moments`**: Provides mean and variance calculations for constellations based on logits, useful for assessing signal qualities such as signal-to-noise ratio (SNR) and for adaptive modulation schemes.
- **`SymbolInds2Bits`**, **`PAM2QAM`**, and **`QAM2PAM`**: Facilitate translation between different modulations and their binary representations, complying with the exact modulation schemes specified in 3GPP TS 38.211.

### Use Case
A particularly beneficial use case for the `sionna.mapping` correlation with 3GPP TS 38.211 is in the development of a soft-input soft-output (SISO) decoder for a 5G NR receiver. Decoders like LDPC or Turbo that require soft bit inputs can greatly exploit the LLR calculations performed by `SymbolLogits2LLRs`. This ensures accurate soft-decision information is provided during the iterative decoding process, potentially enhancing the overall bit error rate (BER) performance of the receiver. Given that the decoding process is integral to the link-level simulation of communication systems, adhering to 3GPP TS 38.211 specifications in soft value calculations is crucial for developing standardized and reliable 5G systems.

