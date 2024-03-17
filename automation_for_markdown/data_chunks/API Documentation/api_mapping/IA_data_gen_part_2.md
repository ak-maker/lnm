INSTRUCTION: Explain how to initialize a `sionna.mapping.Demapper` object, specifying demapping_method, constellation_type, num_bits_per_symbol, and detailing when to use custom constellation.
ANSWER:To initialize a `sionna.mapping.Demapper` object, you will need to specify the `demapping_method`, `constellation_type`, `num_bits_per_symbol`, and optionally, a `constellation` object for custom constellations. Here are the steps and particular conditions under which you should provide certain parameters, including when to use a custom constellation:

1. **Choose a Demapping Method**: 
   - You must specify the `demapping_method` parameter, which determines the algorithm the Demapper will use to calculate log-likelihood ratios (LLRs) for the received symbols. The available methods are `app` for the a posteriori probability method and `maxlog` for the max-log approximation method.

2. **Specify the Constellation Type**: 
   - Set the `constellation_type` parameter to one of the following strings: `qam` for quadrature amplitude modulation, `pam` for pulse amplitude modulation, or `custom` if you intend to use a custom constellation.

3. **Number of Bits per Symbol**: 
   - The `num_bits_per_symbol` parameter is an integer that specifies how many bits each symbol in your constellation represents. For example, in a 16-QAM constellation, there are 4 bits per symbol, so you would set `num_bits_per_symbol=4`. This parameter is required if you use either the `qam` or `pam` as your `constellation_type`.

4. **Using a Custom Constellation**: 
   - If you are employing a modulation scheme not covered by standard types or if you need to shape the constellation for specific requirements, you should use a custom constellation.
   - To use a custom constellation, set the `constellation_type` to `custom` and then create an instance of `sionna.mapping.Constellation`. Pass this instance to the `constellation` parameter when initializing the `Demapper`.
   - A custom constellation allows for a trainable constellation, meaning that you can optimize the placement of the constellation points as part of an end-to-end learning process or adapt the constellation to particular channel conditions.

Here's an example code snippet addressing how to initialize the `sionna.mapping.Demapper` with each of the above parameters:

```python
import sionna
import tensorflow as tf

# Choose demapping method
demapping_method = "app" # or "maxlog"

# Specify constellation type and bits per symbol
constellation_type = "qam" # or "pam" or "custom"
num_bits_per_symbol = 4 # Example value for QAM16

# For custom constellations,
# create a `Constellation` object (this code is indicative, customize as needed):
# constellation = sionna.mapping.Constellation(custom_points, custom_labels)

# Initialize the Demapper without custom constellation
demapper = sionna.mapping.Demapper(demapping_method, constellation_type, num_bits_per_symbol)

# If using a custom constellation, pass the instance instead:
# demapper = sionna.mapping.Demapper(demapping_method, constellation_type="custom", constellation=constellation)
```

When you opt for a custom constellation, you should not provide the `num_bits_per_symbol` parameter; the custom `Constellation` object already contains this information. Instead, pass the `constellation` parameter directly with the `Constellation` instance that defines your custom constellation.

INSTRUCTION: Describe the input requirements when using the `Demapper` class with prior knowledge of bits set by with_prior flag.
ANSWER:When using the `Demapper` class with the `with_prior` flag set to `True`, it is assumed that prior knowledge on the bits is available. The `Demapper` class then requires additional input for the prior knowledge when computing the log-likelihood ratios (LLRs) or hard decisions. Here are the specific input requirements:

1. **Received Symbols (y)**: A tensor of complex values representing the received symbols. This must have the shape `[..., n]`, where `n` is the number of received symbols.

2. **Prior Information (prior)**: A tensor of floating-point values representing the prior LLRs for every bit. This input is required only when `with_prior` is set to `True`. The shape of the `prior` tensor can either be `[num_bits_per_symbol]` for the entire input batch or a broadcastable shape to `[..., n, num_bits_per_symbol]`.

3. **Noise Variance Estimate (no)**: A scalar or tensor providing an estimate of the noise variance. This can be provided either as a single scalar for the entire input batch or as a tensor "broadcastable" to the shape of `y` (`[..., n]`).

The output from the `Demapper` with the `with_prior` flag set will be a tensor of LLRs or hard decisions for every bit, with the shape `[..., n*num_bits_per_symbol]`, where `num_bits_per_symbol` is the number of bits per constellation symbol. The output tensor will have corresponding floating-point data types, such as `tf.float32` or `tf.float64`, depending on the `dtype` specified.

INSTRUCTION: Outline the types of output that the `Demapper` class produces and distinguish between the output when hard_out is set to True versus False.
ANSWER:The `Demapper` class in the provided context is designed to process a tensor of received symbols and output either log-likelihood ratios (LLRs) or hard decisions on bits, depending on the configuration. The class operates based on various parameters, including the `hard_out` parameter, which controls the type of output produced. Here's an outline of the types of output:

When `hard_out` is set to `False` (default setting):
- The `Demapper` class produces LLRs for every bit associated with the received symbols. LLRs represent the natural logarithm of the ratio of probabilities that a bit is 1 given the received symbol over the probability that the same bit is 0, conditioned on the received symbol and any prior information available (if `with_prior` is used). These LLRs can be used for soft-decision decoding processes, which take into account the probability of a bit being a certain value and are typically associated with more sophisticated decoding algorithms. The LLRs are typically used in iterative decoding schemes allowing more complex error correction through interactions between the decoder and the demapper.

When `hard_out` is set to `True`:
- The output of the `Demapper` is a tensor of hard-decided bits instead of soft-values. Hard decision demapping simplifies the decoding process by making a definitive decision on each bit—either a '0' or a '1'—based on which constellation point the received symbol is closest to. This is essentially a straightforward decision that does not take into account the reliability of each bit but can be used with hard-decision decoding algorithms, which are typically simpler and faster but may have reduced error correction capabilities compared to their soft-decision counterparts.

Regardless of the setting for `hard_out`, the class interface and methods are designed to be fully differentiable when soft decisions (LLRs) are computed. This enables the demapper to be used in systems that employ gradient-based optimization algorithms, such as when training neural networks.

INSTRUCTION: Differentiate between the "app" and "maxlog" demapping methods, providing equations for LLR computation for each method in the context of `Demapper`.
ANSWER:The "app" and "maxlog" demapping methods refer to two different algorithms used in a `Demapper` to compute log-likelihood ratios (LLRs) for the received bits based on the received symbols in a communication system.

The "app" method stands for "a posteriori probability" and computes the exact LLR for each bit according to the following expression:

$$
LLR(i) = \ln\left(\frac{\sum_{c \in \mathcal{C}_{i,1}} \Pr(c \lvert \mathbf{p}) \exp\left(-\frac{1}{N_o}|y-c|^2\right)}{\sum_{c \in \mathcal{C}_{i,0}} \Pr(c \lvert \mathbf{p}) \exp\left(-\frac{1}{N_o}|y-c|^2\right)}\right)
$$

Here, $LLR(i)$ is the LLR for the $i^{th}$ bit, $\mathcal{C}_{i,1}$ and $\mathcal{C}_{i,0}$ are the sets of constellation points where the $i^{th}$ bit is 1 or 0, respectively, $y$ is the received symbol, $|\cdot|$ denotes the absolute value, $N_o$ is the noise variance, and $\mathbf{p}$ is the vector of prior LLRs. $\Pr(c \lvert \mathbf{p})$ is the prior probability for the constellation symbol $c$, which can be computed from the prior LLRs using a sigmoid function and the bit labels $\ell(c)_k$ (with '0' replaced by '-1') of the symbol $c$.

The "maxlog" method is an approximation to the "app" method that simplifies computation by taking the maximum term in the numerator and denominator of the LLR expression. The approximation is calculated as:

$$
LLR(i) \approx \max_{c \in \mathcal{C}_{i,0}}\left(\ln\left(\Pr(c \lvert \mathbf{p})\right) - \frac{|y-c|^2}{N_o}\right) - \max_{c \in \mathcal{C}_{i,1}}\left(\ln\left(\Pr(c \lvert \mathbf{p})\right) - \frac{|y-c|^2}{N_o}\right)
$$

In this approximation, the sums are replaced by the maximum term inside the sum, thereby reducing computational complexity while still providing a reasonably accurate estimate for the LLR.

To summarize, the "app" method provides an exact LLR calculation considering all constellation points corresponding to each bit value. In contrast, the "maxlog" method simplifies the calculation by considering only the constellation point that contributes most significantly to the LLR for each bit value, thus providing a computationally efficient approximation.

INSTRUCTION: Explain the functional differences between `Demapper` and `SymbolDemapper`, including discussion on prior knowledge and its effects on outputs.
ANSWER:The functional differences between `Demapper` and `SymbolDemapper` relate to the level at which they operate within a digital communication system. The `Demapper` operates at the bit level, converting received symbols into log-likelihood ratios (LLRs) or hard decisions for individual bits. The `SymbolDemapper`, on the other hand, operates at the symbol level, computing normalized log-probabilities (logits) or hard decisions for the entire symbol.

`Demapper`:
- The `Demapper` class computes LLRs or hard decisions on bits for a tensor of received symbols.
- LLRs are essentially a measure of confidence that a given bit is a '1' or a '0' based on the received symbol and the noise estimate.
- `Demapper` can operate using different demapping methods, such as "app" (a posteriori probability) and "maxlog" (an approximation maximizing the logarithm).
- It has a `with_prior` parameter, indicating that prior knowledge about the bits (given as LLRs) can be used in the demapping process if set to `True`.
- The prior information is used to adjust the computation of LLRs based on the assumed knowledge about the likelihood of each bit before receiving the symbol.

`SymbolDemapper`:
- The `SymbolDemapper` class, on the contrary, computes logits or hard decisions for entire symbols.
- Logits represent the normalized log-probabilities of the received symbol being each possible constellation point.
- It works with received symbols to generate an array of log-probabilities for each of the possible constellation points.
- Similar to `Demapper`, `SymbolDemapper` also has a `with_prior` parameter. This parameter, when set to true, denotes that prior knowledge of the constellation points is available and will be used. This prior information is given in the form of log-probabilities (or logits).
- The `SymbolDemapper` typically does not compute LLRs for individual bits but instead operates at the symbol level, providing a single likelihood figure for the entire symbol's constellation point.

The effect of the prior knowledge in both cases influences the output:
- In the `Demapper`, the prior acts as an additional input that adjusts the computation of LLRs based on previously assumed bit probabilities.
- In the `SymbolDemapper`, the prior represents assumed knowledge about the likelihood of each constellation point directly, before receiving the symbol.

Both demapping methods can provide outputs as either soft-values (probabilistic information, like LLRs or logits) or hard decisions (definitive choices of bits or symbols). The output in the soft-decision case for `Demapper` is a tensor of LLRs with shape `[...,n*num_bits_per_symbol]`, while for `SymbolDemapper`, it's a tensor of logits for every constellation point with shape `[...,n, num_points]`. The computation of these probabilities or LLRs allows for further processing, like decoding where further error correction and interpretation of the data occurs.

INSTRUCTION: Provide code examples using the `SymbolDemapper` class to compute normalized log-probabilities for received symbols without and then with the assumption of prior knowledge.
ANSWER:To use the `SymbolDemapper` class to compute normalized log-probabilities (logits) for received symbols, you would first need to create an instance of the `SymbolDemapper`, specifying the type of constellation and the number of bits per symbol, if not using a custom constellation.

Here's how you can compute log-probabilities without prior knowledge:

```python
import tensorflow as tf
from sionna.mapping import SymbolDemapper, Constellation

# Define the constellation type and number of bits per symbol
constellation_type = "qam" # For example, using Quadrature Amplitude Modulation
num_bits_per_symbol = 4 # For example, 16-QAM
constellation = Constellation(constellation_type, num_bits_per_symbol)

# Create a SymbolDemapper instance without prior knowledge
symbol_demapper = SymbolDemapper(constellation_type=constellation_type,
                                 num_bits_per_symbol=num_bits_per_symbol,
                                 constellation=constellation,
                                 hard_out=False,
                                 with_prior=False)

# Received symbols (y) and noise variance estimate (no)
y = tf.constant([...]) # Replace with actual received symbol tensor
no = tf.constant([...]) # Replace with actual noise variance tensor

# Compute logits
logits = symbol_demapper((y, no))
```

Now, if you want to compute log-probabilities with prior knowledge, the code changes slightly to include the prior log-probabilities (logits) as an additional input:

```python
# Create a SymbolDemapper instance with prior knowledge
symbol_demapper_with_prior = SymbolDemapper(constellation_type=constellation_type,
                                            num_bits_per_symbol=num_bits_per_symbol,
                                            constellation=constellation,
                                            hard_out=False,
                                            with_prior=True)

# Prior log-probabilities for each symbol
prior = tf.constant([...]) # Replace with actual prior logits tensor

# Compute logits with prior knowledge
logits_with_prior = symbol_demapper_with_prior((y, prior, no))
```

Make sure to replace `[...]` with the actual tensors representing your received symbols (`y`), noise variance estimates (`no`), and the prior log-probabilities (`prior`). The computed `logits` or `logits_with_prior` will have the normalized log-probabilities for each received symbol. If `hard_out` is set to `True`, the demapper would provide hard-decisions on the symbols instead of logits.

INSTRUCTION: Clarify how to provide noise variance estimates to `Demapper` or `SymbolDemapper` when the input set is a batch with non-uniform noise variances.
ANSWER:To provide noise variance estimates to `Demapper` or `SymbolDemapper` within Sionna when the input set is a batch with non-uniform noise variances, you should follow these guidelines:

For `Demapper`:
- The noise variance estimate is represented by the parameter `no`.
- The noise variance can be either a scalar or a tensor that is "broadcastable" to the shape of `y`, which represents the received symbols.
- If non-uniform noise variances are present in the batch, you should provide `no` as a tensor with dimensions that match the dimensions of `y`, except for the last dimension, which corresponds to the number of received symbols.

Here is an example of how to give a tensor of noise variances for a batch of received symbols with different noise variances for each batch entry (assuming you have already defined or obtained `y`, `prior`, and `noise_variance_tensor` with the appropriate shapes):

```python
# Create an instance of the Demapper class with the desired configuration
demapper = sionna.mapping.Demapper(demapping_method, constellation_type, num_bits_per_symbol, constellation, hard_out, with_prior, dtype)

# Call the Demapper instance with received symbols and non-uniform noise variances
llrs_or_hard_decisions = demapper((y, prior, noise_variance_tensor))
```

For `SymbolDemapper`:
- Similar to `Demapper`, the noise variance estimate is given using the `no` parameter.
- The noise variance should match the first dimensions of the `y` tensor, not counting the last dimension. This makes it "broadcastable" to the shape of `y`.

Here is an instance of how to use noise variance tensor with `SymbolDemapper` when you have non-uniform noise variances within your batch:

```python
# Create an instance of SymbolDemapper class with the desired configuration
symbol_demapper = sionna.mapping.SymbolDemapper(constellation_type, num_bits_per_symbol, constellation, hard_out, with_prior, dtype)

# Call the SymbolDemapper instance with received symbols and non-uniform noise variances
logits_or_hard_decisions = symbol_demapper((y, prior, noise_variance_tensor))
```

In both examples, the `noise_variance_tensor` should be carefully crafted so that its shape is compatible with the broadcasting rules required by the `Demapper` and `SymbolDemapper` classes. This tensor should typically have the same shape as the `y` tensor, except for the last dimension if `y` holds symbols, or it could possibly omit additional dimensions if the same noise variance applies to multiple symbols. 

For a batch with non-uniform noise variances, you need to ensure that each entry in the batch has a corresponding noise variance. If your received symbols tensor `y` has a shape of `[batch_size, num_symbols]`, then `noise_variance_tensor` should have a shape of `[batch_size, 1]` or `[batch_size, num_symbols]` to allow for either the same noise variance for all symbols within an entry of the batch or different variances for each symbol, respectively.

INSTRUCTION: Demonstrate how to evaluate log-probabilities for a specific constellation point using `SymbolDemapper` and interpret the logits output generated when hard_out flag is false.
ANSWER:To evaluate log-probabilities using `SymbolDemapper` from the provided context, we need to follow a few steps that involve creating an instance of the `SymbolDemapper` class, feeding it with the received symbols and noise variance, and interpreting the output logits.

1. **Instantiate the SymbolDemapper**: Create a SymbolDemapper object with the appropriate parameters, such as `constellation_type`, `num_bits_per_symbol`, and whether to output logits or hard decision. Here, we ensure that `hard_out` is set to `False` to get log-probabilities as the output.

```python
from sionna.mapping import SymbolDemapper

# Instantiate the SymbolDemapper
# Replace 'constellation_type' and 'num_bits_per_symbol' with the actual values
demapper = SymbolDemapper(constellation_type="qam", 
                          num_bits_per_symbol=4, # Example for QAM16
                          hard_out=False)
```

2. **Prepare Input Tensors**: Prepare the input tensors `y` containing the received symbols, and `no` the noise variance estimate. The noise variance is typically known or estimated in the receiver.

```python
import tensorflow as tf

# Example placeholder tensors
y = tf.constant([...], dtype=tf.complex64) # Received complex symbols tensor
no = tf.constant([...], dtype=tf.float32)  # Noise variance tensor
```

3. **Call the SymbolDemapper Layer**: Pass the input tensors to the `SymbolDemapper` instance to calculate log-probabilities (logits).

```python
# Calculate logits
logits = demapper((y, no))
```

4. **Interpret the Output**: The output `logits` is a tensor containing the log-probabilities (logits) for each received symbol against each possible constellation point.

The log-probabilities are normalized such that they can be interpreted as relative likelihoods of each constellation point, given the observed symbol `y` and the noise variance. Assuming the prior is uniform (all constellation points are equally likely a priori), the logits represent:

$$
\ln{\left(\frac{\exp{\left(-\frac{|y-c|^2}{N_0}\right)}}{\sum_{c'\in\mathcal{C}}\exp{\left(-\frac{|y-c'|^2}{N_0}\right)}}\right)}
$$

for each constellation point $c$.

- `[..., n, num_points]`: The last dimension of the output tensor matches the number of constellation points. Each element in this dimension represents the log-probability of the corresponding constellation point.

5. **Optional Steps - Convert Logits to Probabilities**: If you wish to convert logits to actual probabilities, apply the softmax function. This step is not typically necessary for further processing (like demapping to bits), as most algorithms can work directly with log-probabilities due to their numerical stability.

```python
# Convert logits to probabilities if needed
probabilities = tf.nn.softmax(logits, axis=-1)
```

The output `probabilities` now represents the probability distribution over the constellation points for each received symbol.

By following these steps, you would have evaluated the log-probabilities using the `SymbolDemapper` class from the given context. Remember, the softmax step to convert logits to probabilities is often optional, depending on your application's needs.

