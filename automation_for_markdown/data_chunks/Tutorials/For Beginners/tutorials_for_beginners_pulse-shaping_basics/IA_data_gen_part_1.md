INSTRUCTION: Import the necessary modules in Python for pulse shaping using the Sionna package, including matplotlib for plotting, numpy for numerical operations, and specific signal processing and QAM source utilities from Sionna.
ANSWER:To import the necessary modules in Python for pulse shaping using the Sionna package, including matplotlib for plotting, numpy for numerical operations, and specific signal processing and QAM source utilities from Sionna, you can use the following code:

```python
import matplotlib.pyplot as plt
import numpy as np
from sionna.utils import QAMSource
from sionna.signal import Upsampling, Downsampling, RootRaisedCosineFilter, empirical_psd, empirical_aclr
```

This code snippet includes the matplotlib library for plotting graphs (`plt`), numpy for numerical computations (`np`), and specific utilities from the Sionna package for working with signal processing in digital communications. The `QAMSource` class is used for generating QAM symbols, while the `Upsampling`, `Downsampling`, and `RootRaisedCosineFilter` are signal processing utilities. Additionally, `empirical_psd` and `empirical_aclr` are functions for calculating the empirical power spectral density and adjacent channel leakage power ratio, respectively.

INSTRUCTION: Explain how to create a RootRaisedCosineFilter object in Sionna with a specific roll-off factor, span in symbols, and oversampling factor.
ANSWER:To create a RootRaisedCosineFilter object in Sionna with a specific roll-off factor, span in symbols, and oversampling factor, you can use the `RootRaisedCosineFilter` class provided in the `sionna.signal` module.

Here are the steps and a piece of example code to illustrate how to create such an object:

1. Import the required class from the `sionna.signal` module.

```python
from sionna.signal import RootRaisedCosineFilter
```

2. Define the parameters for the roll-off factor, span in symbols, and oversampling factor.

Say you want to set:
- Roll-off factor (`beta`): 0.25
- Span in symbols (`span_in_symbols`): 10
- Oversampling factor (`samples_per_symbol`): 8

You can assign these parameters as variables:

```python
beta = 0.25  # Roll-off factor
span_in_symbols = 10  # Filter span in symbols
samples_per_symbol = 8  # Oversampling factor (number of samples per symbol)
```

3. Instantiate the `RootRaisedCosineFilter` object with the defined parameters by passing them to the constructor.

```python
rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)
```

Here's the complete example code encapsulating the steps above:

```python
# Import the RootRaisedCosineFilter class
from sionna.signal import RootRaisedCosineFilter

# Define the parameters for the filter
beta = 0.25  # Roll-off factor
span_in_symbols = 10  # Filter span in symbols
samples_per_symbol = 8  # Oversampling factor

# Create the RootRaisedCosineFilter object
rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)
```

Now, `rrcf` is an instance of the `RootRaisedCosineFilter` class tailored to your specified roll-off factor, span in symbols, and oversampling factor. You can use this object for pulse shaping and matched filtering operations on signal sequences as part of a digital communication system simulation in Sionna.

INSTRUCTION: Illustrate how to use the RootRaisedCosineFilter's visualization functions to display its impulse response and magnitude response in both logarithmic (dB) and linear scales, normalized by symbol time and bandwidth.
ANSWER:To demonstrate the usage of the RootRaisedCosineFilter's visualization functions to display its impulse response and magnitude response with normalization and on different scales, you can follow these guidelines based on the provided context:

Firstly, initialize the RootRaisedCosineFilter with the desired parameters. This includes the roll-off factor, the filter span in symbols, and the oversampling factor (samples per symbol). In the context given, this initialization is as follows:

```python
beta = 0.22 # Roll-off factor
span_in_symbols = 32 # Filter span in symbols
samples_per_symbol = 4 # Number of samples per symbol, i.e., the oversampling factor
rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)
```

After creating an instance of the RootRaisedCosineFilter, you can use the `show` method to visualize different aspects of the filter's response. Specifically, you can visualize the impulse response, and the magnitude response in both dB scale (logarithmic) and linear scale.

The impulse response can be visualized using the following code:

```python
rrcf.show("impulse")
```

For visualizing the magnitude response on a dB scale and linear scale, you will normalize it by symbol time and bandwidth. In the context provided, the `show` function is used like this:

```python
rrcf.show("magnitude", "db") # Logarithmic scale
rrcf.show("magnitude", "lin") # Linear scale
```

Note that the context does not contain information about direct functions for normalizing by symbol time `T` and bandwidth `W`. However, these could be implied by the library within the visualization functions. The `show` method typically normalizes frequency to the Nyquist frequency (half the sampling rate).

In summary, to use the visualization functions of the RootRaisedCosineFilter for displaying its impulse response and magnitude response normalized by symbol time and bandwidth, you need to:

1. Create an instance of the RootRaisedCosineFilter with appropriate parameters.
2. Use the `show` method with arguments `"impulse"` to display the impulse response.
3. Use the `show` method with arguments `"magnitude", "db"` to display the magnitude response in dB scale.
4. Use the `show` method with arguments `"magnitude", "lin"` to display the magnitude response in a linear scale.

Since the normalization is not explicitly defined in the instructions, it is assumed that the visualization functions internally handle normalization appropriate for their display purposes.

INSTRUCTION: Verify the length of a Sionna filter object and explain why the filter length in Sionna is always an odd number, even when the product of span_in_symbols and samples_per_symbol could be even.
ANSWER:To verify the length of a Sionna filter object, specifically a RootRaisedCosineFilter instance named `rrcf`, you can retrieve the `length` attribute of the filter object. According to the example code given in the context, you would execute the following line of Python code:

```python
print("Filter length:", rrcf.length)
```

The output provided in the context shows that the filter length of the `rrcf` instance is 129:

```
Filter length: 129
```

This length is an odd number. In digital signal processing, filter lengths are typically odd to ensure that the filter has a symmetric impulse response. This symmetry around the central tap allows for what's called a linear phase response, which means that all frequency components of the signal are delayed by the same amount of time when passing through the filter.

In the context of the Sionna’s RootRaisedCosineFilter, regardless of the product of `span_in_symbols` and `samples_per_symbol`, the filter length is odd because a symmetric filter is required to maintain a linear phase. The filter length (`L`) is derived from the product of `span_in_symbols` (which defines the duration of the filter in symbol periods) and `samples_per_symbol` (which defines the oversampling rate), but it is also adjusted to ensure that it ends up being an odd number.

The adjustment to obtain an odd filter length ensures the filter's central tap coincides exactly with the middle of the filter coefficients array, which is crucial for achieving the linear phase characteristic. This central tap represents the peak of the filter's impulse response, making it significant for various signal processing operations, such as matched filtering, where synchronization at the receiver relies on correctly timing the filter's peak response with the incoming signal's symbol period.

INSTRUCTION: Describe the process of upsampling a sequence of QAM symbols in Sionna, including configuring the QAM source layer, inserting zeros after every QAM symbol, and applying the RootRaisedCosineFilter to the upsampled sequence.
ANSWER:To perform upsampling of a sequence of Quadrature Amplitude Modulation (QAM) symbols in Sionna, you need to follow these steps:

1. Configure the QAM source layer:
   First, you need to set up a QAM source which will generate a sequence of QAM symbols. You can define the modulation order (number of bits per symbol) for your QAM constellation.

   ```python
   num_bits_per_symbol = 4  # Example for 16-QAM
   qam = QAMSource(num_bits_per_symbol)  # Layer to generate QAM symbols
   x = qam([batch_size, num_symbols])  # Generate a batch of QAM symbols
   ```

2. Upsample the QAM symbols by inserting zeros:
   Upsampling involves inserting a specific number of zeros between each QAM symbol in your generated sequence. This is typically done by an Upsampling layer that takes as input the number of samples per symbol (upsampling factor).

   ```python
   samples_per_symbol = 4  # Upsampling factor (number of samples per symbol)
   us = Upsampling(samples_per_symbol)  # Create an instance of the Upsampling layer
   x_us = us(x)  # Upsample the QAM symbol sequence
   ```

   In this case, you are increasing the sample rate by inserting `samples_per_symbol - 1` zeros between each symbol.

3. Apply a Root Raised Cosine (RRC) filter:
   Next, apply a root raised cosine filter to the upsampled sequence. This is a form of pulse shaping that reduces intersymbol interference and bandwidth usage. The filter is defined by its roll-off factor, span in symbols, and the upsampling factor (samples per symbol).

   ```python
   beta = 0.22  # Roll-off factor of the root raised cosine filter
   span_in_symbols = 32  # How many symbols the filter's impulse response will span
   rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)  # Instantiate the filter
   x_rrcf = rrcf(x_us)  # Filter the upsampled sequence
   ```

   The RRC filter now shapes the signal by smoothing the transitions between the symbols, making it more suitable for transmission over a physical channel.

Each of these steps involves creating and configuring layers provided by Sionna, a Python package designed for simulating communication systems within the TensorFlow framework. Make sure to set the parameters such as `num_bits_per_symbol`, `samples_per_symbol`, and `beta` as per the requirements of your specific communication system design.

Keep in mind that precise values and configurations may vary depending on your objectives, system specifications, and constraints. The provided code snippets are based on the context given and indicate how these actions can be implemented using Sionna's API.

INSTRUCTION: Show how to recover QAM symbols from a waveform by applying matched filtering followed by downsampling in Sionna, specifying the peak value index calculation, creating the Downsampling layer, and plotting the signals at various stages for visualization.
ANSWER:To recover QAM symbols from a waveform in Sionna, you need to follow a process that includes applying matched filtering to the waveform, followed by downsampling to obtain the original symbol sequence. Here are the steps you need to follow, taking into account the context provided:

1. **Matched Filtering**: Apply matched filtering to the waveform using the same pulse-shaping filter that was used during transmission. This filter is typically a root-raised-cosine (RRC) filter. Matched filtering maximizes the signal-to-noise ratio (SNR) and minimizes intersymbol interference (ISI).

```python
# Apply the matched filter (use the previously defined RRC filter: rrcf)
x_mf = rrcf(x_rrcf)  # Assuming x_rrcf is the received signal after transmission
```

2. **Peak Value Index Calculation**: Before downsampling, you must find the correct symbol timing. The signal's peak value should line up with the symbol timing after matched filtering. This index is determined as follows. The transmit filter has its peak value after `(L-1)/2` samples, where `L` is the filter length. Doubling the filter (match filtering) delays the peak by `L-1` samples.

```python
# Calculate the index of the peak value (assuming rrcf is the RRC filter used for matched filtering)
peak_index = rrcf.length - 1
```

3. **Creating the Downsampling Layer**: Create a Downsampling layer in Sionna, which will downsample the signal to match the original symbol rate. You need to specify the samples per symbol and the starting index based on the peak value obtained above.

```python
# Instantiate a downsampling layer, provided with the number of samples_per_symbol and correct start index
ds = Downsampling(samples_per_symbol, peak_index, num_symbols)  # num_symbols is the number of symbols to recover
```

4. **Downsampling the Signal**: Use the downsampling layer to downsample the signal from the matched filter output to obtain the estimated transmitted symbol sequence.

```python
# Recover the transmitted symbol sequence through downsampling
x_hat = ds(x_mf)
```

5. **Plotting for Visualization**: It can be helpful to plot the signals at various stages (oversampled, after pulse shaping, and after matched filtering) to visualize and verify the processing steps.

```python
# Visualize the different signals
plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.plot(np.real(x_us[0]), "x")
plt.title("Oversampled sequence of QAM symbols $x_{us}$")
plt.subplot(312)
plt.plot(np.real(x_rrcf[0, rrcf.length//2:]))
plt.title("Transmitted sequence after pulse shaping $x_{rrcf}$")
plt.subplot(313)
plt.plot(np.real(x_mf[0, rrcf.length-1:]))
plt.title("Received sequence after matched filtering $x_{mf}$")
plt.tight_layout()
plt.show()
```

These steps, when applied correctly, should allow you to recover the QAM symbols from the waveform in Sionna. Keep in mind that the exact variable names and specific parameters should match those used in your code environment. The code snippets provided here should be closely aligned with the context to implement the described process.

INSTRUCTION: Present the method to calculate and plot the mean-squared error (MSE) between transmitted and recovered QAM symbols in Sionna, and discuss how changes in the filter's span in symbols or roll-off factor influence the MSE.
ANSWER:To calculate and plot the mean-squared error (MSE) between transmitted and recovered QAM symbols in Sionna, you can follow a method similar to the one illustrated in the context provided. Here's a step-by-step guide that adapts the context into a method:

1. **Configure QAM Source and Filters**: Set up your QAM source to generate a batch of symbols, as well as the root-raised-cosine (RRC) filters for pulse shaping and matched filtering. You will need to decide your modulation order, filter span in symbols, roll-off factor, and oversampling factor. In the context, a 16-QAM source (`num_bits_per_symbol = 4`) and a RRC filter with `beta = 0.22`, `span_in_symbols = 32`, and `samples_per_symbol = 4` are used.

2. **Transmitting QAM Symbols**: Upsample your transmitted symbols, apply the pulse-shaping filter, and plot the signals if necessary to visualize the pulse shaping effect. Following the code from the context, this involves using the `QAMSource`, `Upsampling`, and `RootRaisedCosineFilter` classes.

3. **Recovering QAM Symbols**: To recover the transmitted signals, apply the matched filter (the same RRC filter used for transmission) followed by downsampling at the right sampling point. This is done using the `Downsampling` class, taking into account the delay introduced by the matched filter.

4. **Calculate and Plot MSE**: After recovering the symbols, calculate the MSE between the transmitted and recovered symbols. This will involve subtracting the recovered symbols from the transmitted symbols, squaring the difference, averaging the result, and converting it to decibels if necessary. The context provides the following code for calculating MSE:

```python
mse = 10 * np.log10(np.var(x-x_hat))
print("MSE between x and x_hat (dB)", mse)
```

5. **Plot MSE as a Function of Filter Parameters**: To investigate how the MSE changes with different filter parameters, you can create a loop where you vary the filter's span in symbols and/or the roll-off factor, calculate the MSE for each case, and then plot the results.

Discussing how the changes in the filter's span in symbols or roll-off factor influence the MSE begins with an understanding of what these parameters do:

- **Filter Span in Symbols**: This parameter determines the length of the filter. A larger span means a longer filter, which improves the signal's ability to cope with intersymbol interference but will result in more delay and computational complexity. A longer filter span typically reduces the MSE because it truncates the signal less, which reduces distortion.

- **Roll-off Factor (Beta)**: This parameter defines the excess bandwidth of the filter beyond the Nyquist rate. A smaller roll-off factor makes the filter sharper, reducing the bandwidth but potentially increasing intersymbol interference. Conversely, a larger roll-off factor increases the bandwidth and reduces intersymbol interference at the cost of more out-of-band energy. Adjusting the roll-off factor alters the tradeoff between bandwidth efficiency and the robustness of the signal, which can affect the MSE.

To visualize how the MSE changes as you alter these parameters, you can use a two-dimensional plot with the filter span in symbols on one axis and the roll-off factor on the other. Meanwhile, the color intensity or a third axis can represent the MSE. Running simulations with varying parameters will provide the data needed for such a plot, illustrating the relationship between filter design and MSE.

INSTRUCTION: Describe how to visualize the empirical power spectral density (PSD) of a transmitted signal in Sionna and explain what it represents with regards to in-band and out-of-band energy.
ANSWER:To visualize the empirical power spectral density (PSD) of a transmitted signal in Sionna, follow these steps:

1. Import the required modules and functions. In the context provided, the `empirical_psd` function from the `sionna.signal` module is used to compute and plot the power spectral density of a signal.

2. Create a sequence of symbols to be transmitted. This is typically done using a signal source such as QAMSource, as shown in the context. The exact shape and modulation format of these symbols will depend on your specific application.

3. Upsample the symbol sequence to increase the number of samples per symbol, which is necessary for pulse shaping. An upsampling layer, such as `Upsampling` in Sionna, can be used for this purpose.

4. Apply a pulse-shaping filter, such as the root-raised-cosine filter created with `RootRaisedCosineFilter`, to the upsampled signal. This helps to limit the bandwidth of the transmitted signal and control the spectral characteristics, such as the roll-off factor.

5. After passing the signal through the pulse-shaping filter, the ensuing signal's empirical PSD can be calculated. Use the `empirical_psd` function and provide the filtered signal along with the oversampling rate as arguments. For plotting, you may also specify the y-axis limits for better visualization.

Here is a Python snippet based on the context, demonstrating how to visualize the empirical PSD:

```python
# Assuming the transmitted signal after pulse-shaping is stored in `x_rrcf`
# and the oversampling rate is stored in `samples_per_symbol`
empirical_psd(x_rrcf, oversampling=samples_per_symbol, ylim=[-100, 3])
```

This visualization represents the distribution of energy across different frequency components of the transmitted signal. The PSD plot typically shows a main lobe centered at the carrier frequency with side lobes tapering off into the higher frequencies.

Regarding in-band and out-of-band energy:

- **In-band energy** refers to the energy within the signal bandwidth that is typically used for communication. This would be the main lobe in the PSD plot and should align with the specified channel bandwidth. In the context of a root-raised-cosine filter, it is defined by the interval from [-0.5, 0.5] in normalized frequency.

- **Out-of-band energy** represents energy that spills over into adjacent frequencies outside the channel of interest. This is often due to the pulse-shaping filter's characteristics and is visible as side lobes in the PSD plot. Its intensity and decay rate are affected by the filter's roll-off factor.

Control over in-band and out-of-band energy is crucial for meeting spectral efficiency and regulatory requirements, and visualizing the PSD is a key step in signal analysis for communication systems.

INSTRUCTION: Calculate the adjacent channel leakage power ratio (ACLR) empirically in Sionna, discuss its significance for waveform metrics, and compare the empirical ACLR with the theoretical ACLR provided as a property of a Sionna filter object.
ANSWER:To empirically calculate the Adjacent Channel Leakage Power Ratio (ACLR) using Sionna:

1. **Understanding ACLR**: ACLR is a measure of how much power leaks from a channel into adjacent frequency channels. It's an essential metric for evaluating the spectral cleanliness of communication systems, particularly in crowded frequency bands where multiple users operate simultaneously.

2. **Collect Sample Data**: Begin by generating a modulated signal, applying pulse shaping and upsampling to create a realistic communication signal. In Sionna, this would start with creating a QAM source, upsample, and then apply a Root Raised Cosine (RRC) filter.

3. **Empirical Calculation**: Use Sionna's `empirical_aclr` function which computes the ACLR of a sequence by taking the Fast Fourier Transform (FFT) of the signal and then calculating the ratio of the power in the adjacent channel to the power in the main channel.
   
   ```python
   aclr_db = 10*np.log10(empirical_aclr(x_rrcf, oversampling=samples_per_symbol))
   print("Empirical ACLR (db):", aclr_db)
   ```

   Make sure the signal you use (like `x_rrcf` in the example) is the signal after pulse shaping, as it would include both in-band and out-of-band energy.

4. **Discussion**: The significance of ACLR as a waveform metric lies in its ability to quantify the degree to which a signal consumes more bandwidth than intended, which is critical for avoiding interference with other signals and adhering to regulatory requirements on spectral emissions.

5. **Comparison with Theoretical ACLR**: The theoretical ACLR of a filter object in Sionna may be available as a property of the filter. Compare this theoretical value with the empirically calculated ACLR to validate your filter's performance.

   ```python
   print("Filter ACLR (dB)", 10*np.log10(rrcf.aclr))
   ```

6. **Conclusion**: By checking both empirical and theoretical ACLR, you ensure your system’s waveform complies with standards and does not cause undue interference. An ACLR that is too low would indicate excessive leakage, potentially causing harmful interference.

In conclusion, the ACLR provides critical insights into a system's spectral emissions and the potential for interference with adjacent channels. In Sionna, you can empirically calculate this important metric and compare it to the theoretical performance guaranteed by your filter design to ensure adherence to spectrum regulations and minimal interference in your communication system.

INSTRUCTION: Modify the roll-off factor in the creation of a RootRaisedCosineFilter to improve the ACLR and demonstrate the difference in the resulting ACLR value.
ANSWER:To modify the roll-off factor of a RootRaisedCosineFilter for improving the ACLR (Adjacent Channel Leakage Ratio), you need to follow these steps based on the context provided:

1. Update the roll-off factor, originally set to `beta = 0.22`, to a new value that optimizes the ACLR. Let's choose a smaller value such as `beta = 0.1` as a starting point. A smaller roll-off factor can lead to a tighter concentration of signal energy within the desired bandwidth, possibly improving ACLR.

2. Recreate the RootRaisedCosineFilter with the new roll-off factor and recalculate the ACLR to observe the impact of this change.

Here's the modified code snippet to adjust the roll-off factor and demonstrate the effect on ACLR:

```python
# Update roll-off factor
beta_new = 0.1

# Recreate RootRaisedCosineFilter with the new roll-off factor
rrcf_new = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta_new)

# Apply the new filter to the upsampled sequence
x_rrcf_new = rrcf_new(x_us)

# Compute the new empirical ACLR
aclr_db_new = 10*np.log10(empirical_aclr(x_rrcf_new, oversampling=samples_per_symbol))
print("New empirical ACLR (dB):", aclr_db_new)
```

Note that the code sets `beta_new` to 0.1, initializes a new filter object `rrcf_new` with the updated roll-off factor, and then applies this new filter to the upsampled QAM symbol sequence `x_us`, which should be the same sequence used with the original filter. Finally, the new ACLR value `aclr_db_new` is calculated and printed.

By comparing the original ACLR value (`aclr_db`) and the new ACLR value (`aclr_db_new`), you can assess the extent of ACLR improvement due to the modified roll-off factor. It's important to mention that adjusting the roll-off factor may have other effects on signal characteristics, such as the bandwidth requirement and the inter-symbol interference (ISI), which should also be taken into consideration when optimizing filter parameters.

