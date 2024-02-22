### 1. INSTRUCTION:
Configure a single GPU for running a Sionna-based machine learning model and suppress TensorFlow warnings.

**ANSWER:**

To configure a single GPU for running a Sionna-based machine learning model and suppress TensorFlow warnings, follow these steps:

1. Set the environment variable `CUDA_VISIBLE_DEVICES` to `"0"` to specify that you want to use only the first GPU on your system (assuming `0` is the index of your desired GPU).

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

2. To allocate only as much GPU memory as needed, set memory growth on the selected GPU to `True`. This prevents TensorFlow from consuming all available GPU memory upfront.

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to True
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
```

3. To suppress TensorFlow warnings, set the log level to `'ERROR'`, which will suppress all logs except for errors.

```python
# Set TensorFlow to only log errors
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

Here's the combined code snippet that accomplishes both the GPU configuration and the suppression of TensorFlow warnings:

```python
import os
import tensorflow as tf

# Select GPU number 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set TF log level to only display errors to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Configure TensorFlow to only use the selected GPU and allocate memory on demand
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Memory growth must be set before GPUs have been initialized
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print("Exception while setting memory growth:", e)
```

Make sure this code is run before you instantiate any TensorFlow tensors or Sionna-based ML models to ensure the GPU configuration and log level are set correctly.

### 2. INSTRUCTION:
Use the "RootRaisedCosineFilter" from Sionna's signal module to create a pulse-shaping filter with specific parameters such as roll-off factor and oversampling, and visualize its impulse and magnitude responses.

**ANSWER:**

To create a pulse-shaping filter using the `RootRaisedCosineFilter` class from Sionna's signal module with specific parameters, such as roll-off factor and oversampling, and then visualize its impulse and magnitude responses, follow these steps:

1. Define the roll-off factor, the filter span in symbols, and the number of samples per symbol (oversampling factor).
2. Create an instance of the `RootRaisedCosineFilter`.
3. Use the `.show("impulse")` method of the filter to visualize the impulse response.
4. Use the `.show("magnitude")` method of the filter to visualize the magnitude response.

Here's a Python code snippet based on the provided context:

```python
import matplotlib.pyplot as plt
from sionna.signal import RootRaisedCosineFilter

# Define filter parameters
beta = 0.22  # Roll-off factor, for example
span_in_symbols = 32  # Filter span in symbols
samples_per_symbol = 4  # Oversampling factor (number of samples per symbol)

# Create a root-raised-cosine filter instance
rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)

# Visualize the impulse response of the filter
rrcf.show("impulse")

# Visualize the magnitude response of the filter in dB scale
rrcf.show("magnitude", "db")

# If you also want to visualize the magnitude response in linear scale
# Uncomment the line below
# rrcf.show("magnitude", "lin")
```

This code will produce plots for both the impulse response and the magnitude response (in dB) of the previously defined root-raised-cosine filter. If you change the `beta`, `span_in_symbols`, or `samples_per_symbol` variables, the filter characteristics will change accordingly, and you will see these changes reflected in the impulse and magnitude response plots.

Remember that `%matplotlib inline` should be invoked if you are running this code in a Jupyter notebook environment to make sure the plots are displayed inline. The context provided indicates that this matplotlib magic command has already been included, hence it is not repeated in the code snippet.

### 3. INSTRUCTION:
Prepare a Python script that sets up the Sionna environment for training models on a CPU instead of a GPU.

**ANSWER:**

To prepare a Python script that sets up the Sionna environment for training models on a CPU instead of a GPU, we can follow the instructions given in the context's "GPU Configuration and Imports" section. The context suggests setting an environment variable "CUDA_VISIBLE_DEVICES" to an empty string to use the CPU instead of a GPU.

Here is the Python script based on the provided context:

```python
# Sionna CPU Environment Setup Script

# Import the os module
import os

# Set the environment variable to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set the TensorFlow logging level to avoid unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna if it has not been imported
try:
    import sionna
except ImportError as e:
    # Install Sionna if the package is not already installed
    os.system("pip install --quiet sionna")
    import sionna

# Import TensorFlow
import tensorflow as tf

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

# Since we are not using GPU, we do not need to configure memory growth
# If you would like to explicitly list CPUs (unnecessary), you could do:
# cpus = tf.config.list_physical_devices('CPU')
# print("CPUs: ", cpus)

# The rest of the script would go here, including any Sionna model setups
# and training processes, but only using CPU resources.

# Example: Import matplotlib and numpy for any visualization or calculation
# Note: This is just for example purposes; this part of code would be replaced with your actual model setup/training
import matplotlib.pyplot as plt
import numpy as np

# Your model setup and training code would replace the following lines
print("Sionna environment is set up to train on CPU.")
# [... Your model training code ...]

# Save the script as 'sionna_cpu_setup.py' or any name you want.

```

Once written, this script can be saved as a .py file (for example, `sionna_cpu_setup.py`), and it can be run using a Python interpreter to set up the Sionna environment for using the CPU for training models.

### 4. INSTRUCTION:
Implement up-sampling in Sionna for a sequence of QAM symbols using the "Upsampling" layer and validate the process by inspecting the first few elements of the up-sampled output.

**ANSWER:**

To implement up-sampling in Sionna for a sequence of QAM symbols using the "Upsampling" layer and validate the process by inspecting the first few elements of the up-sampled output, follow these steps:

1. **Create the Upsampling Layer**:
To start, we create an instance of the `Upsampling` layer with the desired `samples_per_symbol` parameter — the number of samples per symbol (also known as the oversampling factor).

2. **Generate QAM Symbols**:
We will need a sequence of QAM symbols to be up-sampled. These can be obtained using the `QAMSource` layer provided by Sionna which generates QAM symbols based on the specified modulation order.

3. **Perform Upsampling**:
Pass the sequence of QAM symbols through the `Upsampling` layer to obtain the up-sampled sequence.

4. **Inspect the First Few Elements**:
After upsampling, we can inspect the first few samples of the up-sampled sequence to validate the process.

Using the context provided in the Sionna tutorial, the following sample code demonstrates these steps:

```python
import sionna
import numpy as np
import matplotlib.pyplot as plt

# Configure QAM source
num_bits_per_symbol = 4 # The modulation order (16-QAM in this case)
qam = sionna.utils.QAMSource(num_bits_per_symbol)

# Generate batch of QAM symbol sequences
batch_size = 128
num_symbols = 1000
x = qam([batch_size, num_symbols])

samples_per_symbol = 4 # Define the oversampling factor (already defined in your context, you can reuse it)

# Create instance of the Upsampling layer
us = sionna.signal.Upsampling(samples_per_symbol)

# Upsample the QAM symbol sequence
x_us = us(x)

# Inspect the first few elements of one row of the up-sampled sequence
# Display the absolute values of the first few symbols for visualization
num_samples_to_inspect = 20 # Adjust the number of samples as needed
plt.stem(np.abs(x_us)[0, :num_samples_to_inspect])
plt.xlabel(r"Sample index $i$")
plt.ylabel(r"|$x_{us}[i]$|")
plt.title("First few elements of up-sampled sequence")
plt.show()
```

Run this code snippet after the context you have in your notebook to accomplish the tasks in the instruction. Adjust the `num_samples_to_inspect` if you like to inspect more or fewer elements in the up-sampled sequence. The `plt.stem()` function is used for visualizing the magnitude of the complex QAM symbols.

### 5. INSTRUCTION:
Write code to perform matched filtering on a pulse-shaped signal using the same filter initially used for pulse shaping, and downsample the result with the appropriate index using Sionna's "Downsampling" layer.

**ANSWER:**

Based on the given context from the Sionna tutorial on Pulse-shaping Basics, you've already learned how to create a root-raised-cosine (RRC) filter for both pulse-shaping and as a matched filter for recovering QAM symbols, how to upsample a sequence of QAM symbols, pulse-shape it using the RRC filter, and then perform matched filtering and downsampling to retrieve the original QAM symbols. To accomplish the matched filtering and downsampling operation as specified in the instruction, we can use the existing knowledge and Sionna's functionality.

Please note that the provided code must be executed within a Python environment with `sionna` and its dependencies installed.

Here's the Python code to perform matched filtering on a pulse-shaped signal using the root-raised-cosine filter created earlier and to downsample the result using Sionna's `Downsampling` layer:

```python
# Assuming the pulse-shaped signal is available in the variable x_rrcf
# and we have the root-raised-cosine filter `rrcf` created earlier.

# Apply matched filtering (Use the same RRC filter, as matched filtering implies using a filter matched to the pulse-shaping filter)
x_mf = rrcf(x_rrcf)

# The filter delay (in samples) determines the downsample starting point.
# Since the filter has a delay of (L - 1) / 2, where L is the filter length, the matched filter adds another (L - 1) / 2 delay.
# So the total delay is L - 1, which is our downsample index.
filter_delay = rrcf.length - 1

# Instantiate a Downsampling layer with appropriate starting index
ds = Downsampling(samples_per_symbol, filter_delay)

# Downsample the matched filtered signal:
# Since Sionna expects downsampling to occur at the appropriate time alignment,
# this will effectively pick out the peaks of the filtered "pulses", which correspond to the original QAM symbols.
x_hat = ds(x_mf)

# At this point, x_hat contains the downsampled (and matched filtered) signal which should represent the recovered QAM symbols.
```

Remember to insert this code snippet at the appropriate place in your tutorial such that the variable `x_rrcf` contains the pulse-shaped signal and the filter `rrcf` is already created.

Please adapt variable names and possibly surrounding context if the code needs to be inserted in a part of the tutorial different from the provided context.

### 6. INSTRUCTION:
Design a visual comparison of an oversampled QAM sequence before and after pulse shaping and matched filtering, using Sionna's library functions and Python's matplotlib for plotting.

**ANSWER:**

To design the visual comparison using Sionna's library functions and matplotlib, we will follow these steps:

1. Generate a QAM symbol sequence.
2. Create an oversampling filter with a given roll-off factor (betta), span in symbols, and samples per symbol.
3. Apply the pulse-shaping filter to the upsampled QAM sequences.
4. Apply matched filtering using the same pulse-shaping filter to the pulse-shaped sequence.
5. Plot the oversampled and pulse-shaped sequences before and after matched filtering for visual comparison.


Here's how you'd do it in Python:

```python
import matplotlib.pyplot as plt
import numpy as np
import sionna
from sionna.utils import QAMSource
from sionna.signal import Upsampling, Downsampling, RootRaisedCosineFilter

# Define parameters
beta = 0.22 # Roll-off factor
span_in_symbols = 32 # Filter span in symbols
samples_per_symbol = 4 # Number of samples per symbol, i.e., the oversampling factor

# Create a Root-Raised-Cosine Filter
rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)

# Configure QAM source
num_bits_per_symbol = 4 # e.g. 16-QAM
qam = QAMSource(num_bits_per_symbol)

# Generate a batch of QAM symbol sequences
batch_size = 1
num_symbols = 100
x = qam([batch_size, num_symbols])

# Upsample the QAM sequence
us = Upsampling(samples_per_symbol)
x_us = us(x)

# Pulse-shape with RRCF
x_rrcf = rrcf(x_us)

# Apply matched filter
mf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta) # Matched Filter usually the same as pulse shaping filter
x_mf = mf(x_rrcf) # Convolve with matched filter

# Downsampling the matched filtered signal to obtain the symbols
ds = Downsampling(samples_per_symbol, mf.length-1, num_symbols)
x_hat = ds(x_mf) # The estimated symbols

# Plot the original, pulse shaped and matched filtered signals
plt.figure(figsize=(10, 6))

# Indices for plotting
plot_indices = range(0, 5*samples_per_symbol)

plt.subplot(2, 1, 1)
plt.plot(plot_indices, x_us[0, plot_indices], 'ko-', label='Upsampled QAM')
plt.plot(plot_indices, x_rrcf[0, plot_indices], 'ro-', label='After Pulse Shaping')
plt.title("Oversampled QAM Sequence Before and After Pulse Shaping")
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(plot_indices, x_rrcf[0, plot_indices], 'ro-', label='Pulse Shaped')
plt.plot(plot_indices, x_mf[0, plot_indices+mf.length//2], 'bo-', label='After Matched Filtering')
plt.title("Pulse Shaped Sequence Before and After Matched Filtering")
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()
```

In this code:

- We set up the QAM source and generate a sequence of QAM symbols `x`.
- We upsample the QAM sequence using `Upsampling` to get `x_us`.
- We pulse-shape the upsampled QAM sequence with a Root-Raised-Cosine filter (`rrcf`) to get `x_rrcf`.
- We convolve the shaped sequence with a matched filter (`mf`, which is the same as the pulse-shaping filter) to get `x_mf`.
- We downsample to obtain an estimate of the transmitted symbols (`x_hat`).
- We then create subplot 1 to plot the comparison before and after pulse shaping (`x_us` and `x_rrcf`).
- We create subplot 2 to plot the comparison before and after matched filtering (`x_rrcf` and `x_mf`).
- The visual comparison is provided using `matplotlib` plots, showing the effect of pulse shaping and matched filtering in time.

Please make sure that Sionna is properly installed in your Python environment and it's compatible with TensorFlow if you intend to execute this code.

### 7. INSTRUCTION:
Experiment with the Sionna library to adjust the length of a pulse-shaping filter and observe the change in the mean-squared error (MSE) between the transmitted and received QAM symbols.

**ANSWER:**

To experiment with the Sionna library and adjust the length of the pulse-shaping filter to observe the change in the mean-squared error (MSE) between the transmitted and received QAM symbols, follow these steps:

1. **Adjust the Length of the Pulse-Shaping Filter**:
   Update the `span_in_symbols` variable to the desired length.

2. **Observe the Change in MSE**:
   Calculate the mean-squared error (MSE) between the transmitted (`x`) and received (`x_hat`) QAM symbols.

To illustrate these steps in code, let's consider two different lengths for the filter: the original 32 symbols and a longer one, say 64 symbols.

For the original 32-symbol filter, you can observe that the MSE computation looks like this in the provided context:

```python
beta = 0.22 # Roll-off factor
span_in_symbols = 32 # Filter span in symbols
samples_per_symbol = 4 # Number of samples per symbol

# ... (rest of the code that generates QAM symbols and applies filters)

print("MSE between x and x_hat (dB)", 10*np.log10(np.var(x-x_hat)))
```

Repeat the MSE computation but with the `span_in_symbols` set to 64:

```python
beta = 0.22 # Roll-off factor
span_in_symbols = 64 # Filter span in symbols updated
samples_per_symbol = 4 # Number of samples per symbol

# Re-create the pulse-shaping filter with updated span_in_symbols
rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)

# ... (rest of the code for signal processing, as shown in the provided context)
# Generate the QAM sequence and apply the upsample, filter, downsample etc.

# Re-compute MSE with the longer filter
print("MSE between x and x_hat with longer filter (dB)", 10*np.log10(np.var(x-x_hat)))
```

By executing the code above with different values for `span_in_symbols`, you can observe how the MSE changes when the length of the filter changes. You should expect that increasing the filter length will generally result in a lower MSE, meaning the transmitted and received symbols are closer to each other, at the expense of increased computational complexity.

### 8. INSTRUCTION:
Write a Python program using Sionna to calculate and visualize the empirical power spectral density (PSD) of a pulse-shaped signal, defining specific parameters like oversampling factors.

**ANSWER:**

To fulfill the instruction, we need to create a Python program using Sionna to pulse shape a QAM signal and then calculate and visualize its empirical power spectral density (PSD). The oversampling factors will be defined in the parameters. I'll start by installing Sionna (if needed) and creating a program that generates a pulse-shaped QAM signal and calculates the PSD.

```python
# This Python program must be run in an environment where Sionna is installed.
# If Sionna is not installed, install it using: !pip install sionna

import os
import numpy as np
import matplotlib.pyplot as plt
from sionna.utils import QAMSource
from sionna.signal import Upsampling, RootRaisedCosineFilter, empirical_psd, PulseShaping

# Configure the environment for Sionna
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set the simulation parameters
beta = 0.22                     # Roll-off factor
span_in_symbols = 32            # Filter span in symbols
oversampling_factor = 4         # Oversampling factor
num_bits_per_symbol = 4         # The modulation order of the QAM constellation, i.e., 16QAM
batch_size = 128                # Batch size for the number of symbol sequences (waveforms)
num_symbols = 1000              # Number of symbols per sequence

# Create a root-raised-cosine filter
rrcf = RootRaisedCosineFilter(span_in_symbols, oversampling_factor, beta)

# Configure QAM source and generate a batch of QAM symbol sequences
qam = QAMSource(num_bits_per_symbol)  # Layer to generate batches of QAM symbols
x = qam([batch_size, num_symbols])    # Generate symbols

# Create instances of the Upsampling layer and upsampling the QAM symbol sequence
us = Upsampling(oversampling_factor)
x_us = us(x)  # Apply upsampling

# Pulse shaping the upsampled sequence
pulse_shaping = PulseShaping(rrcf)
x_shape = pulse_shaping(x_us)

# Calculate the empirical power spectral density (PSD) of the pulse-shaped signal
f, psd_db = empirical_psd(x_shape.numpy(), oversampling=oversampling_factor, num_averages=100)

# Visualize the empirical power spectral density (PSD)
plt.figure(figsize=(10, 6))
plt.plot(f, psd_db)
plt.title('Empirical Power Spectral Density of the Pulse-shaped Signal')
plt.xlabel('Frequency [1/symbol]')
plt.ylabel('PSD [dB/Hz]')
plt.grid()
plt.show()
```

In this program, we set up the filter and oversampling parameters, create a sequence of QAM symbols, and upsample and pulse shape them using a root-raised-cosine (RRC) filter. After that, we calculate the empirical PSD using the `empirical_psd` function from Sionna and visualize the results.

Remember you need to run this code on an appropriate Python environment where Sionna is installed with additional dependencies such as TensorFlow, Matplotlib, and NumPy. The exact output will depend on the generated QAM symbols and the chosen parameters for the RRC filter and oversampling factor.

### 9. INSTRUCTION:
Utilize Sionna's empirical ACLR function to compute the adjacent channel leakage power ratio (ACLR) of a signal and compare it with the theoretical ACLR provided by the filter object.

**ANSWER:**

To compute and compare the empirical ACLR with the theoretical ACLR of a signal using Sionna's functions, you can follow these steps:

1. Generate a sequence of QAM symbols and upsample them.
2. Apply a pulse-shaping filter (e.g., Root-Raised-Cosine filter) to the upsampled symbols.
3. Compute the empirical ACLR using Sionna's `empirical_aclr` function.
4. Retrieve the theoretical ACLR from the filter object itself using its `.aclr` property.
5. Compare the two ACLR values.

Here is a step-by-step implementation based on the provided context:

```python
# Assume all necessary libraries and the Sionna library are already imported as per the provided context

# Step 1: Define the parameters and create the RRC filter
beta = 0.22 # Roll-off factor
span_in_symbols = 32 # Filter span in symbols
samples_per_symbol = 4 # Number of samples per symbol, i.e., the oversampling factor
rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)

# Step 2: Configure QAM source and generate QAM symbols
num_bits_per_symbol = 4 # 16-QAM
qam = QAMSource(num_bits_per_symbol)
batch_size = 128
num_symbols = 1000
x = qam([batch_size, num_symbols])

# Upsample the QAM symbol sequence
us = Upsampling(samples_per_symbol)
x_us = us(x)

# Step 3: Apply the pulse-shaping filter
x_rrcf = rrcf(x_us)

# Step 4: Compute the empirical ACLR
empirical_aclr_value = empirical_aclr(x_rrcf, oversampling=samples_per_symbol)
empirical_aclr_db = 10 * np.log10(empirical_aclr_value)
print("Empirical ACLR (dB):", empirical_aclr_db)

# Step 5: Retrieve the theoretical ACLR from the filter object
theoretical_aclr_db = 10 * np.log10(rrcf.aclr)
print("Theoretical ACLR (dB):", theoretical_aclr_db)

# Step 6: Compare the empirical ACLR with the theoretical ACLR
# The comparison is simply a printout here, as both values are already printed above.
# For an automated approach, you may define a threshold or simply assess the difference:
aclr_difference_db = np.abs(empirical_aclr_db - theoretical_aclr_db)
print("Difference between Empirical and Theoretical ACLR (dB):", aclr_difference_db)
```

This code follows the instructions based on your provided context by using the Sionna library to handle QAM symbol generation, upsampling, filtering, and ACLR computation. A look at the power spectral density (PSD) might also be of interest to visually assess the in-band and out-of-band power characteristics which contribute to ACLR computation.

### 10. INSTRUCTION:
Modify the roll-off factor of a RootRaisedCosineFilter in Sionna to understand its impact on the ACLR and document the observed changes.

**ANSWER:**

To modify the roll-off factor of a RootRaisedCosineFilter in Sionna and understand its impact on the ACLR (Adjacent Channel Leakage Ratio), you'll need to follow these steps:

1. Change the roll-off factor (`beta`) used to instantiate the filter.
2. Recalculate the impulse and magnitude responses of the filter.
3. Reshape the input QAM symbols and pass them through the updated pulse-shaping filter.
4. Analyze the empirical PSD (Power Spectral Density) and compute the new ACLR.
5. Document the observed changes by comparing the results to the previous roll-off value.

Here is a step-by-step instruction to do that:

1. **Modify the Roll-Off Factor**:
Replace the existing roll-off factor in the notebook with a new value. For example, change it from 0.22 to 0.35 or another value such as 0.1 as suggested in the notebook.

For the purpose of an example, let’s use 0.35 as the new roll-off factor:
```
beta = 0.35  # New roll-off factor
```
Update the filter with this new value.

2. **Recalculate Filter Responses**:
Visualize the new filter's impulse and magnitude responses, just as you did before with the `show` function. This will enable you to observe how the roll-off factor change affects the filter characteristics.
```
rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)
rrcf.show("impulse")
rrcf.show("magnitude", "db")
rrcf.show("magnitude", "lin")
```
3. **Apply Pulse-Shaping**:
With the modified filter, repeat the pulse-shaping process with the QAM symbols. This involves upsampling the symbols and then filtering the sequence with the new Root Raised Cosine Filter.
```
x_rrcf = rrcf(x_us)  # Filter with the new roll-off factor
```
4. **Analyze PSD and Compute ACLR**:
Use the empirical PSD function to visualize the power spectral density of the new pulse-shaped waveform and then calculate the ACLR.
```
empirical_psd(x_rrcf, oversampling=samples_per_symbol, ylim=[-100, 3])
aclr_db = 10*np.log10(empirical_aclr(x_rrcf, oversampling=samples_per_symbol))
print("New Empirical ACLR (db) with beta=", beta, ":", aclr_db)
```
5. **Document Observed Changes**:
Compare the new ACLR value with the previous one (with the original roll-off factor) to understand the impact of changing the roll-off factor. Document your findings by noting down observations like whether the ACLR has improved or worsened, and describe any changes in the PSD plot, such as bandwidth occupation and sidelobe levels.

You should discuss how the roll-off factor affects the balance between time-domain localization (which impacts intersymbol interference) and frequency-domain localization (which impacts ACLR). When you change the roll-off factor, you'll typically find that larger values of beta result in less spectral efficiency (more bandwidth consumption) but result in better time-domain properties; whereas, smaller values of beta result in more spectral efficiency but might introduce more intersymbol interference due to the sharper filter transitions.

Repeat these steps with several different roll-off factors to gain more thorough insights.

### 11. INSTRUCTION:
Demonstrate the application of windowing to a root-raised-cosine filter in Sionna to enhance its spectral characteristics, and compare the impulse and magnitude responses before and after applying the window.

**ANSWER:**

The given context is part of a tutorial notebook for Sionna, a Python library for simulating communication systems. The excerpt from the notebook provides a comprehensive introduction to pulse shaping with root-raised-cosine (RRC) filters, recovery of symbols through matched filtering and downsampling, investigating the adjacent channel leakage power ratio (ACLR), and finally, the application of windowing to improve filter spectral characteristics. Windowing is briefly introduced, demonstrating its effects on the out-of-band attenuation and ACLR.

Now, to demonstrate how to apply windowing to a root-raised-cosine filter in Sionna and analyze its spectral characteristics both before and after applying the window, follow these general steps, extrapolated from the context:

1. **Create an RRC Filter Without Windowing:**
   Initialize a root-raised-cosine filter with desired parameters such as roll-off factor, span in symbols, and samples per symbol. Display its impulse response and ACLR.

```python
beta = 0.22  # Roll-off factor for RRC filtering
span_in_symbols = 32  # How many symbols the filter spans
samples_per_symbol = 4  # Oversampling factor

# Creating the filter without windowing
rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)

# Visualize the impulse response and initial ACLR
rrcf.show("impulse")
rrcf.show("magnitude", "db")  # Logarithmic magnitude response
acrl_without_window = 10 * np.log10(rrcf.aclr)
```

2. **Define the Window and Apply to the RRC Filter:**
   Select a window function and apply it to the filter. One common choice is the Blackman window, which is mentioned in the context. Create a new RRCF with the window applied and visualize the response.

```python
# Define the window type (Blackman, Hann, Hamming, etc.)
window_type = "blackman"

# Create a root-raised-cosine filter with the specified window
rrcf_with_window = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta, window=window_type)

# Show the impulse and magnitude responses of the windowed filter
rrcf_with_window.show("impulse")
rrcf_with_window.show("magnitude", "db")

# Calculate the ACLR with window
aclr_with_window = 10 * np.log10(rrcf_with_window.aclr)
```

3. **Compare Impulse and Magnitude Responses:**
   Plot the impulse and magnitude responses side by side for both the unwindowed and windowed filters.

```python
# Impulse responses comparison
plt.figure()
plt.subplot(1, 2, 1)
rrcf.show("impulse")
plt.title("Impulse Response - No Windowing")
plt.subplot(1, 2, 2)
rrcf_with_window.show("impulse")
plt.title("Impulse Response - With Windowing")
plt.show()

# Magnitude responses comparison
plt.figure()
plt.subplot(1, 2, 1)
rrcf.show("magnitude", "db")
plt.title("Magnitude Response - No Windowing")
plt.subplot(1, 2, 2)
rrcf_with_window.show("magnitude", "db")
plt.title("Magnitude Response - With Windowing")
plt.show()

# Compare and print ACLRs
print("ACLR without window (dB):", acrl_without_window)
print("ACLR with window (dB):", aclr_with_window)
```

By executing these steps in the Sionna environment, you will see how windowing affects both the time domain (impulse response) and frequency domain (magnitude response) characteristics of the root-raised-cosine filter. Typically, windowing smooths out sidelobes in the frequency domain at the expense of a broader main lobe, which may lead to a slight increase in ACLR, as the energy leaks more into adjacent channels.

### 12. INSTRUCTION:
Analyze the trade-offs of using windowing on pulse-shaping filters in terms of out-of-band attenuation and passband broadening, and calculate the difference in ACLR using Sionna's provided metrics.

**ANSWER:**

Based on the given context, which provides an overview of Sionna's tutorial on pulse-shaping filters including their properties, applications, and optimization techniques such as windowing, we can analyze the trade-offs of using windowing on pulse-shaped filters. The main aspects to consider are out-of-band attenuation and passband broadening. Then, using the ACLR metrics provided, we can estimate the difference in ACLR.

### Trade-offs of using windowing on pulse-shaped filters

1. **Out-of-Band Attenuation:**
    - Without Windowing: The out-of-band energy leakage is higher, which can lead to more interference in adjacent frequency channels. This is crucial in communication systems where spectral efficiency and compliance with regulatory standards on spectral emissions are important.
    - With Windowing: The use of windowing, such as a Blackman window, generally improves the suppression of sidelobes in the frequency response, thereby increasing out-of-band attenuation. This results in less interference with adjacent channels.

2. **Passband Broadening:**
    - Without Windowing: The passband of the pulse-shaped filter is narrower, leading to less distortion of the contained signal, at the expense of higher out-of-band emissions.
    - With Windowing: While a window can improve out-of-band rejection, it can also result in passband broadening. Broadening the passband can reduce the filter's selectivity and cause distortion of the desired signal. This is a trade-off against the improved out-of-band rejection.

### ACLR (Adjacent Channel Leakage Ratio) Calculation

The ACLR is a measure of the ratio of the power within the intended communication channel to the power leaked into adjacent channels.

- The context indicates that the ACLR without windowing is given by the `rrcf_short.aclr` property, which in linear scale is -13.982985019683838 dB.
- The ACLR with a Blackman window applied is provided by the `rrcf_short_blackman.aclr` property, which is -12.624131441116333 dB.

To calculate the difference in ACLR due to windowing, we simply subtract the two values:

Difference in ACLR (dB) = ACLR without window - ACLR with window
Difference in ACLR (dB) = (-13.982985019683838) - (-12.624131441116333)
Difference in ACLR (dB) ≈ -1.358853578567505

This calculation shows that the ACLR has worsened by approximately 1.36 dB due to windowing. This aligns with the context provided which indicates that windowing leads to a reduced out-of-band attenuation but an increased ACLR due to passband broadening.

It's important to keep in mind that these metrics are specific to the tutorial setup and parameters such as filter length, oversampling factor, and the shape of the window used. In practical scenarios, the exact values may vary and should be computed considering all relevant details of the system configuration.

