INSTRUCTION: Explain the purpose of the 5G NR module in the Sionna Python package and its primary focus on simulating the physical uplink shared channel (PUSCH).
ANSWER:The 5G NR module in the Sionna Python package focuses on enabling simulations of 5G New Radio (NR) compliant features, with a particular emphasis on the physical uplink shared channel (PUSCH). The PUSCH is an essential component of the 5G NR specifications as it carries user data from mobile devices to the base station. This module incorporates a subset of physical layer functionalities as specified by the 3GPP standards, specifically in the documents 3GPP TS 38.211, 38.212, and 38.214.

Understanding the workings and performance of the PUSCH is crucial for 5G system design, as it impacts the overall network capacity, reliability, and user experience. The Sionna 5G NR module, therefore, provides users with the necessary tools, such as the `PUSCHTransmitter` and `PUSCHReceiver`, to facilitate high-level simulations of PUSCH transmissions. These tools abstract the required processing blocks for PUSCH, allowing users to easily modify them for different simulation needs.

A core feature of the module is the support for the generation of demodulation reference signals (DMRS) needed for PUSCH, as well as channel estimation using least-squares (LS) estimation methods. The `PUSCHDMRSConfig` object, for example, sets parameters related to DMRS generation in various configurations, affecting the number of DMRS positions, length, scrambling identities, and other properties. Proper DMRS configuration is essential for accurate channel estimation and, subsequently, the successful demodulation of data carried by PUSCH.

The code example provided in the context shows a simple simulation setup where a `PUSCHTransmitter` is instantiated with default settings, followed by the creation of a corresponding `PUSCHReceiver`. An additive white gaussian noise (AWGN) channel simulation is then performed using these components. The transmit signals and information bits are generated, passed through the channel, and finally, the receiver attempts to recover the transmitted bits. This simulation flows into a computation of the bit error rate (BER), which offers a common metric for assessing the quality of the transmission and the efficacy of the system design.

In summary, the 5G NR module in the Sionna Python package is specifically aimed at facilitating the simulation of PUSCH within 5G NR physical layer research, providing tools to model, simulate, and evaluate the performance of PUSCH transmissions in a compliant and customizable manner.

INSTRUCTION: Refer to the 3GPP specifications as a foundation for understanding the physical layer functionalities implemented in the Sionna package for 5G NR PUSCH simulations.
ANSWER:To understand the physical layer functionalities implemented in the Sionna package for 5G NR PUSCH simulations, it is important to have a solid grasp of the 3GPP specifications, which serve as the blueprint for these functionalities. The 3GPP (3rd Generation Partnership Project) specifies the technical details of 5G New Radio (NR) systems, including the physical layer protocols and signal processing procedures.

For PUSCH (Physical Uplink Shared Channel) simulations in Sionna, you would particularly refer to the following 3GPP documents:

- [3GPP TS 38.211](https://nvlabs.github.io/sionna/api/nr.html#gpp38211): This technical specification covers the physical channels and modulation. For PUSCH, it defines aspects such as the reference signals used for channel estimation (e.g. DMRS), the time-frequency structure of the uplink shared channel, and the mapping of data to resource elements.
- [3GPP TS 38.212](https://nvlabs.github.io/sionna/api/nr.html#gpp38212): This document describes the multiplexing and channel coding of data and control information for transport channels, including PUSCH. It specifies how data is encoded, rate-matched, and modulated before getting mapped to the PUSCH.
- [3GPP TS 38.214](https://nvlabs.github.io/sionna/api/nr.html#gpp38214): This part of the specification provides information on the physical layer procedures for data, including aspects such as power control, link adaptation, and HARQ processes which are key for PUSCH.

These specifications outline requirements and configurations essential for simulating a compliant PUSCH transmission. For example, they will guide decisions about DMRS configurations which are vital for channel estimation in the receiver.

In the Sionna package, elements such as `PUSCHConfig`, `PUSCHTransmitter`, and `PUSCHReceiver` provide an abstracted framework that conforms to some parts of the 3GPP standards. For instance, `PUSCHConfig` establishes a configuration for the PUSCH simulation, including parameters like symbol allocation length, and `PUSCHDMRSConfig` sets up the specifications for the generation of DMRS as stated in 3GPP TS 38.211 Section 6.4.1.1. This will impact the channel estimation process, which is crucial for the proper demodulation and decoding of the received PUSCH signal.

When a simulation is run, functions such as `PUSCHTransmitter` and `PUSCHReceiver` carry out the transmission and reception processes by generating transmit signals, adding noise through a channel like `AWGN()`, and then recovering the information bits at the receiver. 

The accuracy and efficiency of simulations in Sionna also rely on LS (Least Squares) channel estimation, which is part of the PUSCH simulation package. Components like `PUSCHLSChannelEstimator` are responsible for estimating the channel using LS estimation at pilot positions and interpolating these estimates across the resource grid. This reflects an implementation of the channel estimation procedures as specified in the 3GPP documentation.

By referring to the 3GPP specifications, you can ensure that the functionalities in Sionna capture the intricacies of PUSCH as accurately as possible and that any modifications or extensions to the simulation environment align with 3GPP-defined procedures.

INSTRUCTION: Locate and review the 5G NR PUSCH Tutorial for practical examples on how to use the module's components effectively.
ANSWER:To effectively learn how to use the components of the 5G NR PUSCH module, you should review the provided 5G NR PUSCH Tutorial. The tutorial is the best resource for discovering the intricacies of the module’s components through practical examples. You can access the tutorial by visiting the following URL: ../examples/5G_NR_PUSCH.html.

In this tutorial, you are likely to find detailed explanations and examples that demonstrate how to configure a PUSCH transmission, simulate the uplink shared channel communication, and analyze the performance of the system under various conditions. The tutorial will help you understand key concepts such as PUSCH resource allocation, reference signal configuration, channel estimation, and data recovery.

The context suggests that the tutorial will also walk you through the necessary steps of creating standard-compliant simulations. These steps might include setting up a PUSCH configuration using the `PUSCHConfig` class, transmitting signals using the `PUSCHTransmitter`, receiving signals with the `PUSCHReceiver`, and adding elements like an AWGN channel for the simulation of transmission over a noisy channel.

Furthermore, the tutorial likely covers in detail the roles played by various configurations and estimators, such as `PUSCHDMRSConfig` and `PUSCHLSChannelEstimator`, which are essential for generating demodulation reference signals (DMRS) and for least-squares (LS) channel estimation, respectively.

To take full advantage of the tutorial, review the code snippets provided, and if available, run them in an appropriate environment to see how the simulation behaves. Experiment with different parameters to observe how they affect the transmission and understand the impact on the performance metrics such as Bit Error Rate (BER).

As the tutorial is a practical guide, you should be able to gain hands-on experience with the 5G NR PUSCH module, enabling you to use the components effectively for your simulation needs.

INSTRUCTION: Write a Python script using Sionna to create a PUSCH configuration and instantiate PUSCHTransmitter and PUSCHReceiver objects, simulate an AWGN channel, and calculate the bit error rate (BER).
ANSWER:Here's a Python script using Sionna to create a PUSCH configuration, instantiate PUSCHTransmitter and PUSCHReceiver objects, simulate an AWGN channel, and calculate the bit error rate (BER):

```python
import tensorflow as tf
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver, compute_ber
from sionna.channel import AWGN

# Create a PUSCH configuration with default settings
pusch_config = PUSCHConfig()

# Instantiate a PUSCHTransmitter from the PUSCHConfig
pusch_transmitter = PUSCHTransmitter(pusch_config)

# Create a PUSCHReceiver using the PUSCHTransmitter
pusch_receiver = PUSCHReceiver(pusch_transmitter)

# AWGN channel
channel = AWGN()

# Number of transmissions to simulate
batch_size = 16

# Noise variance
noise_variance = 0.1 

# Generate transmit signal and info bits
x, b = pusch_transmitter(batch_size) 

# Simulate channel impact
y = channel([x, noise_variance]) 

# Recover the info bits
b_hat = pusch_receiver([y, noise_variance]) 

# Compute the bit error rate
ber = compute_ber(b, b_hat)

# Output the BER
print("BER:", ber.numpy())
```

This script configures the PUSCH with default settings, creates the transmitter and receiver objects, simulates the transmission through an AWGN channel, and computes the BER based on the transmitted and received bits. Make sure you have installed Sionna and TensorFlow in your Python environment before running this script.

INSTRUCTION: Modify the default settings of a PUSCHDMRSConfig object and alter properties like `config_type` and `additional_position`.
ANSWER:To modify the default settings of a `PUSCHDMRSConfig` object in a 5G NR simulation environment like that provided by Sionna, one might follow these steps. Please note that the example provided should conform to the simulation's API requirements, and the values for `config_type` and `additional_position` should be chosen based on the simulation's needs and 3GPP specification requirements.

First, instantiate the `PUSCHDMRSConfig` object with default settings:

```python
# Create a PUSCH DMRS configuration object with default settings
dmrs_config = PUSCHDMRSConfig()
```

Next, modify the `config_type` and `additional_position` properties. For example, if you wish to change the DMRS configuration type to 2 and set the maximum number of additional DMRS positions to 1, you would do the following:

```python
# Modify properties of the PUSCH DMRS configuration
dmrs_config.config_type = 2
dmrs_config.additional_position = 1
```

These modifications must adhere to the constraints defined in the Sionna documentation. For `config_type`, the acceptable values are 1 or 2. For `additional_position`, the valid range is 0, 1, 2, or 3.

Remember that any alterations should be done before using the `PUSCHDMRSConfig` object within a simulation run to ensure that the transmitter and receiver are configured with these specifications in place.

If you find that there are specific attributes of the `PUSCHDMRSConfig` that are read-only and cannot be changed after initialization, you would need to set them when you instantiate the object. For example:

```python
# Initialize PUSCHDMRSConfig with non-default settings
dmrs_config = PUSCHDMRSConfig(config_type=2, additional_position=1)
```

It is important to consult the documentation for the particular simulation environment you are using (in this case, Sionna) to ensure that the properties are being modified correctly and are coherent with the underlying model's requirements. The provided example assumes that `config_type` and `additional_position` are properties that can be altered after the instance has been created, which may or may not align with the actual implementation. If upon reviewing the API these are found to be read-only, the latter method of setting them upon instantiation must be applied.

INSTRUCTION: List the properties of a PUSCHDMRSConfig object, including `additional_position`, `allowed_dmrs_ports`, `beta`, `cdm_groups`, and explain their significance in the PUSCH DMRS configuration.
ANSWER:The `PUSCHDMRSConfig` object configures parameters related to the generation of Demodulation Reference Signals (DMRS) for a Physical Uplink Shared Channel (PUSCH) in 5G New Radio (NR) systems. DMRS are crucial for channel estimation at the receiver side, which is necessary for coherent demodulation of the uplink data signal. Here are the properties of a `PUSCHDMRSConfig` object and their significance:

1. `additional_position`: This parameter specifies the maximum number of additional DMRS positions beyond the minimum required. The number of utilized DMRS positions depends on the length of the PUSCH symbol allocation. In the context of DMRS, additional positions may allow for more frequent reference signals, which can enhance channel estimation accuracy, especially in fast-fading or highly mobile scenarios.

2. `allowed_dmrs_ports`: This is a list of nominal antenna ports. The number of allowed antenna ports (`max_num_dmrs_ports`) a user can have depends on the DMRS `config_type` and `length`. This has implications on multiple-input multiple-output (MIMO) configurations, as each port corresponds to a transmit antenna or layer in the MIMO system. Limiting the number of antenna ports impacts the level of spatial multiplexing and beamforming capabilities.

3. `beta`: This is the ratio of PUSCH energy per resource element (EPRE) to DMRS EPRE denoted as $\beta^{DMRS}_{PUSCH}$. It determines the power scaling of DMRS relative to the data-carrying portion of the signal. Correctly setting this parameter can help balance between DMRS visibility for channel estimation and the overall power budget for data transmission.

4. `cdm_groups`: These are lists of Cyclic Delay Diversity (CDD) groups (represented by $\lambda$) for all ports in the `dmrs_port_set`. CDM groups represent groups of DMRS ports that are subjected to different cyclic shifts to achieve signal diversity. The configuration of CDM groups affects the orthogonality of DMRS sequences between the multiple transmitting antenna ports, which is essential for accurate channel estimation and MIMO operation.

5. `allowed_dmrs_ports`, `beta`, `cdm_groups` are read-only properties, meaning their values are derived from other configurations and cannot be directly set by the user.

The significance of these properties revolves around their impact on the reliability of the uplink transmission. By properly configuring DMRS, the user can influence channel estimation quality, robustness against interference and Doppler shift, and the efficiency of uplink transmissions, all of which are cornerstone functionalities for 5G NR performance.

INSTRUCTION: Utilize the PUSCHDMRSConfig properties `config_type`, `deltas`, `dmrs_port_set`, `length`, `n_id`, `n_scid`, `num_cdm_groups_without_data`, `type_a_position`, `w_f`, and `w_t` to configure the DMRS for a PUSCH transmission.
ANSWER:To configure the DMRS for a PUSCH transmission in a 5G NR system using the `PUSCHDMRSConfig`, you need to set the properties given as `config_type`, `deltas`, `dmrs_port_set`, `length`, `n_id`, `n_scid`, `num_cdm_groups_without_data`, `type_a_position`, `w_f`, and `w_t`. Below is an example in Python using these properties, which should be adapted to fit the specifics of your transmission scenario:

```python
# Import the required module
from sionna.nr import PUSCHDMRSConfig

# Create a PUSCHDMRSConfig with specified parameters
dmrs_config = PUSCHDMRSConfig(
    config_type=1, # Choose 1 or 2 depending on the desired configuration type
    deltas=[0,1,2], # Replace with valid delta values for your configuration
    dmrs_port_set=[0,1], # Replace with the DMRS ports being used
    length=1, # Set to 1 for single-symbol or 2 for double-symbol DMRS
    n_id=(12345, 67890), # Replace with a 2-tuple of scrambling identities
    n_scid=0, # Set the scrambling initialization (choose from 0 or 1)
    num_cdm_groups_without_data=2, # Set the number of CDM groups without data
    type_a_position=2, # Set the position of the first DMRS OFDM symbol (2 or 3)
    # For w_f and w_t, which are read-only and matrix form,
    # we assume they are initialized by the PUSCHDMRSConfig automatically.
    # If you have specific weight vectors, the API or documentation will 
    # guide you on how to set them, but from the context they seem to be read-only.
)

# Print DMRS configuration details
print(f"DMRS frequency density: {'6 subcarriers/PRB' if dmrs_config.config_type == 1 else '4 subcarriers/PRB'}")
print(f"Delta shifts: {dmrs_config.deltas}")
print(f"DMRS ports: {dmrs_config.dmrs_port_set}")
print(f"DMRS length: {'Single-symbol' if dmrs_config.length == 1 else 'Double-symbol'}")
print(f"Scrambling IDs: {dmrs_config.n_id}")
print(f"Scrambling initialization: n_scid = {dmrs_config.n_scid}")
print(f"CDM groups without data: {dmrs_config.num_cdm_groups_without_data}")
print(f"Type A position: {dmrs_config.type_a_position}")
# Since w_f and w_t are read-only, they likely need to be accessed
# but not explicitly provided during initialization.
print(f"Frequency weight vectors (w_f): {dmrs_config.w_f}")
print(f"Time weight vectors (w_t): {dmrs_config.w_t}")
```

Please note that some of the provided values are placeholders and should be replaced with values specific to your scenario. Also, the `w_f` and `w_t` properties are marked as read-only in the provided context, which means you may not be able to set them directly during the `PUSCHDMRSConfig` object initialization. These would be determined by the configuration and are likely generated based on other parameter choices. You would typically access these after initializing `PUSCHDMRSConfig` where they are likely used internally by the DMRS generation process.

This is a simplified example, please ensure that you consult the appropriate 5G NR standard specifications for your scenario or the documentation for the library you are using to determine the correct values for your application.

INSTRUCTION: Instantiate the PUSCHLSChannelEstimator class by providing the required arguments such as `resource_grid`, `dmrs_length`, `dmrs_additional_position`, and `num_cdm_groups_without_data`.
ANSWER:To instantiate the `PUSCHLSChannelEstimator` class, we must provide values for the arguments `resource_grid`, `dmrs_length`, `dmrs_additional_position`, and `num_cdm_groups_without_data`. The context provided indicates that these parameters are part of the PUSCH related classes but does not provide specific values for these parameters. Therefore, I will use placeholder values which should be replaced with the actual values according to your specific setup or data.

Here's an example of how you might instantiate the `PUSCHLSChannelEstimator` class:

```python
# Assuming `resource_grid_obj` is an existing instance of a ResourceGrid class that you've created elsewhere in your code
# The other parameters are placeholders and should be replaced with actual values you have

# Placeholder values for required parameters
resource_grid = resource_grid_obj  # replace with your actual ResourceGrid object
dmrs_length = 1                    # replace with actual DMRS length value, possible values are [1, 2]
dmrs_additional_position = 0       # replace with actual additional DMRS positions, possible values are [0, 1, 2, 3]
num_cdm_groups_without_data = 1    # replace with actual CDM groups without data, possible values are [1, 2, 3]

# Instantiate the PUSCHLSChannelEstimator class
pusch_ls_channel_estimator = PUSCHLSChannelEstimator(resource_grid,
                                                      dmrs_length,
                                                      dmrs_additional_position,
                                                      num_cdm_groups_without_data)
```

Please make sure to replace `resource_grid_obj`, `dmrs_length`, `dmrs_additional_position`, and `num_cdm_groups_without_data` with the actual values from your specific configuration or resource grid. The values provided are only placeholders for the sake of the example.

INSTRUCTION: Explain the usage of the `interpolation_type` and `interpolator` arguments in the PUSCHLSChannelEstimator, including how to select and apply different interpolation functions.
ANSWER:The `PUSCHLSChannelEstimator` in the context provided is a class in Sionna, a library for simulating 5G NR compliant features, particularly the Physical Uplink Shared Channel (PUSCH). The class is responsible for performing channel estimation using the Least Squares (LS) method for NR PUSCH transmissions. After the initial LS estimation at the pilot positions (DMRS, or Demodulation Reference Signals), the channel estimates and error variances need to be interpolated across the resource grid. This is where the `interpolation_type` and `interpolator` arguments come into play.

**`interpolation_type` Argument Usage:**
- The `interpolation_type` argument specifies the method of interpolation to be used if the `interpolator` argument is set to `None`. The interpolation method is applied to the estimated channel values at the pilot positions to obtain channel state information for all positions in the resource grid.
- The argument can take on string values indicating the type of interpolation, which include:
  - `"nn"`: This applies Nearest Neighbor Interpolation, meaning each point in the resource grid that lacks a channel estimate will take the value of the nearest pilot's estimate.
  - `"lin"`: This stands for Linear Interpolation, which interpolates channel values linearly between pilot positions.
  - `"lin_time_avg"`: This option indicates Linear Interpolation with time averaging across OFDM symbols, which can help to smooth out the channel estimates over time.

**`interpolator` Argument Usage:**
- The `interpolator` argument allows for the use of a custom interpolator. If an instance of `BaseChannelInterpolator` (or a subclass thereof) is provided, the class will use this for interpolation instead of the method specified by `interpolation_type`.
- This means that if the `interpolator` is not `None`, it will be prioritized for interpolation, and the `interpolation_type` will be ignored.
- Custom interpolators could be more sophisticated, for example, a `LMMSEInterpolator` (Linear Minimum Mean Square Error), that takes into account statistical properties of the channel and noise to yield potentially better channel estimates.

**How to Select and Apply Different Interpolation Functions:**
Selection:
1. Determine the requirements of your simulation: the complexity, computational overhead, and accuracy you need should guide your choice of interpolator.
2. If your requirements are modest and you simply need a quick estimation, "nn" might be sufficient. 
3. If you need more fine-grained interpolation, consider "lin" or "lin_time_avg". 
4. If the built-in interpolators do not meet your requirements, or if you need to consider specific channel conditions and noise characteristics, create a custom interpolator deriving from `BaseChannelInterpolator`.

Application:
1. Create an instance of `PUSCHLSChannelEstimator`, setting the `interpolation_type` argument to one of the strings mentioned above if you wish to use a built-in interpolator.
2. If using a custom interpolator, instantiate it from the appropriate class and pass the instance to `PUSCHLSChannelEstimator` via the `interpolator` argument.
3. Apply the channel estimation by calling the `PUSCHLSChannelEstimator` instance with the observed resource grid (`y`) and noise variance (`no`) as input. It will perform LS channel estimation on the DMRS positions and use the chosen interpolation method to estimate the channel across the entire grid.

In practice, the selection between different interpolation functions typically considers the trade-off between computational complexity and the accuracy of channel estimates. More sophisticated interpolation techniques, such as those considering spatial and temporal correlations in the channel (potentially available through custom `interpolator` instances), might yield more precise estimates but at a higher computational cost. In contrast, simple interpolation methods like nearest neighbor might be less accurate but computationally less demanding.

INSTRUCTION: Use the PUSCHLSChannelEstimator with a given resource grid and noise variance to perform least-squares channel estimation and obtain channel estimates and error variances across the resource grid.
ANSWER:To perform least-squares channel estimation using the PUSCHLSChannelEstimator with a given resource grid and noise variance, you would typically need to follow these steps:

1. Configure the parameters for the demodulation reference signals (DMRS) by creating a `PUSCHDMRSConfig` object. You will need parameters such as `dmrs_length`, `dmrs_additional_position`, and `num_cdm_groups_without_data`.

2. Instantiate the `PUSCHLSChannelEstimator` object with parameters specifying the details of the resource grid and the DMRS configuration. Parameters could include `resource_grid`, `dmrs_length`, `dmrs_additional_position`, `num_cdm_groups_without_data`, any applicable keyword arguments for interpolation (`interpolation_type` and `interpolator`), and the `dtype` if different from the default.

3. Input the observed resource grid `y` and the noise variance `no` to the `PUSCHLSChannelEstimator`. The result will be the channel estimates (represented by matrix `h_ls`) and the channel estimation error variances (represented by matrix `err_var`) across the entire resource grid.

The actual code might look something like this (fill in the <...> with the appropriate parameters based on your simulation setup):

```python
# Assuming `resource_grid` and `noise_variance` are already given,
# and the DMRS configuration parameters are set appropriately.

# Create a PUSCHDMRSConfig instance with the necessary parameters
dmrs_config = PUSCHDMRSConfig(dmrs_length=<dmrs_length>, 
                              dmrs_additional_position=<dmrs_additional_position>, 
                              num_cdm_groups_without_data=<num_cdm_groups_without_data>, 
                              ...)

# Instantiate the PUSCHLSChannelEstimator object with the resource grid
# and the DMRS configuration parameters
pusch_ls_channel_estimator = PUSCHLSChannelEstimator(resource_grid=<resource_grid>, 
                                                     dmrs_length=dmrs_config.dmrs_length, 
                                                     dmrs_additional_position=dmrs_config.dmrs_additional_position, 
                                                     num_cdm_groups_without_data=dmrs_config.num_cdm_groups_without_data, 
                                                     ...)

# Input the observed resource grid and the noise variance to perform channel estimation
h_ls, err_var = pusch_ls_channel_estimator((<y>, <no>))

# `h_ls` are the channel estimates and `err_var` represents the error variances
```

Please note that this code is a general guideline and it assumes that you have already defined the variables `resource_grid`, `noise_variance`, `y`, and `no` with the necessary values. You will also need the appropriate imports and configurations for the above objects to work within your specific simulation environment.


INSTRUCTION: Ensure the correct datatype by setting the `dtype` argument in the PUSCHLSChannelEstimator to `tf.complex64` for internal calculations and output.
ANSWER:To ensure that the `PUSCHLSChannelEstimator` uses `tf.complex64` as the datatype for internal calculations and output, you should set the `dtype` argument to `tf.complex64` when creating an instance of this class.

As per the context provided, it seems that the `PUSCHLSChannelEstimator` class takes `dtype` as one of its parameters, with the default already being set to `tf.complex64`. However, to explicitly ensure this datatype is set, you would initialize `PUSCHLSChannelEstimator` as follows:

```python
pusch_ls_channel_estimator = PUSCHLSChannelEstimator(
    resource_grid=resource_grid,
    dmrs_length=dmrs_length,
    dmrs_additional_position=dmrs_additional_position,
    num_cdm_groups_without_data=num_cdm_groups_without_data,
    interpolation_type=interpolation_type,
    interpolator=interpolator,
    dtype=tf.complex64,  # Ensure the datatype is set to tf.complex64
    **kwargs
)
```

Ensure you replace `resource_grid`, `dmrs_length`, `dmrs_additional_position`, `num_cdm_groups_without_data`, `interpolation_type`, `interpolator`, and `kwargs` with the actual parameters relevant to your specific use case.

If the creation of the `PUSCHLSChannelEstimator` instance is not shown in the provided context but is part of your code, make sure to follow the pattern shown in the example above, passing `tf.complex64` to the `dtype` argument.

