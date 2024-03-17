INSTRUCTION: Can you provide an overview of the complete API structure of the Sionna library?
ANSWER:
# Forward Error Correction (FEC)
## Linear Codes
### Encoder
- LinearEncoder
- AllZeroEncoder
### Decoder
- OSDecoder
## Low-Density Parity-Check (LDPC)
### LDPC Encoder
- LDPC5GEncoder
### LDPC Decoder
- LDPCBPDecoder
- LDPC5GDecoder
## Polar Codes
### Polar Encoding
- Polar5GEncoder
- PolarEncoder
### Polar Decoding
- Polar5GDecoder
- PolarSCDecoder
- PolarSCLDecoder
- PolarBPDecoder
### Polar Utility Functions
- generate_5g_ranking
- generate_polar_transform_mat
- generate_rm_code
- generate_dense_polar
## Convolutional Codes
### Convolutional Encoding
### Viterbi Decoding
### BCJR Decoding
### Convolutional Code Utility Functions
- Trellis
- polynomial_selector
## Turbo Codes
### Turbo Encoding
### Turbo Decoding
### Turbo Utility Functions
- TurboTermination
- polynomial_selector
- puncture_pattern
## Cyclic Redundancy Check (CRC)
- CRCEncoder
- CRCDecoder
## Interleaving
### Interleaver
- RowColumnInterleaver
- RandomInterleaver
- Turbo3GPPInterleaver
### Deinterleaver
## Scrambling
- Scrambler
- TB5GScrambler
- Descrambler
## Utility Functions
- (Binary) Linear Codes
  - load_parity_check_examples
  - alist2mat
  - load_alist
  - generate_reg_ldpc
  - make_systematic
  - gm2pcm
  - pcm2gm
  - verify_gm_pcm
- EXIT Analysis
  - plot_exit_chart
  - get_exit_analytic
  - plot_trajectory
- Miscellaneous
  - GaussianPriorSource
  - bin2int
  - int2bin
  - bin2int_tf
  - int2bin_tf
  - int_mod_2
  - llr2mi
  - j_fun
  - j_fun_inv
  - j_fun_tf
  - j_fun_inv_tf

# Mapping
## Constellations
- Constellation
- qam
- pam
- pam_gray
## Mapper
## Demapping
- Demapper
- DemapperWithPrior
- SymbolDemapper
- SymbolDemapperWithPrior
## Utility Functions
- SymbolLogits2LLRs
- LLRs2SymbolLogits
- SymbolLogits2LLRsWithPrior
- SymbolLogits2Moments
- SymbolInds2Bits
- PAM2QAM
- QAM2PAM

# Channel
## Wireless
### AWGN
### Flat-fading channel
- FlatFadingChannel
- GenerateFlatFadingChannel
- ApplyFlatFadingChannel
- SpatialCorrelation
- KroneckerModel
- PerColumnModel
### Channel model interface
### Time domain channel
- TimeChannel
- GenerateTimeChannel
- ApplyTimeChannel
- cir_to_time_channel
- time_to_ofdm_channel
### Channel with OFDM waveform
- OFDMChannel
- GenerateOFDMChannel
- ApplyOFDMChannel
- cir_to_ofdm_channel
### Rayleigh block fading
### 3GPP 38.901 channel models
- PanelArray
- Antenna
- AntennaArray
- Tapped delay line (TDL)
- Clustered delay line (CDL)
- Urban microcell (UMi)
- Urban macrocell (UMa)
- Rural macrocell (RMa)
### External datasets
### Utility functions
- subcarrier_frequencies
- time_lag_discrete_time_channel
- deg_2_rad
- rad_2_deg
- wrap_angle_0_360
- drop_uts_in_sector
- relocate_uts
- set_3gpp_scenario_parameters
- gen_single_sector_topology
- gen_single_sector_topology_interferers
- exp_corr_mat
- one_ring_corr_mat
## Optical
- Split-step Fourier method
- Erbium-doped fiber amplifier
## Utility functions
- time_frequency_vector
## Discrete
- BinaryMemorylessChannel
- BinarySymmetricChannel
- BinaryErasureChannel
- BinaryZChannel

# Orthogonal Frequency-Division Multiplexing (OFDM)
## Resource Grid
- ResourceGrid
- ResourceGridMapper
- ResourceGridDemapper
- RemoveNulledSubcarriers
## Modulation & Demodulation
- OFDMModulator
- OFDMDemodulator
## Pilot Pattern
- PilotPattern
- EmptyPilotPattern
- KroneckerPilotPattern
## Channel Estimation
- BaseChannelEstimator
- BaseChannelInterpolator
- LSChannelEstimator
- LinearInterpolator
- LMMSEInterpolator
- NearestNeighborInterpolator
- tdl_time_cov_mat
- tdl_freq_cov_mat
## Precoding
- ZFPrecoder
## Equalization
- OFDMEqualizer
- LMMSEEqualizer
- MFEqualizer
- ZFEqualizer
## Detection
- OFDMDetector
- OFDMDetectorWithPrior
- EPDetector
- KBestDetector
- LinearDetector
- MaximumLikelihoodDetector
- MaximumLikelihoodDetectorWithPrior
- MMSEPICDetector

# Multiple-Input Multiple-Output (MIMO)
## Stream Management
## Precoding
- zero_forcing_precoder
## Equalization
- lmmse_equalizer
- mf_equalizer
- zf_equalizer
## Detection
- EPDetector
- KBestDetector
- LinearDetector
- MaximumLikelihoodDetector
- MaximumLikelihoodDetectorWithPrior
- MMSE-PIC
## Utility Functions
- List2LLR
- List2LLRSimple
- complex2real_vector
- real2complex_vector
- complex2real_matrix
- real2complex_matrix
- complex2real_covariance
- real2complex_covariance
- complex2real_channel
- real2complex_channel
- whiten_channel

# 5G NR
## Carrier
- CarrierConfig
## Layer Mapping
- LayerMapper
- LayerDemapper
## PUSCH
- PUSCHConfig
- PUSCHDMRSConfig
- PUSCHLSChannelEstimator
- PUSCHPilotPattern
- PUSCHPrecoder
- PUSCHReceiver
- PUSCHTransmitter
## Transport Block
- TBConfig
- TBEncoder
- TBDecoder
## Utils
- calculate_tb_size
- generate_prng_seq
- select_mcs

# Ray Tracing
## Scene
- Scene
- compute_paths
- trace_paths
- compute_fields
- coverage_map
- load_scene
- preview
- render
- render_to_file
## Example Scenes
- floor_wall
- simple_street_canyon
- etoile
- munich
- simple_wedge
- simple_reflector
- double_reflector
- triple_reflector
- Box
## Paths
- Paths
## Coverage Maps
- CoverageMap
## Cameras
- Camera
## Scene Objects
- SceneObject
## Radio Materials
- RadioMaterial
- ScatteringPattern
## Radio Devices
- Transmitter
- Receiver
## Antenna Arrays
- AntennaArray
- PlanarArray
## Antennas
- Antenna
- compute_gain
- visualize
- dipole_pattern
- hw_dipole_pattern
- iso_pattern
- tr38901_pattern
- polarization_model_1
- polarization_model_2
## Utility Functions
- cross
- dot
- normalize
- phi_hat
- rotate
- rotation_matrix
- rot_mat_from_unit_vecs
- r_hat
- sample_points_on_hemisphere
- theta_hat
- theta_phi_from_unit_vec

# Signal
## Filters
- SincFilter
- RaisedCosineFilter
- RootRaisedCosineFilter
- CustomFilter
- Filter
## Window functions
- HannWindow
- HammingWindow
- BlackmanWindow
- CustomWindow
- Window
## Utility Functions
- convolve
- fft
- ifft
- Upsampling
- Downsampling
- empirical_psd
- empirical_aclr

# Utility Functions
## Metrics
- BitErrorRate
- BitwiseMutualInformation
- compute_ber
- compute_bler
- compute_ser
- count_errors
- count_block_errors
## Tensors
- expand_to_rank
- flatten_dims
- flatten_last_dims
- insert_dims
- split_dims
- matrix_sqrt
- matrix_sqrt_inv
- matrix_inv
- matrix_pinv
## Miscellaneous
- BinarySource
- SymbolSource
- QAMSource
- PAMSource
- PlotBER
- sim_ber
- ebnodb2no
- hard_decisions
- plot_ber
- complex_normal
- log2
- log10
