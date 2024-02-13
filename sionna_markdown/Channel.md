
# Channel<a class="headerlink" href="https://nvlabs.github.io/sionna/#channel" title="Permalink to this headline"></a>

This module provides layers and functions that implement channel models for <a class="reference external" href="https://nvlabs.github.io/sionna/channel.wireless.html">wireless</a> and <a class="reference external" href="https://nvlabs.github.io/sionna/channel.optical.html">optical</a> communications.

- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html">Wireless</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#awgn">AWGN</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#flat-fading-channel">Flat-fading channel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#flatfadingchannel">FlatFadingChannel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#generateflatfadingchannel">GenerateFlatFadingChannel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#applyflatfadingchannel">ApplyFlatFadingChannel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#spatialcorrelation">SpatialCorrelation</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#kroneckermodel">KroneckerModel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#percolumnmodel">PerColumnModel</a>


- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#channel-model-interface">Channel model interface</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#time-domain-channel">Time domain channel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#timechannel">TimeChannel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#generatetimechannel">GenerateTimeChannel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#applytimechannel">ApplyTimeChannel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#cir-to-time-channel">cir_to_time_channel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#time-to-ofdm-channel">time_to_ofdm_channel</a>


- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#channel-with-ofdm-waveform">Channel with OFDM waveform</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#ofdmchannel">OFDMChannel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#generateofdmchannel">GenerateOFDMChannel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#applyofdmchannel">ApplyOFDMChannel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#cir-to-ofdm-channel">cir_to_ofdm_channel</a>


- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#rayleigh-block-fading">Rayleigh block fading</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#gpp-38-901-channel-models">3GPP 38.901 channel models</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#panelarray">PanelArray</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#antenna">Antenna</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#antennaarray">AntennaArray</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#tapped-delay-line-tdl">Tapped delay line (TDL)</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#clustered-delay-line-cdl">Clustered delay line (CDL)</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#urban-microcell-umi">Urban microcell (UMi)</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#urban-macrocell-uma">Urban macrocell (UMa)</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#rural-macrocell-rma">Rural macrocell (RMa)</a>


- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#external-datasets">External datasets</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#utility-functions">Utility functions</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#subcarrier-frequencies">subcarrier_frequencies</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#time-lag-discrete-time-channel">time_lag_discrete_time_channel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#deg-2-rad">deg_2_rad</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#rad-2-deg">rad_2_deg</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#wrap-angle-0-360">wrap_angle_0_360</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#drop-uts-in-sector">drop_uts_in_sector</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#relocate-uts">relocate_uts</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#set-3gpp-scenario-parameters">set_3gpp_scenario_parameters</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#gen-single-sector-topology">gen_single_sector_topology</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#gen-single-sector-topology-interferers">gen_single_sector_topology_interferers</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#exp-corr-mat">exp_corr_mat</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.wireless.html#one-ring-corr-mat">one_ring_corr_mat</a>




- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.optical.html">Optical</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.optical.html#split-step-fourier-method">Split-step Fourier method</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.optical.html#erbium-doped-fiber-amplifier">Erbium-doped fiber amplifier</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.optical.html#utility-functions">Utility functions</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.optical.html#time-frequency-vector">time_frequency_vector</a>




- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.discrete.html">Discrete</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.discrete.html#binarymemorylesschannel">BinaryMemorylessChannel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.discrete.html#binarysymmetricchannel">BinarySymmetricChannel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.discrete.html#binaryerasurechannel">BinaryErasureChannel</a>
- <a class="reference internal" href="https://nvlabs.github.io/sionna/channel.discrete.html#binaryzchannel">BinaryZChannel</a>

