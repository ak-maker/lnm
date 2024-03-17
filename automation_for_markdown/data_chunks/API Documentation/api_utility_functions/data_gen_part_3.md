# Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#utility-functions" title="Permalink to this headline"></a>
    
The utilities sub-package of the Sionna library contains many convenience
functions as well as extensions to existing TensorFlow functions.

# Table of Content
## Miscellaneous<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#miscellaneous" title="Permalink to this headline"></a>
### PlotBER<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#plotber" title="Permalink to this headline"></a>
### sim_ber<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sim-ber" title="Permalink to this headline"></a>
### ebnodb2no<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#ebnodb2no" title="Permalink to this headline"></a>
  
  

### PlotBER<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#plotber" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.utils.plotting.``PlotBER`(<em class="sig-param">`title``=``'Bit/Block` `Error` `Rate'`</em>)<a class="reference internal" href="../_modules/sionna/utils/plotting.html#PlotBER">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER" title="Permalink to this definition"></a>
    
Provides a plotting object to simulate and store BER/BLER curves.
Parameters
    
**title** (<em>str</em>) – A string defining the title of the figure. Defaults to
<cite>“Bit/Block Error Rate”</cite>.

Input
 
- **snr_db** (<em>float</em>) – Python array (or list of Python arrays) of additional SNR values to be
plotted.
- **ber** (<em>float</em>) – Python array (or list of Python arrays) of additional BERs
corresponding to `snr_db`.
- **legend** (<em>str</em>) – String (or list of strings) of legends entries.
- **is_bler** (<em>bool</em>) – A boolean (or list of booleans) defaults to False.
If True, `ber` will be interpreted as BLER.
- **show_ber** (<em>bool</em>) – A boolean defaults to True. If True, BER curves will be plotted.
- **show_bler** (<em>bool</em>) – A boolean defaults to True. If True, BLER curves will be plotted.
- **xlim** (<em>tuple of floats</em>) – Defaults to None. A tuple of two floats defining x-axis limits.
- **ylim** (<em>tuple of floats</em>) – Defaults to None. A tuple of two floats defining y-axis limits.
- **save_fig** (<em>bool</em>) – A boolean defaults to False. If True, the figure
is saved as file.
- **path** (<em>str</em>) – A string defining where to save the figure (if `save_fig`
is True).




`add`(<em class="sig-param">`ebno_db`</em>, <em class="sig-param">`ber`</em>, <em class="sig-param">`is_bler``=``False`</em>, <em class="sig-param">`legend``=``''`</em>)<a class="reference internal" href="../_modules/sionna/utils/plotting.html#PlotBER.add">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.add" title="Permalink to this definition"></a>
    
Add static reference curves.
Input
 
- **ebno_db** (<em>float</em>) – Python array or list of floats defining the SNR points.
- **ber** (<em>float</em>) – Python array or list of floats defining the BER corresponding
to each SNR point.
- **is_bler** (<em>bool</em>) – A boolean defaults to False. If True, `ber` is interpreted as
BLER.
- **legend** (<em>str</em>) – A string defining the text of the legend entry.





<em class="property">`property` </em>`ber`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.ber" title="Permalink to this definition"></a>
    
List containing all stored BER curves.


<em class="property">`property` </em>`is_bler`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.is_bler" title="Permalink to this definition"></a>
    
List of booleans indicating if ber shall be interpreted as BLER.


<em class="property">`property` </em>`legend`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.legend" title="Permalink to this definition"></a>
    
List containing all stored legend entries curves.


`remove`(<em class="sig-param">`idx``=``-` `1`</em>)<a class="reference internal" href="../_modules/sionna/utils/plotting.html#PlotBER.remove">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.remove" title="Permalink to this definition"></a>
    
Remove curve with index `idx`.
Input
    
**idx** (<em>int</em>) – An integer defining the index of the dataset that should
be removed. Negative indexing is possible.




`reset`()<a class="reference internal" href="../_modules/sionna/utils/plotting.html#PlotBER.reset">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.reset" title="Permalink to this definition"></a>
    
Remove all internal data.


`simulate`(<em class="sig-param">`mc_fun`</em>, <em class="sig-param">`ebno_dbs`</em>, <em class="sig-param">`batch_size`</em>, <em class="sig-param">`max_mc_iter`</em>, <em class="sig-param">`legend``=``''`</em>, <em class="sig-param">`add_ber``=``True`</em>, <em class="sig-param">`add_bler``=``False`</em>, <em class="sig-param">`soft_estimates``=``False`</em>, <em class="sig-param">`num_target_bit_errors``=``None`</em>, <em class="sig-param">`num_target_block_errors``=``None`</em>, <em class="sig-param">`target_ber``=``None`</em>, <em class="sig-param">`target_bler``=``None`</em>, <em class="sig-param">`early_stop``=``True`</em>, <em class="sig-param">`graph_mode``=``None`</em>, <em class="sig-param">`distribute``=``None`</em>, <em class="sig-param">`add_results``=``True`</em>, <em class="sig-param">`forward_keyboard_interrupt``=``True`</em>, <em class="sig-param">`show_fig``=``True`</em>, <em class="sig-param">`verbose``=``True`</em>)<a class="reference internal" href="../_modules/sionna/utils/plotting.html#PlotBER.simulate">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.simulate" title="Permalink to this definition"></a>
    
Simulate BER/BLER curves for given Keras model and saves the results.
    
Internally calls <a class="reference internal" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.sim_ber" title="sionna.utils.sim_ber">`sionna.utils.sim_ber`</a>.
Input
 
- **mc_fun** – Callable that yields the transmitted bits <cite>b</cite> and the
receiver’s estimate <cite>b_hat</cite> for a given `batch_size` and
`ebno_db`. If `soft_estimates` is True, b_hat is interpreted as
logit.
- **ebno_dbs** (<em>ndarray of floats</em>) – SNR points to be evaluated.
- **batch_size** (<em>tf.int32</em>) – Batch-size for evaluation.
- **max_mc_iter** (<em>int</em>) – Max. number of Monte-Carlo iterations per SNR point.
- **legend** (<em>str</em>) – Name to appear in legend.
- **add_ber** (<em>bool</em>) – Defaults to True. Indicate if BER should be added to plot.
- **add_bler** (<em>bool</em>) – Defaults to False. Indicate if BLER should be added
to plot.
- **soft_estimates** (<em>bool</em>) – A boolean, defaults to False. If True, `b_hat`
is interpreted as logit and additional hard-decision is applied
internally.
- **num_target_bit_errors** (<em>int</em>) – Target number of bit errors per SNR point until the simulation
stops.
- **num_target_block_errors** (<em>int</em>) – Target number of block errors per SNR point until the simulation
stops.
- **target_ber** (<em>tf.float32</em>) – Defaults to <cite>None</cite>. The simulation stops after the first SNR point
which achieves a lower bit error rate as specified by
`target_ber`. This requires `early_stop` to be <cite>True</cite>.
- **target_bler** (<em>tf.float32</em>) – Defaults to <cite>None</cite>. The simulation stops after the first SNR point
which achieves a lower block error rate as specified by
`target_bler`.  This requires `early_stop` to be <cite>True</cite>.
- **early_stop** (<em>bool</em>) – A boolean defaults to True. If True, the simulation stops after the
first error-free SNR point (i.e., no error occurred after
`max_mc_iter` Monte-Carlo iterations).
- **graph_mode** (<em>One of [“graph”, “xla”], str</em>) – A string describing the execution mode of `mc_fun`.
Defaults to <cite>None</cite>. In this case, `mc_fun` is executed as is.
- **distribute** (<cite>None</cite> (default) | “all” | list of indices | <cite>tf.distribute.strategy</cite>) – Distributes simulation on multiple parallel devices. If <cite>None</cite>,
multi-device simulations are deactivated. If “all”, the workload
will be automatically distributed across all available GPUs via the
<cite>tf.distribute.MirroredStrategy</cite>.
If an explicit list of indices is provided, only the GPUs with the
given indices will be used. Alternatively, a custom
<cite>tf.distribute.strategy</cite> can be provided. Note that the same
<cite>batch_size</cite> will be used for all GPUs in parallel, but the number
of Monte-Carlo iterations `max_mc_iter` will be scaled by the
number of devices such that the same number of total samples is
simulated. However, all stopping conditions are still in-place
which can cause slight differences in the total number of simulated
samples.
- **add_results** (<em>bool</em>) – Defaults to True. If True, the simulation results will be appended
to the internal list of results.
- **show_fig** (<em>bool</em>) – Defaults to True. If True, a BER figure will be plotted.
- **verbose** (<em>bool</em>) – A boolean defaults to True. If True, the current progress will be
printed.
- **forward_keyboard_interrupt** (<em>bool</em>) – A boolean defaults to True. If False, <cite>KeyboardInterrupts</cite> will be
catched internally and not forwarded (e.g., will not stop outer
loops). If False, the simulation ends and returns the intermediate
simulation results.


Output
 
- **(ber, bler)** – Tuple:
- **ber** (<em>float</em>) – The simulated bit-error rate.
- **bler** (<em>float</em>) – The simulated block-error rate.





<em class="property">`property` </em>`snr`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.snr" title="Permalink to this definition"></a>
    
List containing all stored SNR curves.


<em class="property">`property` </em>`title`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.plotting.PlotBER.title" title="Permalink to this definition"></a>
    
Title of the plot.


### sim_ber<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sim-ber" title="Permalink to this headline"></a>

`sionna.utils.``sim_ber`(<em class="sig-param">`mc_fun`</em>, <em class="sig-param">`ebno_dbs`</em>, <em class="sig-param">`batch_size`</em>, <em class="sig-param">`max_mc_iter`</em>, <em class="sig-param">`soft_estimates``=``False`</em>, <em class="sig-param">`num_target_bit_errors``=``None`</em>, <em class="sig-param">`num_target_block_errors``=``None`</em>, <em class="sig-param">`target_ber``=``None`</em>, <em class="sig-param">`target_bler``=``None`</em>, <em class="sig-param">`early_stop``=``True`</em>, <em class="sig-param">`graph_mode``=``None`</em>, <em class="sig-param">`distribute``=``None`</em>, <em class="sig-param">`verbose``=``True`</em>, <em class="sig-param">`forward_keyboard_interrupt``=``True`</em>, <em class="sig-param">`callback``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#sim_ber">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.sim_ber" title="Permalink to this definition"></a>
    
Simulates until target number of errors is reached and returns BER/BLER.
    
The simulation continues with the next SNR point if either
`num_target_bit_errors` bit errors or `num_target_block_errors` block
errors is achieved. Further, it continues with the next SNR point after
`max_mc_iter` batches of size `batch_size` have been simulated.
Early stopping allows to stop the simulation after the first error-free SNR
point or after reaching a certain `target_ber` or `target_bler`.
Input
 
- **mc_fun** (<em>callable</em>) – Callable that yields the transmitted bits <cite>b</cite> and the
receiver’s estimate <cite>b_hat</cite> for a given `batch_size` and
`ebno_db`. If `soft_estimates` is True, <cite>b_hat</cite> is interpreted as
logit.
- **ebno_dbs** (<em>tf.float32</em>) – A tensor containing SNR points to be evaluated.
- **batch_size** (<em>tf.int32</em>) – Batch-size for evaluation.
- **max_mc_iter** (<em>tf.int32</em>) – Maximum number of Monte-Carlo iterations per SNR point.
- **soft_estimates** (<em>bool</em>) – A boolean, defaults to <cite>False</cite>. If <cite>True</cite>, <cite>b_hat</cite>
is interpreted as logit and an additional hard-decision is applied
internally.
- **num_target_bit_errors** (<em>tf.int32</em>) – Defaults to <cite>None</cite>. Target number of bit errors per SNR point until
the simulation continues to next SNR point.
- **num_target_block_errors** (<em>tf.int32</em>) – Defaults to <cite>None</cite>. Target number of block errors per SNR point
until the simulation continues
- **target_ber** (<em>tf.float32</em>) – Defaults to <cite>None</cite>. The simulation stops after the first SNR point
which achieves a lower bit error rate as specified by `target_ber`.
This requires `early_stop` to be <cite>True</cite>.
- **target_bler** (<em>tf.float32</em>) – Defaults to <cite>None</cite>. The simulation stops after the first SNR point
which achieves a lower block error rate as specified by `target_bler`.
This requires `early_stop` to be <cite>True</cite>.
- **early_stop** (<em>bool</em>) – A boolean defaults to <cite>True</cite>. If <cite>True</cite>, the simulation stops after the
first error-free SNR point (i.e., no error occurred after
`max_mc_iter` Monte-Carlo iterations).
- **graph_mode** (<em>One of [“graph”, “xla”], str</em>) – A string describing the execution mode of `mc_fun`.
Defaults to <cite>None</cite>. In this case, `mc_fun` is executed as is.
- **distribute** (<cite>None</cite> (default) | “all” | list of indices | <cite>tf.distribute.strategy</cite>) – Distributes simulation on multiple parallel devices. If <cite>None</cite>,
multi-device simulations are deactivated. If “all”, the workload will
be automatically distributed across all available GPUs via the
<cite>tf.distribute.MirroredStrategy</cite>.
If an explicit list of indices is provided, only the GPUs with the given
indices will be used. Alternatively, a custom <cite>tf.distribute.strategy</cite>
can be provided. Note that the same <cite>batch_size</cite> will be
used for all GPUs in parallel, but the number of Monte-Carlo iterations
`max_mc_iter` will be scaled by the number of devices such that the
same number of total samples is simulated. However, all stopping
conditions are still in-place which can cause slight differences in the
total number of simulated samples.
- **verbose** (<em>bool</em>) – A boolean defaults to <cite>True</cite>. If <cite>True</cite>, the current progress will be
printed.
- **forward_keyboard_interrupt** (<em>bool</em>) – A boolean defaults to <cite>True</cite>. If <cite>False</cite>, KeyboardInterrupts will be
catched internally and not forwarded (e.g., will not stop outer loops).
If <cite>False</cite>, the simulation ends and returns the intermediate simulation
results.
- **callback** (<cite>None</cite> (default) | callable) – If specified, `callback` will be called after each Monte-Carlo step.
Can be used for logging or advanced early stopping. Input signature of
`callback` must match <cite>callback(mc_iter, snr_idx, ebno_dbs,
bit_errors, block_errors, nb_bits, nb_blocks)</cite> where `mc_iter`
denotes the number of processed batches for the current SNR point,
`snr_idx` is the index of the current SNR point, `ebno_dbs` is the
vector of all SNR points to be evaluated, `bit_errors` the vector of
number of bit errors for each SNR point, `block_errors` the vector of
number of block errors, `nb_bits` the vector of number of simulated
bits, `nb_blocks` the vector of number of simulated blocks,
respectively. If `callable` returns <cite>sim_ber.CALLBACK_NEXT_SNR</cite>, early
stopping is detected and the simulation will continue with the
next SNR point. If `callable` returns
<cite>sim_ber.CALLBACK_STOP</cite>, the simulation is stopped
immediately. For <cite>sim_ber.CALLBACK_CONTINUE</cite> continues with
the simulation.
- **dtype** (<em>tf.complex64</em>) – Datatype of the callable `mc_fun` to be used as input/output.


Output
 
- **(ber, bler)** – Tuple:
- **ber** (<em>tf.float32</em>) – The bit-error rate.
- **bler** (<em>tf.float32</em>) – The block-error rate.


Raises
 
- **AssertionError** – If `soft_estimates` is not bool.
- **AssertionError** – If `dtype` is not <cite>tf.complex</cite>.




**Note**
    
This function is implemented based on tensors to allow
full compatibility with tf.function(). However, to run simulations
in graph mode, the provided `mc_fun` must use the <cite>@tf.function()</cite>
decorator.

### ebnodb2no<a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#ebnodb2no" title="Permalink to this headline"></a>

`sionna.utils.``ebnodb2no`(<em class="sig-param">`ebno_db`</em>, <em class="sig-param">`num_bits_per_symbol`</em>, <em class="sig-param">`coderate`</em>, <em class="sig-param">`resource_grid``=``None`</em>)<a class="reference internal" href="../_modules/sionna/utils/misc.html#ebnodb2no">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/utils.html#sionna.utils.ebnodb2no" title="Permalink to this definition"></a>
    
Compute the noise variance <cite>No</cite> for a given <cite>Eb/No</cite> in dB.
    
The function takes into account the number of coded bits per constellation
symbol, the coderate, as well as possible additional overheads related to
OFDM transmissions, such as the cyclic prefix and pilots.
    
The value of <cite>No</cite> is computed according to the following expression

$$
N_o = \left(\frac{E_b}{N_o} \frac{r M}{E_s}\right)^{-1}
$$
    
where $2^M$ is the constellation size, i.e., $M$ is the
average number of coded bits per constellation symbol,
$E_s=1$ is the average energy per constellation per symbol,
$r\in(0,1]$ is the coderate,
$E_b$ is the energy per information bit,
and $N_o$ is the noise power spectral density.
For OFDM transmissions, $E_s$ is scaled
according to the ratio between the total number of resource elements in
a resource grid with non-zero energy and the number
of resource elements used for data transmission. Also the additionally
transmitted energy during the cyclic prefix is taken into account, as
well as the number of transmitted streams per transmitter.
Input
 
- **ebno_db** (<em>float</em>) – The <cite>Eb/No</cite> value in dB.
- **num_bits_per_symbol** (<em>int</em>) – The number of bits per symbol.
- **coderate** (<em>float</em>) – The coderate used.
- **resource_grid** (<em>ResourceGrid</em>) – An (optional) instance of <a class="reference internal" href="ofdm.html#sionna.ofdm.ResourceGrid" title="sionna.ofdm.ResourceGrid">`ResourceGrid`</a>
for OFDM transmissions.


Output
    
<em>float</em> – The value of $N_o$ in linear scale.



