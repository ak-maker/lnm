# Discover Sionna<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Discover-Sionna" title="Permalink to this headline"></a>
    
This example notebook will guide you through the basic principles and illustrates the key features of <a class="reference external" href="https://nvlabs.github.io/sionna">Sionna</a>. With only a few commands, you can simulate the PHY-layer link-level performance for many 5G-compliant components, including easy visualization of the results.

# Table of Content
## Bit-Error Rate (BER) Monte-Carlo Simulations<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Bit-Error-Rate-(BER)-Monte-Carlo-Simulations" title="Permalink to this headline"></a>
## Conclusion<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Conclusion" title="Permalink to this headline"></a>
  
  

## Bit-Error Rate (BER) Monte-Carlo Simulations<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Bit-Error-Rate-(BER)-Monte-Carlo-Simulations" title="Permalink to this headline"></a>
    
Monte-Carlo simulations are omnipresent in todays communications research and development. Due its performant implementation, Sionna can be directly used to simulate BER at a performance that competes with compiled languages – but still keeps the flexibility of a script language.

```python
[19]:
```

```python
ebno_dbs = np.arange(0, 15, 1.)
batch_size = 200 # reduce in case you receive an out-of-memory (OOM) error
max_mc_iter = 1000 # max number of Monte-Carlo iterations before going to next SNR point
num_target_block_errors = 500 # continue with next SNR point after target number of block errors
# we use the built-in ber simulator function from Sionna which uses and early stop after reaching num_target_errors
sionna.config.xla_compat=True
ber_mc,_ = sionna.utils.sim_ber(run_graph_xla, # you can also evaluate the model directly
                                ebno_dbs,
                                batch_size=batch_size,
                                num_target_block_errors=num_target_block_errors,
                                max_mc_iter=max_mc_iter,
                                verbose=True) # print status and summary
sionna.config.xla_compat=False
```


```python
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 3.4157e-01 | 1.0000e+00 |     1259148 |     3686400 |          600 |         600 |         0.1 |reached target block errors
      1.0 | 3.1979e-01 | 1.0000e+00 |     1178870 |     3686400 |          600 |         600 |         0.1 |reached target block errors
      2.0 | 2.9844e-01 | 1.0000e+00 |     1100177 |     3686400 |          600 |         600 |         0.1 |reached target block errors
      3.0 | 2.7401e-01 | 1.0000e+00 |     1010102 |     3686400 |          600 |         600 |         0.1 |reached target block errors
      4.0 | 2.4763e-01 | 1.0000e+00 |      912849 |     3686400 |          600 |         600 |         0.1 |reached target block errors
      5.0 | 2.2038e-01 | 1.0000e+00 |      812407 |     3686400 |          600 |         600 |         0.1 |reached target block errors
      6.0 | 1.8646e-01 | 1.0000e+00 |      687378 |     3686400 |          600 |         600 |         0.1 |reached target block errors
      7.0 | 1.1909e-01 | 9.9000e-01 |      439008 |     3686400 |          594 |         600 |         0.1 |reached target block errors
      8.0 | 4.1536e-02 | 4.7667e-01 |      306236 |     7372800 |          572 |        1200 |         0.2 |reached target block errors
      9.0 | 1.2096e-02 | 1.4028e-01 |      267553 |    22118400 |          505 |        3600 |         0.6 |reached target block errors
     10.0 | 3.2914e-03 | 3.5278e-02 |      291203 |    88473600 |          508 |       14400 |         2.5 |reached target block errors
     11.0 | 9.5878e-04 | 9.8814e-03 |      298073 |   310886400 |          500 |       50600 |         8.5 |reached target block errors
     12.0 | 2.6973e-04 | 2.7933e-03 |      296647 |  1099776000 |          500 |      179000 |        29.6 |reached target block errors
     13.0 | 9.2277e-05 | 9.6000e-04 |      113390 |  1228800000 |          192 |      200000 |        32.9 |reached max iter
     14.0 | 3.3341e-05 | 3.8000e-04 |       40970 |  1228800000 |           76 |      200000 |        32.6 |reached max iter
```

    
Let’s look at the results.

```python
[20]:
```

```python
sionna.utils.plotting.plot_ber(ebno_dbs,
                               ber_mc,
                               legend="E2E Model",
                               ylabel="Coded BER");
```

## Conclusion<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Conclusion" title="Permalink to this headline"></a>
    
We hope you are excited about Sionna - there is much more to be discovered:
 
- TensorBoard debugging available
- Scaling to multi-GPU simulation is simple
- See the <a class="reference external" href="https://nvlabs.github.io/sionna/tutorials.html">available tutorials</a> for more advanced examples.

    
And if something is still missing - the project is <a class="reference external" href="https://github.com/nvlabs/sionna/">open-source</a>: you can modify, add, and extend any component at any time.