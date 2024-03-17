# Discover Sionna<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#Discover-Sionna" title="Permalink to this headline"></a>
    
This example notebook will guide you through the basic principles and illustrates the key features of <a class="reference external" href="https://nvlabs.github.io/sionna">Sionna</a>. With only a few commands, you can simulate the PHY-layer link-level performance for many 5G-compliant components, including easy visualization of the results.

# Table of Content
## First Link-level Simulation<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#First-Link-level-Simulation" title="Permalink to this headline"></a>
  
  

## First Link-level Simulation<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Discover_Sionna.html#First-Link-level-Simulation" title="Permalink to this headline"></a>
    
We can already build powerful code with a few simple commands.
    
As mentioned earlier, Sionna aims at hiding system complexity into Keras layers. However, we still want to provide as much flexibility as possible. Thus, most layers have several choices of init parameters, but often the default choice is a good start.
    
**Tip**: the <a class="reference external" href="https://nvlabs.github.io/sionna/api/sionna.html">API documentation</a> provides many helpful references and implementation details.

```python
[10]:
```

```python
# system parameters
n_ldpc = 500 # LDPC codeword length
k_ldpc = 250 # number of info bits per LDPC codeword
coderate = k_ldpc / n_ldpc
num_bits_per_symbol = 4 # number of bits mapped to one symbol (cf. QAM)
```

    
Often, several different algorithms are implemented, e.g., the demapper supports <em>“true app”</em> demapping, but also <em>“max-log”</em> demapping.
    
The check-node (CN) update function of the LDPC BP decoder also supports multiple algorithms.

```python
[11]:
```

```python
demapping_method = "app" # try "max-log"
ldpc_cn_type = "boxplus" # try also "minsum"
```

    
Let us initialize all required components for the given system parameters.

```python
[12]:
```

```python
binary_source = sionna.utils.BinarySource()
encoder = sionna.fec.ldpc.encoding.LDPC5GEncoder(k_ldpc, n_ldpc)
constellation = sionna.mapping.Constellation("qam", num_bits_per_symbol)
mapper = sionna.mapping.Mapper(constellation=constellation)
channel = sionna.channel.AWGN()
demapper = sionna.mapping.Demapper(demapping_method,
                                   constellation=constellation)
decoder = sionna.fec.ldpc.decoding.LDPC5GDecoder(encoder,
                                                 hard_out=True, cn_type=ldpc_cn_type,
                                                 num_iter=20)
```

    
We can now run the code in <em>eager mode</em>. This allows us to modify the structure at any time - you can try a different `batch_size` or a different SNR `ebno_db`.

```python
[13]:
```

```python
# simulation parameters
batch_size = 1000
ebno_db = 4
# Generate a batch of random bit vectors
b = binary_source([batch_size, k_ldpc])
# Encode the bits using 5G LDPC code
print("Shape before encoding: ", b.shape)
c = encoder(b)
print("Shape after encoding: ", c.shape)
# Map bits to constellation symbols
x = mapper(c)
print("Shape after mapping: ", x.shape)
# Transmit over an AWGN channel at SNR 'ebno_db'
no = sionna.utils.ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
y = channel([x, no])
print("Shape after channel: ", y.shape)
# Demap to LLRs
llr = demapper([y, no])
print("Shape after demapping: ", llr.shape)
# LDPC decoding using 20 BP iterations
b_hat = decoder(llr)
print("Shape after decoding: ", b_hat.shape)
# calculate BERs
c_hat = tf.cast(tf.less(0.0, llr), tf.float32) # hard-decided bits before dec.
ber_uncoded = sionna.utils.metrics.compute_ber(c, c_hat)
ber_coded = sionna.utils.metrics.compute_ber(b, b_hat)
print("BER uncoded = {:.3f} at EbNo = {:.1f} dB".format(ber_uncoded, ebno_db))
print("BER after decoding = {:.3f} at EbNo = {:.1f} dB".format(ber_coded, ebno_db))
print("In total {} bits were simulated".format(np.size(b.numpy())))
```


```python
Shape before encoding:  (1000, 250)
Shape after encoding:  (1000, 500)
Shape after mapping:  (1000, 125)
Shape after channel:  (1000, 125)
Shape after demapping:  (1000, 500)
Shape after decoding:  (1000, 250)
BER uncoded = 0.119 at EbNo = 4.0 dB
BER after decoding = 0.010 at EbNo = 4.0 dB
In total 250000 bits were simulated
```

    
Just to summarize: we have simulated the transmission of 250,000 bits including higher-order modulation and channel coding!
    
But we can go even faster with the <em>TF graph execution</em>!

