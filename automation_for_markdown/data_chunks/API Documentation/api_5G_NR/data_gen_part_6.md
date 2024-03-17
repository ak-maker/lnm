# 5G NR<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#g-nr" title="Permalink to this headline"></a>
    
This module provides layers and functions to support simulations of
5G NR compliant features, in particular, the physical uplink shared channel (PUSCH). It provides implementations of a subset of the physical layer functionalities as described in the 3GPP specifications <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38211" id="id1">[3GPP38211]</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38212" id="id2">[3GPP38212]</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id3">[3GPP38214]</a>.
    
The best way to discover this module’s components is by having a look at the <a class="reference external" href="../examples/5G_NR_PUSCH.html">5G NR PUSCH Tutorial</a>.
    
The following code snippet shows how you can make standard-compliant
simulations of the 5G NR PUSCH with a few lines of code:
```python
# Create a PUSCH configuration with default settings
pusch_config = PUSCHConfig()
# Instantiate a PUSCHTransmitter from the PUSCHConfig
pusch_transmitter = PUSCHTransmitter(pusch_config)
# Create a PUSCHReceiver using the PUSCHTransmitter
pusch_receiver = PUSCHReceiver(pusch_transmitter)
# AWGN channel
channel = AWGN()
# Simulate transmissions over the AWGN channel
batch_size = 16
no = 0.1 # Noise variance
x, b = pusch_transmitter(batch_size) # Generate transmit signal and info bits
y = channel([x, no]) # Simulate channel output
b_hat = pusch_receiver([x, no]) # Recover the info bits
# Compute BER
print("BER:", compute_ber(b, b_hat).numpy())
```

    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHTransmitter" title="sionna.nr.PUSCHTransmitter">`PUSCHTransmitter`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHReceiver" title="sionna.nr.PUSCHReceiver">`PUSCHReceiver`</a> provide high-level abstractions of all required processing blocks. You can easily modify them according to your needs.

# Table of Content
## Transport Block
### TBConfig<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#tbconfig" title="Permalink to this headline"></a>
  
  

### TBConfig<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#tbconfig" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.nr.``TBConfig`(<em class="sig-param">`**``kwargs`</em>)<a class="reference internal" href="../_modules/sionna/nr/tb_config.html#TBConfig">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig" title="Permalink to this definition"></a>
    
The TBConfig objects sets parameters related to the transport block
encoding, as described in TS 38.214 <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id21">[3GPP38214]</a>.
    
All configurable properties can be provided as keyword arguments during the
initialization or changed later.
    
The TBConfig is configured by selecting the modulation and coding scheme
(MCS) tables and index.
<p class="rubric">Example
```python
>>> tb_config = TBConfig(mcs_index=13)
>>> tb_config.mcs_table = 3
>>> tb_config.channel_type = "PUSCH"
>>> tb_config.show()
```

    
The following tables provide an overview of the corresponding coderates and
modulation orders.
<table class="docutils align-center" id="id46">
<caption>Table 1 MCS Index Table 1 (Table 5.1.3.1-1 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id22">[3GPP38214]</a>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#id46" title="Permalink to this table"></a></caption>
<colgroup>
<col style="width: 22%" />
<col style="width: 23%" />
<col style="width: 29%" />
<col style="width: 26%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head">
MCS Index
$I_{MCS}$
</th>
<th class="head">
Modulation Order
$Q_m$
</th>
<th class="head">
Target Coderate
$R\times[1024]$
</th>
<th class="head">
Spectral Efficiency

</th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td>    
0</td>
<td>    
2</td>
<td>    
120</td>
<td>    
0.2344</td>
</tr>
<tr class="row-odd"><td>    
1</td>
<td>    
2</td>
<td>    
157</td>
<td>    
0.3066</td>
</tr>
<tr class="row-even"><td>    
2</td>
<td>    
2</td>
<td>    
193</td>
<td>    
0.3770</td>
</tr>
<tr class="row-odd"><td>    
3</td>
<td>    
2</td>
<td>    
251</td>
<td>    
0.4902</td>
</tr>
<tr class="row-even"><td>    
4</td>
<td>    
2</td>
<td>    
308</td>
<td>    
0.6016</td>
</tr>
<tr class="row-odd"><td>    
5</td>
<td>    
2</td>
<td>    
379</td>
<td>    
0.7402</td>
</tr>
<tr class="row-even"><td>    
6</td>
<td>    
2</td>
<td>    
449</td>
<td>    
0.8770</td>
</tr>
<tr class="row-odd"><td>    
7</td>
<td>    
2</td>
<td>    
526</td>
<td>    
1.0273</td>
</tr>
<tr class="row-even"><td>    
8</td>
<td>    
2</td>
<td>    
602</td>
<td>    
1.1758</td>
</tr>
<tr class="row-odd"><td>    
9</td>
<td>    
2</td>
<td>    
679</td>
<td>    
1.3262</td>
</tr>
<tr class="row-even"><td>    
10</td>
<td>    
4</td>
<td>    
340</td>
<td>    
1.3281</td>
</tr>
<tr class="row-odd"><td>    
11</td>
<td>    
4</td>
<td>    
378</td>
<td>    
1.4766</td>
</tr>
<tr class="row-even"><td>    
12</td>
<td>    
4</td>
<td>    
434</td>
<td>    
1.6953</td>
</tr>
<tr class="row-odd"><td>    
13</td>
<td>    
4</td>
<td>    
490</td>
<td>    
1.9141</td>
</tr>
<tr class="row-even"><td>    
14</td>
<td>    
4</td>
<td>    
553</td>
<td>    
2.1602</td>
</tr>
<tr class="row-odd"><td>    
15</td>
<td>    
4</td>
<td>    
616</td>
<td>    
2.4063</td>
</tr>
<tr class="row-even"><td>    
16</td>
<td>    
4</td>
<td>    
658</td>
<td>    
2.5703</td>
</tr>
<tr class="row-odd"><td>    
17</td>
<td>    
6</td>
<td>    
438</td>
<td>    
2.5664</td>
</tr>
<tr class="row-even"><td>    
18</td>
<td>    
6</td>
<td>    
466</td>
<td>    
2.7305</td>
</tr>
<tr class="row-odd"><td>    
19</td>
<td>    
6</td>
<td>    
517</td>
<td>    
3.0293</td>
</tr>
<tr class="row-even"><td>    
20</td>
<td>    
6</td>
<td>    
567</td>
<td>    
3.3223</td>
</tr>
<tr class="row-odd"><td>    
21</td>
<td>    
6</td>
<td>    
616</td>
<td>    
3.6094</td>
</tr>
<tr class="row-even"><td>    
22</td>
<td>    
6</td>
<td>    
666</td>
<td>    
3.9023</td>
</tr>
<tr class="row-odd"><td>    
23</td>
<td>    
6</td>
<td>    
719</td>
<td>    
4.2129</td>
</tr>
<tr class="row-even"><td>    
24</td>
<td>    
6</td>
<td>    
772</td>
<td>    
4.5234</td>
</tr>
<tr class="row-odd"><td>    
25</td>
<td>    
6</td>
<td>    
822</td>
<td>    
4.8164</td>
</tr>
<tr class="row-even"><td>    
26</td>
<td>    
6</td>
<td>    
873</td>
<td>    
5.1152</td>
</tr>
<tr class="row-odd"><td>    
27</td>
<td>    
6</td>
<td>    
910</td>
<td>    
5.3320</td>
</tr>
<tr class="row-even"><td>    
28</td>
<td>    
6</td>
<td>    
948</td>
<td>    
5.5547</td>
</tr>
</tbody>
</table>
<table class="docutils align-center" id="id47">
<caption>Table 2 MCS Index Table 2 (Table 5.1.3.1-2 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id23">[3GPP38214]</a>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#id47" title="Permalink to this table"></a></caption>
<colgroup>
<col style="width: 22%" />
<col style="width: 23%" />
<col style="width: 29%" />
<col style="width: 26%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head">
MCS Index
$I_{MCS}$
</th>
<th class="head">
Modulation Order
$Q_m$
</th>
<th class="head">
Target Coderate
$R\times[1024]$
</th>
<th class="head">
Spectral Efficiency

</th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td>    
0</td>
<td>    
2</td>
<td>    
120</td>
<td>    
0.2344</td>
</tr>
<tr class="row-odd"><td>    
1</td>
<td>    
2</td>
<td>    
193</td>
<td>    
0.3770</td>
</tr>
<tr class="row-even"><td>    
2</td>
<td>    
2</td>
<td>    
308</td>
<td>    
0.6016</td>
</tr>
<tr class="row-odd"><td>    
3</td>
<td>    
2</td>
<td>    
449</td>
<td>    
0.8770</td>
</tr>
<tr class="row-even"><td>    
4</td>
<td>    
2</td>
<td>    
602</td>
<td>    
1.1758</td>
</tr>
<tr class="row-odd"><td>    
5</td>
<td>    
4</td>
<td>    
378</td>
<td>    
1.4766</td>
</tr>
<tr class="row-even"><td>    
6</td>
<td>    
4</td>
<td>    
434</td>
<td>    
1.6953</td>
</tr>
<tr class="row-odd"><td>    
7</td>
<td>    
4</td>
<td>    
490</td>
<td>    
1.9141</td>
</tr>
<tr class="row-even"><td>    
8</td>
<td>    
4</td>
<td>    
553</td>
<td>    
2.1602</td>
</tr>
<tr class="row-odd"><td>    
9</td>
<td>    
4</td>
<td>    
616</td>
<td>    
2.4063</td>
</tr>
<tr class="row-even"><td>    
10</td>
<td>    
4</td>
<td>    
658</td>
<td>    
2.5703</td>
</tr>
<tr class="row-odd"><td>    
11</td>
<td>    
6</td>
<td>    
466</td>
<td>    
2.7305</td>
</tr>
<tr class="row-even"><td>    
12</td>
<td>    
6</td>
<td>    
517</td>
<td>    
3.0293</td>
</tr>
<tr class="row-odd"><td>    
13</td>
<td>    
6</td>
<td>    
567</td>
<td>    
3.3223</td>
</tr>
<tr class="row-even"><td>    
14</td>
<td>    
6</td>
<td>    
616</td>
<td>    
3.6094</td>
</tr>
<tr class="row-odd"><td>    
15</td>
<td>    
6</td>
<td>    
666</td>
<td>    
3.9023</td>
</tr>
<tr class="row-even"><td>    
16</td>
<td>    
6</td>
<td>    
719</td>
<td>    
4.2129</td>
</tr>
<tr class="row-odd"><td>    
17</td>
<td>    
6</td>
<td>    
772</td>
<td>    
4.5234</td>
</tr>
<tr class="row-even"><td>    
18</td>
<td>    
6</td>
<td>    
822</td>
<td>    
4.8164</td>
</tr>
<tr class="row-odd"><td>    
19</td>
<td>    
6</td>
<td>    
873</td>
<td>    
5.1152</td>
</tr>
<tr class="row-even"><td>    
20</td>
<td>    
8</td>
<td>    
682.5</td>
<td>    
5.3320</td>
</tr>
<tr class="row-odd"><td>    
21</td>
<td>    
8</td>
<td>    
711</td>
<td>    
5.5547</td>
</tr>
<tr class="row-even"><td>    
22</td>
<td>    
8</td>
<td>    
754</td>
<td>    
5.8906</td>
</tr>
<tr class="row-odd"><td>    
23</td>
<td>    
8</td>
<td>    
797</td>
<td>    
6.2266</td>
</tr>
<tr class="row-even"><td>    
24</td>
<td>    
8</td>
<td>    
841</td>
<td>    
6.5703</td>
</tr>
<tr class="row-odd"><td>    
25</td>
<td>    
8</td>
<td>    
885</td>
<td>    
6.9141</td>
</tr>
<tr class="row-even"><td>    
26</td>
<td>    
8</td>
<td>    
916.5</td>
<td>    
7.1602</td>
</tr>
<tr class="row-odd"><td>    
27</td>
<td>    
8</td>
<td>    
948</td>
<td>    
7.4063</td>
</tr>
</tbody>
</table>
<table class="docutils align-center" id="id48">
<caption>Table 3 MCS Index Table 3 (Table 5.1.3.1-3 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id24">[3GPP38214]</a>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#id48" title="Permalink to this table"></a></caption>
<colgroup>
<col style="width: 22%" />
<col style="width: 23%" />
<col style="width: 29%" />
<col style="width: 26%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head">
MCS Index
$I_{MCS}$
</th>
<th class="head">
Modulation Order
$Q_m$
</th>
<th class="head">
Target Coderate
$R\times[1024]$
</th>
<th class="head">
Spectral Efficiency

</th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td>    
0</td>
<td>    
2</td>
<td>    
30</td>
<td>    
0.0586</td>
</tr>
<tr class="row-odd"><td>    
1</td>
<td>    
2</td>
<td>    
40</td>
<td>    
0.0781</td>
</tr>
<tr class="row-even"><td>    
2</td>
<td>    
2</td>
<td>    
50</td>
<td>    
0.0977</td>
</tr>
<tr class="row-odd"><td>    
3</td>
<td>    
2</td>
<td>    
64</td>
<td>    
0.1250</td>
</tr>
<tr class="row-even"><td>    
4</td>
<td>    
2</td>
<td>    
78</td>
<td>    
0.1523</td>
</tr>
<tr class="row-odd"><td>    
5</td>
<td>    
2</td>
<td>    
99</td>
<td>    
0.1934</td>
</tr>
<tr class="row-even"><td>    
6</td>
<td>    
2</td>
<td>    
120</td>
<td>    
0.2344</td>
</tr>
<tr class="row-odd"><td>    
7</td>
<td>    
2</td>
<td>    
157</td>
<td>    
0.3066</td>
</tr>
<tr class="row-even"><td>    
8</td>
<td>    
2</td>
<td>    
193</td>
<td>    
0.3770</td>
</tr>
<tr class="row-odd"><td>    
9</td>
<td>    
2</td>
<td>    
251</td>
<td>    
0.4902</td>
</tr>
<tr class="row-even"><td>    
10</td>
<td>    
2</td>
<td>    
308</td>
<td>    
0.6016</td>
</tr>
<tr class="row-odd"><td>    
11</td>
<td>    
2</td>
<td>    
379</td>
<td>    
0.7402</td>
</tr>
<tr class="row-even"><td>    
12</td>
<td>    
2</td>
<td>    
449</td>
<td>    
0.8770</td>
</tr>
<tr class="row-odd"><td>    
13</td>
<td>    
2</td>
<td>    
526</td>
<td>    
1.0273</td>
</tr>
<tr class="row-even"><td>    
14</td>
<td>    
2</td>
<td>    
602</td>
<td>    
1.1758</td>
</tr>
<tr class="row-odd"><td>    
15</td>
<td>    
4</td>
<td>    
340</td>
<td>    
1.3281</td>
</tr>
<tr class="row-even"><td>    
16</td>
<td>    
4</td>
<td>    
378</td>
<td>    
1.4766</td>
</tr>
<tr class="row-odd"><td>    
17</td>
<td>    
4</td>
<td>    
434</td>
<td>    
1.6953</td>
</tr>
<tr class="row-even"><td>    
18</td>
<td>    
4</td>
<td>    
490</td>
<td>    
1.9141</td>
</tr>
<tr class="row-odd"><td>    
19</td>
<td>    
4</td>
<td>    
553</td>
<td>    
2.1602</td>
</tr>
<tr class="row-even"><td>    
20</td>
<td>    
4</td>
<td>    
616</td>
<td>    
2.4063</td>
</tr>
<tr class="row-odd"><td>    
21</td>
<td>    
6</td>
<td>    
438</td>
<td>    
2.5564</td>
</tr>
<tr class="row-even"><td>    
22</td>
<td>    
6</td>
<td>    
466</td>
<td>    
2.7305</td>
</tr>
<tr class="row-odd"><td>    
23</td>
<td>    
6</td>
<td>    
517</td>
<td>    
3.0293</td>
</tr>
<tr class="row-even"><td>    
24</td>
<td>    
6</td>
<td>    
567</td>
<td>    
3.3223</td>
</tr>
<tr class="row-odd"><td>    
25</td>
<td>    
6</td>
<td>    
616</td>
<td>    
3.6094</td>
</tr>
<tr class="row-even"><td>    
26</td>
<td>    
6</td>
<td>    
666</td>
<td>    
3.9023</td>
</tr>
<tr class="row-odd"><td>    
27</td>
<td>    
6</td>
<td>    
719</td>
<td>    
4.2129</td>
</tr>
<tr class="row-even"><td>    
28</td>
<td>    
6</td>
<td>    
772</td>
<td>    
4.5234</td>
</tr>
</tbody>
</table>
<table class="docutils align-center" id="id49">
<caption>Table 4 MCS Index Table 4 (Table 5.1.3.1-4 in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id25">[3GPP38214]</a>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#id49" title="Permalink to this table"></a></caption>
<colgroup>
<col style="width: 22%" />
<col style="width: 23%" />
<col style="width: 29%" />
<col style="width: 26%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head">
MCS Index
$I_{MCS}$
</th>
<th class="head">
Modulation Order
$Q_m$
</th>
<th class="head">
Target Coderate
$R\times[1024]$
</th>
<th class="head">
Spectral Efficiency

</th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td>    
0</td>
<td>    
2</td>
<td>    
120</td>
<td>    
0.2344</td>
</tr>
<tr class="row-odd"><td>    
1</td>
<td>    
2</td>
<td>    
193</td>
<td>    
0.3770</td>
</tr>
<tr class="row-even"><td>    
2</td>
<td>    
2</td>
<td>    
449</td>
<td>    
0.8770</td>
</tr>
<tr class="row-odd"><td>    
3</td>
<td>    
4</td>
<td>    
378</td>
<td>    
1.4766</td>
</tr>
<tr class="row-even"><td>    
4</td>
<td>    
4</td>
<td>    
490</td>
<td>    
1.9141</td>
</tr>
<tr class="row-odd"><td>    
5</td>
<td>    
4</td>
<td>    
616</td>
<td>    
2.4063</td>
</tr>
<tr class="row-even"><td>    
6</td>
<td>    
6</td>
<td>    
466</td>
<td>    
2.7305</td>
</tr>
<tr class="row-odd"><td>    
7</td>
<td>    
6</td>
<td>    
517</td>
<td>    
3.0293</td>
</tr>
<tr class="row-even"><td>    
8</td>
<td>    
6</td>
<td>    
567</td>
<td>    
3.3223</td>
</tr>
<tr class="row-odd"><td>    
9</td>
<td>    
6</td>
<td>    
616</td>
<td>    
3.6094</td>
</tr>
<tr class="row-even"><td>    
10</td>
<td>    
6</td>
<td>    
666</td>
<td>    
3.9023</td>
</tr>
<tr class="row-odd"><td>    
11</td>
<td>    
6</td>
<td>    
719</td>
<td>    
4.2129</td>
</tr>
<tr class="row-even"><td>    
12</td>
<td>    
6</td>
<td>    
772</td>
<td>    
4.5234</td>
</tr>
<tr class="row-odd"><td>    
13</td>
<td>    
6</td>
<td>    
822</td>
<td>    
4.8154</td>
</tr>
<tr class="row-even"><td>    
14</td>
<td>    
6</td>
<td>    
873</td>
<td>    
5.1152</td>
</tr>
<tr class="row-odd"><td>    
15</td>
<td>    
8</td>
<td>    
682.5</td>
<td>    
5.3320</td>
</tr>
<tr class="row-even"><td>    
16</td>
<td>    
8</td>
<td>    
711</td>
<td>    
5.5547</td>
</tr>
<tr class="row-odd"><td>    
17</td>
<td>    
8</td>
<td>    
754</td>
<td>    
5.8906</td>
</tr>
<tr class="row-even"><td>    
18</td>
<td>    
8</td>
<td>    
797</td>
<td>    
6.2266</td>
</tr>
<tr class="row-odd"><td>    
19</td>
<td>    
8</td>
<td>    
841</td>
<td>    
6.5703</td>
</tr>
<tr class="row-even"><td>    
20</td>
<td>    
8</td>
<td>    
885</td>
<td>    
6.9141</td>
</tr>
<tr class="row-odd"><td>    
21</td>
<td>    
8</td>
<td>    
916.5</td>
<td>    
7.1602</td>
</tr>
<tr class="row-even"><td>    
22</td>
<td>    
8</td>
<td>    
948</td>
<td>    
7.4063</td>
</tr>
<tr class="row-odd"><td>    
23</td>
<td>    
10</td>
<td>    
805.5</td>
<td>    
7.8662</td>
</tr>
<tr class="row-even"><td>    
24</td>
<td>    
10</td>
<td>    
853</td>
<td>    
8.3301</td>
</tr>
<tr class="row-odd"><td>    
25</td>
<td>    
10</td>
<td>    
900.5</td>
<td>    
8.7939</td>
</tr>
<tr class="row-even"><td>    
26</td>
<td>    
10</td>
<td>    
948</td>
<td>    
9.2578</td>
</tr>
</tbody>
</table>

<em class="property">`property` </em>`channel_type`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.channel_type" title="Permalink to this definition"></a>
    
5G NR physical channel type. Valid choices are “PDSCH” and “PUSCH”.


`check_config`()<a class="reference internal" href="../_modules/sionna/nr/tb_config.html#TBConfig.check_config">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.check_config" title="Permalink to this definition"></a>
    
Test if configuration is valid


<em class="property">`property` </em>`mcs_index`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.mcs_index" title="Permalink to this definition"></a>
    
Modulation and coding scheme (MCS) index (denoted as $I_{MCS}$
in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id26">[3GPP38214]</a>)


<em class="property">`property` </em>`mcs_table`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.mcs_table" title="Permalink to this definition"></a>
    
Indicates which MCS table from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id27">[3GPP38214]</a> to use. Starts with “1”.


<em class="property">`property` </em>`n_id`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.n_id" title="Permalink to this definition"></a>
    
Data scrambling initialization
$n_\text{ID}$. Data Scrambling ID related to cell id and
provided by higher layer. If <cite>None</cite>, the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.PUSCHConfig" title="sionna.nr.PUSCHConfig">`PUSCHConfig`</a> will automatically set
$n_\text{ID}=N_\text{ID}^{cell}$.
Type
    
int, None (default), [0, 1023]




<em class="property">`property` </em>`num_bits_per_symbol`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.num_bits_per_symbol" title="Permalink to this definition"></a>
    
Modulation order as defined by the selected MCS
Type
    
int, read-only




<em class="property">`property` </em>`target_coderate`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.target_coderate" title="Permalink to this definition"></a>
    
Target coderate of the TB as defined by the selected
MCS
Type
    
float, read-only




<em class="property">`property` </em>`tb_scaling`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/nr.html#sionna.nr.TBConfig.tb_scaling" title="Permalink to this definition"></a>
    
TB scaling factor for PDSCH as
defined in <a class="reference internal" href="https://nvlabs.github.io/sionna/api/nr.html#gpp38214" id="id28">[3GPP38214]</a> Tab. 5.1.3.2-2.
Type
    
float, 1. (default), read-only




