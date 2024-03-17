# Introduction to Sionna RT<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#Introduction-to-Sionna-RT" title="Permalink to this headline"></a>
    
In this notebook, you will
 
- Discover the basic functionalities of Sionna’s <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html">ray tracing (RT) module</a>
- Learn how to compute coverage maps
- Use ray-traced channels for link-level simulations instead of stochastic channel models
# Table of Content
## GPU Configuration and Imports
## Conclusion and Outlook
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

```python
[1]:
```

```python
import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Colab does currently not support the latest version of ipython.
# Thus, the preview does not work in Colab. However, whenever possible we
# strongly recommend to use the scene preview mode.
try: # detect if the notebook runs in Colab
    import google.colab
    colab_compat = True # deactivate preview
except:
    colab_compat = False
resolution = [480,320] # increase for higher quality of renderings
# Allows to exit cell execution in Jupyter
class ExitCell(Exception):
    def _render_traceback_(self):
        pass
# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(1) # Set global random seed for reproducibility

```
```python
[2]:
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time
# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement

```

## Conclusion and Outlook<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#Conclusion-and-Outlook" title="Permalink to this headline"></a>
    
In this notebook, you have learned how to use the Sionna RT module. We have seen how paths can be found in complex environments such as in the Munich scene. In a second step, we have calculated the effective CIRs from the paths and used them for link-level simulations.
    
There is one key feature of Sionna RT that was not discussed in this notebook: Automatic gradient computation. Like most components of Sionna, the ray tracer is differentiable with respect to most system parameters, such as radio materials, transmitter and receiver orientations, array geometries, positions, etc. This enables a whole new line of research which is discussed in the <a class="reference external" href="https://nvlabs.github.io/sionna/made_with_sionna.html#sionna-rt-differentiable-ray-tracing-for-radio-propagation-modeling">Sionna RT paper and the related
notebooks</a>.
    
Hopefully you have enjoyed this tutorial on Sionna’s RT module!
    
Please have a look at the <a class="reference external" href="https://nvlabs.github.io/sionna/api/sionna.html">API documentation</a> of the various components or the other available <a class="reference external" href="https://nvlabs.github.io/sionna/tutorials.html">tutorials</a> to learn more.
<script type="application/vnd.jupyter.widget-state+json">
{"state": {}, "version_major": 2, "version_minor": 0}
</script>