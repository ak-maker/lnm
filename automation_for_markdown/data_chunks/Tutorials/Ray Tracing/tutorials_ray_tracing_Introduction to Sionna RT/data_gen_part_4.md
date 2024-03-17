# Introduction to Sionna RT<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#Introduction-to-Sionna-RT" title="Permalink to this headline"></a>
    
In this notebook, you will
 
- Discover the basic functionalities of Sionna’s <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html">ray tracing (RT) module</a>
- Learn how to compute coverage maps
- Use ray-traced channels for link-level simulations instead of stochastic channel models
# Table of Content
## GPU Configuration and Imports
## Runtime vs Depth
## Coverage Map
  
  

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

## Runtime vs Depth<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#Runtime-vs-Depth" title="Permalink to this headline"></a>
    
We will now investigate the complexity of the ray tracing algorithm for different values of `max_depth`, i.e., for a different number of bounces of the rays.

```python
[22]:
```

```python
max_depths = 10 # evaluate performance up to 10 reflections
depths = range(1,max_depths+1)
ts = []
pl_avg = []
for d in depths:
    # save start time
    t = time.time()
    # run the ray tracer
    paths = scene.compute_paths(max_depth=d)
    # and measure the required time interval
    ts.append(time.time()-t)

```
```python
[23]:
```

```python
# and plot results
plt.figure()
plt.plot(depths, ts, color="b");
plt.xlabel("Max. depth")
plt.ylabel("Runtime (s)", color="b")
plt.grid(which="both")
plt.xlim([1, max_depths]);

```


    
As can be seen, the computational complexity increases significantly with the number of ray interactions. Note that the code above does not account for scattering or diffraction. Adding these phenomea adds additional complexity as can be seen below:

```python
[24]:
```

```python
t = time.time()
paths = scene.compute_paths(max_depth=3, diffraction=False)
print("Time without diffraction and scattering:" , time.time()-t)
t = time.time()
paths = scene.compute_paths(max_depth=3, diffraction=True)
print("Time with diffraction:" , time.time()-t)
t = time.time()
paths = scene.compute_paths(max_depth=3, scattering=True)
print("Time with scattering:" , time.time()-t)

```


```python
Time without diffraction and scattering: 2.456580400466919
Time with diffraction: 2.614542245864868
Time with scattering: 3.055100917816162
```

    
Although we have simulated scattering in the last example above, the scattered paths do not carry any energy as none of the materials in the scene has a positive scattering coefficient. You can learn more about scattering and diffraction in the dedicated <a class="reference external" href="https://nvlabs.github.io/sionna/tutorials.html#ray-tracing">tutorial notebooks</a>.

## Coverage Map<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#Coverage-Map" title="Permalink to this headline"></a>
    
Sionna RT can be also used to simulate coverage maps for a given environment. We now put a new transmitter on top of the Frauenkirche and simulate a large-scale coverage map.

```python
[25]:
```

```python
# Remove old transmitter and add new one
scene.remove("tx")
tx = Transmitter(name="tx",
                 position=[-210,73,105], # top of Frauenkirche
                 orientation=[0,0,0])
scene.add(tx)
# We could have alternatively modified the properties position and orientation of the existing transmitter
#scene.get("tx").position = [-210,73,105]
#scene.get("tx").orientation = [0,0,0]

```

    
Let’s have a look at the new setup. The receiver can be ignored for the coverage map simulation.

```python
[26]:
```

```python
 # Open 3D preview (only works in Jupyter notebook)
if colab_compat:
    scene.render(camera="scene-cam-0", num_samples=512, resolution=resolution);
    raise ExitCell
scene.preview()

```

```python
[27]:
```

```python
cm = scene.coverage_map(max_depth=5,
                        diffraction=True, # Disable to see the effects of diffraction
                        cm_cell_size=(5., 5.), # Grid size of coverage map cells in m
                        combining_vec=None,
                        precoding_vec=None,
                        num_samples=int(20e6)) # Reduce if your hardware does not have enough memory

```

    
Once simulated, the coverage map object can be directly visualized with the `preview` or `render` function.

```python
[28]:
```

```python
# Create new camera
tx_pos = scene.transmitters["tx"].position.numpy()
bird_pos = tx_pos.copy()
bird_pos[-1] = 1000 # Set height of coverage map to 1000m above tx
bird_pos[-2]-= 0.01 # Slightly move the camera for correct orientation
# Create new camera
bird_cam = Camera("birds_view", position=bird_pos, look_at=tx_pos)
scene.add(bird_cam)
if colab_compat:
    scene.render(camera="birds_view", coverage_map=cm, num_samples=512, resolution=resolution);
    raise ExitCell
# Open 3D preview (only works in Jupyter notebook)
scene.preview(coverage_map=cm)

```


    
Alternatively, a 2D visualization of the coverage map can be shown.

```python
[29]:
```

```python
cm.show(tx=0); # If multiple transmitters exist, tx selects for which transmitter the cm is shown

```


    
Note that it can happen in rare cases that diffracted rays arrive inside or behind buildings through paths which should not exists. This is not a bug in Sionna’s ray tracing algorithm but rather an artefact of the way how scenes are created which can lead to the false detection of diffraction edges.

