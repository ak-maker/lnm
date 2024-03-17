# Introduction to Sionna RT<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#Introduction-to-Sionna-RT" title="Permalink to this headline"></a>
    
In this notebook, you will
 
- Discover the basic functionalities of Sionna’s <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html">ray tracing (RT) module</a>
- Learn how to compute coverage maps
- Use ray-traced channels for link-level simulations instead of stochastic channel models
# Table of Content
## GPU Configuration and Imports
## Ray Tracing for Radio Propagation
  
  

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

## Ray Tracing for Radio Propagation<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#Ray-Tracing-for-Radio-Propagation" title="Permalink to this headline"></a>
    
We need to configure transmitters and receivers prior to computing propagation paths between them. All transmitters and all receivers are equipped with the same antenna arrays which are defined by the `scene` properties `scene.tx_array` and `scene.rx_array`, respectively. Antenna arrays are composed of multiple identical antennas. Antennas can have custom or pre-defined patterns and are either single- or dual-polarized. One can add multiple transmitters and receivers to a scene which need
to have unique names, a position, and orientation which is defined by yaw, pitch, and roll angles.

```python
[8]:
```

```python
# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="V")
# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="cross")
# Create transmitter
tx = Transmitter(name="tx",
                 position=[8.5,21,27])
# Add transmitter instance to scene
scene.add(tx)
# Create a receiver
rx = Receiver(name="rx",
              position=[45,90,1.5],
              orientation=[0,0,0])
# Add receiver instance to scene
scene.add(rx)
tx.look_at(rx) # Transmitter points towards receiver

```

    
Each <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html#scene-objects">SceneObject</a> has an assigned <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html#radio-materials">RadioMaterial</a> that describes the electromagnetic properties of the object whenever it interacts with a ray. This behavior can be frequency-dependent and the ray tracing is done for a specific frequency.
    
We now set the carrier frequency of the scene and implicitly update all RadioMaterials.

```python
[9]:
```

```python
scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials
scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)

```

    
The default scenes have RadioMaterials assigned to each scene object. However, the RadioMaterial of a specific object can be modified and customized by the user.

```python
[10]:
```

```python
# Select an example object from the scene
so = scene.get("Altes_Rathaus-itu_marble")
# Print name of assigned radio material for different frequenies
for f in [3.5e9, 2.14e9]: # Print for differrent frequencies
    scene.frequency = f
    print(f"\nRadioMaterial: {so.radio_material.name} @ {scene.frequency/1e9:.2f}GHz")
    print("Conductivity:", so.radio_material.conductivity.numpy())
    print("Relative permittivity:", so.radio_material.relative_permittivity.numpy())
    print("Complex relative permittivity:", so.radio_material.complex_relative_permittivity.numpy())
    print("Relative permeability:", so.radio_material.relative_permeability.numpy())
    print("Scattering coefficient:", so.radio_material.scattering_coefficient.numpy())
    print("XPD coefficient:", so.radio_material.xpd_coefficient.numpy())

```


```python

RadioMaterial: itu_marble @ 3.50GHz
Conductivity: 0.017550057
Relative permittivity: 7.074
Complex relative permittivity: (7.074-0.090132594j)
Relative permeability: 1.0
Scattering coefficient: 0.0
XPD coefficient: 0.0
RadioMaterial: itu_marble @ 2.14GHz
Conductivity: 0.0111273555
Relative permittivity: 7.074
Complex relative permittivity: (7.074-0.09346512j)
Relative permeability: 1.0
Scattering coefficient: 0.0
XPD coefficient: 0.0
```

    
Let us run the ray tracing process and compute propagation paths between all transmitters and receivers. The parameter `max_depth` determines the maximum number of interactions between a ray and a scene objects. For example, with a `max_depth` of one, only LoS paths are considered. When the property `scene.synthetic_array` is set to `True`, antenna arrays are explicitly modeled by finding paths between any pair of transmitting and receiving antennas in the scene. Otherwise, arrays are
represented by a single antenna located in the center of the array. Phase shifts related to the relative antenna positions will then be applied based on a plane-wave assumption when the channel impulse responses are computed.

```python
[11]:
```

```python
# Compute propagation paths
paths = scene.compute_paths(max_depth=5,
                            num_samples=1e6)  # Number of rays shot into directions defined
                                              # by a Fibonacci sphere , too few rays can
                                              # lead to missing paths
# Visualize paths in the 3D preview
if colab_compat:
    scene.render("my_cam", paths=paths, show_devices=True, show_paths=True, resolution=resolution);
    raise ExitCell
scene.preview(paths, show_devices=True, show_paths=True) # Use the mouse to focus on the visualized paths

```


    
<em>Remark</em>: only one preview instance can be opened at the same time. Please check the previous preview if no output appears.
    
The <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html#paths">Paths</a> object contains all paths that have been found between transmitters and receivers. In principle, the existence of each path is determininistic for a given position and environment. Please note that due to the stochastic nature of the <em>shoot-and-bounce</em> algorithm, different runs of the `compute_paths` function can lead to different paths that are found. Most importantly, diffusely reflected or scattered paths are obtained through
random sampling of directions after each interaction with a scene object. You can seet TensorFlow’s random seed to a specific value before executing `compute_paths` to ensure reproducibility.
    
The Paths object contains detailed information about every found path and allows us to generated channel impulse responses and apply Doppler shifts for the simulation of time evolution.
    
Let us now inspect some of the available properties:

```python
[12]:
```

```python
# Show the coordinates of the starting points of all rays.
# These coincide with the location of the transmitters.
print("Source coordinates: ", paths.sources.numpy())
print("Transmitter coordinates: ", list(scene.transmitters.values())[0].position.numpy())
# Show the coordinates of the endpoints of all rays.
# These coincide with the location of the receivers.
print("Target coordinates: ",paths.targets.numpy())
print("Receiver coordinates: ",list(scene.receivers.values())[0].position.numpy())
# Show the types of all paths:
# 0 - LoS, 1 - Reflected, 2 - Diffracted, 3 - Scattered
# Note that Diffraction and scattering are turned off by default.
print("Path types: ", paths.types.numpy())

```


```python
Source coordinates:  [[ 8.5 21.  27. ]]
Transmitter coordinates:  [ 8.5 21.  27. ]
Target coordinates:  [[45.  90.   1.5]]
Receiver coordinates:  [45.  90.   1.5]
Path types:  [[0 1 1 1 1 1 1 1 1 1 1 1 1]]
```

    
We can see from the list of path types, that there are 14 paths in total. One LoS and 13 reflected paths.

```python
[13]:
```

```python
# We can now access for every path the channel coefficient, the propagation delay,
# as well as the angles of departure and arrival, respectively (zenith and azimuth).
# Let us inspect a specific path in detail
path_idx = 4 # Try out other values in the range [0, 13]
# For a detailed overview of the dimensions of all properties, have a look at the API documentation
print(f"\n--- Detailed results for path {path_idx} ---")
print(f"Channel coefficient: {paths.a[0,0,0,0,0,path_idx, 0].numpy()}")
print(f"Propagation delay: {paths.tau[0,0,0,path_idx].numpy()*1e6:.5f} us")
print(f"Zenith angle of departure: {paths.theta_t[0,0,0,path_idx]:.4f} rad")
print(f"Azimuth angle of departure: {paths.phi_t[0,0,0,path_idx]:.4f} rad")
print(f"Zenith angle of arrival: {paths.theta_r[0,0,0,path_idx]:.4f} rad")
print(f"Azimuth angle of arrival: {paths.phi_r[0,0,0,path_idx]:.4f} rad")

```


```python

--- Detailed results for path 4 ---
Channel coefficient: (4.429778527992312e-06+1.574603736287372e-08j)
Propagation delay: 0.95107 us
Zenith angle of departure: 1.6485 rad
Azimuth angle of departure: 0.9691 rad
Zenith angle of arrival: 1.6485 rad
Azimuth angle of arrival: 0.1625 rad
```
