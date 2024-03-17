# Introduction to Sionna RT<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#Introduction-to-Sionna-RT" title="Permalink to this headline"></a>
    
In this notebook, you will
 
- Discover the basic functionalities of Sionna’s <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html">ray tracing (RT) module</a>
- Learn how to compute coverage maps
- Use ray-traced channels for link-level simulations instead of stochastic channel models
# Table of Content
## GPU Configuration and Imports
## Background Information
## Loading Scenes
  
  

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

## Background Information<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#Background-Information" title="Permalink to this headline"></a>
    
Ray tracing is a technique to simulate environment-specific and physically accurate channel realizations for a given scene and user position. Please see the <a class="reference external" href="https://nvlabs.github.io/sionna/em_primer.html">EM Primer</a> for further details on the theoretical background of ray tracing of wireless channels.
    
Sionna RT is a ray tracing extension for radio propagation modeling which is built on top of <a class="reference external" href="https://www.mitsuba-renderer.org/">Mitsuba 3</a> and <a class="reference external" href="https://www.tensorflow.org/">TensorFlow</a>. Like all of Sionna’s components, it is differentiable.
    
Mitsuba 3 is a rendering system for forward and inverse light-transport simulation that makes use of the differentiable just-in-time compiler <a class="reference external" href="https://drjit.readthedocs.io/en/latest/">Dr.Jit</a>. Sionna RT relies on Mitsuba 3 for the rendering and handling of scenes, e.g., its XML-file format, as well as the computation of ray intersections with scene primitives, i.e., triangles forming a mesh. The transformations of the polarized field components at each point of interaction between a ray and a
scene object, e.g., reflection, are computed in TensorFlow, which is also used to combine the retained paths into (optionally) time-varying channel impulse responses. Thanks to TensorFlow’s automatic gradient computation, channel impulse responses and functions thereof are differentiable with respect to most parameters of the ray tracing process, including material properties (conductivity, permittivity), antenna patterns, orientations, and positions.
    
Scene files for Mitsuba 3 can be created, edited, and exported with the popular open-source 3D creation suite <a class="reference external" href="https://www.blender.org/">Blender</a> and the <a class="reference external" href="https://github.com/mitsuba-renderer/mitsuba-blender">Mitsuba-Blender add-on</a>. One can rapdily create scenes from almost any place in the world using <a class="reference external" href="https://www.openstreetmap.org/">OpenStreetMap</a> and the <a class="reference external" href="https://prochitecture.gumroad.com/l/blender-osm">Blender-OSM add-on</a>. In Sionna, scenes and radio propagation paths can be
either rendered through the lens of configurable cameras via ray tracing or displayed with an integrated 3D viewer. For more detail on scene creation and rendering, we refer to <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html">Sionna’s API documentation</a> and the available <a class="reference external" href="https://youtu.be/7xHLDxUaQ7c">video tutorial</a>.


## Loading Scenes<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#Loading-Scenes" title="Permalink to this headline"></a>
    
The Sionna RT module can either load external scene files (in Mitsuba’s XML file format) or it can load one of the <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html#example-scenes">integrated scenes</a>.
    
In this example, we load an example scene containing the area around the Frauenkirche in Munich, Germany.

```python
[3]:
```

```python
# Load integrated scene
scene = load_scene(sionna.rt.scene.munich) # Try also sionna.rt.scene.etoile

```

    
To visualize the scene, we can use the `preview` function which opens an interactive preview of the scene. This only works in Jupyter notebooks.
    
You can use the following controls:
 
- Mouse left: Rotate
- Scroll wheel: Zoom
- Mouse right: Move

    
Please note that the preview does not work in Colab and is therefore deactivated when `colab_compat` is set to True. Further, only one preview instance can be open at the same time.

```python
[4]:
```

```python
 # Open 3D preview (only works in Jupyter notebook)
if colab_compat:
    scene.render(camera="scene-cam-0", num_samples=512);
    raise ExitCell
scene.preview()

```


    
It is often convenient to choose a viewpoint in the 3D preview prior to rendering it as a high-quality image. The next cell uses the “preview” camera which corresponds to the viewpoint of the current preview image.

```python
[5]:
```

```python
# The preview camera can be directly rendered as high-quality image
if not colab_compat:
    scene.render(camera="preview", num_samples=512);
else:
    print("Function not available in Colab mode.")

```


```python
Function not available in Colab mode.
```

    
One can also render the image to a file as shown below:

```python
[6]:
```

```python
render_to_file = False # Set to True to render image to file
# Render scene to file from preview viewpoint
if render_to_file:
    scene.render_to_file(camera="scene-cam-0", # Also try camera="preview"
                         filename="scene.png",
                         resolution=[650,500])

```

    
Instead of the preview camera, one can also specify dedicated cameras with different positions and `look_at` directions.

```python
[7]:
```

```python
# Create new camera with different configuration
my_cam = Camera("my_cam", position=[-250,250,150], look_at=[-15,30,28])
scene.add(my_cam)
# Render scene with new camera*
scene.render("my_cam", resolution=resolution, num_samples=512); # Increase num_samples to increase image quality

```


    
Note that each <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html#scene-objects">SceneObject</a> (camera, transmitter,…) needs a unique name. Thus, running the cells above multiple times will lead to an error if the object name is not changed or the object is not removed from the scene.

