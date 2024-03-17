# Tutorial on Diffraction<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Diffraction.html#Tutorial-on-Diffraction" title="Permalink to this headline"></a>
    
In this notebook, you will
 
- Learn what diffraction is and why it is important
- Make various ray tracing experiments to validate some theoretical results
- Familiarize yourself with the Sionna RT API
- Visualize the impact of diffraction on channel impulse responses and coverage maps
# Table of Content
## GPU Configuration and Imports
## Coverage Maps with Diffraction
## References
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Diffraction.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

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
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import sys
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMaterial, Camera
from sionna.rt.utils import r_hat
from sionna.constants import PI, SPEED_OF_LIGHT
from sionna.utils import expand_to_rank
```

## Coverage Maps with Diffraction<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Diffraction.html#Coverage-Maps-with-Diffraction" title="Permalink to this headline"></a>
    
So far, we have obtained a solid microscopic understanding of the effect of scattering. Let us now turn to the macroscopic effects that can be nicely observed through coverage maps.
    
A coverage map describes the average received power from a specific transmitter at every point on a plane. The effects of fast fading, i.e., constructive/destructive interference between different paths, are averaged out by summing the squared amplitudes of all paths. As we cannot compute coverage maps with infinitely fine resolution, they are approximated by small rectangular tiles for which average values are computed. For a detailed explanation, have a look at the <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html#coverage-map">API
Documentation</a>.
    
Let us now load a slightly more interesting scene containing a couple of rectangular buildings and add a transmitter. Note that we do not need to add any receivers to compute a coverage map.

```python
[18]:
```

```python
scene = load_scene(sionna.rt.scene.simple_street_canyon)
# Set the carrier frequency to 1GHz
scene.frequency = 1e9
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")
scene.rx_array = scene.tx_array
scene.add(Transmitter(name="tx",
                      position=[-33,11,32],
                      orientation=[0,0,0]))
# Render the scene from one of its cameras
# The blue dot represents the transmitter
scene.render('scene-cam-1');
```


    
Computing a coverage map is as simple as running the following command:

```python
[19]:
```

```python
cm = scene.coverage_map(cm_cell_size=[1,1], num_samples=10e6)
```

    
We can visualizes the coverage map in the scene as follows:

```python
[20]:
```

```python
# Add a camera looking at the scene from the top
my_cam = Camera("my_cam", position=[10,0,300], look_at=[0,0,0])
my_cam.look_at([0,0,0])
scene.add(my_cam)
# Render scene with the new camera and overlay the coverage map
scene.render(my_cam, coverage_map=cm);
```


    
From the figure above, we can see that many regions behind buildings do not receive any signal. The reason for this is that diffraction is by default deactivated. Let us now generate a new coverage map with diffraction enabled:

```python
[21]:
```

```python
cm_diff = scene.coverage_map(cm_cell_size=[1,1], num_samples=10e6, diffraction=True)
scene.render(my_cam, coverage_map=cm_diff);
```


    
As expected from our experiements above, there is not a single point in the scene that is left blank. In some areas, however, the signal is still very weak and will not enable any form of communication.
    
Let’s do the same experiments at a higher carrier frequency (30 GHz):

```python
[22]:
```

```python
scene = load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = 30e9
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")
scene.rx_array = scene.tx_array
scene.add(Transmitter(name="tx",
                      position=[-33,11,32],
                      orientation=[0,0,0]))
scene.add(my_cam)
cm = scene.coverage_map(cm_cell_size=[1,1], num_samples=10e6)
cm_diff = scene.coverage_map(cm_cell_size=[1,1], num_samples=10e6, diffraction=True)
scene.render(my_cam, coverage_map=cm);
scene.render(my_cam, coverage_map=cm_diff);
```



    
While the 1 GHz and 30 GHz carrier frequency coverage maps appear similar, key differences exist. The dynamic range for 30 GHz has grown by around 16dB due to the reduced diffracted field in deep shadow areas, such as behind buildings. The diffracted field at this frequency is considerably smaller compared to the incident field than it is at 1 GHz, leading to a significant increase in dynamic range.
    
In conclusion, diffraction plays a vital role in maintaining the consistency of the electric field across both reflection and incident shadow boundaries. It generates diffracted rays that form a Keller cone around an edge. As we move away from these boundaries, the diffracted field diminishes rapidly. Importantly, the contributions of the diffracted field become less significant as the carrier frequency increases.
    
We hope you enjoyed our dive into diffraction with this Sionna RT tutorial. We really encourage you to get hands-on, conduct your own experiments and deepen your understanding of ray tracing. There’s always more to learn, so do explore our other <a class="reference external" href="https://nvlabs.github.io/sionna/tutorials.html">tutorials</a> as well.

## References<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Diffraction.html#References" title="Permalink to this headline"></a>
    
[1] J.B. Keller, <a class="reference external" href="https://opg.optica.org/josa/abstract.cfm?uri=josa-52-2-116">Geometrical Theory of Diffraction</a>, Journal of the Optical Society of America, vol. 52, no. 2, Feb. 1962.
    
[2] R.G. Kouyoumjian, <a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/1451581/authors#authors">A uniform geometrical theory of diffraction for an edge in a perfectly conducting surface</a>, Proc. of the IEEE, vol. 62, no. 11, Nov. 1974.
    
[3] D.A. McNamara, C.W.I. Pistorius, J.A.G. Malherbe, <a class="reference external" href="https://us.artechhouse.com/Introduction-to-the-Uniform-Geometrical-Theory-of-Diffraction-P288.aspx">Introduction to the Uniform Geometrical Theory of Diffraction</a>, Artech House, 1990.
    
[4] R. Luebbers, <a class="reference external" href="https://ieeexplore.ieee.org/abstract/document/1143189">Finite conductivity uniform GTD versus knife edge diffraction in prediction of propagation path loss</a>, IEEE Trans. Antennas and Propagation, vol. 32, no. 1, Jan. 1984.