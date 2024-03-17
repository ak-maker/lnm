# Tutorial on Diffraction<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Diffraction.html#Tutorial-on-Diffraction" title="Permalink to this headline"></a>
    
In this notebook, you will
 
- Learn what diffraction is and why it is important
- Make various ray tracing experiments to validate some theoretical results
- Familiarize yourself with the Sionna RT API
- Visualize the impact of diffraction on channel impulse responses and coverage maps
# Table of Content
## GPU Configuration and Imports
## Background Information
  
  

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

## Background Information<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Diffraction.html#Background-Information" title="Permalink to this headline"></a>
    
    
Let’s consider an infinitely long wedge as shown in the figure above. To better visualize this, think of an endless slice of pie. The edge vector is pointed straight out of your screen and the wedge has an opening angle of $n\pi$, where $n$ is a real number between 1 and 2.
    
The wedge has two faces, the 0- and the n-face. They are labeled this way to indicate from which surface the angle $\phi'\in[0,n]$ of an incoming locally planar electromagnetic wave is measured. Both faces are made from possibly different materials, each with their own unique properties. These properties are represented by a term known as complex relative permittivity, denoted by $\eta_0$ and $\eta_n$, respectively. Without diving too deep into the specifics, permittivity
measures how a material reacts to an applied electric field.
    
We can define three distinct regions in this figure: Region $I$, in which the incident field as well as the reflected field from the 0-face are present, Region $II$, in which the reflected field vanishes, and Region $III$, in which the incident field is shadowed by the wedge. The three regions are separated by the reflection shadow boundary (RSB) and the incident shadow boundary (ISB). The former is determined by the angle of specular reflection $\pi-\phi'$, while the
latter is simply the prolongation of the direction of the incoming wave through the edge, i.e., $\pi+\phi'$.
    
Using geometrical optics (GO) alone, the electromagnetic field would abruptly change at each boundary as the reflected and incident field components suddenly disappear. As this is physically not plausible, the geometrical theory of diffraction (GTD) [1], as developed by Joseph B. Keller in the 1960s, introduces a so-called diffracted field which ensures that the total field is continuous. The diffracted field is hence especially important in the transition regions between the different regions
and then rapidly decays to zero. Most importantly, without diffraction, there would be no field beyond the ISB in Region $III$.
    
Diffraction is hence a very important phenomenon which enables wireless coverage behind buildings at positions without a line-of-sight of strong reflected path. As you will see later in this tutorial, the diffracted field is generally much weaker than the incident or reflected field. Moreover, the higher the frequency, the faster the diffracted field decays when moving away from the RSB and ISB.
    
    
According to the GTD, when a ray hits a point on the edge, its energy gets spread over a continuum of rays lying on the Keller cone. All rays on this cone make equal angles with the edge of diffraction at the point of diffraction, i.e., $\beta_0'=\beta_0$. One can think of this phenomenon as an extension of the law of reflection at planar surfaces. The figures above illustrates this concept.
    
The GTD was later extended to the uniform theory of diffraction (UTD) [2,3] which overcomes some of its shortcomings.
    
We will explore in this notebook these effects in detail and also validate the UTD implementation in Sionna RT as a by-product.

**Wedge vs Edge**<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Diffraction.html#Wedge-vs-Edge" title="Permalink to this headline"></a>
    
First, it is important to know the difference between a <em>wedge</em> and an <em>edge</em>, and why we distinguish between them.
    
Sionna defines a <em>wedge</em> as the line segment between two primitives, i.e., the common segment of two triangles. For example, a cubic building would have 12 wedges.
    
For primitives that have one or more line segments that are not shared with another primitive, Sionna refers to such line segments as <em>edges</em>. See <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.floor_wall">sionna.rt.scene.floor_wall</a> for an example scene.
    
By default, Sionna does not simulate diffraction on edges (`edge_diffraction=False`), to avoid problems such as diffraction on the exterior edges of the ground surface (modelled as a rectangular plane).

