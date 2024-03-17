# Tutorial on Scattering<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Scattering.html#Tutorial-on-Scattering" title="Permalink to this headline"></a>
    
In this notebook, you will
 
- Learn what scattering is and why it is important
- Make various ray tracing experiments to validate some theoretical results
- Familiarize yourself with the Sionna RT API
- Visualize the impact of scattering on channel impulse responses and coverage maps
# Table of Content
## GPU Configuration and Imports
## Scattering Basics
## Scattering Patterns
  
  

## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Scattering.html#GPU-Configuration-and-Imports" title="Permalink to this headline"></a>

```python
[1]:
```

```python
import os # Configure which GPU
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
from sionna.channel import cir_to_time_channel
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMaterial, Camera, LambertianPattern, DirectivePattern, BackscatteringPattern
from sionna.rt.utils import r_hat
from sionna.constants import PI, SPEED_OF_LIGHT
from sionna.utils import expand_to_rank
```

## Scattering Basics<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Scattering.html#Scattering-Basics" title="Permalink to this headline"></a>
    
    
When an electromagnetic wave impinges on a surface, one part of the energy gets reflected while the other part gets refracted, i.e., it propagates into the surface. We distinguish between two types of reflection, specular and diffuse. The latter type is also called diffuse scattering. When a rays hits a diffuse reflection surface, it is not reflected into a single (specular) direction but rather scattered toward many different directions.
    
One way to think about scattering is that every infinitesimally small surface element $dA$ (as shown in the figure above) reradiates a part of the energy impinging on it. It essentially behaves like a point source that radiates electromagnetic waves into the hemisphere defined by the surface normal [1]. Similar to the far-field of an antenna which is determined by the antenna pattern, the scattered field is determined by the scattering pattern of the surface element, denoted
$f_\text{s}(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s})$, where $\hat{\mathbf{k}}_\text{i}$ and $\hat{\mathbf{k}}_\text{s}$ are the incomning and scattered directions, respectively. In other words, the scattered field can be stronger in certain directions than others.
    
The most important difference between diffuse and specular reflections for ray tracing is that an incoming ray essentially spawns infinitely many scattered rays while there is only a single specular path. In order to computed the scattered field at a particular position, one needs to integrate the scattered field over the entire surface.
    
Let us have a look at some common scattering patterns that are implemented in Sionna:

```python
[2]:
```

```python
LambertianPattern().visualize();
```


```python
[3]:
```

```python
DirectivePattern(alpha_r=10).visualize(); # The stronger alpha_r, the more the pattern
                                          # is concentrated around the specular direction.
```



    
In order to develop a feeling for the difference between specular and diffuse reflections, let us load a very simple scene with a single quadratic reflector and place a transmitter and receiver.

```python
[4]:
```

```python
scene = load_scene(sionna.rt.scene.simple_reflector)
# Configure the transmitter and receiver arrays
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")
scene.rx_array = scene.tx_array
# Add a transmitter and receiver with equal distance from the center of the surface
# at an angle of 45 degrees.
dist = 5
d = dist/np.sqrt(2)
scene.add(Transmitter(name="tx", position=[-d,0,d]))
scene.add(Receiver(name="rx", position=[d,0,d]))
# Add a camera for visualization
scene.add(Camera("my_cam", position=[0, -30, 20], look_at=[0,0,3]))
# Open 3D preview (only works in Jupyter notebook)
if colab_compat:
    scene.render("my_cam");
    raise ExitCell
scene.preview()
```


    
Next, let us compute the specularly reflected path:

```python
[5]:
```

```python
paths = scene.compute_paths(los=False, reflection=True)
# Open 3D preview (only works in Jupyter notebook)
if colab_compat:
    scene.render("my_cam", paths=paths);
    raise ExitCell
scene.preview(paths=paths)
```


    
As expected from geometrical optics (GO), the specular path goes through the center of the reflector and has indentical incomning and outgoing angles with the surface normal.
    
We can compute the scattered paths in a similar way:

```python
[6]:
```

```python
paths = scene.compute_paths(los=False, reflection=False, scattering=True, scat_keep_prob=1.0)
# Open 3D preview (only works in Jupyter notebook)
if colab_compat:
    scene.render("my_cam", paths=paths);
    raise ExitCell
scene.preview(paths=paths)
```

```python
[7]:
```

```python
print(f"There are {tf.size(paths.a).numpy()} scattered paths")
```


```python
There are 2247 scattered paths
```

    
We can see that there is a very large number paths. Actually, any ray that hits the surface will be scattered toward the receiver. Thus, the more rays we shoot, the more scattered paths there are. You can see this through the following experiment:

```python
[8]:
```

```python
paths = scene.compute_paths(num_samples=2e6, los=False, reflection=False, scattering=True, scat_keep_prob=1.0)
print(f"There are {tf.size(paths.a).numpy()} scattered paths.")
paths = scene.compute_paths(num_samples=10e6, los=False, reflection=False, scattering=True, scat_keep_prob=1.0)
print(f"There are {tf.size(paths.a).numpy()} scattered paths.")
```


```python
There are 4400 scattered paths.
There are 22572 scattered paths.
```

    
The number of rays hitting the surface is proportional to the total number of rays shot and the squared distance between the transmitter and the surface. However, the total received energy across the surface is constant as the transmitted energy is equally divided between all rays.
    
If you closely inspect the code in the above cells, you might have noticed the keyword argument `scat_keep_prob`. This determines the fraction of scattered paths that will be randomly dropped in the ray tracing process. The importance of the remaining paths is increased proportionally. Setting this argument to small values prevents obtaining channel impulse responses with an excessive number of scattered paths.

```python
[9]:
```

```python
paths = scene.compute_paths(num_samples=10e6, los=False, reflection=False, scattering=True, scat_keep_prob=0.001)
print(f"There are {tf.size(paths.a).numpy()} scattered paths.")
```


```python
There are 16 scattered paths.
```

    
In our example scene, each ray hitting the surfaces spawns exactly one new ray which connects to the receiver. Each ray has a random phase and energy that is determined by the scattering pattern and the so-called scattering coefficient $S\in[0,1]$. The squared scattering coefficient $S^2$ determines which portion of the totally reflected energy (specular and diffuse combined) is diffusely reflected. For details on the precise modeling of the scattered field, we refer to the <a class="reference external" href="https://nvlabs.github.io/sionna/em_primer.html#scattering">EM
Primer</a>.
    
By default, all materials in Sionna have a scattering coefficient equal to zero. For this reason, we would expect that all of the scattered paths carry zero energy. Let’s verify that this is indeed the case:

```python
[10]:
```

```python
print("All scattered paths have zero energy:", np.all(np.abs(paths.a)==0))
```


```python
All scattered paths have zero energy: True
```

    
Let us change the scattering coefficient of the radio material used by the reflector and run the path computations again:

```python
[11]:
```

```python
scene.get("reflector").radio_material.scattering_coefficient = 0.5
paths = scene.compute_paths(num_samples=1e6, los=False, reflection=False, scattering=True)
print("All scattered paths have positive energy:", np.all(np.abs(paths.a)>0))
```


```python
All scattered paths have positive energy: True
```
## Scattering Patterns<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Scattering.html#Scattering-Patterns" title="Permalink to this headline"></a>
    
In order to study the impact of the scattering pattern, let’s replace the perfectly diffuse Lambertian pattern (which all radio materials have by default) by the <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.DirectivePattern">DirectivePattern</a>. The larger the integer parameter $\alpha_r$, the more the scattered field is focused around the direction of the specular reflection.

```python
[12]:
```

```python
scattering_pattern = DirectivePattern(1)
scene.get("reflector").radio_material.scattering_pattern = scattering_pattern
alpha_rs = np.array([1,2,3,5,10,30,50,100], np.int32)
received_powers = np.zeros_like(alpha_rs, np.float32)
for i, alpha_r in enumerate(alpha_rs):
    scattering_pattern.alpha_r = alpha_r
    paths = scene.compute_paths(num_samples=1e6, los=False, reflection=False, scattering=True, scat_keep_prob=1.0)
    received_powers[i] = 10*np.log10(tf.reduce_sum(tf.abs(paths.a)**2))
plt.figure()
plt.plot(alpha_rs, received_powers)
plt.xlabel(r"$\alpha_r$")
plt.ylabel("Received power (dB)");
plt.title("Impact of the Directivity of the Scattering Pattern");
```


    
We can indeed observe that the received energy increases with $\alpha_r$. This is because the scattered paths are almost parallel to the specular path directions in this scene. If we move the receiver away from the specular direction, this effect should be reversed.

```python
[13]:
```

```python
# Move the receiver closer to the surface, i.e., away from the specular angle theta=45deg
scene.get("rx").position = [d, 0, 1]
received_powers = np.zeros_like(alpha_rs, np.float32)
for i, alpha_r in enumerate(alpha_rs):
    scattering_pattern.alpha_r = alpha_r
    paths = scene.compute_paths(num_samples=1e6, los=False, reflection=False, scattering=True, scat_keep_prob=1.0)
    received_powers[i] = 10*np.log10(tf.reduce_sum(tf.abs(paths.a)**2))
plt.figure()
plt.plot(alpha_rs, received_powers)
plt.xlabel(r"$\alpha_r$")
plt.ylabel("Received power (dB)");
plt.title("Impact of the Directivity of the Scattering Pattern");
```


