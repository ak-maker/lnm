# Tutorial on Diffraction<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Diffraction.html#Tutorial-on-Diffraction" title="Permalink to this headline"></a>
    
In this notebook, you will
 
- Learn what diffraction is and why it is important
- Make various ray tracing experiments to validate some theoretical results
- Familiarize yourself with the Sionna RT API
- Visualize the impact of diffraction on channel impulse responses and coverage maps
# Table of Content
## GPU Configuration and Imports
## Experiments with a Simple Wedge
  
  

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

## Experiments with a Simple Wedge<a class="headerlink" href="https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Diffraction.html#Experiments-with-a-Simple-Wedge" title="Permalink to this headline"></a>
    
We start by loading a pre-made scene from Sionna RT that contains a simple wedge:

```python
[2]:
```

```python
scene = load_scene(sionna.rt.scene.simple_wedge)
# Create new camera with different configuration
my_cam = Camera("my_cam", position=[10,-100,100], look_at=[10,0,0])
scene.add(my_cam)
# Render scene
scene.render(my_cam);
# You can also preview the scene with the following command
# scene.preview()
```


    
The wedge has an opening angle of $\frac{3}{2}\pi=270^\circ$, i.e., $n=1.5$. The 0-face is aligned with the x axis and the n-face aligned with the negative y axis.
    
For the following experiments, we will configure the wedge to be made of metal, an almost perfect conductor, and set the frequency to 1GHz.

```python
[3]:
```

```python
scene.frequency = 1e9 # 1GHz
scene.objects["wedge"].radio_material = "itu_metal" # Almost perfect reflector
```

    
With our scene being set-up, we now need to configure a transmitter and place multiple receivers to measure the field. We assume that the transmitter and all receivers have a single vertically polarized isotropic antenna.

```python
[4]:
```

```python
# Configure the antenna arrays used by the transmitters and receivers
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")
scene.rx_array = scene.tx_array
```
```python
[5]:
```

```python
# Transmitter
tx_angle = 30/180*PI # Angle phi from the 0-face
tx_dist = 50 # Distance from the edge
tx_pos = 50*r_hat(PI/2, tx_angle)
ref_boundary = (PI - tx_angle)/PI*180
los_boundary = (PI + tx_angle)/PI*180
scene.add(Transmitter(name="tx",
                      position=tx_pos,
                      orientation=[0,0,0]))
# Receivers
# We place num_rx receivers uniformly spaced on the segment of a circle around the wedge
num_rx = 1000 # Number of receivers
rx_dist = 5 # Distance from the edge
phi = tf.linspace(1e-2, 3/2*PI-1e-2, num=num_rx)
theta = PI/2*tf.ones_like(phi)
rx_pos = rx_dist*r_hat(theta, phi)
for i, pos in enumerate(rx_pos):
    scene.add(Receiver(name=f"rx-{i}",
                       position=pos,
                       orientation=[0,0,0]))
```
```python
[6]:
```

```python
# Render scene
my_cam.position = [-30,100,100]
my_cam.look_at([10,0,0])
scene.render(my_cam);
```


    
In the above figure, the blue ball is the transmitter and the green circle corresponds to 1000 receivers uniformly distributed over a segment of a circle around the edge.
    
Next, we compute the channel impulse response between the transmitter and all of the receivers. We deactivate scattering in this notebook as it would require a prohibitive amount of memory with such a large number of receivers.

```python
[7]:
```

```python
# Compute paths between the transmitter and all receivers
paths = scene.compute_paths(num_samples=1e6,
                            los=True,
                            reflection=True,
                            diffraction=True,
                            scattering=False)
# Obtain channel impulse responses
# We squeeze irrelevant dimensions
# [num_rx, max_num_paths]
a, tau = [np.squeeze(t) for t in paths.cir()]
```
```python
[8]:
```

```python
def compute_gain(a, tau):
        """Compute $|H(f)|^2 at f = 0 where H(f) is the baseband channel frequency response"""
        a = tf.squeeze(a, axis=-1)
        h_f_2 = tf.math.abs(tf.reduce_sum(a, axis=-1))**2
        h_f_2 = tf.where(h_f_2==0, 1e-24, h_f_2)
        g_db = 10*np.log10(h_f_2)
        return tf.squeeze(g_db)
```

    
Let’s have a look at the channel impulse response of one of the receivers:

```python
[9]:
```

```python
n = 400
plt.figure()
plt.stem(tau[n]/1e-9, 10*np.log10(np.abs(a[n])**2))
plt.title(f"Angle of receiver $\phi: {int(phi[n]/PI*180)}^\circ$");
plt.xlabel("Delay (ns)");
plt.ylabel("$|a|^2$ (dB)");
```


    
For an angle of around 108 degrees, the receiver is located within Region I, where all propagation effects should be visible. As expected, we can observe three path: line-of-sight, reflected, and diffracted. While the first two have roughly the same strength (as metal is an almost perfect reflector), the diffracted path has significantly lower energy.
    
Next, let us compute the channel frequency response $H(f)$ as the sum of all paths multiplied with their complex phase factors:

$$
H(f) = \sum_{i=1}^N a_i e^{-j2\pi\tau_i f}
$$

```python
[10]:
```

```python
h_f_tot = np.sum(a, axis=-1)
```

    
We can now visualize the path gain $|H(f)|^2$ for all receivers, i.e., as a function of the angle $\phi$:

```python
[11]:
```

```python
fig = plt.figure()
plt.plot(phi/PI*180, 20*np.log10(np.abs(h_f_tot)))
plt.xlabel("Diffraction angle $\phi$ (deg)");
plt.ylabel(r"Path gain $|H(f)|^2$ (dB)");
plt.ylim([-100, -59]);
plt.xlim([0, phi[-1]/PI*180]);
```


    
The most important observation from the figure above is that $H(f)$ remains continous over the entire range of $\phi$, especially at the RSB and ISB boundaries at around $\phi=150^\circ$ and $\phi=209^\circ$, respectively.
    
To get some more insight, the convenience function in the next cell, computes and visualizes the different components of $H(f)$ by their type.

```python
[12]:
```

```python
def plot(frequency, material):
    """Plots the path gain $|H(f)|^2 versus $phi$ for a given
       frequency and RadioMaterial of the wedge.
    """
    # Set carrier frequency and material of the wedge
    # You can see a list of available materials by executing
    # scene.radio_materials
    scene.frequency = frequency
    scene.objects["wedge"].radio_material = material
    # Recompute paths with the updated material and frequency
    paths = scene.compute_paths(num_samples=1e6,
                                los=True,
                                reflection=True,
                                diffraction=True,
                                scattering=False)
    def compute_gain(a, tau):
        """Compute $|H(f)|^2 are f = 0 where H(f) is the baseband channel frequency response"""
        a = tf.squeeze(a, axis=-1)
        h_f_2 = tf.math.abs(tf.reduce_sum(a, axis=-1))**2
        h_f_2 = tf.where(h_f_2==0, 1e-24, h_f_2)
        g_db = 10*np.log10(h_f_2)
        return tf.squeeze(g_db)
    # Compute gain for all path types
    g_tot_db = compute_gain(*paths.cir())
    g_los_db = compute_gain(*paths.cir(reflection=False, diffraction=False, scattering=False))
    g_ref_db = compute_gain(*paths.cir(los=False, diffraction=False, scattering=False))
    g_dif_db = compute_gain(*paths.cir(los=False, reflection=False, scattering=False))
    # Make a nice plot
    fig = plt.figure()
    phi_deg = phi/PI*180
    ymax = np.max(g_tot_db)+5
    ymin = ymax - 45
    plt.plot(phi_deg, g_tot_db)
    plt.plot(phi_deg, g_los_db)
    plt.plot(phi_deg, g_ref_db)
    plt.plot(phi_deg, g_dif_db)
    plt.ylim([ymin, ymax])
    plt.xlim([phi_deg[0], phi_deg[-1]]);
    plt.legend(["Total", "LoS", "Reflected", "Diffracted"], loc="lower left")
    plt.xlabel("Diffraction angle $\phi$ (deg)")
    plt.ylabel("Path gain $|H(f)|^2$ (dB)")
    ax = fig.axes[0]
    ax.axvline(x=ref_boundary, ymin=0, ymax=1, color="black", linestyle="--")
    ax.axvline(x=los_boundary, ymin=0, ymax=1, color="black", linestyle="--")
    ax.text(ref_boundary-10,ymin+5,'RSB',rotation=90,va='top')
    ax.text(los_boundary-10,ymin+5,'ISB',rotation=90,va='top')
    ax.text(ref_boundary/2,ymax-2.5,'Region I', ha='center', va='center',
            bbox=dict(facecolor='none', edgecolor='black', pad=4.0))
    ax.text(los_boundary-(los_boundary-ref_boundary)/2,ymax-2.5,'Region II', ha='center', va='center',
            bbox=dict(facecolor='none', edgecolor='black', pad=4.0))
    ax.text(phi_deg[-1]-(phi_deg[-1]-los_boundary)/2,ymax-2.5,'Region III', ha='center', va='center',
            bbox=dict(facecolor='none', edgecolor='black', pad=4.0))
    plt.title('$f={}$ GHz ("{}")'.format(frequency/1e9, material))
    plt.tight_layout()
    return fig
```
```python
[13]:
```

```python
plot(1e9, "itu_metal");
```


    
The figure above shows the path gain for the total field as well as that for the different path types. In Region $I$, the line-of-sight and reflected paths dominate the total field. While their contributions are almost constant over the range of $\phi\in[0,150^\circ]$, their combined field exhibits large fluctutations due to constructive and destructive interference. As we approach the RSB, the diffracted field increases to ensure continuity at $\phi=150^\circ$, where the
reflected field immediately drops to zero. A similar observation can be made close to the ISB, where the incident (or line-of-sight) component suddenly vanishes. In Region $III$, the only field contribution comes from the diffracted field.
    
Let us now have a look at what happens when we change the frequency to $10\,$GHz.

```python
[14]:
```

```python
plot(10e9, "itu_metal");
```


    
The first observation we can make is that the overall path gain has dropped by around $20\,$dB. This is expected as it is proportional to the square of the wavelength $\lambda$.
    
The second noticeable difference is that the path gain fluctuates far more rapidly. This is simply due to the shorter wavelength.
    
The third observation we can make is that the diffracted field decays far more radpily when moving away from the boundaries as compared to a frequency of $1\,$GHz. Thus, diffraction is less important at high frequencies.
    
We can verify that the same trends continue by plotting the result for a frequency of $100\,$GHz, which is the upper limit for which the ITU Metal material is defined (see the <a class="reference external" href="https://nvlabs.github.io/sionna/api/rt.html#radio-materials">Sionna RT Documentation</a>).

```python
[15]:
```

```python
plot(100e9, "itu_metal");
```


    
It is also interesting to change the material of the wedge. The preconfigured materials in Sionna RT can be inspected with the following command:

```python
[16]:
```

```python
list(scene.radio_materials.keys())
```
```python
[16]:
```
```python
['vacuum',
 'itu_concrete',
 'itu_brick',
 'itu_plasterboard',
 'itu_wood',
 'itu_glass',
 'itu_ceiling_board',
 'itu_chipboard',
 'itu_plywood',
 'itu_marble',
 'itu_floorboard',
 'itu_metal',
 'itu_very_dry_ground',
 'itu_medium_dry_ground',
 'itu_wet_ground']
```

    
Let’s see what happens when we change the material of the wedge to wood and the frequency back to $1\,$GHz.

```python
[17]:
```

```python
plot(1e9, "itu_wood");
```


    
We immediately notice that wood is a bad reflector since the strength of the reflected path has dropped by $10\,$dB compared to the metal. Thanks to the heuristic extension of the diffracted field equations in [2] to non-perfect conductors in [4] (which are implemented in Sionna RT), the total field remains continuous.
    
You might now want to try different materials and frequencies for yourself.

