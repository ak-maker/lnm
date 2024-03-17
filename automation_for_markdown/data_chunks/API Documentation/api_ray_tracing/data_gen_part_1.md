# Ray Tracing<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#ray-tracing" title="Permalink to this headline"></a>
    
This module provides a differentiable ray tracer for radio propagation modeling.
The best way to get started is by having a look at the <a class="reference external" href="../examples/Sionna_Ray_Tracing_Introduction.html">Sionna Ray Tracing Tutorial</a>.
The <a class="reference external" href="../em_primer.html">Primer on Electromagnetics</a> provides useful background knowledge and various definitions that are used throughout the API documentation.
    
The most important component of the ray tracer is the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a>.
It has methods for the computation of propagation <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a>) and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a> (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.coverage_map" title="sionna.rt.Scene.coverage_map">`coverage_map()`</a>).
Sionna has several integrated <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#example-scenes">Example Scenes</a> that you can use for your own experiments. In this <a class="reference external" href="https://youtu.be/7xHLDxUaQ7c">video</a>, we explain how you can create your own scenes using <a class="reference external" href="https://www.openstreetmap.org">OpenStreetMap</a> and <a class="reference external" href="https://www.blender.org">Blender</a>.
You can preview a scene within a Jupyter notebook (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>) or render it to a file from the viewpoint of a camera (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render" title="sionna.rt.Scene.render">`render()`</a> or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render_to_file" title="sionna.rt.Scene.render_to_file">`render_to_file()`</a>).
    
Propagation <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> can be transformed into time-varying channel impulse responses (CIRs) via <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.cir" title="sionna.rt.Paths.cir">`cir()`</a>. The CIRs can then be used for link-level simulations in Sionna via the functions <a class="reference internal" href="channel.wireless.html#sionna.channel.cir_to_time_channel" title="sionna.channel.cir_to_time_channel">`cir_to_time_channel()`</a> or <a class="reference internal" href="channel.wireless.html#sionna.channel.cir_to_ofdm_channel" title="sionna.channel.cir_to_ofdm_channel">`cir_to_ofdm_channel()`</a>. Alternatively, you can create a dataset of CIRs that can be used by a channel model with the help of <a class="reference internal" href="channel.wireless.html#sionna.channel.CIRDataset" title="sionna.channel.CIRDataset">`CIRDataset`</a>.
    
The paper <a class="reference external" href="https://nvlabs.github.io/sionna/made_with_sionna.html#sionna-rt-differentiable-ray-tracing-for-radio-propagation-modeling">Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling</a> shows how differentiable ray tracing can be used for various optimization tasks. The related <a class="reference external" href="https://nvlabs.github.io/sionna/made_with_sionna.html#sionna-rt-differentiable-ray-tracing-for-radio-propagation-modeling">notebooks</a> can be a good starting point for your own experiments.

# Table of Content
## Scene<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#scene" title="Permalink to this headline"></a>
  
  

## Scene<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#scene" title="Permalink to this headline"></a>
    
The scene contains everything that is needed for radio propagation simulation
and rendering.
    
A scene is a collection of multiple instances of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="sionna.rt.SceneObject">`SceneObject`</a> which define
the geometry and materials of the objects in the scene.
The scene also includes transmitters (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>) and receivers (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>)
for which propagation <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> or  channel impulse responses (CIRs) can be computed,
as well as cameras (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>) for rendering.
    
A scene is loaded from a file using the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.load_scene" title="sionna.rt.load_scene">`load_scene()`</a> function.
Sionna contains a few <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#example-scenes">Example Scenes</a>.
The following code snippet shows how to load one of them and
render it through the lens of the preconfigured scene <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> “scene-cam-0”:
```python
scene = load_scene(sionna.rt.scene.munich)
scene.render(camera="scene-cam-0")
```

    
You can preview a scene in an interactive 3D viewer within a Jupyter notebook using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>:
```python
scene.preview()
```

    
In the code snippet above, the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.load_scene" title="sionna.rt.load_scene">`load_scene()`</a> function returns the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a> instance which can be used
to access scene objects, transmitters, receivers, cameras, and to set the
frequency for radio wave propagation simulation. Note that you can load only a single scene at a time.
    
It is important to understand that all transmitters in a scene share the same <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a> which can be set
through the scene property <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.tx_array" title="sionna.rt.Scene.tx_array">`tx_array`</a>. The same holds for all receivers whose <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a>
can be set through <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.rx_array" title="sionna.rt.Scene.rx_array">`rx_array`</a>. However, each transmitter and receiver can have a different position and orientation.
    
The code snippet below shows how to configure the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.tx_array" title="sionna.rt.Scene.tx_array">`tx_array`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.rx_array" title="sionna.rt.Scene.rx_array">`rx_array`</a> and
to instantiate a transmitter and receiver.
```python
# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=8,
                          num_cols=2,
                          vertical_spacing=0.7,
                          horizontal_spacing=0.5,
                          pattern="tr38901",
                          polarization="VH")
# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                          num_cols=1,
                          vertical_spacing=0.5,
                          horizontal_spacing=0.5,
                          pattern="dipole",
                          polarization="cross")
# Create transmitter
tx = Transmitter(name="tx",
              position=[8.5,21,27],
              orientation=[0,0,0])
scene.add(tx)
# Create a receiver
rx = Receiver(name="rx",
           position=[45,90,1.5],
           orientation=[0,0,0])
scene.add(rx)
# TX points towards RX
tx.look_at(rx)
print(scene.transmitters)
print(scene.receivers)
```

```python
{'tx': <sionna.rt.transmitter.Transmitter object at 0x7f83d0555d30>}
{'rx': <sionna.rt.receiver.Receiver object at 0x7f81f00ef0a0>}
```

    
Once you have loaded a scene and configured transmitters and receivers, you can use the scene method
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a> to compute propagation paths:
```python
paths = scene.compute_paths()
```

    
The output of this function is an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> and can be used to compute channel
impulse responses (CIRs) using the method <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.cir" title="sionna.rt.Paths.cir">`cir()`</a>.
You can visualize the paths within a scene by one of the following commands:
```python
scene.preview(paths=paths) # Open preview showing paths
scene.render(camera="preview", paths=paths) # Render scene with paths from preview camera
scene.render_to_file(camera="preview",
                     filename="scene.png",
                     paths=paths) # Render scene with paths to file
```

    
Note that the calls to the render functions in the code above use the “preview” camera which is configured through
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>. You can use any other <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> that you create here as well.
    
The function <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.coverage_map" title="sionna.rt.Scene.coverage_map">`coverage_map()`</a> computes a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a> for every transmitter in a scene:
```python
cm = scene.coverage_map(cm_cell_size=[1.,1.], # Configure size of each cell
                        num_samples=1e7) # Number of rays to trace
```

    
Coverage maps can be visualized in the same way as propagation paths:
```python
scene.preview(coverage_map=cm) # Open preview showing coverage map
scene.render(camera="preview", coverage_map=cm) # Render scene with coverage map
scene.render_to_file(camera="preview",
                     filename="scene.png",
                     coverage_map=cm) # Render scene with coverage map to file
```


