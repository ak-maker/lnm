INSTRUCTION: Understand the `Scene` component's significance in the Sionna ray tracer by studying the `Scene` class, focusing on the `compute_paths()` and `coverage_map()` methods used for propagation modeling.
ANSWER:The `Scene` component is a pivotal element of the Sionna ray tracer, which is a tool designed for radio propagation modeling. A `Scene` in this context likely represents an environment with certain geometries and materials that affect how radio waves propagate, and it connects the physical layout with the computational models that simulate the behavior of radio waves within that environment.

Two key methods of the `Scene` class are `compute_paths()` and `coverage_map()`, both central to the process of propagation modeling:

1. `compute_paths()`: This method is critical for simulating how radio waves propagate from a transmitter to a receiver by calculating the `Paths` that waves can take through the scene. These `Paths` may include direct line-of-sight, reflected, refracted, or diffracted paths, and would encompass information on how the waves interact with various objects and materials in the scene. This could include the angles of arrival and departure, delay, attenuation, phase shift, and so on, which are all important for constructing channel models – representations of the propagation channel.

2. `coverage_map()`: Likely uses the information on propagation paths to compute a `CoverageMap`. This is probably an analysis of the signal strength or quality across different regions of the scene. It could be used to visualize areas of strong and weak signal reception or to optimize the placement and configuration of transmitters to achieve desired coverage and performance characteristics.

Together, these methods allow users to model and understand the complex behavior of radio waves in a given environment, which is fundamental for designing and optimizing wireless networks. For example, in an urban environment, identifying paths that radio waves can take around buildings can inform network design to minimize dead zones and interference. The `coverage_map()` could then be employed to evaluate the efficacy of these design choices.

Using these methods with the Sionna ray tracer, researchers and engineers can carry out advanced simulations to predict and analyze the performance of wireless communication systems in a variety of scenarios, which can have applications ranging from cellular network planning to the design of communication protocols for complex environments like smart cities or automated factories. The `Scene` class, with its `compute_paths()` and `coverage_map()` methods, provides the essential services of propagation path calculation and coverage analysis, and is therefore central to the ray tracing module's capabilities for radio propagation modeling.

INSTRUCTION: Learn how to use the Example Scenes provided by Sionna and investigate how to create custom scenes using OpenStreetMap and Blender for your experiments.
ANSWER:To use the Example Scenes provided by Sionna for ray tracing and to create custom scenes using OpenStreetMap and Blender, follow these steps:

1. **Getting Started with Sionna Ray Tracing Tutorial:**
   Begin with the [Sionna Ray Tracing Tutorial](https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html). This tutorial provides an introduction to Sionna’s ray tracing module and should familiarize you with the basic concepts and functions like `Scene`, `Paths`, and `CoverageMap`, which are crucial for creating and analyzing radio propagation models.

2. **Understanding Example Scenes:**
   Sionna provides a set of Example Scenes for experimentation. You can access these scenes through the [documentation](https://nvlabs.github.io/sionna/api/rt.html#example-scenes). Study these examples to understand how scenes are structured and how the ray tracing functions are used within them. The examples should also show you how to visualize a scene within a Jupyter notebook using `Scene.preview()` and how to render it from the viewpoint of a camera using `Scene.render()` or `Scene.render_to_file()`.

3. **Creating Custom Scenes:**
   There is a [video](https://youtu.be/7xHLDxUaQ7c) provided by Sionna that explains how to create your own scenes using OpenStreetMap and Blender. Watch this video to learn how to:

   - Extract geographical data from OpenStreetMap for the area you are interested in simulating.
   - Import that data into Blender, which is a free and open-source 3D creation suite.
   - Use Blender to create a 3D model of the environment. You would refine the geometry into a format that is suitable for use in Sionna’s ray tracing simulations.

4. **Coding Custom Scenes:**
   Once you have a 3D model from Blender, you would typically export it in a format compatible with Sionna, which might involve converting the model into a mesh of triangles that represents the various surfaces for your ray tracing simulations.

5. **Utilizing Ray Tracing in Sionna:**
   After constructing the scene, you'll need to use Sionna's ray tracing functions to compute propagation paths (`compute_paths()`) and create coverage maps (`coverage_map()`). Learn how these functions work by referring to the Sionna documentation and the given examples.

6. **Analyzing Channel Impulse Responses:**
   Using the paths computed by the ray tracer, you can transform them into time-varying channel impulse responses (CIRs) using the `cir()` function. Subsequently, these CIRs can be employed in link-level simulations using functions like `cir_to_time_channel()` or `cir_to_ofdm_channel()`, or you can create a dataset of CIRs using `CIRDataset`.

7. **Leveraging Sionna's Scientific Papers and Notebooks:**
   For advanced use cases and further research, you may explore Sionna’s related [scientific papers and notebooks](https://nvlabs.github.io/sionna/made_with_sionna.html#sionna-rt-differentiable-ray-tracing-for-radio-propagation-modeling). These resources show how differentiable ray tracing is utilized for various optimization tasks and provide practical examples that can serve as a starting point for your own work.

These steps should help you learn how to use and create scenes with the Sionna ray tracing module, leveraging the inbuilt examples, and extend your work to incorporate complex environments using OpenStreetMap and Blender. Remember to consult the Sionna documentation for detailed information on the functions and classes that you'll be using.

INSTRUCTION: Discover how the `Scene.preview()` and `Scene.render_to_file()` methods can be used within a Jupyter notebook to visualize scenes and the impact of different viewpoints.
ANSWER:The `Scene.preview()` and `Scene.render_to_file()` methods mentioned in the context are part of an API for a differentiable ray tracer used for radio propagation modeling. To use these methods within a Jupyter notebook to visualize scenes and the impact of different viewpoints, follow the steps below:

1. **Scene.preview()**:
   - The `Scene.preview()` method allows you to quickly visualize the 3D scene directly within the Jupyter notebook interface. This is helpful for getting a quick look at the scene layout, locations of objects, and defining the viewpoints of interest.
   - To use this method, you would first need to create a `Scene` object, which represents your 3D environment. Assuming `Scene` is a class in the ray tracer module you're using, you would instantiate it accordingly.
   - After setting up the scene with necessary configurations (like adding objects, defining materials, etc.), you call the `.preview()` method on the scene object to display a 3D preview within the notebook.
   
   Example (assuming you have instantiated a scene object named `scene`):
   ```python
   scene.preview()
   ```

2. **Scene.render_to_file()**:
   - The `Scene.render_to_file()` method is used to render the scene from a specific viewpoint and save the rendered image to a file. This method is especially useful when you want to generate images from different camera positions or viewpoints to study the impact of various geometries and angles on signal propagation.
   - Just like with `.preview()`, you need a configured `Scene` object.
   - You then define the camera viewpoint within the scene—this could be a specific location and orientation that you want to visualize the scene from.
   - You call the `.render_to_file()` method, passing parameters such as the filename to save the image, and camera configuration settings like position, look-at-point, field of view, etc., as required by the method.
   
   Example (assuming you have a `scene` object and defined a camera viewpoint `camera_viewpoint`):
   ```python
   scene.render_to_file(viewpoint=camera_viewpoint, filename="scene_view.png")
   ```

To effectively visualize different viewpoints and their impact on the scene, you can create multiple camera configurations, rendering each one to a file to compare how the scene looks from those different angles. This way, you can analyze how viewpoints affect the understanding of the radio propagation and potentially optimize the configuration for better signal modeling.

Keep in mind that in order to run these methods, you'll need the necessary libraries and modules installed and imported in your Jupyter notebook. Additionally, the `Scene` object needs to be properly configured with the geometries, materials, and other required details relevant for ray tracing in your specific application. Refer to the documentation of the ray tracing API you're using for detailed instructions on setting up the `Scene` object and for the specific parameters that the `preview()` and `render_to_file()` methods might require.

INSTRUCTION: Practice transforming Propagation Paths into channel impulse responses (CIRs) using the `Paths.cir()` method and understand how to use the resulting CIRs for link-level simulations with `cir_to_time_channel()` or `cir_to_ofdm_channel()` functions.
ANSWER:To transform Propagation Paths into channel impulse responses (CIRs) and use them for link-level simulations in a radio propagation model, you'll follow a high-level process that goes something like this:

1. **Compute Propagation Paths**: You start with a `Scene`, using the `compute_paths()` method to compute Propagation Paths between transmitters and receivers within the scene. These paths represent the various ways radio waves propagate from the transmitter to the receiver, such as direct paths, reflected paths, and so on.

2. **Transform Paths to CIRs**: Once you have the Propagation Paths calculated, you can transform them into Channel Impulse Responses (CIRs) using the `Paths.cir()` method. The CIR is a time-domain representation that characterizes how a signal would be affected by the channel. It encapsulates the time delays, amplitudes, and phases of the multiple propagation paths.

3. **Simulate Time and Frequency Channels**: You can then use the CIRs obtained for further simulations. Depending on the type of link-level simulation you are interested in, you could:
    - Use `cir_to_time_channel()` to simulate a time-variant channel for, say, generic modulation schemes.
    - Use `cir_to_ofdm_channel()` when you are working with OFDM (Orthogonal Frequency Division Multiplexing) schemes, which are widely used in modern communications systems like LTE and WiFi.

4. **Experiment with Simulated Channels**: After setting up your simulated channels, you can proceed to perform various experiments such as testing the performance of different communication algorithms or assessing the impact of channel conditions on signal quality.

Here’s a rough outline in code, assuming you have already set up a `Scene` and have the Sionna package installed:

```python
# Assume `scene` is an instance of `sionna.rt.Scene` and has been properly set up.
# Compute paths using the Scene object
paths = scene.compute_paths()

# Transform paths into CIRs
cir = paths.cir()

# Now you can use the resulting CIR for link-level simulations:
# For time-variant channel simulation
time_channel = cir_to_time_channel(cir)

# For OFDM channel simulation
ofdm_channel = cir_to_ofdm_channel(cir)
```

Remember, the code provided here is only illustrative. In practice, you would need to fill in additional parameters and handle the setup required by Sionna for the creation of the scene, the calculations of paths, and subsequent transformations based on the specific details of your simulation environment. Always consult the official documentation and tutorials of Sionna for detailed instructions and guidelines.

INSTRUCTION: Explore how to create a dataset of CIRs that can be used by a channel model using the `CIRDataset` class and understand how this can benefit model training.
ANSWER:To create a dataset of Channel Impulse Responses (CIRs) that can be used by a channel model, you would likely use the `CIRDataset` class in the context provided. Using such a dataset can greatly benefit model training by providing a realistic representation of the propagation environment, captured through the impulse responses between transmitters and receivers. This kind of dataset can be particularly useful for training machine learning models that aim to predict channel characteristics or for simulating communication system performance under various conditions.

Here’s a general approach to creating a dataset of CIRs using the capabilities outlined in the provided context:

1. **Define a Scene**: Initially, you would define a `Scene` using ray tracing. The `Scene` class contains methods for computing propagation paths (`compute_paths()`) and coverage maps (`coverage_map()`). In reality, the scene would represent your environment with all the physical objects, materials, and transmitter and receiver locations.

2. **Compute Paths**: Once the scene is set up, use the `compute_paths()` method of the `Scene` to compute propagation paths for the radio signals. These paths represent the routes that radio waves take as they travel from the transmitter to the receiver, including reflections, diffractions, and scatterings.

3. **Generate CIRs**: Transform these propagation paths into CIRs using the `cir()` method. The computation of CIRs is key as they represent how the channel affects the transmitted signal over time, which is crucial information for characterizing the channel behavior for wireless communication analysis and simulation.

4. **Create a CIRDataset**: When you have a collection of CIRs, you can organize them into a `CIRDataset`. While the code instruction for creating a `CIRDataset` is not provided in the context, it is suggested that such a class exists and facilitates the handling and storage of CIRs in a format that can be easily used by channel models.

5. **Training the Model**: With the `CIRDataset`, you can proceed to train channel models, whether they are physics-based or data-driven machine learning models. Having a realistic dataset can significantly improve the model's ability to generalize to different environments and predict the channel's response accurately.

Understanding how each step contributes to model training:

- **Realism**: By creating CIRs based on a detailed model of the environment, the resulting dataset encapsulates realistic channel effects.
- **Diversity**: Different scenes and configurations can be used to generate CIRs under various conditions, leading to a diverse training set.
- **Accurate Simulations**: Using CIRs for simulations can yield accurate performance predictions for communication systems in the considered environments.
- **Machine Learning**: For data-driven models, the `CIRDataset` provides high-quality labeled data that are key to training algorithms.

It's important to note that the details of creating CIRs and the `CIRDataset` would typically be extracted from the complete API documentation, examples, and tutorials provided by the library you are working with, such as Sionna. This would include further code examples and specific instructions on how to parameterize scenes, path computation, CIR transformation, and dataset management.

INSTRUCTION: Study the research paper "Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling" to examine how differentiable ray tracing can be utilized for optimization tasks in radio propagation modeling.
ANSWER:To examine how differentiable ray tracing can be utilized for optimization tasks in radio propagation modeling, one must first understand the core principles and functionality behind the "Sionna RT" ray tracing module, as discussed in the paper "Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling."

Ray tracing is a technique used to model the propagation of radio waves in an environment by simulating the paths that these waves take when reflecting, diffracting, or scattering from various obstacles. A 'Scene' is essential for ray tracing, as it represents the environment where the propagation is modeled. Within a 'Scene', methods such as 'compute_paths()' calculate the paths that rays take from a transmitter to a receiver, while 'coverage_map()' analyzes coverage over a geographic area.

Differentiable ray tracing refers to the ability to compute gradients of the ray tracing simulation outcomes with respect to various parameters of the environment. This capability is significant for optimization tasks as it allows for the application of gradient-based optimization techniques. By adjusting the parameters in the direction of the gradient, one can optimize for better signal coverage, reduced interference, and improved localization, among other objectives.

The differentiability comes into play particularly with respect to the scattering models that describe how radio waves interact with surfaces. For instance, classes like 'LambertianPattern,' 'DirectivePattern,' and 'BackscatteringPattern' provide models for different scattering behaviors. Each of these models represents a different mathematical abstraction of how the waves scatter, and parameters within these models might be subject to optimization.

For instance, the 'BackscatteringPattern' includes parameters such as `alpha_r` and `alpha_i`, which relate to the width of the scattering lobes in the direction of the specular reflection and incoming direction, respectively, and `lambda_`, which determines the percentage of diffusely reflected energy around the specular reflection. These parameters can be variables or even outputs of a callable, such as a neural network layer, signifying the potential to integrate learnable models into the ray-tracing simulation.

In an optimization context, if one were looking to optimize the parameter `lambda_` to minimize interference or maximize signal strength at a receiver, the differentiable nature of the ray tracer would permit the calculation of the gradient of the objective function with respect to `lambda_`. An optimization algorithm could then adjust `lambda_` iteratively to find an optimal value.

The paper likely provides further insights into how these differentiable aspects of ray tracing have been applied to specific optimization tasks in radio propagation modeling. It might discuss experimental setups, results, and the potential for applying these techniques to real-world scenarios in wireless communication design and deployment. Additionally, the related notebooks mentioned in the context likely offer practical, hands-on examples of how to implement and utilize differentiable ray tracing for specific optimization goals.

Keep in mind that a detailed study of the paper itself, which isn't provided in the context, would be necessary to gain a full understanding of the methodologies, experiments, results, and implications of the research presented in "Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling."

INSTRUCTION: Implement the computation of a `coverage_map` using the `scene.coverage_map()` method in Python, noting parameters such as `rx_orientation`, `max_depth`, and `num_samples`, and grasp their effects on the resulting coverage map.
ANSWER:To implement the computation of a `coverage_map` using the `scene.coverage_map()` method in Python, first ensure you have a scene object created which is an instance of the `Scene` class from the Sionna ray tracing module. The `scene.coverage_map()` method is likely used to compute a coverage map for radio propagation modeling within a defined scene, which maps the signal coverage based on various parameters including the receiver orientation, the maximum depth for ray tracing, and the number of samples considered for estimation.

Although the specific Python code to compute the `coverage_map` is not provided in your context, I can describe how the method might be used, based on the mentioned parameters:

- `rx_orientation`: This parameter is likely to represent the orientation of the receiver in the scene. The orientation could be crucial in determining how the receiver 'sees' the incoming signals, influencing the resulting coverage map. It might be defined by angles representing the azimuth and elevation, or a quaternion representing the receiver's rotation in 3D space.

- `max_depth`: This parameter probably refers to the maximum depth of ray-tracing, essentially how many surfaces the rays are allowed to bounce off before ceasing calculations. A greater `max_depth` would result in a more detailed and accurate representation of the coverage, but at the cost of increased computation time.

- `num_samples`: This parameter signifies the number of samples to be used in the calculation of the coverage. Higher `num_samples` could increase the accuracy of the coverage map by considering more potential paths for the signals to take but would similarly increase the computational effort.

Given these parameters, the `coverage_map` computation might be structured as follows in Python, assuming you have a `Scene` instance:

```python
# Assuming `scene` is an instance of the `Scene` class from Sionna ray tracing module
coverage_map = scene.coverage_map(rx_orientation=my_orientation,
                                  max_depth=my_max_depth,
                                  num_samples=my_num_samples)
```

In this hypothetical code snippet:

- `my_orientation` should be replaced with the actual orientation of the receiver.
- `my_max_depth` should be a value representing the desired maximum number of surface reflections.
- `my_num_samples` should be the number of samples you want to use to compute the coverage.

Please refer to the Sionna ray tracing module documentation for the exact syntax and available options since the actual method signature and parameter names may vary. The documentation will also provide details on acceptable value ranges and data types for these parameters. Additionally, the documentation may include examples that can help illustrate the proper use of the `coverage_map()` method in a practical context.

INSTRUCTION: Learn how to load a scene from a file using the `load_scene()` function, noting the importance of specifying the scene file name and the data type (dtype) for internal computations.
ANSWER:To load a scene from a file using the `load_scene()` function, you will need to follow these general steps, while taking into account the need to specify the scene file name and the data type (dtype) for the internal computations:

1. **Import the necessary module**: First, ensure that you have the required module imported that contains the `load_scene()` function. This is likely to be a part of a ray tracing library or a specific module designed for scene management in the context you are working with.

2. **Specify the scene file name**: You need to know the file name and the path of the scene file you intend to load. The scene file would typically contain the description of the scene geometry, materials, lighting, and other properties required to define the environment for your simulations.

3. **Specify the data type (dtype)**: The internal computations for a ray tracing simulation can be performed using different data types, typically floating-point numbers, which can have different levels of precision. In the provided context, `tf.complex64` and `tf.complex128` are mentioned as possible data types, which correspond to complex numbers with 64-bit and 128-bit precision, respectively. The choice of `dtype` may affect the accuracy and performance of your simulations. `tf.complex64` is the default dtype, which represents a complex number with both real and imaginary parts as 32-bit floating-point numbers. For higher precision, you could use `tf.complex128`, but this would increase computational demands.

4. **Call the `load_scene()` function**: With the file name and data type determined, you can call the `load_scene()` function, passing these as arguments. If the function is not directly shown in the given context, it might look something like this in practice:

```python
scene = load_scene(file_name="path/to/your/scene_file", dtype=tf.complex64)
```

Remember to replace `"path/to/your/scene_file"` with the actual file path of your scene file.

5. **Use the loaded scene**: After loading the scene, you can utilize it to perform various operations such as computing propagation paths, rendering the scene, or simulating ray-tracing based on the functionalities provided by the ray tracing module.

Please note that the specific syntax, function names, and module names may vary depending on the particular ray tracing library or API you are using. However, the general process described here should be adaptable to most ray tracing or scene management setups. If the library you are using is Sionna or similar, refer to their specific documentation for the exact functions and parameters required to load a scene from a file.

INSTRUCTION: Execute a provided Python example code to create and configure antenna arrays for transmitters and receivers within a scene, add a transmitter, calculate a coverage map, and visualize the coverage in a preview.
ANSWER:To execute the provided Python example code and perform the given tasks, you would typically follow these steps:

1. **Setting Up the Environment:**
   Ensure that you have Python installed on your system, along with the necessary libraries mentioned in the context, such as TensorFlow and any specific packages related to ray tracing and visualization.

2. **Importing Libraries:**
   Import the required libraries to your Python script. This may include TensorFlow (typically imported as `tf`), and specific classes from the Sionna library related to ray tracing, like `Scene`, `CoverageMap`, and visualization utilities.

3. **Creating and Configuring Antenna Arrays:**
   Instantiate and configure the antenna arrays for both the transmitter and the receiver using the appropriate classes and methods provided by the Sionna library.

4. **Adding a Transmitter to the Scene:**
   Create a transmitter object and configure it with the necessary parameters such as its location, transmission power, and antenna configuration.

5. **Creating a Scene:**
   Create a `Scene` object which is central to the ray tracing process. The scene includes the physical layout, transmitters, receivers, and materials defining the propagation environment.

6. **Calculating Coverage Map:**
   Use the `coverage_map()` method of the `Scene` object to calculate the coverage map. This method will internally compute the paths and predict signal coverage over the specified area.

7. **Visualizing Coverage in a Preview:**
   Utilize the `preview()` method of the `Scene` object to visualize the coverage within a Jupyter notebook or use `render()` or `render_to_file()` to create a visualization file.

The context you have given does not include the example code itself. With more specific details or actual Python code examples, I could provide you with more detailed guidance on executing each step. If you have access to such an example code, please provide it, and I'll assist you further in executing those specific instructions.

Without the actual code or a clear indication of where the code can be found (e.g., an URL to a tutorial or example), it's not possible to provide you with working Python code. However, you'd typically follow this high-level workflow, aligning your script to the API and examples given in the Sionna library documentation.

INSTRUCTION: Absorb the mathematical foundations behind path coefficient calculations, specifically focusing on how angles of departure and arrival are used along with antenna positions to determine phase shifts and the channel matrix for a given ray path.
ANSWER:Path coefficient calculations are a critical part of understanding the behavior of radio waves as they propagate through an environment, especially in complex scenarios such as urban areas or inside buildings. These coefficients are used to model the propagation path of a radio signal from the transmitter to the receiver, taking into account the interactions with the surrounding environment, like reflections, diffractions, and scattering. The "ray tracer" module appears to be part of a toolset (likely Sionna) for radio propagation modeling using differentiable ray tracing techniques.

Angles of departure and arrival: These angles describe the direction from which a radio wave departs from the transmitter (angle of departure) and the direction in which it arrives at the receiver (angle of arrival). When radio waves interact with objects, such as bouncing off a building or diffracting around a corner, these angles change. Accurately determining these angles is essential for path coefficient calculations because they influence the phase shifts observed at the receiver.

Phase shifts: Phase shifts occur due to the difference in path lengths that each ray travels from the transmitter to the receiver, resulting in signals arriving at different times and potentially interfering with one another. The phase shift for a given path can be calculated by considering the distance the signal has traveled and the wavelength of the signal. It is directly related to the time it takes for the signal to travel from one point to another.

Antenna positions: The physical locations of the antennas (both transmitting and receiving) play a significant role in the ray-tracing process because they define the starting and ending points of the paths that need to be calculated. The relative positions and orientations of the antennas affect the angles of departure and arrival and, consequently, the phase shifts and signal strengths.

Channel Matrix: The channel matrix is a mathematical representation of the effect the propagation environment has on the radio signals between the transmitter and receiver. Each element of the channel matrix represents the complex path gain, which includes both amplitude attenuation and phase shift, for each transmitter-receiver antenna pair. The channel matrix summarizes the behavior of all the paths that were considered in the ray-tracing process and is used to predict the signal at the receiver and to understand multipath effects.

The provided context mentions tools like `Scene`, `Paths`, and different scattering patterns such as `LambertianPattern`, `DirectivePattern`, and `BackscatteringPattern`, which are used in the ray-tracing process. These tools and patterns help define how radio waves interact with the environment and are used to calculate the coefficients and channel matrix. While the context does not provide the exact mathematical formulas, these would typically involve calculations of the electromagnetic wavefront's parameters—amplitude, phase, and polarization—based on the setup of the scene and interactions with the materials and objects within it.

It also seems that the ray tracer is differentiable, suggesting that it can be used in optimization tasks where gradients are required, like in machine learning. This differentiation capability allows for the optimization of certain parameters within the environment or signal processing algorithms to improve system performance.

In summary, the path coefficient calculation process involves complex interactions of the angles of departure and arrival, antenna positions, and the surrounding environment that influence the phase shifts of the signal, culminating in the computation of the channel matrix that represents all these effects for a given ray path. The precise mathematics would require an in-depth understanding of electromagnetics, signal processing, and the specific models and algorithms used by the ray-tracing software, such as the one described in the context.

