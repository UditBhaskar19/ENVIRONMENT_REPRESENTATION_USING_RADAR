
# Grid based representation for Radar perception functions
[Python Code](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/tree/main/P1_static_environment_representation/python) <br>




## Introduction
A **Grid** is a discrete representation of a region in space. It is one of the key components for the AD/ADAS, as it can aid in the following functions.
   - Characterise the perceptual environment around the ego vehicle in-terms of various sensor parameters like accuracy, resolution, FoV boundary limits, FoV overlap regions in-case of multisensor setup. These quantifications can be useful for track management and track quality computation in the Object tracking applications.
   - Compute the clutter intensity for a specific radar sensor empirically from measurements collected offline during data collection phase. The clutter map can be useful for data-association in the Object tracking applications.
   - Integrate multiple sensor scans for Occupancy Grid Mapping.
   - Represent the environment for deep-learning based detection, classification and tracking.

The primary focus of this project is on defining and computing grid in a systematic way. Once the grid is defined and described, two applications are designed that makes use of Grid.
   - **Sensor FoV overlap determination**
   - **Computation of clutter intensity** at different regions in the sensor FoV grid empirically from measurements.
   <br>

   ![](https://github.com/UditBhaskar19/temp/blob/main/readme_artifacts/12_output.PNG)

   <br> 

The results are based on [NuScenes](https://www.nuscenes.org/) dataset.
<br><br>



## Table of Contents <a name="t0"></a>

   - [Defining the Grid](#t1)
   - [Useful quantities derived from Grid definition parameters](#t2)
   - [How the Grid actually helps us in efficient computation](#t3)
   - [Application 1 : Sensor FoV overlap determination](#t4)
   - [Application 2 : Computation of clutter density](#t5)
<br>


## 1. Defining the Grid <a name="t1"></a>
A 2D grid can be defined using the following parameters:
   - $X_{min}, X_{max}$ : Grid longitudinal limits
   - $Y_{min}, Y_{max}$ : Grid lateral limits
   - $\Delta x, \Delta y$ : Grid cell resolutions

Here the assumption is that an arbitrary point $(x, y)$ is considered to be within a Grid if it satisfies the below conditions.
$$X_{min} \leq x < X_{max} + \varepsilon_o$$
$$Y_{min} \leq y < Y_{max} + \varepsilon_o$$
$$\varepsilon_o \to some \ small \ number $$ 

![](https://github.com/UditBhaskar19/temp/blob/main/readme_artifacts/1_grid.PNG)

[Back to TOC](#t0)
<br><br>




## 2. Useful quantities derived from Grid definition parameters <a name="t2"></a>
The following quantities can be derived from the above defined parameters.
   
   - **Number of Grid cells along $X$ and $Y$**

   $$N_X = \bigg \lceil \dfrac{X_{max} + \varepsilon_o - X_{min}}{\Delta x} \bigg \rceil$$ 
   $$N_Y = \bigg \lceil \dfrac{Y_{max} + \varepsilon_o - Y_{min}}{\Delta y} \bigg \rceil$$  

   - **Number of cells**
   $$N = N_X * N_Y$$ 

   - **Cell XY ID of a point $(x_{coord}, y_{coord})$**
   $$C_X = \bigg \lfloor \dfrac{x_{coord} - X_{min}}{\Delta x} \bigg \rfloor$$ 
   $$C_Y = \bigg \lfloor \dfrac{y_{coord} - Y_{min}}{\Delta y} \bigg \rfloor$$ 

   - **Cell scalar ID**
   $$C_{ID} = C_X * N_Y + C_Y$$

   - **Cell XY ID from Cell scalar ID**
   $$C_X = \bigg \lfloor \dfrac{C_{ID}}{N_Y} \bigg \rfloor$$
   $$C_Y = C_{ID} - C_X * N_Y$$

   - **Cell XY Center Coordinate from Cell XY ID**
   $$C_{xcoord} = (C_X + 0.5)\Delta x + X_{min}$$
   $$C_{ycoord} = (C_Y + 0.5)\Delta y + Y_{min}$$

<br>

[Back to TOC](#t0)
<br><br>




## 3. How the Grid actually helps us in efficient computation <a name="t3"></a>
The formulas above helps in **mapping** a point $(x_{coord}, y_{coord})$ to a cell ID $(C_{ID})$ in **constant time** ( O(1) ). Hence the Grid can be used as an computationally efficient lookup table for various functions like integrating multiple scans to the grid, retrieving position dependent measurement parameters like clutter intensity, sensor FoV overlap region etc.

![](https://github.com/UditBhaskar19/temp/blob/main/readme_artifacts/2_lookup_table.PNG)

[Back to TOC](#t0)
<br><br>





## 4. Application 1 : Sensor FoV overlap determination <a name="t4"></a>
   ### Introduction
   In a multisensor setup on the ego-vehicle it is often needed to determine if a point is within the FoV of a sensor. Such information is used to determine if an object is in the blind zone, out of FoV or if it is within the FoV of multiple sensors. Such information can be easily computed if we represent the sensor FoV coverage as a grid of boolean values. The below figure illustrates the concept.
   <br><br><br>
   ![](https://github.com/UditBhaskar19/temp/blob/main/readme_artifacts/2_app1_concept.PNG)
   <br>

   ### Design : [Link](https://github.com/UditBhaskar19/temp/blob/main/application1_design.pdf) 

   ### Output Plot 
   <br> 

   ![](https://github.com/UditBhaskar19/temp/blob/main/readme_artifacts/10_individual_rad_fov.PNG)

   ![](https://github.com/UditBhaskar19/temp/blob/main/readme_artifacts/10_fov.PNG)


[Back to TOC](#t0)
<br><br>




## 5. Application 2 : Computation of clutter density <a name="t5"></a>
   ### Introduction
   It is typical for radar sensor to give clutter measurements. Although it is often assumed that the clutters are distributed uniformly within the sensor FoV, in reality such assumptions are not true. For example if we analyse the clutter distribution from [NuScenes](https://www.nuscenes.org/) radar dataset, we find that the number of clutters are much more near the boundary of the sensor FoV; also the short range sensor mode has much more clutters than the long range mode. A scatter plot of the clutter measurements accumulated from multiple frames is shown below. It clearly indicated the non-uniformity of the clutter distritution in space.
   <br><br> 
   ![](https://github.com/UditBhaskar19/temp/blob/main/readme_artifacts/9_clutters4.PNG)
   <br>
   Hence in this application we use the grid and the sliding window concept to empirically estimate the clutter intensity
   <br>

   ### Design [Link](https://github.com/UditBhaskar19/temp/blob/main/application2_design.pdf) 

   ### Output Visualization
   <br> 

   ![](https://github.com/UditBhaskar19/temp/blob/main/readme_artifacts/11_clutter_map.PNG)


[Back to TOC](#t0)
<br><br> 


