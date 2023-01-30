# Static Environment Representation from Radar Measurements
[detailed design document link](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/tree/main/P1_static_environment_representation/1_radar_static_environment_representation.pdf)


## Introduction
**Static environment modelling** is a key component of autonomous navigation. Unfortunately due to various **Radar** specific phenomologies like clutter, missed-detection and sparsity of the point cloud, the raw radar point cloud cannot be used like a lidar point cloud. So in this project the radar data is first upsampled by random sampling; After which the upsampled data is represented in the form of a Regular Grid. Simmilar to occupancy grid mapping, a log-odds update scheme with a degrading factor is applied for each of the valid grid cells. Here the valid grid cells are those cells whose log-odds value is above a certain threshold. Each valid grid cells is characterized by particle position and log-odd value **$(x_m, y_m, l_m)$**. It turns out this scheme results in low log-odds value for false / clutter detections, hence those can be filtered out by thresholding the log-odds. Finally we show some applications of this modelled environment by computing free-space and road boundary points using basic methods. More sophisticated methods for these application can be designed which will be a part of a different project.  



## Table of Contents <a name="t0"></a>

   - [Sensor Setup and Layout](#t1)
   - [Inputs Considered and Required Outputs](#t2)
   - [Radar Scan Visualization in Ego Vehicle frame](#t3)
   - [High Level Design](#t4)
   - [Sequence Diagram](#t5)
   - [Module Architecture](#t6)
   - [Analysis](#t7)
   - [Results, Plots and Some Observations regarding Plots](#t8)
   - [Conclusion](#t9)

<br>

### 1. Sensor Setup and Layout <a name="t1"></a>
In this project [RadarScenes](https://radar-scenes.com/) dataset is used for validating and generating results. The measurements are not synchronized and the sensor layout doesnot have a full 360&deg; coverage. Nonetheless the dataset is considered here because it is one of the few datasets publickly available that has raw radar point cloud measurements.
<br>
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/0_sensor_setups.PNG)
<br>

[Back to TOC](#t0)
<br>


### 2. Inputs Considered and Required Outputs <a name="t2"></a>
The inputs are the radar measurements in polar coordinates.
<br>
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/1_inputs_outputs.PNG)
<br>

[Back to TOC](#t0)
<br>


### 3. Radar Scan Visualization in Ego Vehicle frame <a name="t3"></a>
The below animation is a brief sequence of radar frames. It can be observed that most of the range-rate is pointed radially towards the radar location. These arrows corrospond to the stationary measurements. The arrows that points away from the sensor location or has length that appears to be of drastically different size corrosponds to measurements from dynamic objects. In this project we use the stationary measuremnets. A method for selecting stationary measurements has been discussed in this [repo](https://github.com/UditBhaskar19/EGO_MOTION_ESTIMATION/tree/main/2_egomotion_radar_polar)

[Animation for longer sequence of radar frames](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/radar_range_rate.gif)
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/radar_range_rate4.gif)
<br>

[Back to TOC](#t0)
<br>


### 4. High Level Design <a name="t4"></a>
   - **Radar $i$ Static Environment Grid Estimation $( i={1,2,3,4} )$** <a name="t41"></a> : A list of valid grid cells are estimated locally corrosponding to each of the radars. Depending on the sensor internal and mounting parameters, a part of the environment might be detected more accurately by one sensor, than the other. In such cases it was found that estimating the cell states locally, and then fusing them centrally gives a more consistent result.<br>
   - **Temporal Allignment** : Since each of the radar has its own grid state estimator, and the radars are operating asynchronously, before grid fusion step we have to represent the cell states for all the radars in the same reference frame. This allignment is achieved in the Temporal Allignment block where we do ego-motion compensation for all the cell states <br>
   - **Grid Fusion** : Finally we combine the local grid state estimates into a single grid <br><br>
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/4_architecture.PNG)
<br>

[Back to TOC](#t0)
<br>


### 5. Sequence Diagram <a name="t5"></a>
The below diagram explains the temporal sequence of the grid cell state estimation and fusion.<br>
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/4_seq_diagram.PNG)
<br>

[Back to TOC](#t0)
<br>

### 6. Module Architecture <a name="t6"></a>
The components in each of the Radar $i$ [Static Environment Grid Estimation](#t41) $( i={1,2,3,4} )$ block is as follows
   - **Stationary Measurement Identification** : The stationary measurements are identified. First the predicted range-rate for stationarity case at each measurement (x,y) location is computed. If the measurement range-rate and the predicted range-rate is 'close' within a certain margin, then that measurement is considered for further processing. Vehicle odometry is utilized for computing the predicted range-rate. <br>
   - **Clutter Removal by RANSAC** : After an preliminary selection of the stationary measurements, Random Sample Consensus (RANSAC) is used to remove clutter measurements. <br>
   - **Convert Measurement from polar to cartesian** : The selected measurements are converted from polar to cartesian coordinates. <br>
   - **Coordinate Transformation Sensor frame to Vehicle Frame** : Here the measurements are coordinate transformed from sensor frame to vehicle frame. <br>
   - **Compute Measurement Grid** : The measurements are first upsampled by random sampling, the probability (weight) and the corrosponding log-odds is computed for each of the samples. The samples are put in the grid cells and we pass the sample position and log-odds $(x_i, y_i, l_i)$ as the output. <br>
   - **Predict Grid States** : Before we can do grid cell state update, the grid cell state in the previous time $(t-1)$ is predicted using ego vehicle localization information at time $(t-1)$ and $t$ so that the grid cell measurements at time $t$ and the previous cell states at time $(t-1)$ are in same ego vehicle frame at current time $t$. <br> 
   - **Update Grid State** : The grid cell measurements and the predicted grid cell states are gated and updated. Since the grid is rectangular with uniformly sized cells. Each grid cell can be indexed like an image leading to efficient gating and state updates. different rules for state update is applied depending on whether the cells are gated , not gated , inside active sensor FOV or outside active sensor FOV. The state update equations are listed below.<br>
        1. Un-Gated Measurement Grid Cell IDs (Initialize new Grid Cell States) <br>
        2. Gated Grid Cell IDs <br>
        3. Un-Gated Predicted Grid Cell within active sensor FOV <br>
        4. Un-Gated Predicted Grid Cell outside active sensor FOV <br>

![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/4_mod_arc.PNG)
<br>

[Back to TOC](#t0)
<br>


### 5. Analysis <a name="t7"></a>
In this section some analysis is done to highlight the importance of two modules in the architecture: **Stationary Measurement Identification** & **Clutter Removal by RANSAC**
   - First, two estimation results are compared, one with and the other without the above two mentioned modules. The plot shows that the system would result in a total failure without these two modules.<br><br>
![](https://github.com/UditBhaskar19/EGO_MOTION_ESTIMATION/blob/main/2_egomotion_radar_polar/readme_artifacts/plot4.PNG)

   - We then compare the measurement range-rates with the predicted range-rates computed from the estimated radar ego-motion (vx, vy). Here the ego-motion is computed by considering all the measurements.<br>
Basically we are computing **$vr_{pred} = -( v_x * cos(theta_{meas}) + v_y * sin(theta_{meas}) )$** and plotting **$vr_{meas}$** & **$vr_{pred}$**. <br><br>
![](https://github.com/UditBhaskar19/EGO_MOTION_ESTIMATION/blob/main/2_egomotion_radar_polar/readme_artifacts/plot_misfit.PNG)

   - Next it is shown how odometry information or some other prior ego-motion estimates can be used to select only those measurements that are most likely stationary. Since RANSAC works better if a significant portion of the data are inliers, this measurement gating step is crucial. <br><br>
![](https://github.com/UditBhaskar19/EGO_MOTION_ESTIMATION/blob/main/2_egomotion_radar_polar/readme_artifacts/plot_odom_prior.PNG)

   - Finally we plot the measurements selected by RANSAC, and as seen below, the predicted range-rate line passes through the stationary measurements.
![](https://github.com/UditBhaskar19/EGO_MOTION_ESTIMATION/blob/main/2_egomotion_radar_polar/readme_artifacts/plot_ransac.PNG)
<br>

[Back to TOC](#t0)
<br>


### 6. Results , Plots and Some Observations regarding Plots ( RadarScenes - scene 105 ) <a name="t8"></a>
   - **Ego motion estimation output Plot** : The estimated yaw-rate seems to be more noisy than the estimated vx<br>
![](https://github.com/UditBhaskar19/EGO_MOTION_ESTIMATION/blob/main/2_egomotion_radar_polar/readme_artifacts/2_plots_results.PNG)

   - **Comparing OLS and KF estimates** :<br>
![](https://github.com/UditBhaskar19/EGO_MOTION_ESTIMATION/blob/main/2_egomotion_radar_polar/readme_artifacts/1_plots_results.PNG)
<br>

[Back to TOC](#t0)
<br>


### 7. Conclusion <a name="t9"></a>
Overall the presented approach for ego-motion estimation looks promising. Further details can be found in the [document](https://github.com/UditBhaskar19/EGO_MOTION_ESTIMATION/blob/main/2_egomotion_radar_polar/1_radar_ego_motion_polar.pdf)
<br>

[Back to TOC](#t0)
<br>


