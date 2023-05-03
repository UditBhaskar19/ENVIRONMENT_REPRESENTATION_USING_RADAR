# Object Detection by Accumulaing and Clustering multiple radar frames
[Python Code](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/tree/main/P2_object_detection_by_clustering_multiple_scans_from_radar/python) <br>
[Result Videos](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/tree/main/P2_object_detection_by_clustering_multiple_scans_from_radar/result_videos)




## Introduction
**Clustering** is a key component for radar based object tracking. When a radar returns multiple measurements from an object, it is necessary to group the measurements so that a unique id or a track can be initialized for that object. Here we employ **DBSCAN** clustering algorithm since it can identify clusters with elongated and non-convex shapes. Due to the sparcity of the radar measurements, DBSCAN often creates multiple clusters per object or erronously identify the object as clutter. Hence in this project we utilize not just the current radar frame, but a sequence of $k$ frames to perform clustering. We create a **accumulated frame buffer** with size $k$. In this buffer we maintain a history of $k-1$ frames and the current $k^{th}$ frame. These $k-1$ frames are extrapolated to the current timestep $t$ so that the entire accumulated frame buffer is temporally alligned. This **frame accumulation** would result in a dense point cloud of radar measurements which is subsequently taken as input to the DBSCAN Algorithm. Here the results are based on [NuScenes](https://www.nuscenes.org/) radar dataset and is shown for individual radars ( front and rear-left ). The concepts can be extended to multiple synchronous radars. Optionally in case we are performing clustering individually for each sensors, one may employ a cluster merge step to combine clusters from overlapped sensor FoVs.

![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/0_output.png)
  




## Table of Contents <a name="t0"></a>

   - [Sensor Setup and Layout](#t1)
   - [Inputs and Outputs](#t2)
   - [Radar Scan Visualization in Ego Vehicle frame](#t3)
   - [Architecture](#t4)
   - [Accumulate radar frames](#t5)
   - [Clustering](#t6)
   - [Merge Clusters](#t7)
   - [Visualization](#t8)
   

<br>




### 1. Sensor Setup and Layout <a name="t1"></a>
In this project [NuScenes](https://www.nuscenes.org/) radar dataset is used for validating and generating results. The sensor layout has a full 360&deg; coverage.
<br><br>
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/1_sensor_setup.PNG)

<br>

[Back to TOC](#t0)
<br>




### 2. Inputs and Outputs <a name="t2"></a>
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/2_inputs.PNG)
<br><br> 
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/3_outputs.PNG)

<br>

[Back to TOC](#t0)
<br>




### 3. Radar Scan Visualization in Ego Vehicle frame <a name="t3"></a>
The below animation is a brief sequence of radar frames. It can be observed that many of the measurements are clutters. Also an object is often miss-detected and in some cases the object returns multiple measurements.

![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/radar_scans_0103_short.gif)

**Long Sequence GIFs (appox 20 sec)**
   - [scene-0655](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/radar_scans_0655_full.gif)   
   - [scene-0103](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/radar_scans_0103_full.gif)   
   - [scene-0796](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/radar_scans_0796_full.gif)   
   - [scene-1077](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/radar_scans_1077_full.gif)    



<br>

[Back to TOC](#t0)
<br>




### 4. Architecture <a name="t4"></a>
<br> 

![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/4_architecture.PNG) 
<br><br> 

   - **Accumulate Radar $i$ multiple frames by sliding window $( i={1,2,3,4,5} )$** <a name="t41"></a> : Here we basically gather/accumulate multiple radar frames in sequence to increase the density of the radar point cloud. A circular queue data-struction can be implemented like a sliding window to store $k$ number of latest radar frames. <br><br> 
   ![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/5_frame_buffer.PNG)
   <br><br> 

   - **DBSCAN Clustering** : DBSCAN algorithm is used for clustering the accumulated radar measurements. DBSCAN is used here because of its ability to handle noise, identify clusters with varying shapes and densities, and automatically determine the number of clusters. <br><br> 
   ![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/6_dbscan.PNG)

   - **Merge Clusters** : In case it is required to combine clusters from multiple radars, we need a cluster merging step. A visual depiction is shown below. <br><br>
   ![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/9_cluster_merge.PNG)


<br>

[Back to TOC](#t0)
<br>




### 5. Accumulate radar frames <a name="t5"></a>
The components in each of the [Accumulate Radar i multiple frames by sliding window](#t41) $( i={1,2,3,4} )$ block are as follows
<br> 

![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/7_acc_frames.PNG)
<br><br> 

   - **Coordinate Transformation** : The measurements are in the sensor frame initially, which needs to be coordinate transformed to the vehicle frame. Let **$X_i = (p_x, p_y, v_x, v_y)$** and **$\Sigma_i$** be a **measurement vector** and **measurement noise covariance**, where $i ={1, 2, 3,..., m}$ indicates that a radar has returned $m$ number of measurements at current time $t$. Let the **mounting info** corrosponding to the radar be **$(X^{mount}, Y^{mount}, \alpha^{mount})$**. The equations for coordinate tranaformation for each of the **measurement $X_i$** and the **covariance $\Sigma_i$** are as follows: <br><br> 

   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Create Rotation matrix $(R)$ and Translation vector $(t)$**

   $$
   R_{2 \times 2} =
   \begin{pmatrix}
   cos(\alpha^{mount}) &  -sin(\alpha^{mount})   \\
   sin(\alpha^{mount}) &   cos(\alpha^{mount})
   \end{pmatrix}
   $$

   $$
   t_{2 \times 1} =
   \begin{pmatrix}
   X^{mount}   \\
   Y^{mount}
   \end{pmatrix}
   $$

   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Perform coordinate transformation of measurement vector $(X_i)$**
   

   $$
   \begin{pmatrix}
   p_{x}^{trans}   \\
   p_{y}^{trans}   \\
   v_{x}^{trans}   \\
   v_{y}^{trans}
   \end{pmatrix} = 
   \begin{pmatrix}
   R_{2 \times 2}   &   O_{2 \times 2}   \\
   O_{2 \times 2}   &   R_{2 \times 2}
   \end{pmatrix} 
   \begin{pmatrix}
   p_{x}   \\
   p_{y}   \\
   v_{x}   \\
   v_{y}
   \end{pmatrix} + 
   \begin{pmatrix}
   t_{2 \times 1}   \\
   O_{2 \times 1}
   \end{pmatrix} 
   $$


   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Perform coordinate transformation of measurement covariance $(\Sigma_i)$**
   

   $$
   \Sigma_i^{trans} = 
   \begin{pmatrix}
   R_{2 \times 2}   &   O_{2 \times 2}   \\
   O_{2 \times 2}   &   R_{2 \times 2}
   \end{pmatrix}  
   \Sigma_i 
   \begin{pmatrix}
   R_{2 \times 2}   &   O_{2 \times 2}   \\
   O_{2 \times 2}   &   R_{2 \times 2}
   \end{pmatrix}^T
   $$

   <br> 


   - **Dynamic Measurement Selection** : Here we select only the dynamic measuremnts. 
   <br> 

   ![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/readme_artifacts/8_sel_dyn_meas.PNG)
   <br><br> 



   - **Temporal Allignment** : To represent the measurements from the frame accumulated buffer at **$t_{k-1}$** in the current timestep **$t_k$**, the measurements are first **ego-compensated** so that the measurements are in the current ego-vehicle frame, and then the measurements are **extrapolated** under the assumption of constant velocity. Let **$\dot \omega_k$** be the ego vehicle yaw-rate in the interval **$(t_{k-1}, t_k]$**. Let **$X_i = (p_x, p_y, v_x, v_y)$** and **$\Sigma_i$** be a **measurement vector** and **measurement noise covariance** from the frame buffer. The temporal allignment is performed as follows <br><br>  

   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Compute the change in yaw $(\Delta\omega)$**

   $$
   \Delta t = t_k - t_{k-1}
   $$

   $$
   \Delta\omega = \dot \omega_k * \Delta t
   $$

   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Compute the rotation matrix $(R)$**

   $$
   R_{2 \times 2} =
   \begin{pmatrix}
   cos(\Delta\omega) &  -sin(\Delta\omega)   \\
   sin(\Delta\omega) &   cos(\Delta\omega)
   \end{pmatrix}
   $$

   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Ego-motion compensation of measurements $(X_i)$ and measurement noise covariance $(\Sigma_i)$**

   $$
   \begin{pmatrix}
   p_{x}^{comp}   \\
   p_{y}^{comp}   \\
   v_{x}^{comp}   \\
   v_{y}^{comp}   
   \end{pmatrix} = 
   \begin{pmatrix}
   R_{2 \times 2}^T &   O_{2 \times 2}  \\
   O_{2 \times 2}   &   R_{2 \times 2}^T 
   \end{pmatrix}  
   \begin{pmatrix}
   p_{x}   \\
   p_{y}   \\
   v_{x}   \\
   v_{y}   
   \end{pmatrix}
   $$

   $$
   \Sigma_i^{comp} = 
   \begin{pmatrix}
   R_{2 \times 2}^T &   O_{2 \times 2}  \\
   O_{2 \times 2}   &   R_{2 \times 2}^T  
   \end{pmatrix} 
   \Sigma_i 
   \begin{pmatrix}
   R_{2 \times 2}^T &   O_{2 \times 2}  \\
   O_{2 \times 2}   &   R_{2 \times 2}^T
   \end{pmatrix}^T
   $$

   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Extrapolate measurements $(X_i)$**

   $$
   \begin{pmatrix}
   p_{x}^{extrap}   \\
   p_{y}^{extrap}   \\
   v_{x}^{extrap}   \\
   v_{y}^{extrap}   
   \end{pmatrix} = 
   \begin{pmatrix}
   1 &   0 &  \Delta t &   0  \\
   0 &   1 &  0 &   \Delta t  \\
   0 &   0 &  1 &   0         \\
   0 &   0 &  0 &   1
   \end{pmatrix}  
   \begin{pmatrix}
   p_{x}^{comp}   \\
   p_{y}^{comp}   \\
   v_{x}^{comp}   \\
   v_{y}^{comp}   
   \end{pmatrix}
   $$

   <br> 



   - **Accumulated Frame Buffer Update** : After alligning temporally the measurements of the previous radar frames from time $t_{k-1}$ to $t_k$ the buffers are updated with the current radar frame. If individual buffers are maintained seperately for each of the radars, then that buffer which corrosponds to the active radar at current time $t_k$ is updated with the measurements. If the buffer is full before the frame update, the frame which is furthest back in time is deleted from the buffer, after which the new frame is inserted in the buffer.

   <br> 

[Back to TOC](#t0)
<br>


   



### 6. Clustering <a name="t6"></a>
Here only the important points are discussed regarding the implementation of DBSCAN Clustering. 
<br> 

   - **Distance Metric** : Mahalanobis distance is used as a measure of simmilarity between two points. Given two vectors $X_i = (p_x^i, p_y^i, v_x^i, v_y^i)$ & $X_j = (p_x^j, p_y^j, v_x^j, v_y^j)$ and the measurement noise covariances $\Sigma_i$ & $\Sigma_j$, the mahalanobis distance is computed as follows: <br> 

   $$
   d_{ij} = (X_i - X_j)(\Sigma_i + \Sigma_j)^{-1}(X_i-X_j) 
   $$

   <br> 


   - **Cluster Properties** : Let a cluster $C_i$ is formed from a set of measurements with state $X = (x_1, x_2, ... x_n)$ and covariances $\Sigma = (\Sigma_1, \Sigma_2, \Sigma_3, ... , \Sigma_n)$. For each of the clusters $C_i$ the below properties are calculated.

   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Cluster Mean $(\mu_i)$**

   $$
   \mu_i = \dfrac{1}{n} \sum_{j=1}^{n} x_j
   $$

   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Cluster Average Covariance $(\sigma_i^{mean})$**

   $$
   \sigma_i^{mean} = \dfrac{1}{n} \sum_{j=1}^{n} \Sigma_j
   $$

   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Cluster Sample Covariance $(\sigma_i^{sample})$**

   $$
   \sigma_i^{sample} = \dfrac{1}{n-1} \sum_{j=1}^{n} (x_j - \mu_i)(x_j - \mu_i)^T
   $$

<br>

[Back to TOC](#t0)
<br>



### 7. Merge Clusters <a name="t7"></a>
If a certain number of clusters belong to the same object, then those clusters are merged. The merged cluster properties are recalculated. Let $C = (c_1, c_2, ... c_n)$ be a set of $n$ clusters that corrospond to the same object. Each of the cluster $c_i$ has the properties: mean, average covariance, sample covariance and number of samples $( \mu_i , \Sigma_i^{avg} , \Sigma_i^{sample}, n_i )$. The merged cluster properties $( \mu , \Sigma^{avg} , \Sigma^{sample}, N )$ are calculated using the below formulas: <br> 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Number of samples $(N)$** 

$$
N = \sum_{i=1}^{n} n_i
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Mean $(\mu)$** 

$$
\mu = \sum_{i=1}^{n} \Bigl(\dfrac{n_{i}}{N}\Bigr) \mu_i 
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Average Covariance $(\Sigma^{avg})$** 

$$
\Sigma^{avg} = \sum_{i=1}^{n} \Bigl(\dfrac{n_{i}}{N}\Bigr)  \Sigma_i^{avg}
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Sample Covariance $(\Sigma^{sample})$** 

$$
\Sigma^{sample} = \sum_{i=1}^{n} \Bigl(\dfrac{n_{i} - 1}{N - 1}\Bigr) \Sigma_i^{sample} +  \sum_{i=1}^{n} \Bigl(\dfrac{n_{i}}{N - 1}\Bigr) (\mu_i-\mu)(\mu_i-\mu)^T
$$


<br>

[Back to TOC](#t0)
<br>




### 8. Visualization <a name="t8"></a>
In this section we show videos of the results. The long sequences are in the below link. Here only short sequences are shown.

- **Long Sequence GIFs of the clustering output (appox 20 sec)**
   - [scene-0655](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/tree/main/P2_object_detection_by_clustering_multiple_scans_from_radar/result_videos/radar_clusters_0655_long.gif)   
   - [scene-0103](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/tree/main/P2_object_detection_by_clustering_multiple_scans_from_radar/result_videos/radar_clusters_0103_long.gif)   
   - [scene-0796](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/tree/main/P2_object_detection_by_clustering_multiple_scans_from_radar/result_videos/radar_clusters_0796_long.gif)   
   - [scene-1077](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/tree/main/P2_object_detection_by_clustering_multiple_scans_from_radar/result_videos/radar_clusters_1077_long.gif)    


- **scene-0655** 
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/result_videos/radar_clusters_0655_short.gif)   

- **scene-0103** 
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P2_object_detection_by_clustering_multiple_scans_from_radar/result_videos/radar_clusters_0103_short.gif)   

- **scene-0796** 
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/tree/main/P2_object_detection_by_clustering_multiple_scans_from_radar/result_videos/radar_clusters_0796_short.gif)   

- **scene-1077** 
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/tree/main/P2_object_detection_by_clustering_multiple_scans_from_radar/result_videos/radar_clusters_1077_short.gif)   

<br>

[Back to TOC](#t0)


