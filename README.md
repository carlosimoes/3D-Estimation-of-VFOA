# 3D-Estimation-of-VFOA
The dissertation project is named 3D Estimation of VFOA, where Attention was calculated based on the person's Gaze and Head Pose. 

Humanoid and social robots may provide valuable resources to society in the most diverse and complex activities and challenges, thanks to their increasing mechanical and decisionmaking abilities. However, robots must comprehend and acquire information about their surroundings for proper interaction with humans. The VFOA can be used as the primary conversational cue. To tackle these challenges, we develop a novel approach that estimates and tracks the VFOA.
The proposed model stems from the consideration that the eye gaze and head pose carry information about actions and interactions. The proposed formulation leads to a 3D algorithm that considers: 
(i) A bounding box of every object in the field of view of the robot’s camera;
(ii) A ray casting algorithm that considers head and gaze directions;
(iii) A Kalman filter performs the tracking of the gaze.
(iv) Finally, the VFOA algorithm estimates the object of attention based on a weighted sum of gaze and head pose information. 
We study the parameters of 3D VFOA algorithm, running simulated scenarios for a selection of the most adequate parameters. The novel approach is validated, tested and benchmarked on the public MPI Sintel dataset containing animated real-world interactions.


Index Terms — VFOA, Eye Gaze, Head Pose, Object Detection, Human-Robot Interaction

