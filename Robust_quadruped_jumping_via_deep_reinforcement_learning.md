Contents lists available at ScienceDirect
Robotics and Autonomous Systems
journal homepage: www.elsevier.com/locate/robot
Robust quadruped jumping via deep reinforcement learning
Guillaume Bellegardaa,1,∗, Chuong Nguyenb,1, Quan Nguyenb
aEcole Polytechnique Federale de Lausanne (EPFL), Lausanne, 1015, VD, Switzerland
bUniversity of Southern California, Los Angeles, 90007, CA, USA
A R T I C L E I N F O
Keywords:
Quadruped jumping
Reinforcement learning
Trajectory optimization
Agile robotsA B S T R A C T
In this paper, we consider a general task of jumping varying distances and heights for a quadrupedal robot in
noisy environments, such as off of uneven terrain and with variable robot dynamics parameters. To accurately
jump in such conditions, we propose a framework using deep reinforcement learning that leverages and
augments the complex solution of nonlinear trajectory optimization for quadrupedal jumping. While the
standalone optimization limits jumping to take-off from flat ground and requires accurate assumptions of
robot dynamics, our proposed approach improves the robustness to allow jumping off of significantly uneven
terrain with variable robot dynamical parameters and environmental conditions. Compared with walking and
running, the realization of aggressive jumping on hardware necessitates accounting for the motors’ torque-
speed relationship as well as the robot’s total power limits. By incorporating these constraints into our learning
framework, we successfully deploy our policy sim-to-real without further tuning, fully exploiting the available
onboard power supply and motors. We demonstrate robustness to environment noise of foot disturbances of
up to 6 cm in height, or 33% of the robot’s nominal standing height, while jumping 2 xthe body length in
distance.
1. Introduction
Legged robots have potential to accomplish many tasks that may be
unsafe for humans, in which overcoming uneven terrain or high obsta-
cles may be necessary. Towards real world deployment, recent works
have shown highly dynamic and agile motions such as biped [ 1] and
quadruped [ 2] backflips, wheel-legged biped jumping [ 3,4], quadruped
running and obstacle jumping [ 5], and continuous jumping on un-
even terrain [ 6] or on stepping stones [ 7]. Such methods have used
either a simple model for real-time planning, or there is no associated
publication.
With respect to optimized jumping, our prior work optimizes over a
full quadruped model to perform highly dynamic jumps [ 8,9]. A teth-
ered quadruped model shows potential for energy efficient lunar jump-
ing with flight phase pitch control through a reaction wheel [ 10]. Other
works have shown single legged [ 11] and/or dynamic miniature [ 12,
13] jumping, for which more recent work shows SALTO performing
prolonged jumping in non-laboratory settings [ 14].
To make jumping more robust to external disturbances and new un-
seen environments, deep learning offers an attractive and generalizable
formulation. Deep reinforcement learning in particular has recently
shown impressive results in learning control policies for quadrupeds
[15–18]. Typically such methods train from scratch (i.e. use little or
∗Corresponding author.
E-mail addresses: guillaume.bellegarda@epfl.ch (G. Bellegarda), vanchuong.nguyen@usc.edu (C. Nguyen), quann@usc.edu (Q. Nguyen).
1These authors contributed equally to this work.no prior information about the system) and rely on extensive sim-
ulation with randomized environment parameters. To facilitate the
sim-to-real transfer, additional techniques are employed such as online
parameter adaptation [ 18,19], learned state estimation modules [ 20],
teacher–student training [ 16,19,21], and careful Markov Decision Pro-
cess choices [ 22–24]. With respect to quadruped jumping, some recent
works have designed complex multi-stage training processes with cur-
riculums and carefully crafted reward functions to learn to jump over
a gap, either in simulation [ 25] or on hardware [ 26].
In contrast, in this paper we seek to use deep reinforcement learn-
ing to improve upon jumping motions produced with a trajectory
optimization framework. Under ideal conditions (i.e. starting on flat
ground with a high enough coefficient of friction), the motions pro-
duced from the optimization can be accurately tracked on hardware,
as shown in our prior work [ 8,9]. However, under disturbances in
foot heights (i.e. <0.05 m ), a feedforward controller using only the
reference trajectory will lead to taking off at an incorrect pitch angle,
causing significant deviation from the desired motion. In addition to
the challenges associated with highly dynamic motions, such potential
errors come with high risk and can have very costly consequences such
as robot damage.
https://doi.org/10.1016/j.robot.2024.104799
Received 11 August 2023; Received in revised form 29 March 2024; Accepted 27 August 2024Robotics  and Autonomous  Systems  182 (2024)  104799  
Available  online  4 September  2024  
0921-8890/©  2024  The Authors.  Published  by Elsevier  B.V. This is an open access  article  under  the CC BY-NC-ND  license  ( http://creativecommons.org/licenses/by-  
nc-nd/4.0/  ). 

G. Bellegarda et al.
Fig. 1. Unitree A1, a mini quadruped robot with leg length of 0.2 m, successfully
jumping off of an unknown 0.06 m block (red line), with a goal distance of 0.6 mand
height 0.2 m. Experiment video link: https://youtu.be/fVxXTofT1f4 . (For interpretation
of the references to color in this figure legend, the reader is referred to the web version
of this article.)
Many of the legged robots used to deploy learned locomotion poli-
cies in sim-to-real make use of direct current (DC) motors due to their
ability to deliver high mobility with on-board batteries, because they
offer a wide range of speeds and high torque. For learning locomotion,
common assumptions include that (i) motor torque and velocity are in-
dependent in the operating region, and (ii) the on-board battery always
has sufficient power to execute the learned policy (e.g. [ 17,18,23,24]).
These assumptions are reasonable for locomotion tasks such as walking
or running, which require high joint velocities rather than torques, and
do not need to use the full available power of the battery. In contrast
to walking or running, highly aggressive motions such as jumping
require both high joint torque and high joint velocity. However, DC
motors typically do not allow both torque and velocity reaching their
maximum values at the same time due to their inherent relationship in
motor dynamics constraints. Moreover, in order to accomplish jumping
motions, robots typically require total power that rapidly reaches the
limits of the on-board power supply. Therefore, it is critical to consider
these constraints when trying to deploy learned control policies to
successfully jump in sim-to-real. This motivates us to integrate motor
dynamics and power constraints into our learning framework. Our
integration considers both the torque–velocity relationship and the on-
board power supply to represent the true system limits, which enables
effective sim-to-real transfer for highly agile jumping motions.
Contribution: We present a method for improving the performance
of the feedforward controller used on optimal jumping trajectories.
We learn a general controller to track multiple desired jumping tra-
jectories with deep reinforcement learning, to successfully jump in
noisy environments with uneven ground, as shown in Fig. 1. This
controller is trained on, and able to track, many different jumping
trajectories, and works with different joint gains. By learning a single
controller capable of achieving many different jumps, this also avoids
re-running potentially computationally expensive optimization routines
at run time for relatively small differences in initial state. Moreover, in
contrast to our prior work on MIT Cheetah 3 [ 8] and Unitree A1 [ 9],
our DRL controller is run as a real-time feedback controller, making our
novel approach both more reliable and more robust. Importantly, we
also incorporate motor dynamic constraints and power limits into the
learning framework, allowing effective sim-to-real deployment with-
out further tuning for robust and highly dynamic jumping motions.
Our hardware results demonstrate that our method fully exploits the
available onboard power supply and takes the motors to their limits to
achieve robust and dynamic jumping motions under noisy conditions.
The rest of this paper is organized as follows. Section 2 provides
background details on the robot model, reinforcement learning, and
gives a brief overview of the jumping trajectory optimization. Sec-
tion 3 describes our learning framework design choices, including
the integration of motor dynamics and power constraints to achieveTable 1
Motor and on-board battery parameters.
Parameter Value Units
Motor gear ratio 9 –
Max joint torque 33.5 N m
Max joint speed 21 Rad ∕s
Max battery voltage 21.5 V
Max battery current 60 A
Max battery power 1290 W
robust jumping. Section 4 shows extensive numerical simulations and
experimental results from learning our general jumping controller, and
a brief conclusion is given in Section 5.
2. Background
2.1. Robot model
In this paper, we validate our jumping controller on the Unitree
A1 [ 27] quadruped robot. The A1 robot has low-inertial legs and
high torque density DC motors with planetary gear reduction, and it
is capable of ground force control without using any force or torque
sensors. A1 uses these high-performance actuators for each hip, thigh,
and knee joint to enable full 3D control of ground reaction forces. It is
also equipped with contact sensors on each foot.
The A1 legs feature a large range of motion: the hip joints have a
range of motion of ±46◦, the thigh joints have a range of motion from
−60◦to240◦, and the knee joints have a range of motion from −154.5◦
to−52.5◦. Each of A1’s actuators consist of a custom high torque density
electric motor coupled to a single-stage 9:1 planetary gear reduction.
The lower link is driven by a bar linkage which passes through the
upper link. The legs are serially actuated, but to keep leg inertia low,
the hip and knee actuators are co-axially located at the hip of each leg.
The actuation capabilities of the A1 robot and battery power supply
limits are summarized in Table 1.
2.2. Reinforcement learning
In the reinforcement learning framework [ 28], an agent interacts
with an environment modeled as a Markov Decision Process (MDP).
An MDP is given by a 4-tuple (,,,), where is the set of states,
is the set of actions available to the agent, ∶××→R
is the transition function, where (𝑠𝑡+1|𝑠𝑡,𝑎𝑡)gives the probability of
being in state 𝑠𝑡, taking action 𝑎𝑡, and ending up in state 𝑠𝑡+1, and
∶××→Ris the reward function, where (𝑠𝑡,𝑎𝑡,𝑠𝑡+1)gives the
expected reward for being in state 𝑠𝑡, taking action 𝑎𝑡, and ending up
in state𝑠𝑡+1. The goal of an agent is to interact with the environment
by selecting actions that will maximize future rewards. In this paper,
we use Soft-Actor Critic (SAC) [ 29] to learn the optimal policy 𝜋to
maximize jumping performance.
SAC learns a policy, 𝜋(𝑎|𝑠), and a critic, 𝑄𝜙(𝑠,𝑎), and aims to
maximize a weighted objective of the reward and the policy entropy,
E𝑠𝑡,𝑎𝑡∼𝜋[∑
𝑡𝑟𝑡+𝛼(𝜋(⋅|𝑠𝑡))]. The critic parameters are learned by mini-
mizing the squared Bellman error using transitions, 𝜏𝑡= (𝑠𝑡,𝑎𝑡,𝑠𝑡+1,𝑟𝑡),
replayed from an experience buffer, :
𝑄(𝜙) =E𝜏∼[(𝑄𝜙(𝑠𝑡,𝑎𝑡) − (𝑟𝑡+𝛾𝑉(𝑠𝑡+1)))2]
(1)
The target value of the next state can be estimated by sampling an
action using the current policy:
𝑉(𝑠𝑡+1) =E𝑎′∼𝜋[𝑄̃𝜙(𝑠𝑡+1,𝑎′) −𝛼log𝜋(𝑎′|𝑠𝑡+1)](2)
where𝑄̃𝜙represents a more slowly updated copy of the critic. The
policy is learned by minimizing the divergence from the exponential
of the soft-Q function at the same states:
𝜋(𝜓) = −E𝑎∼𝜋[𝑄𝜙(𝑠𝑡,𝑎) −𝛼log𝜋(𝑎|𝑠𝑡)](3)
This is done via the reparameterization trick for the newly sampled
action, and 𝛼is learned against a target entropy.Robotics  and Autonomous  Systems  182 (2024)  104799  
2 

G. Bellegarda et al.
Fig. 2. Jumping motion phases from the trajectory optimization.
2.3. Jumping trajectory optimization
In this section we briefly describe the trajectory optimization frame-
work to generate quadruped jumping motions, as well as the associated
jumping controller. For full details, please see our prior work [ 8].
The robot model used in the trajectory optimization framework is
a simplified sagittal plane quadruped model consisting of 5 links. The
body link coordinates are represented by the position of the center
of mass [𝑝𝑥,𝑝𝑧]and the rotational angle (pitch) of the body 𝜃, while
the configuration of the other links (limbs) is denoted by 𝒒. The
optimization problem is divided into 3 contact phases: double contact
(pre-flight preparation), single contact (rear-leg), and flight, as shown
in Fig. 2. The duration of each phase is manually determined based on
desired jumping distance and height.
At a high level, the resulting discrete time optimization can be
formulated as follows:
minimize
𝒙𝑘,𝒖𝑘;𝑘=1...𝑁𝐽(𝒙𝑁)+ℎ𝑁∑
𝑘=1𝑤(𝒙𝑘,𝒖𝑘)
subject to𝑑(𝒙𝑘,𝒖𝑘,𝒙𝑘+1) = 0, 𝑘= 1...𝑁− 1
𝜙(𝒙𝑘,𝒖𝑘)= 0, 𝑘= 1...𝑁
𝜓(𝒙𝑘,𝒖𝑘)≤0, 𝑘= 1...𝑁
where𝒙𝑘=[𝑝𝑥,𝑘;𝑝𝑧,𝑘;𝜃𝑘;𝒒𝑘]is the full state of the system at sample
𝑘along the trajectory, 𝒖𝑘is the corresponding control input, 𝐽and𝑤
are final and additive costs to jump to a particular height and distance
while minimizing energy, ℎis the time between sample points 𝑘and
𝑘+ 1, and𝑁is the total number of samples along the trajectory. The
constraints are specified as follows:
•The function 𝑑(⋅)captures the full-body dynamic constraints [ 8],
which is discretized from
[𝑴 −𝑱𝑇
𝑐
−𝑱𝑇
𝑐𝟎][̈𝒙
𝒇𝑐]
=[−𝑪̇𝒙−𝒈+𝑺𝝉+𝑺𝑓𝑟𝑖𝑐𝝉𝑓𝑟𝑖𝑐
̇𝑱𝑐(𝒙)̇𝒙]
,
where𝑴is the mass matrix, 𝑪represents the Coriolis and
centrifugal terms, 𝒈denotes the gravity vector, 𝑱𝑐is the spatial
Jacobian expressed at the foot contact, 𝑺and𝑺𝑓𝑟𝑖𝑐are distribu-
tion matrices of actuator torques 𝝉and joint friction torques 𝝉𝑓𝑟𝑖𝑐,
𝒇𝑐is the spatial force at the foot contact. The dimensions of 𝑱𝑐
and𝒇𝑐depend on the contact phases.
•The function 𝜙(⋅)represents equality constraints on initial joint
and body configurations, pre-landing configuration, and final
body configuration.
•The function 𝜓(⋅)captures inequality constraints including joint
angle/velocity/torque limits, friction cone limits, minimum
ground reaction forces, and geometric constraints related to the
ground and obstacle clearance.
The optimization produces desired joint angles (𝒒𝑑), joint velocities
(̇𝒒𝑑)and feed-forward joint torques (𝝉𝑑)at a sampling time of 10 ms,
which are then linearly interpolated to 1 ms. These can be tracked by
the following joint PD controller running at 1 kHz as:
𝝉ff=𝑲𝑝,𝑗𝑜𝑖𝑛𝑡 (𝒒𝑑−𝒒) +𝑲𝑑,𝑗𝑜𝑖𝑛𝑡 (̇𝒒𝑑−̇𝒒) +𝝉𝑑 (4)where𝑲𝑝,𝑗𝑜𝑖𝑛𝑡 and𝑲𝑑,𝑗𝑜𝑖𝑛𝑡 are diagonal matrices of proportional and
derivative gains in the joint coordinates.
To improve tracking performance, a Cartesian PD controller is
added. From the desired joint angle (𝒒𝑑)and joint velocity (̇𝒒𝑑)tra-
jectories, we extract desired foot positions (𝒑𝑑)and foot velocities (𝒗𝑑)
in the leg frame. Thus the Cartesian PD and full controllers for tracking
the desired jumping trajectory become:
𝝉Cartesian =𝑱(𝒒)⊤[𝑲𝑝(𝒑𝑑−𝒑)+𝑲𝑑(𝒗𝑑−𝒗)](5)
𝝉opt=𝝉Cartesian +𝝉ff (6)
where𝑱(𝒒)is the foot Jacobian at joint configuration 𝒒,𝑲𝑝and𝑲𝑑
are diagonal matrices of proportional and derivative gains in Cartesian
coordinates, and 𝝉ffis the feed-forward torque from Eq. (4).
3. Robust jumping with reinforcement learning
Given the already established Cartesian and joint space controller
for tracking jumping motions, in this section we describe our process
and reinforcement learning framework for learning to modify and track
these optimal trajectories in the presence of environmental noise and
disturbances. Our learning framework integrates motor dynamics and
power constraints, taking into account the motor torque-speed rela-
tionship and maximum on-board power supply, to represent the actual
system limits. This integration enables effective sim-to-real transfer for
robust and agile jumping on the robot hardware.
3.1. Learning optimal trajectory offsets
In order to provide an intuitive mapping between actions and their
effects on the system, we propose learning to appropriately offset the
jumping trajectories to cope with disturbances in the environment.
Specifically, we consider learning in Cartesian space, with the idea that
the agent can more directly observe the effects of its actions, as well
as more easily map offsets based on the environmental observation
than it can in joint space. In particular, we consider learning Cartesian
space offsets (𝛥𝒑𝑅𝐿)to modify the existing optimal trajectory, which
will be combined with the existing jumping controller in Eq. (6). The
corresponding torque contribution from these Cartesian space offsets
can be written as:
𝝉RL Cartesian =𝑱(𝒒)⊤[𝑲𝑝(𝛥𝒑𝑅𝐿−𝒑)](7)
Such Cartesian space offsets could result in significant deviations in
the optimal joint trajectories (𝒒𝑑). To avoid joint space gain feedback
from counteracting these deviations, we also add offsets in joint space
corresponding to those desired in Cartesian space as:
𝛥𝒒𝑅𝐿=𝑱(𝒒𝑑)⊤𝛥𝒑𝑅𝐿 (8)
This makes the joint space reinforcement learning torque contribution
as follows:
𝝉RL Joint =𝑲𝑝,𝑗𝑜𝑖𝑛𝑡 (𝛥𝒒𝑅𝐿−𝒒) (9)
The full controller for tracking the desired trajectories with learned
reinforcement residual offsets is then the summation of the original
jumping controller (Eq. (6)) with the offset contributions ( Eqs. (7) and
(9)):
𝝉=𝝉opt+𝝉RL Cartesian +𝝉RL Joint (10)
3.2. Reinforcement learning details
There are several challenging aspects of jumping that motivate our
observation space, action space, and reward function choices. Firstly,
while in the air, it is very difficult to meaningfully adjust body po-
sition or orientation. Secondly, even small noise or deviations in the
trajectory before take-off can have a large effect on landing locationRobotics  and Autonomous  Systems  182 (2024)  104799  
3 

G. Bellegarda et al.
Fig. 3. Control diagram for learning to jump robustly with reinforcement learning by leveraging the optimized jumping controller, motor dynamics, and power constraints. The
combination of trajectory reference from the optimization (dotted red lines) and trajectory offsets from the DRL policy are then tracked by joint PD and Cartesian PD controllers.
The dotted lines and gray lines execute at 1kHz, while the solid blue lines execute at 50Hz during contact phases. The motor dynamics and power constraints module is designed
to represent the hardware capability, which limits the final torque reference 𝝉𝑝for the robot’s motors.
and orientation. Thirdly, the entire motion happens very quickly, with
the pre-jump phase taking only 0.8 s, and the flight phase roughly
0.4s, depending on distance and height. To mitigate these issues, we
learn actions to modulate movement while in contact only, and apply
𝝉optin the air. Fig. 3 shows the full block diagram for integrating our
reinforcement learning action with the optimal jumping controller, and
we describe the MDP below.
Action Space : The action space consists of the desired trajectory
offsets 𝐚∈R12for the contact phase of the jumping trajectory, which
are updated at 50 Hz . The agent chooses offsets in [−0.05,0.05] m
from each foot’s local desired (𝑥,𝑦,𝑧 )Cartesian positions from the
optimization.
Observation Space : The observation space consists of the full robot
state at the initial state, trajectory end state (goal), as well as a history
of states in the previous 0.2 s. The history of states is a stack of 10
observations updated at 50 Hz . Each of these states consists of: body
state (position, orientation, linear and angular velocities), joint state
(positions, velocities), foot state (positions, velocities), and foot contact
booleans. All values are first normalized before being used for training
purposes by reinforcement learning.
Reward : We give a sparse, single reward at the end of the jumping
trajectory based on the error between the desired and actual landing
position and orientation. We give a sparse reward, rather than dense
rewards for tracking the optimized jumping trajectory at every time
step, as significant deviations to the offline optimized trajectory can
be expected (and will be needed) for large environmental noise. If
there is large noise such as low coefficient of friction or very uneven
terrain (large blocks under either the front or rear feet), executing the
trajectory optimized for flat terrain will not result in a successful jump.
For example, a large block under the front feet will result in over-pitch
at take-off, therefore being unable to jump the desired distance (see
Fig. 8 and video). Therefore, with DRL, we need to learn offsets to
significantly deviate from the original trajectory to perform a successful
jump, as can be seen in Fig. 12.
More precisely, the reward function attempts to minimize deviations
in the body position (𝑥𝑏,𝑦𝑏,𝑧𝑏)and orientation (𝜙𝑏,𝜃𝑏,𝜓𝑏)from the final
desired states in the optimal trajectory: body position (𝑥𝑁,𝑦𝑁,𝑧𝑁)and
orientation (𝜙𝑁,𝜃𝑁,𝜓𝑁). The final orientation (𝜙𝑁,𝜃𝑁,𝜓𝑁)is always
(0,0,0)as we would like the agent to land upright at its standing
orientation. The reward function is written as:
𝑅(𝑠𝑡,𝑎𝑡,𝑠𝑡+1) =𝑤(1 −‖(𝑥𝑏,𝑦𝑏,𝑧𝑏) − (𝑥𝑁,𝑦𝑁,𝑧𝑁)‖
−‖(𝜙𝑏,𝜃𝑏,𝜓𝑏)‖) (11)
where𝑤is a terminal weight. This reward scheme ensures a reward
of𝑤for perfect tracking, and will decrease from there, and even be
negative, for very poor tracking.Table 2
SAC hyperparameters.
Parameter Value
Optimizer Adam
Learning rate 3⋅10−4
Discount (𝛾) 0.99
Replay buffer size 106
Initial steps 1000
Number of hidden layers (all networks) 2
Number of hidden units per layer 512
Nonlinearity tanh
Batch size 64
Target smoothing coefficient ( 𝜏) 0.005
Target update interval 1
Gradient steps 1
3.3. Training details
We first generate 13 jumping trajectories, with final desired posi-
tions ranging in distance in [0.5,0.8] mand in height in [0,0.4] m. At the
beginning of each episode, one of the trajectories is randomly selected
to track, and random noise is added to the environment. The noise
consists of blocks of up to 0.1 min height under each foot, and the
body mass and inertia are each varied randomly by up to 5% of their
nominal values.
We use PyBullet [ 30] as the physics engine for training and simula-
tion purposes, and the A1 quadruped model introduced in Section 2.1.
For SAC [ 29], our neural networks are multi-layer perceptrons with two
hidden layers of 512 neurons each, with tanh activation. Other training
hyperparameters are listed in Table 2.
3.4. Motor dynamics and power constraints
Since legged robots must rapidly reach their motor and on-board
power supply limits to accomplish dynamic jumping maneuvers, it
is critical to model and integrate the motor dynamics and power
constraints during training to represent the true system limits. This
integration in turn limits the reference torque individually applied to
each motor, which enables successful sim-to-real transfer for aggressive
motions such as jumping.
3.4.1. Motor and power modeling
First, we revisit a simplified DC motor model which captures the
inherent torque-velocity relationship. Since the inductance of stator
windings is typically small (approximately 1 mH for an A1 robot
motor [ 27]), the voltage applied to each motor 𝑖∈ {1,…,𝑛}can be
simplified as follows
𝑉𝑖(𝜏𝑚
𝑖, ̇ 𝑞𝑚
𝑖) =𝐼𝑚
𝑖(𝜏𝑚
𝑖)𝑅𝑖+𝜑𝑖(̇ 𝑞𝑚
𝑖), (12)
where𝑅𝑖is the resistance of the coil windings, and ̇ 𝑞𝑚
𝑖is the motor
velocity. The back electromotive force (EMF) of the windings generatedRobotics  and Autonomous  Systems  182 (2024)  104799  
4 

G. Bellegarda et al.
by the rotation of the motor is estimated by 𝜑𝑖(̇ 𝑞𝑚
𝑖) =𝐾𝑣̇ 𝑞𝑚
𝑖, and the
current𝐼𝑚
𝑖(𝜏𝑚
𝑖)flowing in the windings relates to the motor torque via
𝐼𝑚
𝑖=𝜏𝑚
𝑖∕𝐾𝜏. Here,𝐾𝑣and𝐾𝜏are the electric motor velocity constant
and torque constant, respectively. Considering the gear ratio 𝑔𝑟which
relates
𝜏𝑖=𝜏𝑚
𝑖𝑔𝑟, ̇ 𝑞𝑖=̇ 𝑞𝑚
𝑖∕𝑔𝑟
we can rewrite the voltage Eq. (12) as a linear combination of joint
torque and joint velocity as
𝑉𝑖(𝜏𝑖, ̇ 𝑞𝑖) =𝛼𝜏𝑖+𝛽 ̇ 𝑞𝑖, (13)
where𝛼=𝑅𝑖∕(𝐾𝜏𝑔𝑟)and𝛽=𝐾𝑣𝑔𝑟, respectively.
Moreover, it is noteworthy that jumping maneuvers are highly
demanding and normally quickly drain the battery’s power capacity.
Hence, it is also essential to consider the total power required to run
all of the robot’s motors in the learning framework. The total power
can be estimated by:
𝑛∑
𝑖=1𝑃𝑖(𝜏𝑖, ̇ 𝑞𝑖) =𝑛∑
𝑖=1𝑉𝑖𝐼𝑖=𝑛∑
𝑖=1(𝜏𝑖
𝐾𝜏𝑔𝑟)2
𝑅𝑖+𝐾𝑣𝜏𝑖̇ 𝑞𝑖
𝐾𝜏(14)
The power consists of two parts: the first part is power dissipation on
the windings, which is proportional to 𝜏2
𝑖𝑅𝑖; the second part is the
power of rotation 𝜏𝑖̇ 𝑞𝑖.
Having revisited the torque-speed relationship and total power esti-
mation, we will propose and integrate motor dynamic constraints and
power limits into the simulation environment.
3.4.2. Implementation of motor dynamics and power constraints
We incorporate the motor dynamics and power constraints into the
DRL framework in order to enforce restrictions on the final reference
torque that is applied to the motors, as depicted in Fig. 3. In particular,
the final reference needs to satisfy the following conditions:
i.Motor dynamic constraints (MDC) establish a key relationship be-
tween joint torque and velocity in conjunction with the available
voltage supply capability 𝑉𝑏𝑎𝑡, i.e.,
|𝑉𝑖(𝜏𝑖, ̇ 𝑞𝑖)|=|𝛼𝜏𝑖+𝛽 ̇ 𝑞𝑖|≤𝑉𝑏𝑎𝑡 (15)
ii.Power limits : The total power supplied to all 𝑛motors is con-
strained by the power supply capability. This requires that the
total power for operating all motors is limited by the battery
power𝑃𝑏𝑎𝑡, i.e.,
𝑛∑
𝑖=1𝑃𝑖(𝜏𝑖, ̇ 𝑞𝑖)≤𝑃𝑏𝑎𝑡 (16)
It is noted that the motor dynamic constraints (15) imply that the
joint torques and joint velocities cannot simultaneously reach their
respective limits. In particular, the DC motor reaches maximum velocity
when running at no load, and the back EMF approaches the supply
voltage. Approximately, ̇ 𝑞𝑚𝑎𝑥
𝑖=𝑉𝑏𝑎𝑡∕𝛽, giving rise to the following
constraints:
𝑉𝑏𝑎𝑡≥𝛽 ̇ 𝑞𝑖,−𝑉𝑏𝑎𝑡≤𝛽 ̇ 𝑞𝑖 (17)
With these conditions established, we integrate the motor dynamics
and power constraints into the simulation environment, as described
in Algorithm 1.
Algorithm 1 should be executed sequentially, starting with the MDC
block, followed by the power limits. In case the torque 𝜏𝑣
𝑖obtained from
the MDC block violates the power constraints (16), we will proportion-
ally decrease this torque by setting 𝜏𝑝
𝑖=𝜂𝜏𝑣
𝑖. This modified reference
will then be utilized for the motor in simulation. Consequently, the task
is to find a value of 𝜂∈ (0,1)that satisfies the following quadratic
equation:
𝑓(𝜂)≜𝜂2𝑛∑
𝑖=1𝑅𝑖(𝜏𝑣
𝑖)2
𝐾2
𝜏𝑔2
𝑟+𝜂𝑛∑
𝑖=1𝜏𝑣
𝑖̇ 𝑞𝑖
𝐾𝜏𝐾𝑣=𝑃𝑏𝑎𝑡 (18)Algorithm 1: Integration of Motor Dynamics and Power
Constraints into the Simulation Environment
1Input : The total torque 𝝉(Eq. (10)), feedback joint velocity ̇𝒒,
battery and motor parameters.
2Output : Final reference torque for each motor
3 (i) Motor Dynamic Constraints (MDC):
4Compute:𝑉𝑖←𝛼𝜏𝑖+𝛽 ̇ 𝑞𝑖
5if𝑉𝑖>𝑉𝑏𝑎𝑡then𝜏𝑣
𝑖←(𝑉𝑏𝑎𝑡−𝛽 ̇ 𝑞𝑖)∕𝛼
6else if𝑉𝑖<−𝑉𝑏𝑎𝑡then𝜏𝑣
𝑖←(−𝑉𝑏𝑎𝑡−𝛽 ̇ 𝑞𝑖)∕𝛼
7else𝜏𝑣
𝑖=𝜏𝑖
8return𝝉𝑣;
9 (ii) Power Limits:
10Compute𝑃𝑑
𝑡𝑜𝑡𝑎𝑙←∑𝑛
𝑖=1𝑃𝑖(𝜏𝑣
𝑖, ̇ 𝑞𝑖)
11if𝑃𝑑
𝑡𝑜𝑡𝑎𝑙>𝑃𝑏𝑎𝑡then
12𝜏𝑝
𝑖←𝜂𝜏𝑣
𝑖: reduce torque proportionally ( 0<𝜂< 1)
13 where𝜂=√
𝐵2+4𝐴𝑃𝑏𝑎𝑡−𝐵
2𝐴,
14else if𝑃𝑑
𝑡𝑜𝑡𝑎𝑙≤𝑃𝑏𝑎𝑡then
15𝜏𝑝
𝑖←𝜏𝑣
𝑖;
16end
17return𝝉𝑝;
Let𝐴=∑𝑛
𝑖=1𝑅𝑖(𝜏𝑣
𝑖)2
𝐾2𝜏𝑔2𝑟and𝐵=∑𝑛
𝑖=1𝜏𝑣
𝑖̇ 𝑞𝑖
𝐾𝜏𝐾𝑣, then (18) yields:
𝜂=√
𝐵2+ 4𝐴𝑃𝑏𝑎𝑡−𝐵
2𝐴(19)
In the following, we will prove that the final output 𝝉𝑝at the
end of Algorithm 1 also satisfies the voltage constraints in Eq. (15),
i.e.,|𝑉𝑖(𝜏𝑝
𝑖, ̇ 𝑞𝑖)|≤𝑉𝑏𝑎𝑡,∀𝑖∈ {1,…,𝑛}. Indeed, if 𝑃𝑑
𝑡𝑜𝑡𝑎𝑙≤𝑃𝑏𝑎𝑡, then
Algorithm 1 assigns 𝜏𝑝
𝑖=𝜏𝑣
𝑖. This output torque 𝜏𝑝
𝑖trivially satisfies
the voltage constraints
|𝑉𝑖(𝜏𝑝
𝑖, ̇ 𝑞𝑖)|=|𝛼𝜏𝑝
𝑖+𝛽 ̇ 𝑞𝑖|=|𝛼𝜏𝑣
𝑖+𝛽 ̇ 𝑞𝑖|≤𝑉𝑏𝑎𝑡 (20)
Therefore, it remains to be shown that |𝑉𝑖(𝜏𝑝
𝑖, ̇ 𝑞𝑖)|≤𝑉𝑏𝑎𝑡when𝜏𝑝
𝑖=
𝜂𝜏𝑣
𝑖, 𝜂∈ (0,1)for the case 𝑃𝑑
𝑡𝑜𝑡𝑎𝑙>𝑃𝑏𝑎𝑡.
We start with the voltage values, which are obtained from the MDC
block:|𝑉𝑖(𝜏𝑣
𝑖, ̇ 𝑞𝑖)|=|𝛼𝜏𝑣
𝑖+𝛽 ̇ 𝑞𝑖|≤𝑉𝑏𝑎𝑡. This inequality is equivalent to(−𝑉𝑏𝑎𝑡−𝛽 ̇ 𝑞𝑖)∕𝛼≤𝜏𝑣
𝑖≤(𝑉𝑏𝑎𝑡−𝛽 ̇ 𝑞𝑖)∕𝛼. Then, by multiplying all sides of
the inequalities by 𝛼𝜂, we obtain 𝜂(−𝑉𝑏𝑎𝑡−𝛽 ̇ 𝑞𝑖)≤𝛼𝜏𝑝
𝑖≤𝜂(𝑉𝑏𝑎𝑡−𝛽 ̇ 𝑞𝑖).
Therefore, adding the term 𝛽 ̇ 𝑞𝑖to the inequalities yields
(1 −𝜂)𝛽 ̇ 𝑞𝑖−𝜂𝑉𝑏𝑎𝑡≤𝑉𝑝
𝑖≤(1 −𝜂)𝛽 ̇ 𝑞𝑖+𝜂𝑉𝑏𝑎𝑡 (21)
Combining with Eq. (17) and 0<𝜂< 1, one can verify that
𝑉𝑝
𝑖−𝑉𝑏𝑎𝑡≤(1 −𝜂)(𝛽 ̇ 𝑞𝑖−𝑉𝑏𝑎𝑡)≤0 (22a)
𝑉𝑝
𝑖+𝑉𝑏𝑎𝑡≥(1 −𝜂)(𝛽 ̇ 𝑞𝑖+𝑉𝑏𝑎𝑡)≥0 (22b)
This yields |𝑉𝑝
𝑖|≤𝑉𝑏𝑎𝑡,∀𝑖∈ {1,…,𝑛}, and the proof is complete.
In conclusion, our proposed algorithm computes final torque 𝝉𝑝that
theoretically guarantees both motor dynamic constraints (Eq. (15)) and
power limits (Eq. (16)). In hardware experiments, we will verify the key
role of these constraints in attaining effective sim-to-real transfers.
4. Results
In this section, we discuss results from using our method to achieve
robust jumping. Example snapshots of the jumping task are shown in
Figs. 1, 4, 5, 8, and the reader is encouraged to watch the supple-
mentary videos2,3for clearer visualizations. In particular, we show the
2Simulation video: https://youtu.be/Y44GK___QuY.
3Experiment video: https://youtu.be/fVxXTofT1f4.Robotics  and Autonomous  Systems  182 (2024)  104799  
5 

G. Bellegarda et al.
Fig. 4. Motion snapshots of a jump of distance 0.7 mand height 0.4 m. The front feet have a 0.05 m block beneath them, and the rear feet have a 0.01 m block beneath them.
Top: The learned policy successfully outputs trajectory offsets to jump onto the platform. Bottom: The feedforward controller results in overpitch and overjumps vertically, falling
short of the platform.
Fig. 5. Motion snapshots of a jump of distance 0.7 mand height 0.4 m. The front feet have a 0.01 m block beneath them, and the rear feet have a 0.1 mblock beneath them.
Top: The learned policy successfully outputs trajectory offsets to jump onto the platform. Bottom: The feedforward controller results in underpitch and overjumps horizontally,
making the rear legs catch on the edge of the platform, resulting in falling off.
Fig. 6. Episode reward mean while training under ideal conditions. The baseline
feedforward controllers’ performance are shown as dotted lines. Our framework is able
to track the trajectories accurately for either set of joint gains studied.
results of zero-shot sim-to-real transfers of the learned trajectory-offset
policies from PyBullet to the Unitree A1 hardware.
For our experiments, we are specifically interested in the following
questions:
1. How does choice of joint gain affect tracking performance?
2. Can we improve upon tracking performance in ideal conditions?
3. How does (magnitude of) noise affect the agent’s ability to learn?
4. What is the importance of integrating both motor dynamics con-
straints and power constraints into the learning environment?
We consider two different sets of joint gains, which we name ‘‘high’’
(𝑲𝑝,𝑗𝑜𝑖𝑛𝑡 = 300𝑰3,𝑲𝑑,𝑗𝑜𝑖𝑛𝑡 = 3𝑰3)and ‘‘low’’ (𝑲𝑝,𝑗𝑜𝑖𝑛𝑡 = 100𝑰3,𝑲𝑑,𝑗𝑜𝑖𝑛𝑡 =
2𝑰3)gains. Oftentimes these gains must be tuned by hand, and may
also need to be adapted for different trajectories. Thus, our goal with
learning with different gains is that it may give some insight on if
we can indirectly tune these all at once for multiple trajectories by
Fig. 7. Episode reward mean while training with noisy environment conditions: either
up to 0.05 m or0.1 mblocks under each foot, and base mass/inertia vary by up to 5% of
their nominal values. While the feedforward controllers’ performance is extremely poor,
our method is able to learn to jump accurately through significantly noisy environment
conditions.
selecting trajectory offsets, rather than manual human trial and error.
We set the Cartesian gains as 𝑲𝑝= 500𝑰3,𝑲𝑑= 10𝑰3.
4.1. Simulation results
Fig. 6 shows training results for learning to offset the trajectories
under ideal conditions. With the default baseline ‘‘high’’ joint gain
jumping controller, the tracking is already very accurate, getting close
to𝑤= 100 rewards. On the other hand, the baseline ‘‘low’’ gain
controller does not perform as well, primarily due to errors in pitch
when at the end of the trajectory, as well as falling short in distance.
However, through our framework, we are able to accurately track the
desired jumping motions using either set of gains, though the ‘‘high’’
gains still result in slightly better performance. This shows that our
framework is general enough to learn to improve several jumping
behaviors without the need to explicitly tune gains on a per-motion
basis, as may often be needed in general.Robotics  and Autonomous  Systems  182 (2024)  104799  
6 

G. Bellegarda et al.
Fig. 8. Jumping different target heights and distances under unknown disturbances. The figures show baseline experiments (feedforward controller) and sim-to-real transfers for
different subsets of motor dynamics and power constraints integration during the learning process. For each jumping target, the robot starts with the same initial configuration,
and an unknown disturbance of a 6 cm block (red line) is placed under the front feet. We use the same controller gains for all experiments: 𝐾𝑝,𝑗𝑜𝑖𝑛𝑡 = 300,𝐾𝑑,𝑗𝑜𝑖𝑛𝑡 = 3. Experiment
video link: https://youtu.be/fVxXTofT1f4 . (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)
Fig. 7 shows training results for learning to offset the trajectories
under the noisy conditions described in 3.3. We train under two sets of
noisy environment conditions: with either up to 0.05 m height noise, or
up to 0.1 mheight noise under each foot, in addition to the variability
in body mass and inertia. The baseline controllers for either set of gains
are not able to accurately track the desired motions, predominantly
due to over/under pitching during the single contact rear back phase.
This becomes especially apparent as we increase the environment noise
to0.1 m, where under our reward scheme, the feedforward controller
averages approximately 0 reward across 100 random trials, for either
set of joint gains, corresponding to extremely poor performance where
the robot is not even close to the goal location.
The bottom row of Fig. 4 shows the over-pitching behavior of the
baseline controller when the front legs start higher than the rear ones,
during one of the more difficult jumps in terms of height and distance.
This results in jumping vertically and not coming close to landing on
the platform.
The bottom of Fig. 5 shows the opposite result (under-pitching)
when the rear feet start at a higher 𝑧height than that of the front
feet. In this case, the baseline feedforward controller does not pitch
enough before take off, leading to a more horizontal jump that crashes
horizontally into the platform. For both of these scenarios, our learned
controller is able to successfully jump onto the platform, as can be seen
in the top rows of Figs. 4 and 5.
A noteworthy observation is that while the ‘‘low’’ gain baseline
performance (as well as when training with our method) is not as good
as the ‘‘high’’ gain controller for ideal conditions, as the noise increases
significantly, we see that the agent is able to exploit the lower gain
joint controller to outperform the policy using the high gain controller,
as can be seen in Fig. 7.
These results show that through our method, using either set of
gains, we are able to learn offsets to significantly improve jumping
performance under noisy environmental conditions, close to as well as
under ideal conditions.
4.2. Experimental verification
We validate the effectiveness of our proposed learning framework,
which incorporates motor dynamics and power constraints, in en-
abling robust jumping on the Unitree A1 robot hardware. We con-
duct various experiments with different jumping targets of (𝑥,𝑧) ∈
{(60,20),(60,0),(70,10)} ( cm), and different block disturbances {3,6}
(cm)introduced under the robot’s feet. We focus our discussion on the
6 cm disturbance, as illustrated in Fig. 8. This disturbance amounts to
33% of the robot’s initial height and is not explicitly known by the
agent.In the baseline experiments, we only use the joint PD and Cartesian
PD controller to track the joint and foot trajectory references from the
full-body trajectory optimization, as described in Section 2.3. Since
there is no feedback controller to compensate for the uneven ter-
rain disturbance, robot trajectory errors propagate during the jumping
process, resulting in overpitching before take-off and failed jumps
(Fig. 8-Baseline).
In contrast to the baseline, our learned jumping policy is run as a
real-time feedback controller to compensate jumping trajectory errors.
In order to verify the effectiveness of integrating motor dynamics and
power constraints into the learning environment, we compare the sim-
to-real transfer performance of training controllers with the following
subset of constraints:
1. No motor dynamics nor power constraints.
2. Only motor dynamics constraints.
3. Both motor dynamics and power constraints.
Case 1 - Learning with No Constraints : In the first learning experiment
for each target, we only consider a naïve implementation of torque
limits that is widely utilized for learning locomotion (e.g., [ 17,18,23,
24]), in which only a saturation function is applied for the final torque
command. As can be seen in Fig. 8, the robot has learned to compensate
for an unknown noise of a 6 cm block, thereby jumping farther than
the baseline cases. However, jumping onto the (𝑥,𝑧) = (60,20) cm box,
for example, requires high voltage of up to 30 V for the motors of
the rear right leg at the time of taking off, as illustrated in Fig. 9b&d
(roughly at 840 ms ). It also demands a significant total power supply
of approximately 3750 W (Fig. 10). These requirements exceed the
battery capability of 𝑉𝑏𝑎𝑡= 21.5 V,𝑃𝑏𝑎𝑡= 1290 W (i.e. violates both
motor dynamics and battery power limits). These violations cause poor
tracking performance and result in the robot falling short of the target,
as can be seen in Fig. 8.
Case 2 - Learning with Only MDC : For the second learning experi-
ment, our policy learns to output actions which are then constrained by
the torque-speed relationship throughout the whole jump, as described
in Eq. (15). Therefore, the voltages for both thigh and calf motors are
always within the limits ( Fig. 9b&d). However, the robot still fails to
reach the target because this aggressive motion did not consider the
power limits. In particular, this jumping motion requires up to 2500 W
when taking off, which is nearly double the maximum battery power
(Fig. 10).
Case 3 - Learning with MDC and Power Constraints : The third learning
experiment demonstrates the importance of considering both Motor
Dynamic Constraints and Power Constraints in order to realize success-
ful sim-to-real transfers for highly aggressive jumping maneuvers. TheRobotics  and Autonomous  Systems  182 (2024)  104799  
7 

G. Bellegarda et al.
Fig. 9. Experiments for jumping on box (𝑥,𝑧) = (60,20)cm. Estimated voltage demands
and total torque commands for the Rear Right leg thigh and calf motors. For this
motion, the rear legs typically require more torques than front legs, so it is sufficient
to consider the torque and voltage profiles for the rear legs. The maximum motor
torques and battery voltage are specified by dotted black lines. The single-leg contact
and flight phases start at approximately 520 ms and 840 ms , which are specified as
vertical black lines.
proposed integration of both motor dynamics and power limits ensure
(i) the voltage demanded for operating the motors can be supplied by
the battery and (ii) the required total power for all motors satisfies
the on-board power supply. Both of these limits can be verified to
not be violated in Figs. 9b&d and 10 for the example target jump
of(𝑥,𝑧) = (60,20)cm. Additionally, our method enables the robot to
successfully reach various jumping targets while ensuring robustness
against unknown and large disturbances (e.g. 6 cm block), as illustrated
in Fig. 8.
A noteworthy observation from Fig. 11 is that all jumping motions
rapidly reach the battery power limits, even for jumping forward with-
out a desired height goal (𝑥,𝑧) = (60,0)cm. Our learning framework,
which integrates motor dynamics and power limits, provides a practical
solution to achieve various jumping targets despite the limited power
capacity of the onboard battery.
Fig. 12 shows that the learned policy significantly deviates from
the original optimized trajectory in order to perform a successful jump.
With the 0.06 m block underneath the front feet, the baseline optimized
trajectory causes an over-pitch of the robot, which is then only able to
jump half of the desired distance (green line). We also note that the
DRL method has learned to significantly modulate the foot position in
order to successfully complete the jump, as can be seen by the front
and rear foot trajectories compared with the baseline in Fig. 12b&c.
4.3. Ablation studies
In this section we perform several hardware ablation studies on
different components of our framework: high vs. low joint gains, dense
vs. sparse rewards, and varying policy update rates.
Fig. 10. Estimated total power requirement to operate all motors for jumping onto a
(𝑥,𝑧) = (60,20)cm box, associated with Fig. 9. The dotted black line represents the
maximum power of the onboard battery supply.
Fig. 11. Experiments for different jumping targets. The estimated total power demand
for jumping to different targets with policies trained with both motor dynamics
and power constraints, corresponding to Fig. 8. The maximum power of the battery
(𝑃𝑚𝑎𝑥≈ 1290 W) is specified as the dotted black line.
4.3.1. Hardware joint gains
We validate the importance of selecting the ‘‘high’’ joint gains (vs.
the ‘‘low’’ joint gains) for the hardware studies by performing several
jumps with both sets of gains. During simulations, employing soft gains
(𝐾𝑝,𝐾𝑑) = (100,2)yields satisfactory joint tracking performance. How-
ever, transitioning this soft gain configuration to the hardware setup
fails to achieve comparable tracking performance. Fig. 13 illustrates the
disparity observed when employing different joint gains for a jumping
task of 60 cm under disturbance in the hardware environment. Specifi-
cally, we focus on the joint tracking of the rear legs for discussion. As
depicted in Fig. 13, adopting higher joint gains (Fig. 13b&d) demon-
strates superior joint tracking performance compared to lower joint
gains (Fig. 13a&c). The issue with tracking performance associated with
lower joint gains is attributed to hardware motor properties. Inaccurate
joint tracking has the potential to induce undesirable foot bouncing
motions, thereby adversely affecting the overall jumping performance.
4.3.2. Dense vs. sparse rewards
In our method, we train policies to perform jumps with only a
single sparse reward at the end of the jump to evaluate the error
with the desired landing location and orientation. We now inves-
tigate dense rewards, where the agent receives feedback after ev-
ery timestep based on the error between the base position (𝑥𝑏,𝑦𝑏,𝑧𝑏)
and orientation ( 𝜙𝑏,𝜃𝑏,𝜓𝑏) with respect to the corresponding desired
position/orientation at that time index in the optimized trajectory
(subscript𝑑):
𝑟𝚍𝚎𝚗𝚜𝚎(𝑠𝑡,𝑎𝑡,𝑠𝑡+1) = (1 −‖(𝑥𝑏,𝑦𝑏,𝑧𝑏) − (𝑥𝑑,𝑦𝑑,𝑧𝑑)‖
−‖(𝜙𝑏,𝜃𝑏,𝜓𝑏) − (𝜙𝑑,𝜃𝑑,𝜓𝑑)‖) (23)
We train policies with dense rewards and compare the final tracking
performance across several different jumps to evaluate the importance
of the reward function decision. We consider 10 different jumping tests
to the target (𝑥,𝑧) = (60,0)cm with the following block disturbance
heights (in cm) underneath the front and rear legs:
(ℎfront,ℎrear) = {(10,0),(8,1),(8,3),(7,5),(5,7),
(6,4),(4,6),(3,8),(1,8),(0,10)} (24)
We compute the error norm of the final location and orientation
of the actual jump with the target location and orientation for (1)
the sparse reward policy and (2) the dense reward policy. As shownRobotics  and Autonomous  Systems  182 (2024)  104799  
8 

G. Bellegarda et al.
Fig. 12. Baseline and learned policy comparison: body position and orientation trajectories, foot XZ trajectories, and rear foot position time-based trajectories for the 0.06 m front
foot jumping disturbance. The learned policy significantly deviates from the original baseline trajectory in order to successfully complete the jump.
Fig. 13. Joint gain comparison hardware experiments: comparison of joint tracking
performance when using low vs. high joint gains.
in Fig. 14, the policies trained with sparse rewards yield smaller
error norms than those trained with dense rewards. Therefore, unlike
most learning-based locomotion controllers, our observations during
the jumping task suggest that dense rewards do not offer significant
advantages over sparse rewards. In fact, for the jumping task, prior-
itizing target jumping proves more crucial than tracking the entire
reference trajectory. With dense rewards, the agent may compromise
some degree of final target tracking for overall reference tracking.
Conversely, employing sparse rewards that only consider the final
jumping goal location enhances target jumping performance.
4.3.3. Policy update rate
While our nominal policy runs at 50 Hz, we investigate training
additional policies at different rates, including both at a lower update
rate (20 Hz), and at a higher update rate (100 Hz). We find it is possible
to have both a higher and lower frequency update for the learning
policy, without any observable (quantitative or qualitative) differences
in performance. We use the same history observation window size of
𝐿= 10, but also test different window sizes in order to correspond to
the same history time interval. All frequency rates result in successful
training and jumping. This is inline with recent studies on locomotion
(without jumping), such as [31,32], which find that a range of policy
update rates and action delays can still produce successful locomotion
gaits. As shown in the video, we validate the successful jumping
experiments with different frequency rates in hardware experiments.
Fig. 14. Tracking performance with dense and sparse reward policies: comparison of
the actual error of the final position and orientation with both policies for the 10 jumps
from initial disturbance heights in order shown in Eq. (24).
5. Conclusion
In this work, we have proposed a method to improve jumping
performance of optimal trajectories with deep reinforcement learning.
Instead of learning from scratch, we learn to modify and augment the
existing trajectories in Cartesian space, which proved to be robust to
significantly varying environmental conditions. In addition to robust-
ness to environmental perturbations, we showed robustness to different
joint gains, and further benefits include avoiding re-running potentially
expensive optimization routines at run time for changes to the initial
robot state, uncertainty in the system dynamics, and uncertainty in the
environment such as varying uneven terrain and coefficients of friction.
In order to realize highly aggressive jumps on hardware, we proposed
and integrated motor dynamics and power limits as key components
of the learning environment, enabling effective sim-to-real transfers
without any further tuning. Our results demonstrate full exploitation
of the available hardware power and motor limits to jump twice the
body length in distance while subject to uneven terrain noise of 33%
of the nominal standing height.
CRediT authorship contribution statement
Guillaume Bellegarda: Conceptualization, Formal analysis, Inves-
tigation, Methodology, Software, Validation, Visualization, Writing –
original draft. Chuong Nguyen: Conceptualization, Formal analysis, In-
vestigation, Methodology, Software, Validation, Visualization, Writing
– review & editing. Quan Nguyen: Conceptualization, Formal analysis,
Funding acquisition, Methodology, Writing – review & editing.Robotics  and Autonomous  Systems  182 (2024)  104799  
9 

G. Bellegarda et al.
Declaration of competing interest
The authors declare that they have no known competing finan-
cial interests or personal relationships that could have appeared to
influence the work reported in this paper.
Data availability
Data will be made available on request.
Acknowledgments
The authors would like to thank Zhuochen Liu, Yiyu Chen, and
Hiep Hoang from the Dynamic Robotics and Control Laboratory for the
insightful discussion on the experimental setup. This work is supported
by USC Viterbi School of Engineering.
Appendix A. Supplementary data
Supplementary material related to this article can be found online
at https://doi.org/10.1016/j.robot.2024.104799 .
References
[1] Boston Dynamics, Atlas gets a grip, 2023, https://www.youtube.com/watch?v=-
e1_QhJ1EhQ .
[2] B. Katz, J. Di Carlo, S. Kim, Mini cheetah: A platform for pushing the limits of
dynamic quadruped control, in: 2019 International Conference on Robotics and
Automation, ICRA, IEEE, 2019, pp. 6295–6301.
[3] Boston Dynamics, Introducing handle, 2017, https://www.youtube.com/watch?
v=-7xvqQeoA8c .
[4] V. Klemm, A. Morra, C. Salzmann, F. Tschopp, K. Bodie, L. Gulich, N. Küng,
D. Mannhart, C. Pfister, M. Vierneisel, et al., Ascento: A two-wheeled jumping
robot, in: 2019 International Conference on Robotics and Automation, ICRA,
IEEE, 2019, pp. 7515–7521.
[5] H.-W. Park, P.M. Wensing, S. Kim, High-speed bounding with the MIT Cheetah
2: Control design and experiments, Int. J. Robot. Res. 36 (2) (2017) 167–192.
[6] G. Bellegarda, M. Shafiee, M.E. Özberk, A. Ijspeert, Quadruped-Frog: Rapid online
optimization of continuous quadruped jumping, in: 2024 IEEE International
Conference on Robotics and Automation, ICRA, 2024, pp. 1443–1450.
[7] C. Nguyen, L. Bao, Q. Nguyen, Continuous jumping for legged robots on stepping
stones via trajectory optimization and model predictive control, in: 2022 IEEE
61th Conference on Decision and Control, CDC, 2022, pp. 93–99.
[8] Q. Nguyen, M.J. Powell, B. Katz, J.D. Carlo, S. Kim, Optimized jumping on
the MIT Cheetah 3 robot, in: 2019 International Conference on Robotics and
Automation, ICRA, 2019, pp. 7448–7454, http://dx.doi.org/10.1109/ICRA.2019.
8794449 .
[9] C. Nguyen, Q. Nguyen, Contact-timing and trajectory optimization for 3D
jumping on quadruped robots, in: 2022 IEEE/RSJ International Conference on
Intelligent Robots and Systems, IROS, 2022, pp. 11994–11999.
[10] H. Kolvenbach, E. Hampp, P. Barton, R. Zenkl, M. Hutter, Towards jumping
locomotion for quadruped robots on the Moon, in: 2019 IEEE/RSJ International
Conference on Intelligent Robots and Systems, IROS, IEEE, 2019, pp. 5459–5466.
[11] Y. Ding, H.-W. Park, Design and experimental implementation of a quasi-direct-
drive leg for optimized jumping, in: 2017 IEEE/RSJ International Conference on
Intelligent Robots and Systems, IROS, IEEE, 2017, pp. 300–305.
[12] D.W. Haldane, M.M. Plecnik, J.K. Yim, R.S. Fearing, Robotic vertical jumping
agility via series-elastic power modulation, Science Robotics 1 (1) (2016).
[13] M. Noh, S.-W. Kim, S. An, J.-S. Koh, K.-J. Cho, Flea-inspired catapult mechanism
for miniature jumping robots, IEEE Trans. Robot. 28 (5) (2012) 1007–1018.
[14] J.K. Yim, E.K. Wang, R.S. Fearing, Drift-free roll and pitch estimation for
high-acceleration hopping, in: 2019 International Conference on Robotics and
Automation, ICRA, IEEE, 2019, pp. 8986–8992.
[15] J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, M. Hutter,
Learning agile and dynamic motor skills for legged robots, Science Robotics 4
(26) (2019) http://dx.doi.org/10.1126/scirobotics.aau5872 .
[16] J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, M. Hutter, Learning quadrupedal
locomotion over challenging terrain, Science Robotics 5 (47) (2020) http://dx.
doi.org/10.1126/scirobotics.abc5986 .
[17] J. Tan, T. Zhang, E. Coumans, A. Iscen, Y. Bai, D. Hafner, S. Bohez, V.
Vanhoucke, Sim-to-Real: Learning agile locomotion for quadruped robots, in:
Proceedings of Robotics: Science and Systems, Pittsburgh, Pennsylvania, 2018,
http://dx.doi.org/10.15607/RSS.2018.XIV.010 .
[18] X.B. Peng, E. Coumans, T. Zhang, T.-W.E. Lee, J. Tan, S. Levine, Learning Agile
robotic locomotion skills by imitating animals, in: Robotics: Science and Systems,
2020, http://dx.doi.org/10.15607/RSS.2020.XVI.064 .[19] A. Kumar, Z. Fu, D. Pathak, J. Malik, Rma: Rapid motor adaptation for legged
robots, in: Robotics: Science and Systems, 2021.
[20] G. Ji, J. Mun, H. Kim, J. Hwangbo, Concurrent training of a control policy and
a state estimator for dynamic and robust legged locomotion, IEEE Robot. Autom.
Lett. 7 (2) (2022) 4630–4637.
[21] T. Miki, J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, M. Hutter, Learning robust
perceptive locomotion for quadrupedal robots in the wild, Science Robotics
(2022).
[22] G. Bellegarda, K. Byl, Training in task space to speed up and guide reinforcement
learning, in: 2019 IEEE/RSJ International Conference on Intelligent Robots and
Systems, IROS, 2019, pp. 2693–2699.
[23] G. Bellegarda, Y. Chen, Z. Liu, Q. Nguyen, Robust high-speed running for
quadruped robots via deep reinforcement learning, in: 2022 IEEE/RSJ In-
ternational Conference on Intelligent Robots and Systems, IROS, 2022, pp.
10364–10370, http://dx.doi.org/10.1109/IROS47612.2022.9982132 .
[24] G. Bellegarda, A. Ijspeert, CPG-RL: Learning central pattern generators for
quadruped locomotion, IEEE Robot. Autom. Lett. 7 (4) (2022) 12547–12554.
[25] A. Iscen, G. Yu, A. Escontrela, D. Jain, J. Tan, K. Caluwaerts, Learning agile
locomotion skills with a mentor, in: 2021 IEEE International Conference on
Robotics and Automation, ICRA, IEEE, 2021, pp. 2019–2025.
[26] V. Atanassov, J. Ding, J. Kober, I. Havoutis, C. Della Santina, Curriculum-based
reinforcement learning for quadrupedal jumping: A reference-free design, 2024,
arXiv preprint arXiv:2401.16337 .
[27] Unitree Robotics, A1, 2021, https://www.unitree.com/products/a1/ .
[28] R.S. Sutton, A.G. Barto, Reinforcement Learning - An Introduction, in: Adaptive
Computation and Machine Learning, MIT Press, 1998.
[29] T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, Soft actor-critic: Off-policy maximum
entropy deep reinforcement learning with a stochastic actor, in: International
Conference on Machine Learning, PMLR, 2018, pp. 1861–1870.
[30] E. Coumans, Y. Bai, PyBullet, a Python module for physics simulation for games,
robotics and machine learning, 2016–2019, http://pybullet.org .
[31] S. Gangapurwala, L. Campanaro, I. Havoutis, Learning low-frequency motion
control for robust and dynamic robot locomotion, in: 2023 IEEE International
Conference on Robotics and Automation, ICRA, IEEE, 2023, pp. 5085–5091.
[32] G. Bellegarda, M. Shafiee, A. Ijspeert, Visual CPG-RL: Learning central pattern
generators for visually-guided quadruped locomotion, in: 2024 IEEE International
Conference on Robotics and Automation, ICRA, IEEE, 2024, pp. 1420–1427.
Guillaume Bellegarda is a postdoctoral researcher in the
Institute of Mechanical Engineering at École Polytechnique
Fédérale de Lausanne (EPFL). He was previously a post-
doctoral researcher at University of Southern California,
and received his Ph.D. and M.S. degrees in Electrical
and Computer Engineering from University of California,
Santa Barbara, and his B.S. degree in Electrical Engineer-
ing and Computer Science from University of California,
Berkeley. His research draws inspiration from machine
learning, model-based control, and neuroscience to maxi-
mize explainable performance for dynamic robotic systems,
as well as to deepen understanding of their biological system
counterparts to adapt to real world situations.
Chuong Nguyen is a Ph.D. student in the Department of
Aerospace and Mechanical Engineering at the University of
Southern California. He earned M.Sc. degrees in Mechanical
Engineering from University of Southern California in 2023,
and from Gwangju Institute of Science and Technology in
2018. Before that, he achieved a B.Sc degree in Control
and Automation Engineering in 2014 from Hanoi University
of Science and Technology, Hanoi, Vietnam. His research
interests span the intersection of control, optimization
and learning approaches for dynamic legged systems and
multi-agent dynamic systems.
Quan Nguyen is an Assistant Professor of Aerospace and
Mechanical Engineering at the University of Southern Cali-
fornia. Prior to joining USC, he was a Postdoctoral Associate
in the Biomimetic Robotics Lab at the Massachusetts In-
stitute of Technology (MIT). He received his Ph.D. from
Carnegie Mellon University (CMU) in 2017 with the Best
Dissertation Award. His research interests span different
control and optimization approaches for highly dynamic
robotics including nonlinear control, trajectory optimization,
real-time optimization-based control, robust and adaptive
control.Robotics  and Autonomous  Systems  182 (2024)  104799  
10 

