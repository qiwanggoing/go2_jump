Towards Quadrupedal Jumping and Walking for Dynamic Locomotion
using Reinforcement Learning
Jørgen Anker Olsen, Lars Rønhaug Pettersen, and Kostas Alexis
Abstract— This paper presents a curriculum-based rein-
forcement learning framework for training precise and high-
performance jumping policies for the robot ‘Jumper’. Separate
policies are developed for vertical and horizontal jumps, leverag-
ing a simple yet effective strategy. First, we densify the inherently
sparse jumping reward using the laws of projectile motion. Next,
a reference state initialization scheme is employed to accelerate
the exploration of dynamic jumping behaviors without reliance
on reference trajectories. We also present a walking policy that,
when combined with the jumping policies, unlocks versatile and
dynamic locomotion capabilities. Comprehensive testing validates
walking on varied terrain surfaces and jumping performance that
exceeds previous works, effectively crossing the Sim2Real gap.
Experimental validation demonstrates horizontal jumps up to
1.25 m with centimeter accuracy and vertical jumps up to 1.0 m.
Additionally, we show that with only minor modifications, the
proposed method can be used to learn omnidirectional jumping.


## I. INTRODUCTION

Quadruped robots can navigate complex terrains and over-
come obstacles not only through walking but also through
powerful jumps. The combination of robust walking and
precise jumping capabilities is particularly valuable for plan-
etary exploration [1], [2]. In reduced gravity environments, a
quadrupedal robot could use walking for normal traversal and
jumping to overcome obstacles much larger than itself [3].
Deep Reinforcement Learning (DRL) has proven to be a
powerful tool for training quadrupedal walking policies that
demonstrate unprecedented robustness and capabilities com-
pared to non-learning-based methods [4]. Jumping presents
a more challenging dynamic task requiring precise coordi-
nation, while operating at the physical limits of the robot
hardware. Additionally, due to extended periods of low control
authority (e.g., during the “flight” phase), jumping controllers
are required to plan ahead over a substantial horizon. This
has traditionally motivated the use of trajectory optimization
for controller synthesis [5]. The corresponding optimization
problems suffer from high dimensionality and non-linearity,
making them challenging to solve and numerically inten-
sive [6]. Multiple simplifications are often necessary to make
the problem tractable, which compromises performance.
Learning-based approaches overcome the need for model
simplifications by instead learning directly from environment
interaction. However, when encountering complex dynamics
and sparse rewards, learning algorithms struggle to avoid local
minima, where only parts of the reward are optimized. As
demonstrated in [7], DRL performs poorly for quadrupedal
The authors are with the Autonomous Robots Lab, NTNU, O.S. Bragstads
Plass 2D, 7034, Trondheim, Norway.jorgen.a.olsen@ntnu.no
Fig. 1. The Jumper quadruped standing in a Mars analog environment.
jumping with random environment interaction alone, but suc-
ceeds when tailoring the training environment using auxiliary
rewards and a curriculum. In this work, we demonstrate how
to systematically tailor the training environment to obtain
precise, versatile, and robust jumping policies using DRL. The
method leverages projectile motion equations and enhances
the exploration capabilities of the agent through an elaborate
reference state initialization scheme.
We demonstrate that DRL enables quadrupeds to perform
powerful and precise jumping to target heights and posi-
tions, as well as walking locomotion, through experimental
validation under Earth gravity conditions. These combined
capabilities position robots like Jumper (Fig. 1) for scenarios
requiring versatile and dynamic locomotion. The contributions
of this work include:
•A high-performing Reinforcement Learning (RL) policy
for horizontal jumping that operates without reference
trajectories, achieving stable landings and demonstrating
jumps up to 1.25 m, with centimeter accuracy.
•A vertical jumping policy capable of reaching specified
heights with precision, tested up to 1.0 m in height.
•A walking policy trained via reinforcement learning and
validated on diverse terrain surfaces using a custom
Visual-Inertial Odometry (VIO) hardware setup.
The remainder of this paper is organized as follows: Section
II reviews related work. Section III describes the Jumper
quadruped. Section IV details the kinematic constraints. Sec-
tion V outlines the training methodology. Section VI shows
simulation results, while Section VII presents experimental
validation. Section VIII concludes the work.


## II. RELATED WORK

Traditionally, legged locomotion for quadrupeds has relied
heavily on model-based approaches [8], including trajectory
optimization [9] and model predictive control [10]. However,
in recent years, reinforcement learning has taken the stagearXiv:2510.24584v1  [cs.RO]  28 Oct 2025
and proven its robustness and versatility, enabling traversal of
unstructured terrain [11] and even parkour [4].
The main body of the literature on jumping quadrupeds con-
sists of model-based methods where trajectory optimization is
deployed to synthesize jumping maneuvers [5]. Due to high
system complexity, these methods often require offline compu-
tation, specialized solvers, and advanced low-level controllers
to compensate for model inaccuracies, reducing their gener-
alizability and versatility [6]. However, in conjunction with
the development of learning-based methods, recent years have
shown great improvement in model-based methods, enabling
robust real-time controllers. This was demonstrated in [12]
where an MIT Mini Cheetah performed a barrel roll using a
model-based controller. In addition, contact implicit methods
[13] represent a promising direction to remove the need for a
human-provided gait schedule.
For learning-based quadruped control, two main approaches
have emerged. Imitation learning leverages reference trajecto-
ries captured from real demonstrations or synthesized using
classical approaches [14]. This makes the diversity and quality
of the reference data a major concern, which naturally con-
strains the generalization and versatility of these methods [15].
Pure reinforcement learning approaches avoid these limitations
by learning directly from environmental interaction [11], [16].
As these learning-based methods mature, researchers have
begun to explore their potential for extreme environments,
including planetary exploration [1], where reduced gravity am-
plifies jumping performance and enables traversal of obstacles
larger than the robot itself, which would be impossible through
walking alone [2], [3]. Recent advances in curriculum-based
RL have demonstrated complex dynamical behaviors, with [7]
showing the potential for pure RL-based policies capable of
forward and sideways jumps, through task-based curriculum
and reward shaping. Their approach employs a three-stage
curriculum that requires learning vertical jumping first before
progressing to horizontal jumping, using specialized rewards,
including squat rewards. This was driven by the fact that direct
training of horizontal jumping was found to be ineffective in
learning proper jumping behaviors. In contrast, our method
uses reference state initialization and the projectile motion
equations to densify sparse rewards, enabling direct training
of horizontal jumping without sequential curriculum behavior
stages. Furthermore, we demonstrate greater jump distances
and accuracy alongside the potential for omnidirectional ca-
pabilities.


## III. EXPERIMENTAL PLATFORM

For experimental validation, the Jumper quadruped platform
was used, shown in Fig. 1 and Fig. 3. This robot configuration
was selected for its jumping capabilities, particularly its design
optimized for reduced gravity locomotion using walking and
jumping [3]. Table I lists the key robot specifications, with
main characteristics including high torque and fast actuators
integrated into a 5-bar linkage leg design.
The robot’s leg design, shown in Fig. 2, consists of three ac-
tuated and three unactuated joints per leg. Each leg’s actuated
joints include: the lateral motor jointθ lthat connects the motorTABLE I
KEYROBOTPARAMETERS
Parameter Value Parameter Value
Robot weight 14.5 kg Nominal height 0.35 m
Body length 0.67 m Lateral peak torque 18.0 N m
Body width 0.38 m Transversal peak torque 24.8 N m
Thigh length 0.175 m Lateralθdef
m 45°
Shank length 0.3 m Transversalθdef
m 0°
Motor Housing
Inner Thigh Outer Thigh
Inner Shank
PawOuter ShankPiaPoaθit θot
θik θokθl
Ppawxz
y
Actuated joints
Passive jointsOuter Knee Inner Knee
Fig. 2. Illustration of Jumper’s leg configuration, with key points, compo-
nents, and angles. This exact leg is the left back leg.
housing to the robot base, and the inner and outer transversal
motor jointsθ itandθ otthat connect the respective thighs
to the motor housing. Each leg also contains three passive
joints: the inner kneeθ ik, the outer kneeθ ok, and the ankle
joint. We define the following notation:θ mas the stack of all
actuated joint positions,θ lfor all four lateral joint positions,
θtfor all eight transversal joint positions (two per leg), and
θ∗
△,△ →m, l, t,denoting the desired quantities. The ankle
joint placement on the inner and outer shank is denotedP ia
andP oa, respectively. For kinematic analysis, we virtually
separate the ankle joint, though mechanicallyP ia=Poa
The system utilizes the NVIDIA Jetson Orin NX as its
onboard computer, which interfaces with the CAN-Bus com-
munication and executes the PD motor controllers at 500 Hz,
while the RL policy inference runs at 60 Hz. Fig. 4 illustrates
the control setup used to deploy the different policies on the
robot, with observationsoand policy actionsa∈[−1,1].
Actions are rescaled and interpreted as offsets to nominal
motor positionsθdef
m, with task-specific rescaling. The target
motor angles are computed asθtarget
m =a⊙s+θdef
m, where
srepresents the action scales, and⊙denotes element-wise
multiplication. A motor command filter then processes the
target motor angles to produce safe motor targetsθsafe
m, which
are sent to the PD controllers to generate commanded torques
τ∗
m. For walking, all motors use 60° as action scales to allow
for an appropriate range of motion, while for jumping, lateral
motors use 15°, and transversal motors use 90° to enable
larger errors in the motor position targets and generate higher
commanded torque from the PD controllers.
We now detail the motor command filter shown in Fig. 4.
The motor position targets are filtered before being sent to
the PD controller to ensure safe motion with minimal risk
of collision. The actuated joint references are constrained to
specific joint limits,θ min≤θm≤θmax. Additionally, due
to the five-bar linkage leg design, we have the additional
constraintl≤θ it+θot≤u, wherelanduare the lower
and upper bounds for the sum of transversal joint angles. The
simplest filter would be to always constrain the motor setpoint
1
2
3
4
5Fig. 3. The robot performing: 1) Jump down from 0.15 m, 2) Landing on uneven ground, 3) Vertical jump of 0.75 m. 4) Sequential jumps of 0.85 m forward
and alternating side component of 0.2 m, 5) 1.25 m jump in simulation. See supplementary video for full sequences, including maximum jumps.
θm,θm
m
target
defτPD Controller
θmMotor controller 500 Hz
θm
Mocap/VIORobot
Policy inference 60 HzO Observations
User input  - Sensors - State estimateMotor command filter
aθmScale
&
Offset*
+safe
Fig. 4. Control setup in simulation and on robot.
within these ranges, however, doing so would necessitate very
aggressive PD gains in order to obtain the torque bandwidth
required to execute dynamic maneuvers, such as jumping.
Therefore, inspired by [7], we implement a predictive filtering
approach that calculates the time-to-violation based on current
joint positions and velocities. The filter then gradually enforces
the limits as they are approached. This enables the policy to
exhibit close to maximum torque across the whole operating
range, which is crucial for jumping, without sacrificing posi-
tion control’s safety and learning capabilities.


## IV. KINEMATIC CONSTRAINTS

Due to Jumper’s five-bar linkage leg design, not all config-
urations are valid kinematic states. By virtually cutting open
the closed kinematic chain (CKC), following [17], we can
formulate the following kinematic constraint,
Cckc(q).=Mpia(q)−Mpoa(q) =0,(1)
whereqdenotes the complete robot configuration vector
including base pose and all joint positions,Mdenotes the
motor housing frame, whileMpiaandMpoadenote the vector
from an arbitrary reference point to the inner and outer ankle
joint. One such constraint is formulated for each leg module
and stacked together. Note that by design (see Fig. 2), Equation
1 is always fulfilled in theycoordinate, making the problem
planar.We choose to treat Equation 1 numerically, necessitating
the computation of the first-order derivatives. Computing
kinematic derivatives of an articulated system like Jumper is a
well-studied problem. To express these quantities in the motor
housing frame,Euler’s rule of differentiationis used
Jckc.=M∂Cckc
∂q=MJpia−MJpoa+ [C ckc]×MJrot,(2)
whereMJpiaandMJpoacorrespond to the translational part
of geometric Jacobians of the inner and outer ankle joint
respectively,[·] ×denotes the skew-symmetric matrix operator,
andMJrotis the rotational part of the geometric Jacobian of
the motor housing. Additionally, the last term in Equation 2
is equivalent to zeroing out all columns of the Jacobians not
corresponding to the leg joint indices.
Stacking the CKC constraint from Equation 1 for each
leg and, in addition, considering the current and desired
paw positions,Bppaw andBp∗
paw, yields the following total
kinematic constraint
C(q).=MCckc
Bppaw−Bp∗
paw
,(3)
withBdenoting the body frame. In addition, we can obtain
the full constraint JacobianJ cby stacking the Jacobian of
each term. This enables applying a classical weighted inverse
kinematic (IK) scheme. Using JAXSIM [18] enables parallel
deployment across thousands of environments on the GPU.


## V. LEARNING FRAMEWORK

We train walking and jumping policies for Jumper using
separate reward functions, each comprising a sum of terms.
All policies are also subject to a series of regularization
rewards, which encourage smoother and safer motions. These
rewards are listed in Table II. For brevity, we introduce the
TABLE II
REGULARIZATIONREWARDS
Reward formulation
Action clip||θtarget
m −θsafe
m||2
Motor torque||τ m||2
Joint acceleration|| ¨θm||2
Action rate||at−at−1||2
Jerk||I[sgn(τt
m)̸=sgn(τt−1
m)]||1
TABLE III
WALKINGREWARDS
Reward formulation
Linear velocity trackingϕ σ1(||Bvxy−Bcxy||)
Yaw rate trackingϕ σ2(Bωz−Bω∗
z)
Vertical velocity L2v2
z
Lateral stability L2||Bωxy||2
Flat L2||Bgxy||2
Standϕ σ3(||θt−θ∗
t||)
Lateral positionϕ σ4(||θt−θ∗
t||4)−1
Transversal positionϕ σ5(||θl−θ∗
l||10)−1
following notation for the Exponential and Laplacian Kernel,
respectively:ϕ σ(x) := exp(−x2
σ2)andψ σ(x) := exp(−|x|
σ).


### A. Walking

Although the focus of this work is on jumping policies,
walking is also facilitated for completeness. In this case,
the objective of Jumper’s RL-based locomotion policy is to
achieve accurate velocity tracking performance. The policy
is rewarded by tracking the command vector, consisting of
commanded linear velocities inx,y, and yaw rate:Bc=
[v∗
x, v∗
y, ω∗
z]. The observations for the walking policy are
o= [Bv,Bω,Bg,Bc,θrel
m,˙θm,at−1],(4)
whereBvis the body linear velocity,Bωis the body angular
velocity,Bgis the gravity vector,θrel
mare the joint positions
relative to default, ˙θmare the joint velocities, andat−1are
the previous actions.
Table III summarizes the rewards used during training.
TheLinear velocity trackingandYaw rate trackingrewards
track the desired velocity command while theVertical velocity
L2reward avoids movement along the undesired axis. To
achieve a more natural gait, theStand,Lateral position, and
Transversal positionrewards regularize joint positions around
nominal values close to the default positions. Different targets
and reward shapes for moving versus stationary states are used
to distinguish between moving and standing. In addition to the
rewards listed in Table III, we incorporate two rewards that
encourage longer steps and lower gait frequency, inspired by
[11]. Additionally, any collisions of the robot with itself or the
environment are penalized, while any collision involving the
robot’s base causes termination. To increase the robustness of
the policy, we randomize the initial configuration of the robot
by sampling random base poses and paw positions. Using the
IK scheme presented in Section IV, the corresponding robot
state is calculated.


### B. Vertical Jumping

The vertical jumping policy is trained to execute controlled
jumps that reach a specified target height and achieve a safe
landing. The observation vector is given by
o= [h∗, c, h,Bv,Bω,Bg,θrel
m,˙θm,at−1],(5)TABLE IV
JUMPINGREWARDS
Vertical jump rewards
Jump heightϕ σ6(hmax−h∗) + 3·ψ σ7(hmax−h∗))
Est jump heightϕ σ8(ˆhmax−h∗) + 3·ψ σ9(ˆhmax−h∗))
Joint symmetryϕ σ10(Var(θ t))·ϕ σ11(||θl||)
Horizontal jump rewards
Trackingϕ σ12(|e|)
Est trackingϕ σ13(|ˆ e|) + 0.1·ϕ σ14(ˆ e))
Joint symmetryϕ σ15(||θ left transversal −θ right transversal ||)
Common rewards
Angular velocityϕ σ16(||Bω||)
Orientation errorϕ σ17(α2
error )
Desired joint posϕ σ18(||θm−θ∗
m||)
Ground force L2||F ground ||2
Soft impactmax(0,1− |min(0,abody
amax·˜vbody)|)
Catch landing clamp(−v z,0,1)
Damp landing clamp( ˙θt,0,1)
whereh∗is the desired jump height,c∈ {0,1}is the jump
command,his the current base height, and the remaining
terms are the same as in Section V-A. The jumping command
cis necessary to synthesize a difference in the robot’s state
before and after a vertical jump is performed.
The rewards for the RL training are listed in Table IV.
Note that not all rewards are necessarily applicable at all
stages of a jump. We therefore track the agent’sjump state
(stance, in-flight, landed), and apply the different rewards
where applicable. The main reward is theJump heightreward,
which rewards the agent based on the error between the
maximum achieved jump heighth max, and the desired jump
heighth∗. This reward is handed out discontinuously, once per
jump. To achieve a denser reward signal, we also estimate the
anticipated jump height ˆhmax continuously between takeoff
and when the peak jump height is achieved using the projectile
equations of motion for a single rigid body. We refer to
this term as theEst jump heightreward. In addition, motion
dynamics indicate that vertical jumping maneuvers should
exhibit strong symmetry, which we encourage through the
Joint Symmetryreward. The policy is also encouraged to
minimize the angular velocity of the base while keeping a
flat orientation, whereα error represents the rotation error
relative to the desired orientation. Furthermore, the agent
is biased towards predefined desired joint positions while
airborne and after landing. Note that before and during takeoff,
no such reference signals are provided. On real hardware, as
opposed to simulation, there is a chance of mechanical strain
and failure when executing jumping maneuvers. To address
this, several rewards are introduced to minimize mechanical
stress. Ground contact forcesF ground are regularized to
reduce actuator loading, whileSoft impactsare encouraged
by rewarding deceleration in the direction of motion, using
body accelerationa body, L2 normalized body velocity ˜vbody,
and a maximum acceleration thresholda max. Finally, the first
0.3 s after touchdown, the robot is encouraged to retract its
legs (Damp landing) while letting the base fall downwards
(Catch landing).


### C. Horizontal Jumping

The horizontal jumping policy is tasked to steer the center
of the quadruped to a position in the planep∗by performing
In-Flight Standing TouchdownFig. 5. Examples of RSI for different stages of a jump,standing,in-flight,
andtouchdown. The red dot indicates the desired landing position.
a jump. The observation vector is given by
o= [RT
yawe, h,Bv,Bω,Bg,θrel
m,˙θm,at−1],(6)
whereR yaw is the 2D rotation matrix corresponding to the
yaw component of the base orientation, andeis the horizontal
(x and y) component of the tracking errorp∗−pbetween
target and current robot position,p∗andp, respectively. The
multiplication with the transpose of the rotation matrix effec-
tively expresses the tracking error in the robot’s yaw-aligned
frame. The remaining terms have already been described in
Sections V-B and V-A.
The horizontal jumping rewards involve mostly adaptations
to those used for vertical jumping and are listed in Table IV.
The main difference is theTrackingreward, which acts as a
navigation reward driving the robot to the desired position.
This reward is applied continuously, but the signal strength
naturally decreases as the distance to the goal increases. To
address this, we make the reward signal denser by estimating
the final landing position using the projectile equations of
motion while the robot is in-flight and applying theEst
trackingreward based on the estimate of the error at the end
of the jumpˆ e.


### D. Curriculum-based Reference State Initialization

To help the agent learn how to jump, an elaborate ref-
erence state initialization (RSI) scheme is employed, where
the agents are initialized randomly at the different stages
of a jumping maneuver, namelystanding/squatting,in-flight,
just beforetouchdown, andlandedclose to the goal posi-
tion, as illustrated in Fig. 5. To generate the base state of
the airborne agents we sample base positions and velocities
along projectile trajectories corresponding to the desired jump.
Section IV, joint configurations are simultaneously random-
ized. Especially for horizontal jumping, the agents that are
initialized intouchdownhave their paws positioned slightly
forward, which helps the policy learn an effective bracing
strategy. Robots initialized instandingbegin with randomized
standing height and base orientation, initially favoring deep
squatting positions in early training, then expanding to cover
broader standing configurations as performance improves. For
horizontal jumps, the robots are also initialized with a slight
forward pitch. As the training proceeds, randomization occurs
over all reasonable standing configurations. This transition
is controlled by performance-based curriculum thresholds. A
separate curriculum progressively increases the commanded
jump distance/height. The RSI scheme removes the need for
auxiliary rewards such as squat depth used in [7], where,
instead of forcing a reference behavior, we help the agent
explore efficient strategies.
To further improve the training pipeline, we introduce
multiple termination criteria to prevent unwanted behavior andTABLE V
RANDOMIZATIONVARIABLES
Variable Walking Jumping
Static friction [0.8, 0.95] [0.9, 1.2]
Dynamic friction [0.7, 0.8] [0.8, 0.9]
Base mass [kg] [-1.0, 2.0] [-1.0, 2.0]
Link mass±20%±20%
Center of Mass shift [m] [-0.03, 0.03] [-0.03, 0.03]
Actuator gains±40%±40%
No-load speed scaled [0.6, 1.2] [0.8, 1.2]
Cutoff speed scaled [0.6, 1.4] [0.8, 1.4]
Motor friction [N m s] [0.0, 0.04] [0.005, 0.04]
Motor armature±40%±40%
Joint offsets [deg] [-2.0, 2.0] [-2.0, 2.0]
Latency [ms] [0.0, 32.0] [0.0, 16.0]
External force [N] [-10.0,10.0] [-5.0, 5.0]
External torque [N m] [-3.0, 3.0] [-3.0, 3.0]
TABLE VI
OBSERVATIONNOISE
Variable Walking Jumping
Body linear velocity [m s−1] 0.1 0.1
Body angular velocity [s−1]0.1 0.1
Projected gravity [m s−2] 0.5 0.5
Joint position [deg]3 3
Joint velocity [deg]10 10
guide the agent away from unproductive states. These include
terminating the agent if it has not performed a jump after a
specified time limit, or if the predicted or measured jumping
performance falls below defined thresholds. The agent is also
terminated by any collisions with itself or the environment, or
if the deceleration during touchdown is above a set threshold.
To force jumping behavior, the agent is terminated if the
base has moved beyond a threshold distance from the starting
position without performing a jump.


### E. Neural Architectures and Implementation

We represent the policies using a three-layer Multilayer
Perceptron (MLP) with Exponential Linear Unit (ELU) acti-
vations. The layer widths are[512,256,128]for walking and
[256,128,128]for jumping. IsaacLab [19] provides the simu-
lation environment, while RL Games [20] implements the PPO
algorithm. Training is parallelized across4096environments
on an NVIDIA RTX 3090.
F . Crossing the Sim2Real Gap
Deploying the learned policy on real hardware requires
bridging the Sim2Real gap. In this work, the three main strate-
gies employed to achieve this are: a)System identification:
improving simulation accuracy to reflect the actual dynamical
behavior of the quadruped through motor parameter identifica-
tion and actuator characterization; b)Domain randomization:
incorporating noise and parameter uncertainty during training
for walking and jumping policies; c)Comprehensive state
coverage:All policies are initialized across all reasonable
configurations, enabling comprehensive state coverage of the
system’s operational envelope during training. Tables V and
VI detail the domain randomization variables and the variance
of the Gaussian noise applied during training. For jumping,
the latency randomization is only applied to actions, but for
walking, it is also applied to observations.
0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2
Target height [m]0.60.81.01.2Achieved peak height [m]N= 200 Jumps
Correlation: r = 0.976
Success rate: 91.00
Mean error: 0.040m
Achived height
Target height
±0.1m toleranceFig. 6. Achieved vertical jump height vs target height in simulation. Training
range: 0.6 m to 1.1 m.


## VI. SIMULATION STUDIES

We conduct a series of simulation studies to assess the
trained policies across wide variations of states and inputs. For
all tests, a successful jump execution is defined as reaching
within 0.1 m of the commanded target height/position. All
jumping tests in simulation are performed with a torque
saturation of 18 N m.
A. Vertical jump
To validate the capabilities of the trained vertical jumping
policy, we simulate 200 jumps with target heights spanning
from 0.5 m to 1.2 m, which exceeds the commands observed
during training by±0.1 m. Fig. 6 shows the high tracking
performance of the policy with a mean absolute tracking error
of 0.04 m. The policy demonstrated generalization slightly
beyond the training range, achieving100% success rate for
heights between 0.5 m to 1.1 m. Performance declined only
at the upper extreme where the torque limits are reached,
resulting in an overall 91% success rate across the full 0.5 m
to 1.2 m test range.
B. Horizontal Jump
We evaluated the horizontal jumping policy by simulating
200 jumps with target landing position covering forward
distances of 0.3 m to 1.5 m, which is outside the training
distribution by−0.1 and+0.5 m, respectively. Fig. 7 shows
high tracking performance with a mean tracking error of
0.026 m and 97% overall success rate, achieving a 100%
success rate up to 1.4 m, thus showing strong generalization.
Tracking performance decreases as the target approaches the
extreme distances at the system limits.
Diagonal jumping precision was tested using a grid of
target positions with forward distances ranging from 0.35 m
to 1.4 m, each combined with a sideways component varying
from−0.35 m to 0.35 m. Two jumps were executed per target
location. Fig. 8 shows that the strong tracking performance
persists when considering diagonal jumping, achieving a mean
tracking error of 0.025 m and a 96.8% success rate. All failures
occurred at the 1.4 m distance, consisting of two cases where
no jump was executed and five unsuccessful jumps that fell
short of the target, with all failures attributed to the distance
far exceeding what was seen during training.


## VII. EXPERIMENTAL VALIDATION

This section covers the experimental validation and testing
of the trained DRL policies. For the walking policy, tests are
conducted both with and without onboard state estimation,
while for the jumping policies, a motion capture (Mocap)
0.4 0.6 0.8 1.0 1.2 1.4
Target distance [m]0.40.60.81.01.21.4Achieved distance [m]N= 200 Jumps
Correlation: r = 0.984
Success rate: 97.0
Mean error: 0.026m
Landing position
Target distance
±0.1m toleranceFig. 7. Achieved forward jump distance vs target distance (no lateral/y
component). Training range: 0.4 m to 1.0 m.
0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4
Forward Position [m]−0.3−0.2−0.10.00.10.20.3Lateral Position [m]Landing positions
No jump executed
Jump targets
Success Rate: 96.8
Mean Error: 0.025m
Failed: 2/220
0.20.40.60.8Distance Error [m]
Fig. 8. Jump landing positions vs jump targets in simulation. The×
showcases the grid of jump targets. The color of◦represents the distance
from the target. Training range is x∈[0.4, 1.0] m and y∈[−0.3, 0.3] m.
system is used. All walking and jumping tests were conducted
with a torque saturation of 18 N m to ensure safer operation.


### A. Walking

We validate the walking policyπ Wacross diverse terrain
conditions using two state estimation approaches. Indoor tests
utilized Mocap for body state feedback, while outdoor valida-
tion employed a custom VIO system consisting of a VectorNav
VN100 IMU and stereo FLIR Blackfly S GigE cameras.
ROVIO [21] provided body state estimates, with MSF [22]
delivering fused estimates at the IMU rate of 200 Hz. The
policy was tested across varied surfaces: indoor environments
included stone flooring, mats, and uneven wooden surfaces,
while outdoor testing encompassed stone and dirt paths, gravel
surfaces, and uneven grass terrain, as shown in Fig. 9. Com-
mand inputs ranged from[v∗
x, v∗
y] = [−0.8, 0.8] m s−1and
ω∗
z= [−0.8, 0.8] rad s−1during policy testing. Quantitative
tracking performance for the walking policy was measured
during integrated locomotion tests combining walking and
jumping maneuvers. The policy achieved RMSE values of
0.17 m s−1, 0.05 m s−1, and 0.12 rad s−1forv x,vy, andω ztrack-
ing, this tracking performance is shown in Fig. 10. Qualitative
behavior observed in simulation closely matched hardware
performance, demonstrating effective Sim2Real transfer.
B. Vertical Jump
The vertical jumping policyπ V Jwas first evaluated through
three repeated jumps at a target height of 0.75 m, achieving
an average absolute error of 0.023 m from the target height.
The jumping capabilities were also tested at their limits by
performing a jump with a target height of 1.0 m, reaching a
height of 1.01 m. Fig. 11 shows the resulting jump trajectories
demonstrating jump accuracy. The policy exhibits adaptive
behavior, learning to squat deeper as the target jump height
1 2
3
4
4Fig. 9. Jumper walking outside on 1) grass, 2) uneven grass, 3) gravel path,
4) dirt path.
−0.50.00.5Linear Velocity [m/s]v∗
x
vxv∗
y
vy
0 10 20 30 40 50 60
Time [s]−1.0−0.50.00.5Angular Velocity [rad/s]Vertical Jump
Forward Jumpω∗
z
ωz
Fig. 10. Walking policy commanded vs actual velocity, tracking performance
during combined test with vertical and forward jump maneuvers.
increases. During landing, the policy demonstrated controlled
deceleration with minimal base rotation. Fig. 3 shows the robot
performing a vertical jump of 0.75 m. Note that to prevent
excessive motor/gear wear and tear, the robot was caught mid-
air with a rope during the maximum height test.
To validate simulation accuracy and establish system per-
formance limits, open-loop experiments were conducted using
pre-programmed trajectories that execute maximum squat-
jump motions with position commands that saturate motor
torque. Using identical control gains and motor filter settings
as the RL policy tests, these trajectories achieved a maximum
jump height of 1.075 m at 18 N m torque saturation. Compar-
ing the maximum open-loop jump of 1.075 m with the RL
policy jump of 1.01 m demonstrates that the policy achieves
94% of the theoretical system limits under the set torque
constraints. Simulation studies with stiffer gains, less strict
filter settings, and maximum motor torque (24.8 N m) suggest
a maximum possible jump height of 1.42 m for the system.
C. Horizontal Jump
We demonstrate the capabilities of the horizontal jump
policyπ HJthrough a series of tasks. These include: 1)
Max forward jump (1.25 m), 2) Forward jump with a lateral
component (0.75 m forward,±0.35 m sideways), 3) The same
jump (0.85 m forward, 0.1 m sideways) repeated three times,
4) Jumping off a 0.15 m platform, 5) Jumping onto an uneven
and movable surface, and 6) Consecutive jumps.
All experiments were initialized by manually configuring
the robot at the desired starting position. Standing in place
is then achieved by setting the target position close to the
desired starting position. This process naturally introduces
some variation in the initial configuration, which is robustly
−0.2 0.0 0.2 0.4 0.6 0.8 1.0 1.2
Time [s]0.20.30.40.50.60.70.80.91.0Z Position [m]Vertical jump 1
Vertical jump 2
Vertical jump 3
Max vertical jumpTarget repeatability
Target max height
Jump trigger
±5cm toleranceFig. 11. Jump height vs target height with 18 N m torque saturation.
0.00.20.40.60.81.01.2X Position [m]Repeat 1
Repeat 2
Repeat 3
Max jump
Diagonal 1
Diagonal 2
Sideways 1Sideways 2
Backwards
Target repeat
Target max
Target diagonal
Target sideways
Target backwards
0.0 0.5 1.0 1.5 2.0 2.5 3.0
Time [s]−0.50.00.5Y Position [m]
Fig. 12. Jump position vs jump target in x and y.
handled by the policy. Jump commands are then generated by
specifying target landing positions at various distances relative
to the starting location. Fig. 3 shows experiments 4, 5, and 6,
along with a simulated forward jump of 1.25 m.
The tracking performance for experiments 1, 2, and 3 are
presented in Fig. 12. The policy achieved a maximum jump
distance of 1.25 m with a landing error of 0.004 m from the
target, which, to the best of the authors’ knowledge, exceeds
the state-of-the-art for comparable works (see Table VII for
jump distance comparison). Note that due to variations in
hardware, this is not an exact comparison.
Horizontal jumping experiments demonstrate high precision
across multiple target configurations. For the two 0.75 m for-
ward jumps with lateral targets at±0.35 m, the mean landing
error the policy achieved was 0.0015 m. For the repeatability
tests targeting 0.85 m forward and a 0.1 m lateral distance,
yields a mean landing error of 0.005 m. Experiments4and5
tested the system’s robustness to real-world terrain variations,
including elevation changes and surface irregularities. Both
scenarios validated the policy’s ability to maintain stable
jumping performance despite non-ideal ground conditions.
Experiment6evaluated sequential jumping capabilities us-
ing two test setups. The diagonal tests consisted of a first jump
of 0.8 m forward with+0.2 m lateral, followed by a second
jump of 0.75 m forward with−0.2 m lateral. The straight tests
consisted of two consecutive 0.8 m forward jumps. The policy
achieved a mean landing error of 0.0054 m across all jumps.
Note that the first jump could still be completed successfully
if the initial takeoff occurs on low-friction surfaces or with
TABLE VII
MAXIMUMJUMPDISTANCECOMPARISON
Work: [23] [24] [25] [26] [7] Ours
Jump length [m]: 0.5 m 0.66 m 0.8 m 0.8 m 0.9 m 1.25 m
targets near the edge of the policy’s capabilities. However,
these cases might result in an off-nominal configuration when
the robot lands the first jump, which increases the chance of
failure for the second jump. This can be mitigated by switching
to the walking policy to recover the nominal stance.
Additional experiments demonstrated that the proposed RL
pipeline could learn omnidirectional jumping through curricu-
lum modifications alone. Separate policies were trained with
sideways and backwards commands included in the curricu-
lum, successfully achieving these multi-directional jumping
movements with high accuracy, as shown in Fig. 12. However,
this generalization compromised peak jumping performance,
indicating a trade-off between specialized performance and
omnidirectional versatility.
D. Hierarchical policy deployment
The integration of multiple policies for complex locomotion
was evaluated through a hierarchical test combining walking
with vertical (0.75 m) and horizontal (0.85 m) jumps, using
manually triggered transitions between policies. Figure 10
shows the walking policy’s velocity tracking during this
test. This test demonstrates the potential for combining such
policies to fully leverage the robot’s dynamic locomotion
capabilities for complex traversal and navigation tasks.


## VIII. CONCLUSION

This work demonstrates the training and deployment of
walking and jumping RL policies for the Jumper quadruped.
Fundamental to training the jumping policies is an elaborate
RSI scheme, which is enabled by a GPU-parallelized IK-
solver, and densifying the rewards using projectile motion
equations. Experimental validation on real hardware shows
the capabilities of the trained policies to walk across various
terrains with onboard state estimation. The jumping policies
exhibit excellent tracking performance, both for vertical and
horizontal jumps, with horizontal jumps up to 1.25 m with
centimeter accuracy. The policies successfully handle diagonal
jumps, jumping from heights, and onto unstructured and loose
terrain. Quantitative analysis, both in the real world and in
simulations, demonstrates effective Sim2Real transfer. The
jumping policies also demonstrate generalization to jumping
maneuvers not seen during training. Additionally, the training
pipeline demonstrates the potential for omnidirectional jump-
ing through curriculum modifications.


## REFERENCES

[1] P. Armet al., “Scientific exploration of challenging planetary analog
environments with a team of legged robots,”Science robotics, vol. 8,
no. 80, p. eade9548, 2023.
[2] A. Spiridonovet al., “Spacehopper: A small-scale legged robot for
exploring low-gravity celestial bodies,” inProc. IEEE Int. Conf. Robot.
Autom., 2024, pp. 3464–3470.
[3] J. A. Olsen, G. Malczyk, and K. Alexis, “Olympus: A jumping
quadruped for planetary exploration utilizing reinforcement learning for
in-flight attitude control,”arXiv preprint arXiv:2503.03574, 2025.[4] N. Rudin, J. He, J. Aurand, and M. Hutter, “Parkour in the wild:
Learning a general and extensible agile locomotion policy using multi-
expert distillation and rl fine-tuning,”arXiv preprint arXiv:2505.11164,
2025.
[5] Q. Nguyen, M. J. Powell, B. Katz, J. Di Carlo, and S. Kim, “Optimized
jumping on the mit cheetah 3 robot,” inIEEE Int. Conf. Robotics and
Automation (ICRA), 2019, pp. 7448–7454.
[6] C. Nguyen and Q. Nguyen, “Contact-timing and trajectory optimization
for 3d jumping on quadruped robots,” in2022 IEEE/RSJ international
conference on intelligent robots and systems (IROS). IEEE, 2022, pp.
11 994–11 999.
[7] V . Atanassov, J. Ding, J. Kober, I. Havoutis, and C. D. Santina,
“Curriculum-based reinforcement learning for quadrupedal jumping: A
reference-free design,” 2024.
[8] M. H. Raibert,Legged robots that balance. MIT press, 1986.
[9] M. Neunertet al., “Trajectory optimization through contacts and auto-
matic gait discovery for quadrupeds,”IEEE Robotics and Automation
Letters, vol. 2, no. 3, pp. 1502–1509, 2017.
[10] J. Di Carloet al., “Dynamic locomotion in the mit cheetah 3 through
convex model-predictive control,” inIEEE/RSJ Int. Conf. on intelligent
robots and systems (IROS), 2018.
[11] N. Rudin, D. Hoeller, P. Reist, and M. Hutter, “Learning to walk
in minutes using massively parallel deep reinforcement learning,” in
Conference on Robot Learning. PMLR, 2022, pp. 91–100.
[12] H. Li and P. M. Wensing, “Cafe-mpc: A cascaded-fidelity model pre-
dictive control framework with tuning-free whole-body control,”IEEE
Transactions on Robotics, vol. 41, pp. 837–856, 2025.
[13] G. Kimet al., “Contact-implicit model predictive control: Controlling
diverse quadruped motions without pre-planned contact modes or tra-
jectories,”Int. J. Robot. Res., vol. 44, no. 3, pp. 486–510, 2025.
[14] E. Xiao, Y . Dong, J. Ma, and P. Lu, “Stable imitation of multigait and
bipedal motions for quadrupedal robots over uneven terrains,”Advanced
Robotics Research, p. 202500036, 2025.
[15] C. Liet al., “Learning agile skills via adversarial imitation of rough
partial demonstrations,” inConf. on Robot Learning (CoRL). PMLR,
2023, pp. 342–352.
[16] J. Leeet al., “Learning quadrupedal locomotion over challenging
terrain,”Science robotics, vol. 5, no. 47, p. eabc5986, 2020.
[17] N. N. Balaji, “Dynamics and control of closed kinematic chains: A
numerical investigation.”
[18] D. Ferigoet al., “JaxSim: A differentiable physics engine and
multibody dynamics library for control and robot learning,” 2022.
[Online]. Available: http://github.com/ami-iit/jaxsim
[19] M. Mittalet al., “Orbit: A unified simulation framework for interactive
robot learning environments,”IEEE Robotics and Automation Letters,
vol. 8, no. 6, pp. 3740–3747, 2023.
[20] D. Makoviichuk and V . Makoviychuk, “rl-games: A high-performance
framework for reinforcement learning,” https://github.com/Denys88/rl
games, May 2021.
[21] M. Bloeschet al., “Iterated extended kalman filter based visual-inertial
odometry using direct photometric feedback,”The International Journal
of Robotics Research, vol. 36, no. 10, pp. 1053–1072, 2017.
[22] S. Lynenet al., “A robust and modular multi-sensor fusion approach
applied to mav navigation,” inProc. of the IEEE/RSJ Conference on
Intelligent Robots and Systems (IROS), 2013.
[23] K. Caluwaertset al., “Barkour: Benchmarking animal-level agility with
quadruped robots,”arXiv preprint arXiv:2305.14654, 2023.
[24] G. B. Margoliset al., “Learning to jump from pixels,”arXiv preprint
arXiv:2110.15344, 2021.
[25] Q. Liuet al., “Distance-controllable long jump of quadruped robot based
on parameter optimization using deep reinforcement learning,”IEEE
Access, vol. 11, pp. 98 566–98 577, 2023.
[26] X. Cheng, K. Shi, A. Agarwal, and D. Pathak, “Extreme parkour
with legged robots,” in2024 IEEE Int. Conf. Robotics and Automation
(ICRA), pp. 11 443–11 450.
