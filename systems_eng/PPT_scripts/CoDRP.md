# Work Breakdown Structure

The WBS highlights the major components of work and we are following a product model which mirrors
the subsystems we will be developing.

1. The AMR development and Offboard infrastructure will be the largest chunk of work as it comprises
   our core systems. 
2. The The Payload dock will be the next critical step to perfect as it will test our state machine
3. The Simulation and HRI are steps to aid the development while being cognizant that the AMR nor the
   testbed environment may not be ready in spring.
4. Project Management has so far been split amongst all team members, but moving forward we plan to
   do ________________________

# Gantt Chart

Our ideology behind scheduling revolved around this aspect of not having the testbed environment
nor the AMR ready before mid-March.

- In this aspect, we start with simulating the testbed environment using available gazebo models
  for the robot. This will create a platform to test our global and local planners and mock our 
  pipeline for object detection and obstacle avoidance

- The Payload dock and pallet protyping will begin in parallel since one major risk we identified 
  was the tolerances during docking and undocking. The pallet design will also be an input for the
  simulation of the testbed environment

- Offboard infrastructre specifically includes the global planner which gives the rough routes as 
  well as the workflow coordinator and fleet management systems (these systems will mostly be fully
  tested towards the end of the simuation or soon after procuring the robot)

- The milestone of AMR development being towards the end of the semester has enabled us to prepare
  the stack, test it on sim ---> hopefully only tuning controls, sensor mounting and drivers will
  form the chunk of **Integration**. This is captured in the Integration and testing period which
  spans almost ~1 month


# Closing Notes

Having already seen how the gantt chart is structured to tackle some foreseeable risks, we will now
define these risks formally ....