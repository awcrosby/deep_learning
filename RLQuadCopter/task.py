import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
#         # linear z_position to reward, normalized -1.0 to 1.0
#         reward = (2 * self.sim.pose[2] / self.target_pos[2]) - 1
#         reward = np.clip(reward, -1.0, 1.0)
#         reward = reward / self.action_repeat
#         return reward

        # linear distance to reward, normalized -1.0 to 1.0
        dist = np.linalg.norm(abs(self.sim.pose[:3] - self.target_pos[:3]))
        start_dist = self.target_pos[2]
        reward = 1 - 2 * (dist / start_dist)
        reward = np.clip(reward, -1.0, 1.0)
        return reward
        
#         # linear z velocity to reward, normalized -1.0 to 1.0
#         vz_range = (-4, 15)  # approx range that will be mapped from -1.0 to 1.0
#         reward = 2 * (self.sim.v[2] / vz_range[1] - vz_range[0]/vz_range[1]) - 1
#         reward = np.clip(reward, -1.0, 1.0)
#         reward = reward / self.action_repeat
#         return reward    
    
    
#         # add multiple to x and y to greater punish those differences, and get distance from target
#         x_y_mult = 2.0
#         pos_diff = (self.sim.pose[:3] - self.target_pos[:3]) * np.array([x_y_mult, x_y_mult, 1.0])
#         dist = np.linalg.norm(pos_diff)
        
#         # linear distance to reward, normalized -1.0 to 1.0
#         start_dist = self.target_pos[2]
#         reward = 1 - 1.5 * (dist / start_dist)  # reward at start_dist will be -0.5
#         reward = np.clip(reward, -1.0, 1.0)
#         reward = reward / self.action_repeat
#         return reward
  

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state