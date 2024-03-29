#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from hima.envs.coppelia.arm import ARMS
from pyrep.objects.shape import Shape, PrimitiveShape

from os.path import dirname, join, abspath
from typing import Union
import numpy as np
from math import sqrt

from hima.envs.env import Env


class ArmEnv:
    def __init__(self,
                 arm_name: str,
                 scene_file: str,
                 joints_to_manage: Union[list[int], str],
                 observation: list[str],
                 max_steps: int,
                 action_time_step: float,
                 action_cost: float,
                 goal_reward: float,
                 position_threshold: float,
                 workspace_limits: dict,
                 collision_penalty: float = 0,
                 initial_pose: list[float] = None,
                 initial_target_position: list[float] = None,
                 joints_speed_limit: float = 80,
                 camera_resolution: list[int] = None,
                 headless: bool = False,
                 responsive_ui: bool = True,
                 reward_type: str = 'sparse',
                 action_type: str = 'joints',
                 seed=None):
        self.action_time_step = action_time_step
        self.pr = PyRep()
        scene_file = join(dirname(abspath(__file__)), 'scenes', scene_file)
        self.pr.launch(scene_file, headless=headless, responsive_ui=responsive_ui)
        self.pr.start()
        self.goal = Shape.create(type=PrimitiveShape.SPHERE,
                                 size=[2 * position_threshold] * 3,
                                 color=[1.0, 0.0, 0.0],
                                 static=True, respondable=False)
        self.goal.set_model_collidable(False)

        if arm_name in ARMS:
            self.agent = ARMS[arm_name]()
        else:
            raise NotImplemented(f'{arm_name} not found!')
        self.agent.set_model_collidable(True)

        self.reward_type = reward_type
        self.action_type = action_type

        if initial_pose is not None:
            self.initial_joint_positions = np.radians(initial_pose)
        else:
            self.initial_joint_positions = self.agent.get_joint_positions()

        if initial_target_position is not None:
            self.initial_target_position = initial_target_position
        else:
            self.initial_target_position = self.goal.get_position()

        self.camera = VisionSensor('camera')
        if camera_resolution is not None:
            self.camera.set_resolution(camera_resolution)

        self.observation = set(observation)
        self.workspace_limits = workspace_limits
        self.is_first = True
        self.should_reset = False
        self.n_sim_steps_for_action = int(action_time_step / self.pr.get_simulation_timestep())
        assert self.n_sim_steps_for_action > 0
        self.action_cost = action_cost
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.position_threshold = position_threshold
        self.joints_speed_limit = np.pi * joints_speed_limit / 180
        self.max_steps = max_steps
        self.n_steps = 0
        self.max_action_distance = sqrt(
            (self.workspace_limits['r'][1] * 2) ** 2 +
            (self.workspace_limits['h'][1] - self.workspace_limits['h'][0]) ** 2
        )
        self.action_distance = self.max_action_distance
        # should use all joints if you use IK
        # maybe we will remove this in future
        assert (action_type != 'tip') or (joints_to_manage == 'all')

        if isinstance(joints_to_manage, list):
            joints_mask = np.zeros(self.agent.get_joint_count(), dtype=bool)
            joints_mask[joints_to_manage] = True
        elif joints_to_manage == 'all':
            joints_mask = np.ones(self.agent.get_joint_count(), dtype=bool)
        else:
            raise ValueError

        self.joints_to_manage = joints_mask
        self.n_joints = int(sum(joints_mask))

        self.rng = np.random.default_rng(seed)

        self.reset()

    def reset(self):
        self.goal.set_position(self.initial_target_position, relative_to=self.agent.base)
        self.agent.set_joint_positions(self.initial_joint_positions, disable_dynamics=True)
        self.is_first = True
        self.should_reset = False
        self.n_steps = 0

    def act(self, action: Union[list[float], np.ndarray]):
        self.n_steps += 1
        self.is_first = False

        if self.action_type == 'joints':
            target_positions = np.zeros(self.agent.get_joint_count())
            target_positions[self.joints_to_manage] = np.array(action)

            for i, joint in enumerate(self.agent.joints):
                if self.joints_to_manage[i]:
                    joint.set_joint_target_position(target_positions[i])
                else:
                    joint.set_joint_target_velocity(0.0)
        elif self.action_type == 'tip':
            self.action_distance = np.linalg.norm(
                np.array(action) - self.get_tip_position()
            )
            self.agent.target.set_position(action, relative_to=self.agent.base)
        else:
            raise ValueError

        for step in range(self.n_sim_steps_for_action):
            self.pr.step()

    def observe(self):
        if self.should_reset:
            self.reset()
        obs = list()
        if 'camera' in self.observation:
            obs.append(self.camera.capture_rgb())
        if 'joint_pos' in self.observation:
            obs.append(self.get_joint_positions())
        if 'joint_vel' in self.observation:
            obs.append(self.get_joint_velocities())
        if 'target_pos' in self.observation:
            obs.append(self.goal.get_position())
        if 'target_vel' in self.observation:
            obs.append(self.goal.get_velocity())

        tip_in_loc = False
        x, y, z = self.agent.tip.get_position()
        tx, ty, tz = self.goal.get_position()
        if ((abs(x - tx) < self.position_threshold) and
                (abs(y - ty) < self.position_threshold) and
                (abs(z - tz) < self.position_threshold)):
            tip_in_loc = True
            self.should_reset = True
        elif self.n_steps > self.max_steps:
            self.should_reset = True

        reward = -self.action_cost * self.action_distance / self.max_action_distance
        # collision
        # reward += -self.collision_penalty * self.agent.check_collision()
        if self.reward_type == 'sparse':
            if tip_in_loc:
                reward += self.goal_reward
        elif self.reward_type == 'gaus_dist':
            r_2 = (x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2
            d_2 = self.position_threshold ** 2
            reward += self.goal_reward * np.exp(-r_2 / d_2)

        return reward, obs, self.is_first

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def get_joint_positions(self):
        joint_pos = np.array(self.agent.get_joint_positions())
        return joint_pos[self.joints_to_manage]

    def get_joint_velocities(self):
        joint_vel = np.array(self.agent.get_joint_velocities())
        return joint_vel[self.joints_to_manage]

    def get_joints_speed_limit(self):
        return self.joints_speed_limit

    def get_target_position(self):
        return self.agent.target.get_position(relative_to=self.agent.base)

    def get_tip_position(self):
        return self.agent.tip.get_position(relative_to=self.agent.base)

    def set_goal_position(self, pos):
        self.initial_target_position = pos
        self.goal.set_position(pos, relative_to=self.agent.base)


if __name__ == '__main__':
    EPISODES = 1
    EPISODE_LENGTH = 1
    EPS = 0.01
    SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/pulse75_tip.ttt')

    env = ArmEnv(SCENE_FILE,
                 joints_to_manage='all',
                 observation=['joint_pos'],
                 max_steps=200,
                 action_time_step=1,
                 action_cost=-0.1,
                 goal_reward=1,
                 position_threshold=EPS,
                 action_type='tip',
                 initial_pose=[0.0, 0.0, 0.0, 0.0, 1.57, 0.0],  # initial robot joint positions
                 initial_target_position=[0.500, 0.2763, 1.85274],
                 headless=False)

    for e in range(EPISODES):
        print('Starting episode %d' % e)
        env.reset()
        for i in range(EPISODE_LENGTH):
            print(f'Step {i}')
            action = [0.500, 0.2763, 1.85274]
            env.act(action)
            state = env.observe()
            # save images from camera
            # plt.imshow(state)
            # plt.savefig(join(dirname(abspath(__file__)), f'image_{e}_{i}.png'))
    print('Done!')
    env.shutdown()
