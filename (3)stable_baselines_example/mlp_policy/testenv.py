# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import gym
from gym import spaces
import numpy as np
import math

class JetBotEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=1500,
        seed=0,
        headless=True,
        goal_threshold=50,
        times = [],
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        self._goal_threshold = goal_threshold
        from omni.isaac.core import World
        from omni.isaac.jetbot import Carter
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.contact_sensor import _contact_sensor
        from omni.isaac.core.utils.nucleus import find_nucleus_server
        from omni.isaac.core.utils.stage import add_reference_to_stage

        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=0.01)
        self._my_world.scene.add_default_ground_plane()
        self.carter = self._my_world.scene.add(
            Carter(
                prim_path="/carter",
                name="my_carter",
                position=np.array([0, 0.0, 200.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )

        result, nucleus_server = find_nucleus_server()
        if result is False:
            # Use carb to log warnings, errors and infos in your application (shown on terminal)
            carb.log_error("Could not find nucleus server with /Isaac folder")

        self.usd_path = nucleus_server + "/Isaac/Environments/CityView/Props/S_Park_Grounds.usd"
        self.stand = add_reference_to_stage(usd_path=self.usd_path, prim_path="/World/ParkGrounds")
        #env_path = nucleus_server + "/Isaac/Environments/Simple_Warehouse/warehouse_test.usd"
        #env_path = nucleus_server + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        #env_path = nucleus_server + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
        #env_path = nucleus_server + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
        #add_reference_to_stage(usd_path=env_path, prim_path="/World/Env")
        #add_reference_to_stage(usd_path=env_path, prim_path="/World")

        # Add collision properties between carter and walls
        #self.add_walls(750,250) # input: distance and height of walls (cm) from center of domain

        # Add collision properties between carter and walls
        import omni
        from pxr import Usd, UsdGeom, Gf
        from omni.physx.scripts import utils
        stage = omni.usd.get_context().get_stage()
        # Traverse all prims in the stage starting at this path
        curr_prim = stage.GetPrimAtPath("/")
        for prim in Usd.PrimRange(curr_prim):
            # only process shapes and meshes
            if (
                prim.IsA(UsdGeom.Cylinder)
                or prim.IsA(UsdGeom.Capsule)
                or prim.IsA(UsdGeom.Cone)
                or prim.IsA(UsdGeom.Sphere)
                or prim.IsA(UsdGeom.Cube)
            ):
                # use a ConvexHull for regular prims
                utils.setCollider(prim, approximationShape="convexHull")
        pass

        # Add contact sensor to carter
        import carb
        self._cs = _contact_sensor.acquire_contact_sensor_interface()
        self.sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self._on_update)
        props = _contact_sensor.SensorProperties()
        props.radius = -50 # (12) Sensor radius. Negative values indicate it’s a full body sensor. (float)
        props.minThreshold = 0 # Minimum force that the sensor can read. Forces below this value will not trigger a reading. (float)
        props.maxThreshold = 1000000000000 # Maximum force that the sensor can register. Forces above this value will be clamped. (float)
        props.sensorPeriod = 1 / 50.0 # (1/100.0) Sensor reading period in seconds. zero means sync with simulation timestep (float)
        props.position = carb.Float3(150, 0, 0) # Offset sensor 40cm in X direction from rigid body center
        self._sensor_handle = self._cs.add_sensor_on_body("/carter/chassis_link", props)
        self.collided = False

        # Add goal cube
        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=np.array([100, 100, 5]),
                size=np.array([40, 40, 80]),
                color=np.array([1.0, 0, 0]),
            )
        )

        # set RL parameters
        self.seed(seed)
        self.sd_helper = None
        self.viewport_window = None
        self._set_camera()
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        #self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        self._times = []
        # create vector of time steps to add negative rewards
        for i in range(1,11):
            self._times.append((max_episode_length / 10) * i)
        return

    def add_walls(self,distance,height):
        # Add walls to scene:
        from omni.isaac.core.objects import VisualCuboid
        self._my_world.scene.add(VisualCuboid(prim_path="/walls/wall_1", name="wall_1",position=np.array([distance,0,5]),size=np.array([10,2*distance,height]),color=np.array([1.0,1.0,1.0])))
        self._my_world.scene.add(VisualCuboid(prim_path="/walls/wall_2", name="wall_2",position=np.array([-distance,0,5]),size=np.array([10,2*distance,height]),color=np.array([1.0,1.0,1.0])))
        self._my_world.scene.add(VisualCuboid(prim_path="/walls/wall_3", name="wall_3",position=np.array([0,distance,5]),size=np.array([2*distance,10,height]),color=np.array([1.0,1.0,1.0])))
        self._my_world.scene.add(VisualCuboid(prim_path="/walls/wall_4", name="wall_4",position=np.array([0,-distance,5]),size=np.array([2*distance,10,height]),color=np.array([1.0,1.0,1.0])))
        return

    def add_cubes(self,x,y):
        # Add walls to scene:
        from omni.isaac.core.objects import VisualCuboid
        self._my_world.scene.add(VisualCuboid(prim_path="/objects/cube_1", name="cube_1",position=np.array([x,y,0]),size=np.array([50,50,50]),color=np.array([1,1,1])))
        self._my_world.scene.add(VisualCuboid(prim_path="/objects/cube_2", name="cube_2",position=np.array([x,-y,0]),size=np.array([50,50,50]),color=np.array([1,1,1])))
        self._my_world.scene.add(VisualCuboid(prim_path="/objects/cube_3", name="cube_3",position=np.array([-x,y,0]),size=np.array([50,50,50]),color=np.array([1,1,1])))
        self._my_world.scene.add(VisualCuboid(prim_path="/objects/cube_4", name="cube_4",position=np.array([-x,-y,0]),size=np.array([50,50,50]),color=np.array([1,1,1])))
        return

    def get_dt(self):
        return self._dt

    def step(self, action):
        previous_jetbot_position, _ = self.carter.get_world_pose()
        for i in range(self._skip_frame):
            from omni.isaac.core.utils.types import ArticulationAction
            self.carter.apply_wheel_actions(ArticulationAction(joint_velocities=action * 10.0))
            self._my_world.step(render=False)
        observations = self.get_observations()
        success = False
        collision = False
        time_limit = False
        done = False

        goal_world_position, _ = self.goal.get_world_pose()
        current_jetbot_position, _ = self.carter.get_world_pose()
        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_jetbot_position)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_jetbot_position)
        reward = 1.0*(previous_dist_to_goal - current_dist_to_goal)

        # set reward functions
        if self.collided == True:
            # collision occurred
            reward -= 150
            print('Collision Occurred')
            collision = True
            done = True

        if self._my_world.current_time_step_index in self._times:
            # time penalty
            reward -= 10

        if current_dist_to_goal <= self._goal_threshold:
            # goal reached
            print("Goal reached!")
            reward += 150
            success = True
            done = True

        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            # max time length reached, end trial
            print("Time Limit Reached")
            time_limit = True
            done = True

        info = {"Successes":success,"Collisions":collision,"Time Limit":time_limit}
        
        return observations, reward, done, info

    def reset(self):
        self._my_world.reset()
        # randomize goal location in circle around robot
        alpha = 2 * math.pi * np.random.rand()
        mindistance = 300
        r = 100 * math.sqrt(np.random.rand()) + mindistance # radius = randomdist + minimumdistance
        self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, 5]))
        observations = self.get_observations()
        return observations

    def get_observations(self):
        self._my_world.render()
        # wait_for_sensor_data is recommended when capturing multiple sensors, in this case we can set it to zero as we only need RGB
        gt = self.sd_helper.get_groundtruth(
            ["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
        )
        return gt["rgb"][:, :, :3]

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def _set_camera(self):
        import omni.kit
        from omni.isaac.synthetic_utils import SyntheticDataHelper

        camera_path = "/carter/chassis_link/camera_mount/carter_camera_first_person"
        if self.headless:
            viewport_handle = omni.kit.viewport.get_viewport_interface()
            viewport_handle.get_viewport_window().set_active_camera(str(camera_path))
            viewport_window = viewport_handle.get_viewport_window()
            self.viewport_window = viewport_window
            viewport_window.set_texture_resolution(128, 128)
        else:
            viewport_handle = omni.kit.viewport.get_viewport_interface().create_instance()
            new_viewport_name = omni.kit.viewport.get_viewport_interface().get_viewport_window_name(viewport_handle)
            viewport_window = omni.kit.viewport.get_viewport_interface().get_viewport_window(viewport_handle)
            viewport_window.set_active_camera(camera_path)
            viewport_window.set_texture_resolution(128, 128)
            viewport_window.set_window_pos(1000, 400)
            viewport_window.set_window_size(420, 420)
            self.viewport_window = viewport_window
        self.sd_helper = SyntheticDataHelper()
        self.sd_helper.initialize(sensor_names=["rgb"], viewport=self.viewport_window)
        self._my_world.render()
        self.sd_helper.get_groundtruth(["rgb"], self.viewport_window)
        return

    def _on_update(self, dt):
        # read contact sensor data
        reading = self._cs.get_sensor_readings(self._sensor_handle) # returns (timestamp, force_value, inContact)
        self.collided = False
        if len(reading) > 0:
            #print(reading)
            if reading[0][2] == True:
                self.collided = True
        return self.collided


