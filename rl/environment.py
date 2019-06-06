from random import choice

from rl.simulator_car_interface import VelocityControlledAICar
from simulator.stuff_on_road.normal_car import RuleBasedCar
from simulator.stuff_on_road.road import InfiniteRoad
from simulator.utils.constants import NUM_LANES, NUM_ORDINARY_CARS

"""
IRTrafficEnv uses the simulator to create an openAI gym style API to be interfaced with RL algorithms
"""


class IRTrafficEnv:
    def __init__(self, episode_len=1000):
        self._setup()
        self._state = self.aicar.get_states()
        self._episode_ended = False
        self._episode_len = episode_len

    def _setup(self):
        self.iroad = InfiniteRoad(NUM_LANES)
        self.iroad.object_count = 0
        self.steps = 0

        for _ in range(NUM_ORDINARY_CARS):
            c = RuleBasedCar(self.iroad.object_count)
            self.iroad.object_count += 1
            self.iroad.place_object(c, choice(range(self.iroad.num_lanes)), self.iroad.object_count * 100)

        self.aicar = VelocityControlledAICar(self.iroad.object_count)
        self.iroad.object_count += 1
        self.iroad.place_object(self.aicar, 2, 0)

        cam_x, cam_y = self.aicar.get_xy()
        self.iroad.set_camera_xy(cam_x, cam_y)

        self.pause = False

    def _reset(self):
        self._setup()
        self._state = self.aicar.get_states()
        self._episode_ended = False
        return self._state

    def step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self._reset()

        # Make sure episodes don't go on forever.
        if self.steps >= self._episode_len:
            self._episode_ended = True
        else:
            self.steps += 1
            movable_objects_on_road = [vehicle for lane in self.iroad.vehicles for vehicle in lane if isinstance(vehicle, RuleBasedCar)]
            for ob in movable_objects_on_road:
                ob.move(None, None)
            self.aicar.move(action)
            self.iroad.update_road_state()

        reward = self.aicar.get_reward()
        self._state = self.aicar.get_states()
        if self._episode_ended:
            reward = self.aicar.get_reward()
            return self._state, reward, self._episode_ended
        else:
            return self._state, reward, self._episode_ended
