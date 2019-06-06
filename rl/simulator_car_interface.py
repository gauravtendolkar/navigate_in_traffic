# -*- coding: utf-8 -*-
"""
=========
**velocity_controlled_ai_car** - Intelligent car whose gas is intelligently controlled. Has an openAI gym style API
=========
This module describes intelligent car whose gas is intelligently controlled. It extends BaseCar and has
the same geometrical reference

Source
----------------
"""

from simulator.stuff_on_road.normal_car import BaseCar
from simulator.utils.constants import CAR_MAX_SAFE_VEL, CAR_SAFETY_DIST, CAR_COLLISION_D, \
    CAR_IMAGE_WHITE_H, ALPHA, BETA, GAMMA, INFINITE_ROAD_MAX_LENGTH, DELTA


class VelocityControlledAICar(BaseCar):
    """Intelligent car whose gas is intelligently controlled

    It extends BaseCar and has an openAI gym style API to integrate with reinforcement learning agents

    Source
    ----------------
    """
    def __init__(self, *args, **kwargs):
        BaseCar.__init__(self, *args, **kwargs)
        self.image = CAR_IMAGE_WHITE_H
        self.distance_covered_till_now = 0
        self.collided = False
        self.collision_count = 0
        self.traffic_rule_violation = False
        self.traffic_rule_violation_count = 0
        self.distance_covered_in_dt = 0
        self._reward = 0
        self._cumulative_reward = 0

    def has_collided_in_dt(self):
        """
        Checks if car collided in current time step and updates self.collided.

        * Collision implies distance to front car is less than CAR_COLLISION_D (which is a small value)
        * Collisions can span multiple time steps, and so unless car gets out of current collision,
        the collision is not double counted. This is ensured with tracking collision in
        previous timestep (previous_dt_collision)

        :return: self.collided: boolean indicating whether car collided in current time step

        TODO:
        Add logic to check collision from behind by another car
        """
        previous_dt_collision = self.collided
        fc = self._get_front_vehicle()
        if fc and 0 <= fc.y - self.y < CAR_COLLISION_D:
            self.collided = True
            if not previous_dt_collision:
                self.collision_count += 1
        else:
            self.collided = False
        return self.collided

    def has_violated_traffic_rule(self):
        """
        Checks if car violated speed limit rule in current time step and updates self.traffic_rule_violation.

        * Traffic rule violation is self.speed > CAR_MAX_SAFE_VEL
        * Traffic rule violations can span multiple time steps, and so unless car slows to below speed limit,
        the violation is not double counted. This is ensured with tracking violation in
        previous timestep (previous_dt_violation). For example if a car overshoots speed limit and
        then never slows down, it is still counted as one violation. This may not be suitable for many RL situations.
        For instance, the car may learn to overspeed to extreme speeds just to compensate for the cost of one violation.
        In such cases appropriate contraction factor is necessary

        :return: self.traffic_rule_violation: boolean indicating whether car violated speed limit rule in current time step
        """
        previous_dt_violation = self.traffic_rule_violation
        if self.speed > CAR_MAX_SAFE_VEL:
            self.traffic_rule_violation = True
            if not previous_dt_violation:
                self.traffic_rule_violation_count += 1
        else:
            self.traffic_rule_violation = False
        return self.traffic_rule_violation

    def _decide_control_inputs(self):
        """Computes control actions from states

        This particular car always uses steering = 0. The gas is computed by an external Agent logic.
        The gas return value does not matter here

        Returns:
            steering, gas (float, float) in radians and m/s^2
        """
        steering, gas = 0, 0
        return steering, gas

    def move(self, gas, steering=None):
        """Computes control actions from states

        Uses gas argument, and steering from self._decide_control_inputs() to control the car.
        It also tracks self.distance_covered_in_dt, self.distance_covered_till_now,
        self.traffic_rule_violation, self.collided for reward computation
        """
        previous_y = self.y
        steering, _ = self._decide_control_inputs()
        self._control_car(steering, gas)

        self.distance_covered_in_dt = self.y-previous_y if self.y >= (previous_y-0.1) else self.y + INFINITE_ROAD_MAX_LENGTH - previous_y
        self.distance_covered_till_now += self.distance_covered_in_dt
        self.has_collided_in_dt()
        self.has_violated_traffic_rule()
        return

    def get_states(self):
        """Returns state of the car

        State space consists of 3 continuous valued states - speed, distance to front car and front car speed.
        distance to front car is clipped to CAR_SAFETY_DIST

        :return: list of 3 continuous valued states
        """
        front_car = self._get_front_vehicle()
        if not front_car or abs(front_car.y - self.y) > CAR_SAFETY_DIST:
            _state = [self.speed, CAR_SAFETY_DIST, CAR_MAX_SAFE_VEL]
        else:
            _state = [self.speed, front_car.y-self.y, front_car.speed]
        return _state

    def get_reward(self):
        """Returns reward in current timestep

        :return: self._reward: reward in current timestep
        """
        self._reward = ALPHA*self.has_collided_in_dt() + \
                       BETA*self.has_violated_traffic_rule() + GAMMA*self.distance_covered_in_dt \
                       + DELTA*int(abs(self.speed)<0.1)
        self._cumulative_reward += self._reward
        return self._reward

    def act(self, action):
        """helper function to call self.move(steering, gas) with passed parameters"""
        steering = action[0]
        gas = action[1]
        self.move(steering, gas)