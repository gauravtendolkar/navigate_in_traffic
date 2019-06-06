# -*- coding: utf-8 -*-
"""
=========
**normal_car** - Physics and decision making
=========
This module provides access to `BaseCar` and `RuleBasedCar` classes

Geometry Reference
----------------

The geometry of environment is defined in `road` module. To summarise
* Y axis is along the infinite road and X axis is perpendicular to the road
* The clockwise direction of rotation is considered positive
* (Note for Americans) All units are SI


Car Physics
-------------

All 4 wheeled vehicles extend the `BaseCar`. `BaseCar` includes the dynamics of
a four wheeled car using a simple model described below
* Both front wheels face in the same direction (let's say DF) which is car heading + steering angle
* Both back wheels face in the direction of car heading (let's say DH)
* Front wheels move in direction DF
* Back wheels move in direction DH
* All wheel movements move the wheel by (car's speed multiplied by simulation timestep) distance


Notes
----------------

Each car HAS to maintain its ID, Road, X, Y, Speed, Heading, Length, Width. ID and Road are fed as arguments
during BaseCar construction. Length and Width can be configured via the configuration file constants
`NORMAL_CAR_LENGTH` and `NORMAL_CAR_WIDTH`. X, Y, Speed and Heading are updated every timestep by vehicle dynamics
described in `BaseCar._control_car`. Overload this function to upgrade to a better dynamics model.

Each car also has to maintain an image attribute on the `RuleBasedCar` or `SmartCar` IF you want to display the
simulation.

The car can be controlled via Steering and Acceleration whose limits are specified in the configuration as
`CAR_MAX_STEERING`, `CAR_MIN_STEERING` and `CAR_MAX_ACC`, `CAR_MIN_ACC`

TODO:
* Make X, Y, Speed and Heading private

Source
----------------
"""

from math import pi, sin, cos, atan2
from random import choice

from simulator.utils.constants import CAR_MAX_ACCELERATION, CAR_HARD_BRAKE, \
    SIM_DT, ROAD_DRAG_COEFF, GAS_PEDAL_TO_ACC, CAR_MAX_SAFE_VEL, CAR_IMAGE_GREY_H, \
    CAR_SAFETY_DIST, LANE_WIDTH


class BaseCar:
    """BaseCar includes the dynamics of a four wheeled car using a simple model described below

    * Both front wheels face in the same direction (let's say DF) which is car heading + steering angle
    * Both back wheels face in the direction of car heading (let's say DH)
    * Front wheels move in direction DF
    * Back wheels move in direction DH
    * All wheel movements move the wheel by (car's speed multiplied by simulation timestep) distance

    The car can be controlled via Steering and Acceleration whose limits are specified in the
    configuration as `CAR_MAX_STEERING`, `CAR_MIN_STEERING` and `CAR_MAX_ACC`, `CAR_MIN_ACC`

    Each car decides control inputs and then applies to the dynamics. The job of deciding control inputs
    is the intelligence of the car. The `BaseCar` always decides (0,0) as the control inputs which wont
    move the car. To make the car intelligent, you need to overload the `_decide_control_inputs` method.
    Based on this structure, there are two types of cars which inherit from `BaseCar` namely `RuleBasedCar`
    and `SmartCar`. `RuleBasedCar` has a fixed set of rules (if-else conditions) that it uses to decide the
    control inputs. They never collide and never violate traffic rules. Their behaviour can be controlled
    via aggresivness parameter which modifies the safety limits. `SmartCar` can be used to implement any
    intelligent (or dumb) algorithm for choosing the control inputs.

    Use `RuleBasedCar` or `SmartCar` to create a car if you want it to do something in the simulation.

    """
    def __init__(self, idx):
        """Initialize `BaseCar`

        The initial heading is set to pi/2. (x,y) set to (0,0). Car is generally independent of road.
        After a car is initialized, it is placed on road. The placement method on road class then sets
        appropriate values of (x,y).

        Example:
        >>> car = BaseCar(1)
        >>> car.get_xy()
        >>> (0,0)
        >>> car._control_car(steering=0,gas=5)
        >>> car.get_xy()
        >>> (3.0616169978683886e-18, 0.04999999999999993)

        Args:
            idx: Unique string ID assigned to the car. Uniqueness of idx is neither checked
            nor necessary for the simulation

        Returns:
            `BaseCar` object
        """
        self.id = str(idx)
        self.road = None
        self.x = 0
        self.y = 0
        self.speed = 0
        self.heading = pi / 2
        self.lane = None
        self.length = 4
        self.width = 2
        self.image = None
        self.pos_in_lane = None
        self.left_indicator = False
        self.right_indicator = False
        self.target_lane = None
        self.turning = False
        self.max_speed = CAR_MAX_SAFE_VEL*choice([0.5, 1.0, 1.5])
        self.last_gas = 0
        self.last_steering = 0

    def get_xy(self):
        """Gives location of the car in frame of road geometry

        Returns:
            Tuple of floats (x,y)
        """
        return self.x, self.y

    def get_heading(self):
        """Gives heading of the car in frame of road geometry

        Returns:
            heading (float) in radians
        """
        return self.heading

    def _decide_control_inputs(self):
        """Computes control actions from states

        This method always returns (0,0). Overload this method to implement intelligence.
        For example, policy learnt via reinforcement learning is implemented by this method

        Returns:
            steering, gas (float, float) in radians and m/s^2
        """
        steering, gas = 0, 0
        return steering, gas

    def _control_car(self, steering, gas):
        """Physics of 2D four wheeled car

        The dynamics are as follows

        - Both front wheels face in the same direction (let's say DF) which is car heading + steering angle
        - Both back wheels face in the direction of car heading (let's say DH)
        - Front wheels move in direction DF
        - Back wheels move in direction DH
        - All wheel movements move the wheel by (car's speed multiplied by simulation timestep) distance

        """
        self.last_steering, self.last_gas = steering, gas

        front_wheel_x = self.x + self.length / 2 * cos(self.heading)
        front_wheel_y = self.y + self.length / 2 * sin(self.heading)
        back_wheel_x = self.x - self.length / 2 * cos(self.heading)
        back_wheel_y = self.y - self.length / 2 * sin(self.heading)

        self.speed += (
            gas * GAS_PEDAL_TO_ACC * SIM_DT - ROAD_DRAG_COEFF * self.speed * SIM_DT)
        self.speed = self.speed if self.speed > 0 else 0

        # update wheel positions
        front_wheel_x += self.speed * SIM_DT * cos(self.heading + steering)
        front_wheel_y += self.speed * SIM_DT * sin(self.heading + steering)
        back_wheel_x += self.speed * SIM_DT * cos(self.heading)
        back_wheel_y += self.speed * SIM_DT * sin(self.heading)

        # update car position and heading
        self.x = (front_wheel_x + back_wheel_x) / 2
        self.y = (front_wheel_y + back_wheel_y) / 2
        self.heading = atan2((front_wheel_y - back_wheel_y), (front_wheel_x - back_wheel_x))
        self.lane = self.road.get_lane_from_x(self.x)

        return

    def _get_nearby_left_vehicles(self):
        """Returns a list of vehicles in immediate left lane which are within CAR_SAFETY_DIST

        There may be cars in left lane not within CAR_SAFETY_DIST. Those cars wont be part of list

        Returns:
            list of vehicles in immediate left lane which are within CAR_SAFETY_DIST
            OR empty list if no such vehicle found
        """
        if self.road.get_lane_from_x(self.x) > 0:
            return [v for v in self.road.vehicles[self.road.get_lane_from_x(self.x)-1] if abs(v.y-self.y)<CAR_SAFETY_DIST]
        else:
            return []  # return any non empty list

    def _get_nearby_right_vehicles(self):
        if self.road.get_lane_from_x(self.x) < self.road.num_lanes - 1:
            return [v for v in self.road.vehicles[self.road.get_lane_from_x(self.x)+1] if abs(v.y-self.y)<CAR_SAFETY_DIST]
        else:
            return [] # return any non empty list

    def _shift_to_left_lane(self, target_lane):
        self.left_indicator = True
        if self.x <= target_lane*LANE_WIDTH + LANE_WIDTH/2.0:
            self.left_indicator = False
        return

    def _shift_to_right_lane(self, target_lane):
        self.right_indicator = True
        if self.x >= target_lane*LANE_WIDTH + LANE_WIDTH/2.0:
            self.right_indicator = False
        return

    def _get_front_vehicle(self):
        """Sensing vehicle in front in the same lane

        This method can only be called after it has been placed on the road. It returns the car in front.
        Refer to the `InfiniteRoad` data structure to understand how the computation is O(1)

        Returns:
            Car (`RuleBasedCar` or `SmartCar`)
        """
        return self.road.vehicles[self.road.get_lane_from_x(self.x)][self.pos_in_lane - 1] if self.pos_in_lane>0 else None

    def move(self, gas, steering=None):
        """Moves the car one time step

        Calls `_decide_control_inputs` and passes its outputs to `_control_car`
        """
        if steering is None and gas is None:
            steering, gas = self._decide_control_inputs()
        elif steering is None:
            steering, _ = self._decide_control_inputs()
        elif gas is None:
            _, gas = self._decide_control_inputs()

        self._control_car(steering, gas)

        return steering, gas


class RuleBasedCar(BaseCar):
    """Extends `BaseCar` with some fixed rule based intelligence

    `RuleBasedCar` has a fixed set of rules (if-else conditions) that it uses to decide the
    control inputs. They never collide and never violate traffic rules. Their behaviour can be controlled
    via aggresivness parameter which modifies the safety limits.
    """
    def __init__(self, *args, **kwargs):
        """Initialize `RuleBasedCar`

        Initializes `BaseCar` and sets the image attribute. The image can be changed via configuration
        RULE_BASED_CAR_IMAGE

        Returns:
            `RuleBasedCar` object
        """
        BaseCar.__init__(self, *args, **kwargs)
        self.image = CAR_IMAGE_GREY_H

    def _decide_control_inputs(self):
        """Rule Based intelligence to decide control inputs from states

        Overview of rules:
        - Check immediate blocking (Car of Hurdle) front object in own lane
        - Check immediate non-blocking (Cue) front object. Cues are applicable to all lanes
        - If (cue is before blocking objects and distance to cue is less than safety distance) OR
        (there is no blocking object but distance to cue is less than safety distance), then
        follow cue recommendation and return appropriate steering and gas
        - Else if no blocking object exists, accelerate to reach max speed of road
        - Else (blocking object exists and is before cue), if distance to blocking object is less
        than safety distance, brake to halt before the blocker

        Returns:
            steering, gas (float, float) in radians and m/s^2
        """
        steering, gas = 0, 0
        front_vehicle = self._get_front_vehicle()

        closest_blocking_object = front_vehicle
        distance_to_closest_blocking_object = closest_blocking_object.y - self.y if bool(closest_blocking_object) else float("Inf")

        if ((bool(closest_blocking_object) and distance_to_closest_blocking_object < 2*CAR_SAFETY_DIST and closest_blocking_object.speed < self.speed) or
                self.left_indicator) and self.road.get_lane_from_x(self.x) > 0 and len(self._get_nearby_left_vehicles()) == 0:

            if not self.left_indicator:
                self.target_lane = self.lane - 1
            self._shift_to_left_lane(self.target_lane)
            if self.left_indicator:
                steering = 0.1*pi/180
            else:
                steering = 0

        if not self.left_indicator:
            if (((bool(closest_blocking_object) and distance_to_closest_blocking_object < 2*CAR_SAFETY_DIST and closest_blocking_object.speed < self.speed) or
                    self.right_indicator) and
                    (self.road.get_lane_from_x(self.x) == 0 or len(self._get_nearby_left_vehicles()) != 0) and
                        len(self._get_nearby_right_vehicles()) == 0 and
                        self.road.get_lane_from_x(self.x) < self.road.num_lanes-1) or \
                    (self.road.get_lane_from_x(self.x) == 0 and len(self._get_nearby_right_vehicles()) == 0):

                if not self.right_indicator:
                    self.target_lane = self.lane + 1
                self._shift_to_right_lane(self.target_lane)
                if self.right_indicator:
                    steering = -0.1*pi/180
                else:
                    steering = 0

        if self.target_lane is not None and self.target_lane == self.lane and abs(self.x - self.road.get_x_from_lane(self.lane))<0.05*LANE_WIDTH:
            self.left_indicator = False
            self.right_indicator = False
            self.target_lane = None

        if closest_blocking_object:
            distance_to_closest_blocking_object = closest_blocking_object.y - self.y

            if 0 < distance_to_closest_blocking_object < CAR_SAFETY_DIST:
                gas = -CAR_HARD_BRAKE
            elif distance_to_closest_blocking_object >= CAR_SAFETY_DIST and self.speed <= self.max_speed:
                gas = CAR_MAX_ACCELERATION
            else:
                gas = 0
        else:
            if self.speed <= self.max_speed:
                gas = CAR_MAX_ACCELERATION
            elif self.speed > self.max_speed:
                gas = -CAR_HARD_BRAKE
            else:
                gas = 0

        if not (self.left_indicator or self.right_indicator):
            steering = 0.5 * pi / 180 * (self.x - self.road.get_x_from_lane(self.lane)) + 1.0 * self.speed * cos(self.heading) / self.max_speed

        return steering, gas


class AICar(BaseCar):
    """Extends `BaseCar` with some fixed rule based intelligence

    `RuleBasedCar` has a fixed set of rules (if-else conditions) that it uses to decide the
    control inputs. They never collide and never violate traffic rules. Their behaviour can be controlled
    via aggresivness parameter which modifies the safety limits.
    """
    def __init__(self, *args, **kwargs):
        """Initialize `RuleBasedCar`

        Initializes `BaseCar` and sets the image attribute. The image can be changed via configuration
        RULE_BASED_CAR_IMAGE

        Returns:
            `RuleBasedCar` object
        """
        BaseCar.__init__(self, *args, **kwargs)
        self.image = CAR_IMAGE_GREY_H

    def _decide_control_inputs(self):
        """Rule Based intelligence to decide control inputs from states

        Overview of rules:
        - Check immediate blocking (Car of Hurdle) front object in own lane
        - Check immediate non-blocking (Cue) front object. Cues are applicable to all lanes
        - If (cue is before blocking objects and distance to cue is less than safety distance) OR
        (there is no blocking object but distance to cue is less than safety distance), then
        follow cue recommendation and return appropriate steering and gas
        - Else if no blocking object exists, accelerate to reach max speed of road
        - Else (blocking object exists and is before cue), if distance to blocking object is less
        than safety distance, brake to halt before the blocker

        Returns:
            steering, gas (float, float) in radians and m/s^2
        """
        steering, gas = 0, 0

        return steering, gas
