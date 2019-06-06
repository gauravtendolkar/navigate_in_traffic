# -*- coding: utf-8 -*-
"""
=========
**road** - Infinite road
=========
This module describes the infinite road

Geometry Reference
----------------

The geometry of environment is defined as follows

* Y axis is along the infinite road and X axis is perpendicular to the road
* The clockwise direction of rotation is considered positive
* (Note for Americans) All units are SI

Tackling infinity
----------------

The road has a camera and the camera is attached to a particular x,y. For example, the camera could be
attached to the instance of SmartCar or could be fixed at a location.

The length of road is fixed using configuration parameter INFINITE_ROAD_MAX_LENGTH. If camera
coordinates (which start at 0,0) reach the max length, the camera coordinates are reset to x,0 and
the x,y of all other objects in the simulation are reset maintaining relative position with camera.

If any of the objects' absolute distance from camera coordinates exceeds road's max length,
the objects are removed from the simulation. Therefore if you have a static camera, eventually all moving cars
will be removed from the simulation

Notes
----------------

Placing objects on road changes the `road` attribute of object to current road

Source
----------------
"""

from simulator.utils.ds import get_index_from_y

from simulator.utils.constants import LANE_WIDTH, INFINITE_ROAD_MAX_LENGTH


class InfiniteRoad:
    """Infinite road

    The geometry of environment is defined as follows

    * Y axis is along the infinite road and X axis is perpendicular to the road
    * The clockwise direction of rotation is considered positive
    * (Note for Americans) All units are SI

    The road has a camera and the camera is attached to a particular x,y. For example, the camera could be
    attached to the instance of SmartCar or could be fixed at a location.

    The length of road is fixed using configuration parameter INFINITE_ROAD_MAX_LENGTH. If camera
    coordinates (which start at 0,0) reach the max length, the camera coordinates are reset to x,0 and
    the x,y of all other objects in the simulation are reset maintaining relative position with camera.

    If any of the objects' absolute distance from camera coordinates exceeds road's max length,
    the objects are removed from the simulation. Therefore if you have a static camera, eventually all moving cars
    will be removed from the simulation

    Placing objects on road changes the `road` attribute of object to current road

    Source
    ----------------
    """
    def __init__(self, num_lanes):
        self.num_lanes = num_lanes
        self.vehicles = [[] for _ in range(num_lanes)]
        self.camera_xy = (0, 0)
        self.object_count = 0

    def place_object(self, ob, lane, y, x=None):
        """
        Places object on road. Objects are seggregated as follows

        * If object has type VEHICLE/HURDLE, it is stored in vehicles/hurdles property of road (a list of list).
        Each list within corresponds to a lane (goes from 0 to NUM_LANES). The objects are sorted in reverse order
        according to Y coordinate
        * If object has type CUE/ROW_OBJECT, it is stored in cues/row_objects property of road (a single list).
        The objects are sorted in reverse order according to Y coordinate

        Placing an object is an O(n) operation. A better way is to place randomly and then sort the resulting array.
        Random placement is almost sorted since very few cars change lanes/overtake. Sorting an almost sorted array in python's
        timsort is superfast (Although still O(nlogn))

        :param ob: The object to be placed (Car, Cue, Hurdle, Row object)
        :param t: Type of object (CUE, VEHICLE, ROW_OBJECT, HURDLE as defined in constants)
        :param lane: (optional) Lane to put the object in. X is set to X of center line of lane
        :param x: X coordinate of point of placement
        :param y: Y coordinate of point of placement
        :return: Object that was placed with attributes `road`, `x`, `y` modified

        TODO:
        Change object placement algorithm
        """
        ob.road = self
        i = get_index_from_y(self.vehicles[lane], y)
        ob.pos_in_lane = i
        self.vehicles[lane].insert(i, ob)
        if x and self.get_lane_from_x(x)!=lane:
            raise ValueError("Incorrect lane/x cobination")

        ob.x, ob.y = self.get_x_from_lane(lane) if x is None else x, y
        ob.lane = lane
        return ob

    @staticmethod
    def get_x_from_lane(lane):
        """
        Returns X coordinate of lane's centerline
        """
        return lane*LANE_WIDTH + LANE_WIDTH/2.0

    def get_lane_from_x(self, x):
        """
        Returns lane number from X coordinate. Its clipped to 0, NUM_LANES-1
        """
        return max(min(self.num_lanes-1, int(x // LANE_WIDTH)), 0)

    def set_camera_xy(self, x, y):
        """
        Sets camera X, Y coordinate to (x, y)
        """
        self.camera_xy = (x, y)

    def get_camera_xy(self):
        """
        Returns camera's coordinate tuple (x,y)
        """
        return self.camera_xy

    def reset_on_camera_overflow(self, vehicles):
        substract_y = 0
        camera_y_overflow = False

        if self.camera_xy[1] > INFINITE_ROAD_MAX_LENGTH:
            print("Camera Y went beyond max road length. Resetting Y of all objects within simulation")
            substract_y = INFINITE_ROAD_MAX_LENGTH
            camera_y_overflow = True

        if not camera_y_overflow:
            vehicles = [vehicle for lane in self.vehicles for vehicle in lane]
            return vehicles

        self.set_camera_xy(self.camera_xy[0], self.camera_xy[1] - substract_y)

        for lane in self.vehicles:
            for vehicle in lane:
                vehicle.x, vehicle.y = vehicle.x, vehicle.y - substract_y
                vehicles.append(vehicle)

        return vehicles

    def reset(self):
        """
        Resets roads storage (`cues`, `row_objects`, `hurdles` and `vehicles` properties)
        :return:
        """
        self.vehicles = [[] for _ in range(self.num_lanes)]
        return

    def update_road_state(self):
        vehicles = []

        vehicles = self.reset_on_camera_overflow(vehicles)

        self.reset()

        for vehicle in vehicles:
            self.place_object(vehicle, vehicle.lane, vehicle.y, vehicle.x)
        return



