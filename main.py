from random import choice

import pygame as pg
from rl.simulator_car_interface import VelocityControlledAICar
from simulator.stuff_on_road.normal_car import RuleBasedCar

from display.offline_display import OfflineDisplay
from simulator.stuff_on_road.road import InfiniteRoad
from simulator.utils.constants import NUM_LANES, NUM_ORDINARY_CARS

'''
GEOMETRIC CONVENTIONS

X axis is horizontal and Y vertical on the display
X increases towards right
Y increases in upward direction
Therefore, pi/2 heading implies facing up and 0 implies facing right
X=0 implies left edge of road

Everything uses SI units
'''


class Simulation:
    def __init__(self):
        self.iroad = InfiniteRoad(NUM_LANES)
        self.iroad.object_count = 0
        self.steps = 0

        for _ in range(NUM_ORDINARY_CARS):
            c = RuleBasedCar(self.iroad.object_count)
            self.iroad.object_count += 1
            self.iroad.place_object(c, choice(range(self.iroad.num_lanes)), self.iroad.object_count*100)

        self.aicar = VelocityControlledAICar(self.iroad.object_count)
        self.iroad.object_count += 1
        self.iroad.place_object(self.aicar, 2, 0)

        cam_x, cam_y = self.aicar.get_xy()
        self.iroad.set_camera_xy(cam_x, cam_y)

        self.pause = False

    def step(self, a=(None, None)):
        self.steps += 1
        movable_objects_on_road = [vehicle for lane in self.iroad.vehicles for vehicle in lane]
        for ob in movable_objects_on_road:
            if ob == self.aicar:
                pstates, preward = ob.get_states(), ob.get_reward()
                ob.move(a[0], a[1])
                if ob.went_out():
                    return pstates, preward, True, None
                continue
            ob.move()
        self.iroad.update_road_state()
        return self.aicar.get_states(), self.aicar.get_reward(), self.aicar.went_out(), None


d = OfflineDisplay()
sim = Simulation()
steps = 0
previous_action = [0,0]
history = []
while True:

    time_passed = d.clock.tick(30)
    if time_passed > 100:
        continue

    for event in pg.event.get():
        if event.type == pg.QUIT:
            d.quit()
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                d.paused = not d.paused

    if not d.paused:
        steering = None
        gas = None
        action = (steering, gas)
        sim.step(a=action)
        previous_action = action
        if sim.aicar.went_out():
            total_reward = sim.aicar.get_reward()
            print("AI Car went out of road. Simulation ended")
            print("AI car reward earned = {}, collisions = {}, speed = {}, distance covered = {}"
                  .format(total_reward, sim.aicar.collision_count, sim.aicar.speed,
                          sim.aicar.distance_covered_till_now))
            break

        agent_car = {"x": sim.aicar.x, "y": sim.aicar.y, "heading": sim.aicar.heading, "image": sim.aicar.image,
                     "speed": sim.aicar.speed, "last_gas": sim.aicar.last_gas, "last_steering": sim.aicar.last_steering}

        all_cars = [{"x": car.x, "y": car.y, "heading": car.heading, "image": car.image,
                     "speed": car.speed, "last_gas": car.last_gas, "last_steering": car.last_steering}
                    for lane in sim.iroad.vehicles for car in lane]

        history.append({"agent_car": agent_car, "all_cars": all_cars, "num_lanes": sim.iroad.num_lanes, "camera_xy": sim.aicar.get_xy()})

        # d.draw(sim.iroad)
        #d.draw(agent_car, all_cars, sim.iroad.num_lanes, sim.aicar.get_xy())
        steps += 1
        print("AI car reward earned = {}, collisions = {}, speed = {}, distance covered = {}"
              .format(sim.aicar.get_reward(), sim.aicar.collision_count, sim.aicar.speed,
                      sim.aicar.distance_covered_till_now))


    # if steps % 100==0:
    #     sim.object_count += 1
    #     steps = StopSign(sim.iroad, sim.object_count)
    #     sim.object_count += 1
    #     if cw.direction==-1:
    #         sim.iroad.place_object(cw, on_road=True, start_lane=sim.iroad.num_lanes, start_y=choice(sim.iroad.crossings[1:5]))
    #     else:
    #         sim.iroad.place_object(cw, on_road=True, start_lane=1, start_y=choice(sim.iroad.crossings[1:5]))
    #     # sim.iroad.place_object(stps, on_road=False, start_y=400)


