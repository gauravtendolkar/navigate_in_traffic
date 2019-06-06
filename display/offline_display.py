import sys
from math import sqrt, atan2, cos, sin, pi

import pygame as pg

from simulator.utils.constants import SCREEN_HEIGHT, SCREEN_WIDTH, BLACK, CAR_LENGTH, PIXELS_PER_METER, CAR_WIDTH, \
    ROAD_GREY


class OfflineDisplay:
    def __init__(self):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        pg.init()

        pg.font.init()
        self.font = pg.font.SysFont('Comic Sans MS', 22)

        self.screen = pg.display.set_mode(
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        )
        pg.display.set_caption("Bitrus")

        self.clock = pg.time.Clock()
        self.paused = False

    def draw_background(self):
        self.screen.fill((0, 0, 0))

    def __draw_dashed_line(self, coords, camera_xy, wallcolor=BLACK):
        length = sqrt((coords[1][1] - coords[0][1]) ** 2 + (coords[1][0] - coords[0][0]) ** 2)
        dash_length = 10.0
        slope = atan2((coords[1][1] - coords[0][1]), (coords[1][0] - coords[0][0]))

        camera_y = camera_xy[1]
        offset = camera_y % (dash_length * 2)

        for index in range(0, int(length / dash_length)+5, 2):
            start = (coords[0][0] + (index * dash_length) * cos(slope), coords[0][1] +
                     (index * dash_length + offset) * sin(slope))
            end = (coords[0][0] + ((index + 1) * dash_length) * cos(slope), coords[0][1] +
                   ((index + 1) * dash_length + offset) * sin(slope))
            pg.draw.line(self.screen, wallcolor, start, end, 1)

    def draw_road(self, num_lanes, camera_xy):
        road_start = int(SCREEN_WIDTH/4)
        road_end = road_start + int(SCREEN_WIDTH/2)
        pg.draw.polygon(self.screen, ROAD_GREY, ((road_start, 0), (road_end, 0), (road_end, SCREEN_HEIGHT),
                                                 (road_start, SCREEN_HEIGHT)))
        lane_width = SCREEN_WIDTH/(2*num_lanes)
        for l in range(1, num_lanes):
            self.__draw_dashed_line(((l*lane_width+road_start, 0), (l*lane_width+road_start, SCREEN_HEIGHT)), camera_xy)

        return

    def draw_car(self, camera_xy, car, scale=(CAR_WIDTH*PIXELS_PER_METER, CAR_LENGTH*PIXELS_PER_METER)):
        road_start = int(SCREEN_WIDTH / 4)
        x, y = car["x"], car["y"]
        x, y = (PIXELS_PER_METER*x)+road_start-scale[0]/2.0, -y+scale[1]/2.0
        image = pg.transform.rotate(pg.image.load(car["image"]), 90)
        image = pg.transform.smoothscale(image, (scale[1], scale[0]))

        image = pg.transform.rotozoom(image, -(-car["heading"]*180/pi+180), 1)
        rel_x, rel_y = camera_xy
        rel_x, rel_y = (PIXELS_PER_METER * rel_x) + road_start, -rel_y
        x, y = x, y - rel_y + 3*SCREEN_HEIGHT/4
        self.screen.blit(image, (x, y))

    def draw_stats(self, car):
        self.stat_font = pg.font.SysFont('Comic Sans', 16)
        textsurface = self.stat_font.render('Speed: {0:.1f}'.format(car["speed"]), True, (255, 255, 255), )
        self.screen.blit(textsurface, (10, 10))
        textsurface = self.stat_font.render('Accelerator: {0:.1f}'.format(car["last_gas"]), True, (255, 255, 255), )
        self.screen.blit(textsurface, (10, 25))
        textsurface = self.stat_font.render('Steering: {0:.1f}'.format(car["last_steering"] * 180 / pi), True, (255, 255, 255), )
        self.screen.blit(textsurface, (10, 40))
        textsurface = self.stat_font.render('Reward: {0:.1f}'.format(car["cumulative_reward"]), True,
                                            (255, 255, 255), )
        self.screen.blit(textsurface, (10, 55))
        textsurface = self.stat_font.render('dDistance: {0:.1f}'.format(car["distance_in_dt"]), True,
                                            (255, 255, 255), )
        self.screen.blit(textsurface, (10, 70))
        textsurface = self.stat_font.render('Trf Violation: {0:.1f}'.format(car["traffic_violation"]), True,
                                            (255, 255, 255), )
        self.screen.blit(textsurface, (10, 85))
        textsurface = self.stat_font.render('Collision: {0:.1f}'.format(car["collision"]), True,
                                            (255, 255, 255), )
        self.screen.blit(textsurface, (10, 100))

    def draw(self, agent_car, all_cars, num_lanes, camera_xy):
        self.draw_background()
        self.draw_road(num_lanes, camera_xy)

        camera_xy = (agent_car["x"], agent_car["y"])

        self.draw_stats(agent_car)

        for car in all_cars:
            self.draw_car(camera_xy, car)

        pg.display.flip()

    def quit(self):
        sys.exit()