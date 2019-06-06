import sys
from math import sqrt, atan2, cos, sin, pi

import pygame as pg

from simulator.stuff_on_road.normal_car import RuleBasedCar
from simulator.utils.constants import SCREEN_HEIGHT, SCREEN_WIDTH, BLACK, CAR_LENGTH, PIXELS_PER_METER, CAR_WIDTH, \
    ROAD_GREY


class Display:
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

    def __draw_dashed_line(self, coords, infinite_road, wallcolor=BLACK):
        length = sqrt((coords[1][1] - coords[0][1]) ** 2 + (coords[1][0] - coords[0][0]) ** 2)
        dash_length = 10.0
        slope = atan2((coords[1][1] - coords[0][1]), (coords[1][0] - coords[0][0]))

        camera_y = infinite_road.get_camera_xy()[1]
        offset = camera_y % (dash_length * 2)

        for index in range(0, int(length / dash_length)+5, 2):
            start = (coords[0][0] + (index * dash_length) * cos(slope), coords[0][1] +
                     (index * dash_length + offset) * sin(slope))
            end = (coords[0][0] + ((index + 1) * dash_length) * cos(slope), coords[0][1] +
                   ((index + 1) * dash_length + offset) * sin(slope))
            pg.draw.line(self.screen, wallcolor, start, end, 1)

    def draw_road(self, infinite_road):
        road_start = int(SCREEN_WIDTH/4)
        road_end = road_start + int(SCREEN_WIDTH/2)
        pg.draw.polygon(self.screen, ROAD_GREY, ((road_start, 0),(road_end, 0),(road_end, SCREEN_HEIGHT),(road_start, SCREEN_HEIGHT)))
        nl = infinite_road.num_lanes
        lane_width = SCREEN_WIDTH/(2*nl)
        for l in range(nl):
            if l==0:
                continue
            self.__draw_dashed_line(((l*lane_width+road_start,0), (l*lane_width+road_start,SCREEN_HEIGHT)), infinite_road)

        return

    def transform_xy(self, road, x, y, scale=(0,0)):
        road_start = int(SCREEN_WIDTH / 4)
        x, y = (PIXELS_PER_METER * x) + road_start - scale[0] / 2.0, -y + scale[1] / 2.0
        rel_x, rel_y = road.get_camera_xy()
        rel_x, rel_y = (PIXELS_PER_METER * rel_x) + road_start, -rel_y
        x, y = x, y - rel_y + 3 * SCREEN_HEIGHT / 4
        return x,y

    def draw_car(self, infinite_road, car, rotate=-90, scale=(CAR_WIDTH*PIXELS_PER_METER, CAR_LENGTH*PIXELS_PER_METER)):
        road_start = int(SCREEN_WIDTH / 4)
        x,y = car.get_xy()
        x,y = (PIXELS_PER_METER*x)+road_start-scale[0]/2.0, -y+scale[1]/2.0
        image = pg.transform.rotate(car.image, 90)
        image = pg.transform.smoothscale(image, (scale[1],scale[0]))

        image = pg.transform.rotozoom(image, -(-car.heading*180/pi+180), 1)
        rel_x,rel_y = infinite_road.get_camera_xy()
        rel_x, rel_y = (PIXELS_PER_METER * rel_x) + road_start, -rel_y
        x, y = x, y - rel_y + 3*SCREEN_HEIGHT/4

        self.screen.blit(image, (x,y))

    def draw_background(self, infinite_road):
        self.screen.fill((0, 0, 0))

    def draw_stats(self, car):
        self.stat_font = pg.font.SysFont('Comic Sans', 16)
        textsurface = self.stat_font.render('Speed: {0:.1f}'.format(car.speed), True, (255, 255, 255), )
        self.screen.blit(textsurface, (10, 10))
        textsurface = self.stat_font.render('Accelerator: {0:.1f}'.format(car.last_gas), True, (255, 255, 255), )
        self.screen.blit(textsurface, (10, 25))
        textsurface = self.stat_font.render('Steering: {0:.1f}'.format(car.last_steering * 180 / pi), True, (255, 255, 255), )
        self.screen.blit(textsurface, (10, 40))

    def draw(self, infinite_road):
        self.draw_background(infinite_road)
        self.draw_road(infinite_road)

        cars = [c for lane in infinite_road.vehicles for c in lane]
        for car in cars:
            if not isinstance(car, RuleBasedCar):
                el_x, el_y = car.get_xy()
                infinite_road.set_camera_xy(el_x, el_y)
                self.draw_stats(car)
                break
        for car in cars:
            self.draw_car(infinite_road, car)

        pg.display.flip()

    def quit(self):
        sys.exit()

