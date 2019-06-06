from display.offline_display import OfflineDisplay
import pickle as pkl
import pygame as pg
import sys

simulation_name = experiment_name = sys.argv[1]
ep_num = sys.argv[2]
history_file = 'tmp/{}/ep_{}.pkl'

d = OfflineDisplay()
step = 0
history = pkl.load(open(history_file.format(simulation_name, ep_num), 'rb'))

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
        if step >= len(history):
            d.quit()
        d.draw(**history[step])
        step += 1