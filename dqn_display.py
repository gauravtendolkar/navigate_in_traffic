from display.offline_display import OfflineDisplay
import pickle as pkl
import pygame as pg

history_file = 'tmp/testing/ep_{}.pkl'

d = OfflineDisplay()
ep_num = 8400#934 #528# 459, 819
step = 0
history = pkl.load(open(history_file.format(ep_num), 'rb'))

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