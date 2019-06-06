import csv
import os
import pickle as pkl
import sys

import numpy as np

from rl.agent import Agent
from rl.environment import IRTrafficEnv

if __name__ == '__main__':
    experiment_name = sys.argv[1]
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    if not os.path.exists('tmp/{}'.format(experiment_name)):
        os.mkdir('tmp/{}'.format(experiment_name))
    env = IRTrafficEnv(episode_len=500)
    load_checkpoint = False
    agent = Agent(gamma=0.8, epsilon=1.0, alpha=0.002, input_dims=(3,),
                  mem_size=2500, batch_size=32, replace_target=10000)
    if load_checkpoint:
        agent.load_models()
    scores = []
    num_games = 10000
    score = 0

    print("Loading up the agent's memory with random driving")

    while agent.mem_cntr < 5000:
        done = False
        observation = env._reset()
        while not done:
            action = np.random.choice(list(range(agent.n_actions)))
            observation_, reward, done = env.step(agent.action_space[action])
            agent.store_transition(observation, action, reward, observation_, int(done))
            observation = observation_

    print("Done with random driving. Learning...")

    history_file = 'tmp/'+experiment_name+'/ep_{}.pkl'
    writer = csv.writer(open('tmp/'+experiment_name+'scores.csv', 'w'), delimiter=',')
    writer.writerow(['episode', 'score', 'epsilon'])

    for i in range(num_games):
        done = False
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-10):(i+1)])
            print('episode: ', i,'score: ', score, ' average score %.3f' % avg_score, 'epsilon %.3f' % agent.epsilon)
            agent.save_models()
        else:
            print('episode: ', i, 'score: ', score, 'epsilon %.3f' % agent.epsilon)

        observation = env._reset()
        score = 0
        history = []
        while not done:
            action = agent.choose_action(observation)

            agent_car = {"x": env.aicar.x, "y": env.aicar.y, "heading": env.aicar.heading, "image": env.aicar.image,
                         "speed": env.aicar.speed, "last_gas": env.aicar.last_gas,
                         "last_steering": env.aicar.last_steering, "cumulative_reward": env.aicar._cumulative_reward,
                         "distance_in_dt": env.aicar.distance_covered_in_dt,
                         "traffic_violation": env.aicar.has_violated_traffic_rule(),
                         "collision": env.aicar.has_collided_in_dt()}

            all_cars = [{"x": car.x, "y": car.y, "heading": car.heading, "image": car.image,
                         "speed": car.speed, "last_gas": car.last_gas, "last_steering": car.last_steering}
                        for lane in env.iroad.vehicles for car in lane]
            display_object = {"agent_car": agent_car, "all_cars": all_cars, "num_lanes": env.iroad.num_lanes,
                              "camera_xy": env.aicar.get_xy()}

            history.append(display_object)
            observation_, reward, done = env.step(agent.action_space[action])
            score += reward
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))
            observation = observation_
            agent.learn()
        pkl.dump(history, open(history_file.format(i), 'wb'))

        writer = csv.writer(open('tmp/'+experiment_name+'scores.csv', 'a+'), delimiter=',')
        writer.writerow([i, score, agent.epsilon])