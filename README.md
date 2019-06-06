# Reinforcement Learning for Navigating in Urban Traffic

Reinforcement Learning Environment for Autonomous Cars in Traffic (Infinite Straight Road version)

This repository is used for the CATS project at Ohio State University. The repository consists of four parts - 
1. A simulator which simulates a multi-lane infinite straight road. The road has some rule based probabilistic cars and one smart car. The rule based cars follow all traffic rules, but with different levels of aggressiveness (distance to maintain with front car, overspeeding etc). They also exhibit complex behavious like overtaking from left only, staying on right unless to overtake, etc. The smart car takes its sorrounding as input state and decides on gas/brake and steering angle. Smart car's decision function can be implemented as desired
2. An openAI gym style interface to the simulator environment and the smart car
3. A template Deep Q-Learning code that uses above 2 to control acceleration of the smart car.
4. [PyGame](https://www.pygame.org/news) code to display the simulation

Example policy learnt by smart car (Car learns to break as it reaches close to speed limit of 25)

[![Learnt Policy](https://j.gifs.com/YWM22n.gif)]

## Directory Structure
The following directory structure is followed -
```
simulator/ (Contains simulator related code)
    media/
        images/
            (Images used in simulation)
    stuff_on_road/
        normal_car.py (Contains code for basic car mechanics and logic of rule based cars)
        road.py (Contains road related code)
    utils/
        ds.py (Contains utility functions and custom data structures)
        constants.py (Contains all simulation related constants)
display/ (Contains simulation visualisation related code)
    display.py (Display while running simulation)
    offline_display.py (Display from simulation logs)
rl/ (Contains model architecture)
    agent.py (Agent and the Q-Network classes)
    environment.py (OpenAI gym style environment for the simulator)
    simulator_car_interface.py (Interface to simulator car)
dqn_main.py (Run DQN training)
dqn_main.py (Visualise a particular episode of training)
main.py (Run simulation where smart car just goes straight at speed limit)
```

## Requirements
The repository uses Tensorflow 1.8.0. May throw some innocuous deprecated API warnings.
The display part also uses Pygame 1.9.4
Rest of the requirements are specified in requirements.txt

## Setup
1. Setup python 3 virtual environment. If you dont have ```virtualenv```, install it with

```
pip install virtualenv
```

2. Then create the environment with

```
virtualenv -p $(which python3) env
```

3. Activate the environment

```
source env/bin/activate
```

5. Clone the repository

```
git clone https://gitlab.com/codetendolkar/navigate_in_traffic.git
cd navigate_in_traffic
```

4. Install requirements

```
pip install -r requirements.txt
```

6. Run the training script

```
python dqn_main.py <name_for_simulation>
```

This will store logs in ```./tmp/name_for_simulation/``` directory as ```ep_<episode_number>.pkl```. You can visualise the episode by running
```
python dqn_display.py <name_for_simulation> <episode_number>
```
