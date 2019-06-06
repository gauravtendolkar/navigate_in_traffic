import os

import numpy as np
import tensorflow as tf

from simulator.utils.constants import CAR_MAX_ACCELERATION, CAR_HARD_BRAKE, CAR_SMOOTH_BRAKE

"""
Following analogy shows the purpose of each class - 
simulator_car_interface.VelocityControlledAICar is the car
Agent is the driver
AgentDQN are components of driver's brain
"""


class AgentDQN(object):
    def __init__(self, lr, name, n_actions=3, hidden_size=10, num_states=(3,), chkpt_dir='tmp/dqn', summary_dir='tmp/summary'):

        # The neural net takes states as input and outputs Q values for each action
        # DQN architecture restricts us to re-formulate problem into a single discretised action problem
        self.num_states = num_states  # number of states
        self.n_actions = n_actions  # number of actions
        self.name = name  # name of the network. In DQN, we use 2 networks - target network and evaluation network
        self.hidden_size = hidden_size  # number of units in hidden layer

        self.lr = lr  # learning rate of SGD

        # Standard Tensorflow 1.x workflow - build graph, start session, initialise graph variables, run computation
        self.summary_dir = summary_dir

        self.sess = tf.Session()  # start session
        self.build_network()  # build graph
        self.sess.run(tf.global_variables_initializer())  # initialise graph variables

        # Saving/check-pointing and Tensorboard
        self.saver = tf.train.Saver()  # initialise saver
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, 'dqn.ckpt')

        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def build_network(self):

        # Name scoping is important since we are creating multiple graphs programatically
        # Without scoping, the neurons will be shared among multiple graphs
        with tf.variable_scope(self.name):

            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None, self.n_actions],
                                           name='q_value')

            # Create a NN with 2 hidden layers, each containing self.hidden_size number of units
            # The output layer is of size self.n_actions, with no activation
            # Output represents Q values associated with actions for the input state

            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.num_states],
                                        name='inputs')

            dense1 = tf.layers.dense(self.input, units=self.hidden_size,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2))

            dense2 = tf.layers.dense(dense1, units=self.hidden_size,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.variance_scaling_initializer(scale=2))

            self.Q_values = tf.layers.dense(dense2,
                                            units=self.n_actions,
                                            kernel_initializer=tf.variance_scaling_initializer(scale=2))

            self.Q_summary = tf.summary.histogram('Q_values', self.Q_values)

            # q_target input will be equal to Q_values at all indices except at the index of action taken
            # At that index, it will be the ( reward earned + gamma * max(Q_values at next state) )

            self.loss = tf.reduce_mean(tf.square(self.Q_values - self.q_target))

            # add loss to tensorboard summary
            self.loss_summary = tf.summary.scalar('loss', self.loss)

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)


class Agent(object):
    def __init__(self, alpha, gamma, mem_size,
                 epsilon, batch_size, replace_target=10000,
                 input_dims=(3,), q_next_dir='tmp/q_next', q_eval_dir='tmp/q_eval'):

        # Action space goes from [-CAR_HARD_BRAKE, -CAR_SMOOTH_BRAKE, 0, 1, 2, ..., CAR_MAX_ACCELERATION]
        # The reason for this is that CAR_HARD_BRAKE (maximum de-acceleration) is much higher than CAR_MAX_ACCELERATION
        # An action space like [-CAR_HARD_BRAKE, -CAR_HARD_BRAKE+1, ..., -1, 0, 1, 2, ..., CAR_MAX_ACCELERATION] will
        # therefore sample more of brakes than acceleration during exploration phase
        self.action_space = [-CAR_HARD_BRAKE, -CAR_SMOOTH_BRAKE]+list(range(CAR_MAX_ACCELERATION))
        self.n_actions = len(self.action_space)

        self.gamma = gamma  # Contraction factor in Bellman equation
        self.mem_size = mem_size  # Memory size of the replay buffer
        self.mem_cntr = 0  # Memory counter
        self.epsilon = epsilon  # Rate of exploration.
        # Initially, exploration rate is 100% and it exponentially decays after some initial number of steps

        self.batch_size = batch_size
        self.replace_target = replace_target  # number of iterations after which to replace
        # target network with the evaluation network. A lower value implies higher TD error variance

        # Target network used to calculate Q values of next state and choose next action corresponding to max Q value
        self.q_next = AgentDQN(alpha, name='q_next', n_actions=self.n_actions,
                               num_states=input_dims, chkpt_dir=q_next_dir)

        # Evaluation network used for training and current action calculation
        # Target network is replaced by evaluation network every replace_target steps
        self.q_eval = AgentDQN(alpha, name='q_eval', n_actions=self.n_actions,
                               num_states=input_dims, chkpt_dir=q_next_dir)

        # Replay buffers for state, action, reward, next_state
        # We also store boolean indicating if simulation ended after taking action
        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)

    def store_transition(self, state, action, reward, state_, terminal):
        # overwrite if mem_cntr goes above replay buffer size
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        # Actions are stored in one hot encoded fashion
        actions = np.zeros(self.n_actions, dtype=np.int)
        actions[action] = 1
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal
        self.mem_cntr += 1

    def choose_action(self, state):
        # Exploration/Exploitation decision
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(list(range(self.n_actions)))
        else:
            actions = self.q_eval.sess.run(self.q_eval.Q_values, feed_dict={self.q_eval.input: [state]})
            action = np.argmax(actions)
        return action

    def learn(self):
        # Replace target network after few iterations
        if self.mem_cntr % self.replace_target == 0:
            self.update_graph()

        # It is important to randomly sample from replay buffer to remove serial correlations
        max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size
        batch = np.random.choice(max_mem, self.batch_size)
        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        action_values = np.array(range(self.n_actions), dtype=np.int8)
        action_indices = np.dot(action_batch, action_values)
        reward_batch = self.reward_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        q_eval = self.q_eval.sess.run(self.q_eval.Q_values,
                                     feed_dict={self.q_eval.input: state_batch})
        q_next = self.q_next.sess.run(self.q_next.Q_values,
                                feed_dict={self.q_next.input: new_state_batch})

        # For training the network, q_target has the same values as Q_values of q_eval for all indices except
        # at index of action taken
        q_target = q_eval.copy()

        idx = np.arange(self.batch_size)

        q_target[idx, action_indices] = reward_batch + \
                                self.gamma*np.max(q_next, axis=1)*terminal_batch

        loss_summary, Q_summary, _ = self.q_eval.sess.run([self.q_eval.loss_summary,
                                                           self.q_eval.Q_summary,
                                                           self.q_eval.train_op],
                                          feed_dict={self.q_eval.input: state_batch,
                                                     self.q_eval.q_target: q_target})

        self.q_eval.summary_writer.add_summary(loss_summary, self.mem_cntr)
        self.q_eval.summary_writer.add_summary(Q_summary, self.mem_cntr)

        # Reduce exploration rate
        if self.mem_cntr > 100000:
            self.epsilon *= 0.9999997

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def update_graph(self):
        t_params = self.q_next.params
        e_params = self.q_eval.params

        for t, e in zip(t_params, e_params):
            self.q_eval.sess.run(tf.assign(t,e))