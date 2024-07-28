import numpy as np
import tensorflow as tf
from environment import Environment as Env
from time import sleep

def get_action(Q, state):
    state = np.array([state])
    return np.argmax(Q.predict(state, verbose=0))

def play(Q, env):
    state = env.reset(1, Q, 0)
    env.render()
    while True:
        action_1 = get_action(Q, state)

        state, reward, done = env.single_step(action_1, 1)

        env.render()
        print("Reward: {}".format(reward))
        print("Action: {}".format(action_1))
        if done:
            break

        action_2 = int(input("Enter column: "))
        state, reward, done = env.single_step(action_2, -1)

        env.render()
        print("Reward: {}".format(reward))
        print("Action: {}".format(action_2))
        sleep(1)
        if done:
            break

if __name__ == "__main__":
    # Load Model with Keras
    Q = tf.keras.models.load_model('results/Q_MinMax_10000')
    env = Env(num_columns=5, num_rows=4, num_win=3)
    play(Q, env)