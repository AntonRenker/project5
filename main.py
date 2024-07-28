import numpy as np
from time import time

from replay import ReplayMemory
from actionValueFunction import ActionValueFunction
from environment import Environment
from environmentMinMax import EnvironmentMinMax

def get_action(columns, epsilon, Q, state):
    rand_val = np.random.rand()
    if rand_val < epsilon:
        return np.random.randint(columns)
    else:
        state = np.array([state])
        return np.argmax(Q.predict(state))
    
def train_min_max(N: int, num_wins: int, rows: int, columns: int, alpha: float, num_episodes: int , epsilon: float, batch_size: int, gamma: float, C: int, epsilon_min: float, epsilon_decay: float)->None:
    env = EnvironmentMinMax(rows, columns, num_wins)
    
    # Initialize replay memory D with capacity N
    D = ReplayMemory(N)

    # Initialize action-value function Q with random weights h
    Q = ActionValueFunction(input_size=rows*columns, output_size=columns, alpha=alpha)

    # Initialize target action-value function Q' with weights h' = h for both agents
    Q_target = ActionValueFunction(input_size=rows*columns, output_size=columns, alpha=alpha)
    Q_target.model.set_weights(Q.model.get_weights())

    t = 0

    for episode in range(num_episodes):
        print(f'Episode: {episode}')
        
        state = env.reset().copy()

        done = False

        while not done:
            # With probability epsilon select a random action a_t
            action = np.random.randint(columns) if np.random.rand() < epsilon else np.argmax(Q.predict(np.array([state])))

            # Execute action a_t and observe reward r_t and next state s_t+1
            next_state, reward, done = env.step(action)

            # Store transition in D
            D.add(state, action, reward, next_state.copy(), done)

            # Sample random minibatch of transitions
            states, _, rewards, next_states, dones = D.sample(batch_size)

            # Compute target for each minibatch transition
            targets = rewards + gamma * np.amax(Q_target.predict([next_states])[0]) * np.array(dones)

            # Fit Q
            Q.model.fit(states, targets, epochs=1, verbose=0)

            # Every C steps update the target network
            if t % C == 0:
                Q_target.model.set_weights(Q.model.get_weights())
            else:
                t += 1

            # Update state to next state
            state = next_state.copy()

        # update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 100 == 0:
            # Store the trained models
            name = 'results/Q_MinMax_' + str(episode)
            Q.model.save(name)

def train(N: int, num_wins: int, rows: int, columns: int, alpha: float, num_episodes: int , epsilon: float, batch_size: int, gamma: float, C: int, epsilon_min: float, epsilon_decay: float)->None:
    env = Environment(rows, columns, num_wins)
    
    # Initialize replay memory D with capacity N
    D = ReplayMemory(N)

    # Initialize action-value function Q with random weights h
    Q = ActionValueFunction(input_size=rows*columns, output_size=columns, alpha=alpha)

    # Initialize target action-value function Q' with weights h' = h for both agents
    Q_target = ActionValueFunction(input_size=rows*columns, output_size=columns, alpha=alpha)
    Q_target.model.set_weights(Q.model.get_weights())

    player = 1
    t = 0

    for episode in range(num_episodes):
        print(f'Episode: {episode}')
        
        state = env.reset(player, Q, epsilon).copy()

        done = False

        while not done:
            # With probability epsilon select a random action a_t
            action = np.random.randint(columns) if np.random.rand() < epsilon else np.argmax(Q.predict(np.array([state])))

            # Execute action a_t and observe reward r_t and next state s_t+1
            next_state, reward, done = env.step(action, player, Q, epsilon)

            # Store transition in D
            D.add(state, action, reward, next_state.copy(), done)

            # Sample random minibatch of transitions
            states, _, rewards, next_states, dones = D.sample(batch_size)

            # Compute target for each minibatch transition
            targets = rewards + gamma * np.amax(Q_target.predict([next_states])[0]) * np.array(dones)

            # Fit Q
            Q.model.fit(states, targets, epochs=1, verbose=0)

            # Every C steps update the target network
            if t % C == 0:
                Q_target.model.set_weights(Q.model.get_weights())
            else:
                t += 1

            # Update state to next state
            state = next_state.copy()

        # update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # update player every 100 episodes
        if episode % 100 == 0:
            player *= -1

        if episode % 100 == 0:
            # Store the trained models
            name = 'results/Q_' + str(episode)
            Q.model.save(name)

if __name__ == '__main__':
    rows = 4
    columns = 5
    num_wins = 3

    N = 1_000_000
    num_episodes = 100_000
    gamma = 1
    alpha = 0.001
    epsilon = 1
    batch_size = 32
    C = 1000

    start_time = time()
    # train(N, num_wins, columns, rows, alpha, num_episodes, epsilon, batch_size, gamma, C, 0.1, 0.999)
    train_min_max(N, num_wins, columns, rows, alpha, num_episodes, epsilon, batch_size, gamma, C, 0.05, 0.999)
    end_time = time.time()
    print(f'Time: {time.strftime("%M:%S", time.gmtime(end_time - start_time))}')