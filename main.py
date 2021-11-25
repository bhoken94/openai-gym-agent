import gym
import numpy as np
from gym import wrappers
from egreedy.agent import Agent
from softmax.agent import Agent as AgentSoftmax
from utils import plot_learning_epsilon


def start():
    env = gym.make("LunarLander-v2")
    policy = "softmax"  # or greedy
    num_games = 500
    scores, eps = [], []
    if policy == "greedy":
        agent = Agent(num_input=env.observation_space.shape[0],
                      num_action=env.action_space.n,
                      eps_min=0.05,
                      gamma=0.99,
                      lr=0.001,
                      batch_size=64,
                      epsilon=1.0)
    else:
        agent = AgentSoftmax(num_input=env.observation_space.shape[0],
                             num_action=env.action_space.n,
                             gamma=0.99,
                             lr=0.001,
                             batch_size=64,
                             t=100)
    for episode in range(num_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            # env.render()
            action = agent.select_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(new_state=new_observation, last_state=observation, action=action, reward=reward,
                                   done=done)
            agent.learn()
            observation = new_observation
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        if policy == "greedy":
            eps.append(agent.epsilon)
            print("=> Episode", episode, "score %.2f" % score, "media punteggio %.2f" % avg_score,
                "epsilon %.2f" % agent.epsilon)
        else:
            print("=> Episode", episode, "score %.2f" % score, "media punteggio %.2f" % avg_score)
    env.close()
    x = [i + 1 for i in range(num_games)]
    plot_learning_epsilon(x, scores, eps, f"charts/lunar_lavander_{policy}.png")


if __name__ == '__main__':
    start()
