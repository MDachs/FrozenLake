import tensorflow as tf
import numpy as np
import gym


def main():

    stepmap = (1, 1, 2, 1, 1, 2, 2, 2, 0, 2 ,2 )
    env = gym.make('FrozenLake-v0')
    env.__init__(is_slippery=False)

    obsersvation = env.reset()

    #returns possible actions and obersvations
    print(env.action_space)
    print(env.observation_space)

    for x in range(len(stepmap)):
        print("Durchgang: " , x)
        env.render()

        print(obsersvation)

        obsersvation = env.step(stepmap[x])  # take a random action
        print("-------------\n")







if __name__ == "__main__":
    main()