import tensorflow as tf
import numpy as np
import random
import gym


def main():

    env = gym.make('FrozenLake-v0')
    env.__init__(is_slippery=False)

    obsersvation = env.reset()

    # Init Q Values to zero
    Q = np.zeros((env.action_space.n, env.observation_space.n))

    Q[3, 1] = 2
    Q[3, 15] = 9
    print(Q[:,:])



    for _ in range(1):
        for x in range(10):
            print("Durchgang: ", x)

            if (random.uniform(0, 1) < 0.25):
                next_action = random.randint(0, 3)
                print("Is Random!")

            else:
                next_action = np.argmax(Q[:, obsersvation])

            print("Observation: ", obsersvation)
            print("Next Action Index: ", next_action)

            env.render()
            print("-----------------\n")

            obsersvation, reward, done, info = env.step(next_action)


            if done:
                env.reset()
                #update Q Values
                break





if __name__ == "__main__":
    main()