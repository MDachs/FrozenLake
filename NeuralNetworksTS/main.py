import tensorflow as tf
import numpy as np
import random
import gym

eta = 0.1
gamma = 0.9




def getDeltaValue(state, actualQValue, reward, npArray):
   return reward + gamma * npArray[np.argmax(npArray[:, state]), state] - actualQValue


def main():

    # Init Q Values to zero
    env = gym.make('FrozenLake-v0')
    env.__init__(is_slippery=False)
    Q = np.zeros((env.action_space.n, env.observation_space.n))
    print(Q[:, :])



    for _ in range(100000):
        obsersvation = env.reset()
        for x in range(10000):
            #print("Durchgang: ", x)


            if (random.uniform(0, 1) < 0.25):
                next_action = random.randint(0, 3)
                #print("Is Random!")

            else:
                next_action = np.argmax(Q[:, obsersvation])





            print("Observation: ", obsersvation)
            print("Next Action Index: ", next_action)

            #env.render()
            print("-----------------\n")

            obsersvation, reward, done, info = env.step(next_action)
            print("Delta: ", getDeltaValue(obsersvation, Q[next_action,obsersvation],reward, Q))
            Q[next_action, obsersvation] = eta * getDeltaValue(obsersvation, Q[next_action, obsersvation], reward,  Q)

            if done:
                env.reset()
                break





if __name__ == "__main__":
    main()