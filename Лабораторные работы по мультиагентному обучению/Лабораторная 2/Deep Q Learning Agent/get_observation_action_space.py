import gym

from gym.spaces import *
import sys
import collections

def print_spaces(space):
    print(space)
    if(isinstance(space, Box)):
        count_low = 0
        count_high = 0
        for i in range(1, space.shape[0]):
            count_av = 0
            for j in range(0, space.shape[1]):
                countl_cycle = 0
                for k in range(0, space.shape[2]):
                    if(space.low[i-1][j][k] == space.low[i][j][k]):
                        countl_cycle += 1
                if(countl_cycle == space.shape[2]):
                    count_av += 1
            if(count_av == space.shape[1]):
                count_low += 1

        for i in range(1, space.shape[0]):
            count_v = 0
            for j in range(0, space.shape[1]):
                counl_cycle = 0
                for k in range(0, space.shape[2]):
                    if (space.high[i-1][j][k] == space.high[i][j][k]):
                        counl_cycle += 1
                if (countl_cycle == space.shape[2]):
                    count_v += 1
            if (count_v == space.shape[1]):
                count_high += 1
        if (count_low == space.shape[0]-1):
            print(space.low[0][0])

        if (count_high == space.shape[0]-1):
            print(space.high[0][0])
        print(f"\nMatrix parametres {space.shape[0]}x{space.shape[1]}x{space.shape[2]}")




if __name__ == "__main__":
    env = gym.make("BankHeist-v0")
    print("Observation space:")
    print_spaces(env.observation_space)
    print("Action space:")
    print_spaces(env.action_space)

    try:
        print("Action description/meaning", env.unwrapped.get_action_meanings())
    except AttributeError:
        pass

    #try:
    #    print("Action description/meaning", env.unwrapped.get_keys_to_action())
    #except AttributeError:
    #    pass