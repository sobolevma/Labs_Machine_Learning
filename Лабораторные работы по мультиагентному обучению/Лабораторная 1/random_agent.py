#################
# RANDOM AGENT #
#################
import gym
import numpy as np
import pickle
import os
from tqdm import tqdm
from time import sleep, time

import matplotlib.pyplot as plt
import gym_toytext

# Загружаем среду рулетки gym
env = gym.make('Roulette-v0')

print(f"\nProcessing in {str(env.spec)[8:-1]}...\n")

MAX_STEPS = 100


# Число выходов из игры
go_out = 0
# Число выигрышных ставок нулей
zeros = 0
# Число побед
win = 0
# Число поражений
defeat = 0

# Для визуализации на win cmd
show_play = True

# Список частот побед
list_winRate = []
# Максимальная частота побед
max_winRate = -1
# Сумма частот побед
sum_winRate = 0
# Сумма всех шагов во всех эпизодах.
count = 0


# Start
for episode in range(10):
    # В программе присутствует единственное состояние: 0.
    state = env.reset()
    step = 0

    while step < MAX_STEPS:
        # Действие, выполненное агентом
        action = env.action_space.sample()
        # Получаем новое состояние, награду, выпонение условия задачи и информацию (пустое поле).
        # В программе присутствует единственное состояние: 0.
        new_state, reward, done, info = env.step(action)

        # Визуализация Windows CLI
        if show_play:
            os.system('cls')


            try:
                # Вычисляем частоту побед
                win_rate = win / (win + defeat + go_out) * 100
                # Вычисляем сумму частот побед
                sum_winRate += win_rate
                # Вычисляем максимальную частоту побед
                if(max_winRate < win_rate):
                    max_winRate = win_rate
                # Добавляем частоту побед в список частот побед.
                list_winRate.append(win_rate)

                print("Win rate: {}%".format(win_rate))
                print(f"Wins: {win}\n"
                      f"Go out:{go_out}\n"
                      f"Defeats: {defeat}\n"
                      f"Steps: {step}\n"
                      f"Episode: {episode}\n"
                      )
            except ZeroDivisionError:   # Если (win + defeat + go_out) знаменатель равен нулю
                print("Win rate: 0.0%")
                print(f"Wins: {win}\n"     
                      f"Go out:{go_out}\n"
                      f"Defeats: {defeat}\n"
                      f"Steps: {step}\n"
                      f"Episode: {episode}\n"
                      )
            
            #Определяем к чему приводит агента выполнение определённого действия: Выиграл (WIN), Проиграл (DEFEAT) или Вышел из игры (GO_OUT)
            if reward > 0:
                strr = "\n\tWIN"
            elif reward < 0:
                strr = "\n\tDEFEAT"
            else:
                strr = "\n\tGO_OUT"
            print(strr)
            sleep(0.6)

        # В игре присутствет единственное состояние: 0.
        state = new_state

        # По полученной награде определяем, что необходимо сделать:
        if reward == 1:
            # Инкрементировать число побед.
            win += 1

        elif reward == 36:
            # Инкрементировать число побед и кол-во выигрышной ставки ноль.
            win += 1
            zeros += 1

        elif reward == 0:
            # Инкрементировать число выходов из игры.
            go_out += 1
            break
        else:
            # Инкрементировать число поражений
            defeat += 1


        # Инкрементируем число шагов.
        step += 1

        # Кол-во шагов во всех эпизодая шагов.
        count += 1


        # Если было превышено максимальное число шагов
        if show_play and (step >= MAX_STEPS):
            print("\nTIME IS OVER (steps > 100)")
            # sleep(1)
            break


# Вычисляем итоговою частоту побед
win_rate = win / (win + defeat + go_out) * 100
# Вычисляем сумму частот побед
sum_winRate += win_rate
# Вычисляем максимальную частоту побед
if (max_winRate < win_rate):
    max_winRate = win_rate
# Добавляем частоту побед в список частот побед.
list_winRate.append(win_rate)
# Вычисляем итоговою частоту выходов из игры
go_out_rate = go_out / (win + defeat + go_out) * 100


print("Win rate: {}%".format(win_rate))
print("Max win_rate: {}%".format(max_winRate))
print("Average winrate: {}%".format(sum_winRate/count))
print("Go out rate: {}%".format(go_out_rate))
print(f"Wins: {win}\n"
      f"Zeros: {zeros}\n"
      f"Go out:{go_out}\n"
      f"Defeats: {defeat}\n")

# Строим графики
fig, ax = plt.subplots()
# Строим графики частот побед.
ax.plot(range(0, len(list_winRate)), list_winRate)
#  Добавляем подписи к осям:
ax.set_xlabel('Шаги')
ax.set_ylabel('Частота побед')

plt.show()