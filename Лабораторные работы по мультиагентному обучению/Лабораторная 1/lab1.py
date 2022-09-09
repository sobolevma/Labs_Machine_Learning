##################
# LEARNING STAGE #
##################

import gym

import numpy as np
import pickle
import os
from tqdm import tqdm
from time import sleep, time

import matplotlib.pyplot as plt

# Загружаем среду рулетки gym
env = gym.make('Roulette-v0')

# Очищаем экран
os.system('cls')
print(f"\nLearning in {str(env.spec)[8:-1]}...\n")

# Константные параметры
TOTAL_EPISODES = 14_000
MAX_STEPS = 100
EPSILON_DECAY_RATE = 1 / 20000
MIN_EPSILON = 0.001

# RL параметры
gamma = 0.3
lr_rate = 0.005 # alpha
epsilon = 1

# Инициализация Q-таблицы
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Жадный эпсилон
def epsilon_greedy(state):
    # Вычисление действия агента: число (0-36) или выход из игры.
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

# Q-обучение
def learn(state, state2, reward, action):
    # same as formula for Q-learning
    # Q_new[s, a] <- Q_old[s, a] + alpha * (r + gamma * max_a(Q_old[s_new, a]) - Q_old[s, a])

    Q[state, action] = Q[state, action] + lr_rate * \
                       (reward + gamma * np.max(Q[state2, :]) - Q[state, action])

# Вычисление дисперсии дискретного значения
def calc_dispersion(rew_mas, rew_sum, num_elems):
    dispersion = 0
    summ = 0
    rew_average = rew_sum / num_elems
    for rev_elem in rew_mas:
        summ += (rev_elem - rew_average) ** 2
        dispersion = summ / (num_elems - 1)
    return dispersion

# Максимальная скорость
max_speed = -1
# Маcсив скоростей обучения
learning_speed=[]
# Сумма всех скоростей обучения.
sp_sum = 0
# Массив сумм наград по эпизодам
rew_mas = []

# Обучение агента
print(gym.spec('Roulette-v0'))
start = time()
print(start)
# Start
for episode in tqdm(range(TOTAL_EPISODES), ascii=True, unit="episode"):
    # В игре присутствет единственное состояние: 0.
    state = env.reset()

    # Сумма наград по эпизодам
    rew_sum = 0
    # Кол-во элементов в эпизоде
    count_elems = 0
    # Число шагов
    step = 0
    # Новая награда при выходе из игры, если награды, получаемые из среды равномерно распределены.
    reward_val = 4


    # Уменьшаем эпсилон
    if epsilon > MIN_EPSILON:
        epsilon -= EPSILON_DECAY_RATE
    else:
        epsilon = MIN_EPSILON

    # Цикл в эпизоде
    while step < MAX_STEPS:
        # Действие, выполненное агентом
        action = epsilon_greedy(state)
        # Получаем новое состояние, награду, выпонение условия задачи и информацию (пустое поле).
        # В программе присутствует единственное состояние: 0.
        new_state, reward, done, info = env.step(action)
        '''
        if(reward == 0):
            print(action)
        '''

        if reward != 0:
            rew_sum += reward
            count_elems += 1
        else:
            if (count_elems >= 3) and abs(calc_dispersion(rew_mas, rew_sum, count_elems) - (count_elems**2 - 1)/12) <= 8:
                reward = reward_val

        # Обучаем агента
        learn(state, new_state, reward, action)

        # В игре присутствет единственное состояние: 0.
        state = new_state
        # Инкрементируем число шагов
        step += 1

        # Если агент решает выйти из игры.
        if reward == 0 or reward == reward_val:
            break
    # Текущее время с момента начала отсчёта времени
    l_time = time()
    end = l_time - start

    # Новое начальное время
    start = l_time
    # Вычсляем скорость,
    if (end > 0): # когда текущее время больше нуля,
        speed = 1 / end
    else:# иначе
        if (max_speed > 0): # когда макс. скорость больше нуля,
          speed = max_speed
        else: # иначе
          speed = 0

    # Считаем сумму скоростей обучения.
    sp_sum += speed

    # Вычисляем максимальную скорость обучения.
    if(speed > max_speed):
        max_speed = speed

    # Добавляем скорость в список скоростей.
    learning_speed.append(speed)

    # Добавляем сумму наград в эпизоде в массив сумм наград по эпизодам.
    rew_mas.append(rew_sum)

# Выводим Q-таблицу на экран
print("\nQ-table:\n", Q)
sleep(3)

# Сохранить Q-таблицу в файл на диске (в том же каталоге)
with open("frozenLake_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)


#################
# PLAYING STAGE #
#################

# comment all below, if you only need to train agent

print(f"\nPlaying {str(env.spec)[8:-1]}...\n")

# Загрузить Q-таблицу из файла
with open("frozenLake_qTable.pkl", 'rb') as f:
    Q = pickle.load(f)

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
        action = np.argmax(Q[state, :])
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

            #env.render()  # show env


            #print("\n\tWIN" if reward > 0 else "\n\tDEFEAT")
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

if go_out != 0:
    print("Max learning speed: {}%".format(max_speed))
    print("Average learning speed: {}%".format(sp_sum / TOTAL_EPISODES))
else:
    print("Max win_rate: {}%".format(max_winRate))
    print("Average learning speed: {}%".format(sum_winRate/count))

print("Win rate: {}%".format(win_rate))
print("Go out rate: {}%".format(go_out_rate))
print(f"Wins: {win}\n"
      f"Zeros: {zeros}\n"
      f"Go out:{go_out}\n"
      f"Defeats: {defeat}\n")

# Строим графики
fig, ax = plt.subplots()

# Строим график с зависимостью наград от эпизода обучения.
fig_rew, ax_rew = plt.subplots()
ax_rew.plot(range(0, len(rew_mas)), rew_mas)
#  Добавляем подписи к осям:
ax_rew.set_xlabel('Эпизоды')
ax_rew.set_ylabel('Суммарная награда')

if go_out == 0: # Строим график частоты побед.
    ax.plot(range(0, len(list_winRate)), list_winRate)
    #  Добавляем подписи к осям:
    ax.set_xlabel('Шаги')
    ax.set_ylabel('Частота побед')
elif win == 0 and defeat == 0: # Строим график скоростей обучения
    ax.plot(range(0, len(learning_speed)), learning_speed)
    #  Добавляем подписи к осям:
    ax.set_xlabel('Эпизоды')
    ax.set_ylabel('Скорость обучения (эп/с)')
plt.show()
