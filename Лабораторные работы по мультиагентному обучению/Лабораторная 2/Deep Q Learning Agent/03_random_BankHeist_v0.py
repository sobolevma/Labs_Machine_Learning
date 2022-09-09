# Подключаем необходимые библиотеки
import gym
import matplotlib.pyplot as plt

# Список cуммарной награды
list_totalreward = []

if __name__ == "__main__":
    # Создаём среду для игры "BankHeist-v0"
    env = gym.make("BankHeist-v0")

    # Инициализируем число шагов, приведшее к ненулевой награде
    count_notnullReward = 0
    # Сбрасываем состояние среды
    obs = env.reset()
    # Общая награда
    total_reward = 0.0
    # Число всех шагов
    total_count = 0

    while True:
        # Совершаем, какое-то случайное действие возможное в среде
        action = env.action_space.sample()
        # Получаем состояние, награду, флаг окончания цикла (done) и информацию о среде
        obs, reward, done, info = env.step(action)
        # Инкрементируем число всех шагов
        total_count += 1
        # Увеличиваем общую награду на величину, полученной награды
        total_reward += reward
        # Добавляем общую награду в список суммарной награды
        list_totalreward.append(total_reward)

        if reward != 0:# Если была получена ненулевая награда
            # Инкрементируем число шагов, приведшее к ненулевой награде
            count_notnullReward += 1
            # Выводим дествие, награду и информацию
            print("\nДействие:", action)
            print("Полученная за действие награда: {}$".format(reward))
            print("Информация (оставшееся число жизней): {}".format(info))

        env.render()  # show env
        if done:
            break

    # Выводим на экран итоговую информацию
    print("\n\nОбщая полученная награда: {}$" .format(total_reward))
    print("Число шагов, приведшее к ненулевой награде: {}".format(count_notnullReward))
    print("Число всех шагов: {}".format(total_count))

    # Строим графики
    fig, ax = plt.subplots()
    # Строим графики общей награды.
    ax.plot(range(0, len(list_totalreward)), list_totalreward)
    #  Добавляем подписи к осям:
    ax.set_xlabel('Шаги')
    ax.set_ylabel('Суммарная награда')

    plt.show()
