# Добавление необходимых внешних библиотек
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage.color as color
from collections import Counter

"""Набор параметров для прорисовки эллипсов, которые необходимы для выделения границ блюд. """
"""Где каждая четвёрка: [Y0, X0, Axis_B, Axis_A],
   где X0 - координата по оси Х, проходящей через левый верхний угол изображения и правый верхний угол изображения,
   где Y0 - координата по оси Y, проходящей через левый верхний угол изображения и правый нижний угол изображения,
   где Axis_B - ось эллипса, параллелная оси У,
   где Axis_А - ось эллипса, параллелная оси Х.
"""
params = [[[546, 920, 380, 620], [680, 2070, 240, 315], [1450, 880, 430, 510], [1500, 2036, 300, 400]],
          [[570, 850, 350, 350], [600, 1930, 430, 510], [1500, 850, 240, 315], [1620, 2070, 300, 400]],
          [[600, 860, 430, 510], [780, 2000, 410, 600], [1510, 890, 408, 595], [1580, 2030, 300, 400]],
          [[550, 860, 300, 400], [655, 1950, 415, 580], [1310, 874, 290, 467], [1540, 2065, 300, 300]],
          [[550, 925, 408, 595], [660, 2045, 300, 400], [1422, 860, 240, 315], [1423, 1995, 408, 595]],
          [[590, 890, 408, 595], [600, 2036, 300, 400], [1528, 914, 380, 620], [1400, 2060, 350, 350]],
          [[502, 730, 300, 300], [630, 1690, 408, 595], [1200, 884, 300, 400], [1420, 1880, 350, 350]],
          [[575, 930, 415, 580], [475, 2040, 300, 400], [1540, 934, 380, 620], [1195, 2005, 290, 467]],
          [[662, 930, 290, 467], [500, 1938, 300, 300], [1510, 865, 300, 400], [1340, 1950, 415, 580]],
          [[610, 1425, 430, 510], [1420, 887, 300, 400], [1430, 1993, 350, 350]]]


# Список наименований уникальных блюд, которые могут встретиться на изображениях блюд.
different_typesOfFood = ["Сыр", "Огурцы", "Помидоры", "Котлеты", "Чёрная икра", "Свекла", "Малина", "Колбаса Докт.", "Апельсин", "Белый хлеб"]

""" Набор блюд, которые на самом деле присутствуют на каждом из изображений, содержащих различные блюда."""
""" Где [Блюдо1, ...] - список блюд, которые на самом деле присутствуют на изображении."""
idealFoodSet = [["Сыр", "Огурцы", "Помидоры", "Белый хлеб"],
                ["Котлеты", "Помидоры", "Огурцы", "Белый хлеб"],
                ["Помидоры", "Колбаса Докт.", "Свекла", "Белый хлеб"],
                ["Белый хлеб", "Апельсин", "Малина", "Чёрная икра"],
                ["Свекла", "Белый хлеб", "Огурцы", "Колбаса Докт."],
                ["Колбаса Докт.", "Белый хлеб", "Сыр", "Котлеты"],
                ["Чёрная икра", "Свекла", "Белый хлеб", "Котлеты"],
                ["Апельсин", "Белый хлеб", "Сыр", "Малина"],
                ["Малина", "Чёрная икра", "Белый хлеб", "Апельсин"],
                ["Помидоры", "Белый хлеб", "Котлеты"]]



# Собственная функция сортировки на основе длины каждого элемента
def keyFunc_Item_Length(item):
   return len(item)


# Функция получения списка изображений из директории.
def get_List_of_Images(directory):
    # Получаем список файлов-изображений в переменную images_files
    images_files = [file for file in os.listdir(directory) if file.endswith('.jpg')]

    # Сортировка списка файлов-изображений
    images_files.sort(key=keyFunc_Item_Length)

    # Возвращаем полученный список
    return images_files

# Функция вывода изображения на экрана
def image_show(image, nrows=1, ncols=1):
    # Функция вывода изображения на экран:
    # fig - контейнер изображения
    # ax - область на которой отражаются графики
    # nrows/ncols - количество строк / столбцов сетки подграфика.
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))

    # Прорисовка изображения в области ax
    ax.imshow(image)

    # Возвращаем контейнер изображения и область с графиком
    return fig, ax


# Функция возвращающая координаты эллипса
# Входные параметры:
# resolution - кол-во точек эллипса,
# center - центр эллипса,
# param_A - диагональ эллипса, параллельная оси Х,
# param_B - диагональ эллипса, параллельная оси У.
def ellipse_points(resolution, center, param_A, param_B):

    """
        Создание точек, которые определяют эллипс на изображении. Центр относится к центру эллипса.
    """
    # Получаем одномерный массив радиан в диапазоне [0, 2pi], количество которых равно resolution
    radians = np.linspace(0, 2 * np.pi, resolution)

    # Полярные координаты
    # x[i] = x0 + b * cos(t) - координата точек эллипса по оси х.
    x = center[1] + param_B * np.cos(radians)

    # y[i] = x0 + b * sin(t) - координата точек эллипса по оси у.
    y = center[0] + param_A * np.sin(radians)


    # Получаем массив координат эллипса в траспонированном виде
    return np.array([x, y]).T


# Функция получения 4-ых крайних координат точек эллипса, находящихся на осях эллипса.
# Входные параметры:
# centerX - координата центра эллипса по оси Х,
# center - координата центра эллипса по оси У,
# param_A - диагональ эллипса, параллельная оси Х,
# param_B - диагональ эллипса, параллельная оси У.
def get_4Extreme_Coords_Of_Ellipse(centerX, centerY, ellipse_axisA, ellipse_axisB):
    coords = []

    # Проверка того, что все входные параметры являются целыми числами
    if (isinstance(centerX, int) and isinstance(centerY, int) and isinstance(ellipse_axisA, int) and isinstance(ellipse_axisB, int)):
        # minX = centerX - param_A/2
        coord_centerX_Minus_axisA_Div2 = centerX - (ellipse_axisA >> 1)

        # maxX = centerX + param_A/2
        coord_centerX_Plus_axisA_Div2 = centerX + (ellipse_axisA >> 1)

        # minY = centerY - param_B/2
        coord_centerY_Minus_axisB_Div2 = centerY - (ellipse_axisB >> 1)

        # maxY = centerY + param_B/2
        coord_centerY_Plus_axisB_Div2 = centerY + (ellipse_axisB >> 1)

        # Возвращаем массив из 4-ых крайних точек эллипса
        coords = [coord_centerX_Minus_axisA_Div2, coord_centerX_Plus_axisA_Div2, coord_centerY_Minus_axisB_Div2, coord_centerY_Plus_axisB_Div2]
    return coords


# Функция возвращающая расстояние до оси эллипса
# Входные параметры:
# x0 - координата точки по оси Х внутри эллипса или на самом эллипсе,
# y0 - координата точки по оси У внутри эллипса или на самом эллипсе,
# x1 - координата 1-ой точки кординаты оси эллипса по оси Х,
# y1 - координата 1-ой точки кординаты оси эллипса по оси У,
# x2 - координата 2-ой точки кординаты оси эллипса по оси Х,
# y2 - координата 2-ой точки кординаты оси эллипса по оси У.
def get_distance_to_EllipseAxis(x0, y0, x1, y1, x2, y2):
    # Начальное значение расстояния до оси эллипса
    distance = -2

    # Проверка того, что на вход были поданы целые числа
    check_Pars = isinstance(x0, int) and isinstance(y0, int) and isinstance(x1, int) and isinstance(y1, int) and isinstance(x2, int) and isinstance(y2, int)
    if(check_Pars):
        # Общая формула вычисления расстояния от точки до прямой:
        # distance = abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)) / sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

        """ Вид данной формулы для оси b эллипса, которая параллельна оси У.
            Т.е. проверка координат x1, x2, y1 и y2. """
        if (x1 == x2 and y1 != y2):
            distance = abs(x1 - x0)
        # Вид данной формулы для оси b эллипса, которая параллельна оси У.
        # Т.е. проверка координат x1, x2, y1 и y2.
        elif (y1 == y2 and x1 != x2):
            distance = abs(y1 - y0)

        # Возвращаем расстояние от точки до оси эллипса
    return distance


# Функция проверяющая, что точка находится внутри эллипса.
# Возвращаемое значение: True - внутри эллипса или на границе эллипса и False - вне эллипса.
# Входные параметры:
# pointX - координата проверяемой точки по оси Х,
# pointY - координата проверяемой точки по оси У,
# center_X0 - координата центра эллипса по оси Х,
# center_Y0 - координата центра по оси У,
# ellipse_axisA - ось эллипса А, параллельная оси Х,
# ellipse_axisB - ось эллипса В, параллельная оси У.
def check_If_Point_Inside_Ellipse(pointX, pointY, center_X0, center_Y0, ellipse_axisA, ellipse_axisB):
    # Начальное значение параметра, характеризующего положение точки
    inside = False

    # Проверка параметров, переданных функции
    # координат точки
    check_Point = isinstance(pointX, int) and isinstance(pointY, int)
    # координат центра эллипса
    check_Center = isinstance(center_X0, int) and isinstance(center_Y0, int)
    # координат осей эллипса
    check_Axises = isinstance(ellipse_axisA, int) and isinstance(ellipse_axisB, int)
    # Проверка параметров
    if check_Point and check_Center and check_Axises:
        # Выражение для проверки положения точки expression = (X - center_X0)^2 / a^2 + (Y - center_Y0)^2 / b^2.
        expression = ((pointX - center_X0) ** 2) / (ellipse_axisA ** 2) + ((pointY - center_Y0) ** 2) / (ellipse_axisB ** 2)

        # Если значение данного выражения меньше или равно единице, значит точка лежит внутри эллипса.
        if expression <= 1:
            # Выставляем True в параметр положения точки
            inside = True

    # Возвращаем значения параметра положения точки
    return inside


# Функция нахождения индексов 4-ых наибольщих элементов в массиве
def findIndexes_Of4MaxPersents(persents):
    # Список для индексов 4-ох наибольших элементов
    indexes = []

    # Начальное значение максимального индекса элемента из массива
    maxIndex = -1
    # Начальное значение максимального элемента из массива
    maxVal = -9

    # Проверка длины массива
    if(len(persents) > 4):
        # Выполняем поиск максимального 4 раза
        for iteration in range(4):
            # Перебираем индексы всех элементов массива
            for indx in range(len(persents)):
                """ Проверка того что элемент с данным индексом является число типа float и того, 
                что данныq индекс не содержится в массиве индексов. """
                if(isinstance(persents[indx], float) and not(indx in indexes)):
                    """ Получаем значение наибольшего из 2-ух элементовпри помощи сравнения максимального значения 
                    с текущим элементом массива"""
                    element = max(maxVal, persents[indx])

                    """ Проверка того, что значение полученного элемента не рано текущему максимальному значению,
                     т.е. это означает, что текущий элемент массива больше максимального значения."""
                    if(element != maxVal):
                        # Максимальный индекса становится равным индексу текущего элемента массива.
                        maxIndex = indx

                        # Максимальный элемент из массива становится равным текущему элементу массива.
                        maxVal = element

            # Добавляем макисмальный индекс в массив максимальных индексов
            indexes.append(maxIndex)

            # Присваиваем начальное значение максимальному индексу элемента из массива.
            maxIndex = -1
            # Присваиваем начальное значение максимальному элементу из массива.
            maxVal = -9

    # Возвращаем список индексов 4-ох наибольших элементов
    return indexes


# Функция получаюшая набор точек, неаходящихся внутри эллипса
# На вход подаётся изображение в цветовом пространстве YUV и параметры эллипса
def get_imagePixels_Inside_Ellipse(imageYUV, ellipse_Params):
    # Набор пикселей, находящихся внутри эллипса
    pixels_Set = []

    # Проверка того, что на вход были переданы 4 параметра эллипса.
    if len(ellipse_Params) == 4:
        # Х-овая координата центра эллипса.
        centerX = ellipse_Params[0]
        # У-овая координата центра эллипса.
        centerY = ellipse_Params[1]
        # Ось А эллипса, параллельна оси Х.
        ellipse_axisA = ellipse_Params[2]
        # Ось В эллипса, параллельна оси У.
        ellipse_axisB = ellipse_Params[3]

        # Проверка, того что координаты центра эллипса являются целыми числами
        checkCenter = isinstance(centerX, int) and isinstance(centerY, int)
        # Проверка, того что оси центра эллипса являются целыми числами
        checkAxises = isinstance(ellipse_axisA, int) and isinstance(ellipse_axisB, int)
        # Проверка параметров
        if checkCenter and checkAxises:
            # Получаем 4 крайних координаты точек эллипса, находящихся на осях эллипса.
            extreme_Coords = get_4Extreme_Coords_Of_Ellipse(centerX, centerY, ellipse_axisA, ellipse_axisB)

            # Получаем минимальную крайнюю координату эллипса по оси Х.
            minX = extreme_Coords[0]
            # Получаем макисмальную крайнюю координату эллипса по оси Х.
            maxX = extreme_Coords[1]
            # Получаем минимальную крайнюю координату эллипса по оси У.
            minY = extreme_Coords[2]
            # Получаем макисмальную крайнюю координату эллипса по оси У.
            maxY = extreme_Coords[3]

            # Обходим правую нижнюю часть эллипса, начиная из центра эллипса.
            for coordY in range(centerY, maxY):
                for coordX in range(centerX, maxX):
                    """ Проверяем, что точка лежит внутри эллипса, передавая на вход координаты центра эллипса 
                    и оси эллипса"""
                    # И если точка находится внутри эллипса:
                    if (check_If_Point_Inside_Ellipse(coordX, coordY, centerX, centerY, ellipse_axisA, ellipse_axisB)):
                        # Добавляем текущий пиксель в набор пикселей, находящихся внутри эллипса
                        pixels_Set.append(imageYUV[coordY][coordX])

                        """ Получаем расстояние от точки до оси А эллипса, передавая на вход координаты точки (coordX, coordY)
                        и координты оси А (minX, centerY, maxX, centerY)."""
                        distToAxisA = get_distance_to_EllipseAxis(coordX, coordY, minX, centerY, maxX, centerY)

                        """ Получаем расстояние от точки до оси В эллипса, передавая на вход координаты точки (coordX, coordY)
                        и координты оси В (centerX, minY, centerX, maxY)."""
                        distToAxisB = get_distance_to_EllipseAxis(coordX, coordY, centerX, minY, centerX, maxY)

                        # Если точка находится не на осях эллипса
                        if(distToAxisA > 0 and distToAxisB > 0):
                            # то добавляем следующие пиксели в набор пикселей, находящихся внутри эллипса:
                            #с координатой [centerY - distToAxisA][coordX],
                            pixels_Set.append(imageYUV[centerY - distToAxisA][coordX])

                            # с координатой [coordY][centerX - distToAxisB],
                            pixels_Set.append(imageYUV[coordY][centerX - distToAxisB])

                            # с координатой [centerY - distToAxisA][centerX - distToAxisB]
                            pixels_Set.append(imageYUV[centerY - distToAxisA][centerX - distToAxisB])
                        # Если точка находится не на оси А эллипса
                        elif (distToAxisA == 0 and distToAxisB > 0):
                            # то добавляем пиксель с координатой [centerY][centerX - distToAxisB] в набор пикселей.
                            pixels_Set.append(imageYUV[centerY][centerX - distToAxisB])

                        # Если точка находится не на оси В эллипса
                        elif (distToAxisA > 0 and distToAxisB == 0):
                            # то добавляем пиксель с координатой [centerY - distToAxisA][coordX] в набор пикселей.
                            pixels_Set.append(imageYUV[centerY - distToAxisA][coordX])
                    else: # Иначе, прерываем проход по координате Х
                        break

    # Возвращаем набор пикселей, находящихся внутри эллипса.
    return pixels_Set


# Функция получающающая на вход 3 координаты пикселя в модели YUV и возвращающая название найденного цвета в данной модели.
def getPixelColor(paramY, paramU, paramV):
    # Название цвета
    colorName = []

    # Если обнаружен зелёный цвет,
    if(paramU <= -0.04 and paramV <= -0.001):
        colorName = "green"
    # если обнаружен синий цвет,
    elif(paramU >= 0.04 and paramV <= 0.02):
        colorName = "blue"
    # если обнаружен бирюзовый цвет,
    elif (paramU >= -0.1 and paramU <= 0.2 and paramV <= -0.18 and paramV >= -0.34):
        colorName = "turquoise"
    # если обнаружен сине-зелёный цвет,
    elif (paramU >= -0.1 and paramU <= 0.2 and paramV <= -0.34):
        colorName = "blue-green"
    # если обнаружен красный цвет,
    elif (paramU <= -0.085 and paramV >= 0.3 and paramV <= 0.6):
        colorName = "red"
    # если обнаружен оранжевый цвет,
    elif(paramU <= -0.08 and paramV >= 0.2 and paramV <= 0.3):
        colorName = "orange"
    # если обнаружен жёлтый цвет,
    elif(paramU <= -0.08 and paramV >= -0.001 and paramV <= 0.2) :
        colorName = "yellow"

        # Получение грязно-жёлтого цвета из жёлтого цвета путём проверки координаты У пикселя.
        if (paramY <= 0.42):
            colorName = "dirtyYellow"
    # если обнаружен розовый цвет,
    elif (paramU > -0.1 and paramU < -0.0 and paramV >= 0.1):
        colorName = "pink"

        # Получение грязно-розового цвета из розового цвета путём проверки координаты У пикселя.
        if (paramY <= 0.25):
            colorName = "dirtyPink"
    # если обнаружен малиновый цвет,
    elif (paramU >= 0.1 and paramV >= 0.1):
        colorName = "crimson"
    # если обнаружен фиолетовый цвет,
    elif(paramU >= 0.0 and paramU <= 0.1 and paramV >= 0.1):
        colorName = "violet"
    # если обнаружен грязно-фиолетовый цвет,
    elif(paramU >= 0.04 and paramU <= 0.13 and paramV>= 0 and paramV <= 0.1):
        colorName = "dirtyViolet"
    # если обнаружен серый цвет,
    elif (paramU >= -0.08 and paramU <= 0.04 and paramV >= -0.18 and paramV <= 0.12):
        colorName = "gray"

        # Получение чёрного цвета из серого цвета путём проверки координаты У пикселя.
        if (paramY <= 0.34):
            colorName = "black"
    else: # если цвет не был распознан
        colorName = "undefined"

    # Возвращаем название цвета
    return colorName


# Функция принимающая на вход набор пикселей, относящихся к конкретному блюду и возвращающая название блюда
def findFood_andShowAll_FoodColors(pixels_Set):
    # Название найденного блюда
    foodName = []


    # Список процентов найденных цветов блюда
    colors_Persents = []

    # Кол-во пикселей зелёного цвета
    countGreen = 0
    # Кол-во пикселей синего цвета
    countBlue = 0
    # Кол-во пикселей бирюзового цвета
    countTurquoise = 0
    # Кол-во пикселей сине-зелёного цвета
    countBlue_Green = 0
    # Кол-во пикселей фиолетового цвета
    countViolet = 0
    # Кол-во пикселей грязно-фиолетового цвета
    countDirtyViolet = 0
    # Кол-во пикселей красного цвета
    countRed = 0
    # Кол-во пикселей оранжевого цвета
    countOrange = 0
    # Кол-во пикселей жёлтого цвета
    countYellow = 0
    # Кол-во пикселей грязно-жёлтого цвета
    countDirtyYellow = 0
    # Кол-во пикселей малинового цвета
    countCrimson = 0
    # Кол-во пикселей серого цвета
    countGray = 0
    # Кол-во пикселей розового цвета
    countPink = 0
    # Кол-во пикселей грязно-розового цвета
    countDirtyPink = 0
    # Кол-во пикселей чёрно-розового цвета
    countBlack = 0
    # Кол-во пикселей неопределённого цвета
    undefColor = 0


    # Проверка длины набора пикселей блюда и то, что первый элемент данного набоа является пикселем.
    if(len(pixels_Set) > 0) and len(pixels_Set[0] == 3):
        # Размер набора пикселей
        setLength = len(pixels_Set)

        # Проверяем все пиксели из набора пикселей блюда
        for pixel in pixels_Set:
            # Координата У пикселя
            coordY = pixel[0]
            # Координата U пикселя
            coordU = pixel[1]
            # Координата V пикселя
            coordV = pixel[2]

            # Получаем название цвета по 3-м координатам пикселя в модели YUV.
            colorName = getPixelColor(coordY, coordU, coordV)


            # Если был найден зелёный цвет
            if colorName == "green":
                # Инкремент кол-ва пикселей зелёного цвета
                countGreen = countGreen + 1
            # Если был найден синий цвет
            elif colorName == "blue":
                # Инкремент кол-ва пикселей синего цвета
                countBlue = countBlue + 1
            # Если был найден бирюзовый цвет
            elif colorName == "turquoise":
                # Инкремент кол-ва пикселей бирюзового цвета
                countTurquoise = countTurquoise + 1
            # Если был найден сине-зелёный цвет
            elif colorName == "blue-green":
                # Инкремент кол-ва пикселей сине-зелёного цвета
                countBlue_Green = countBlue_Green + 1
            # Если был найден фиолетовый цвет
            elif colorName == "violet":
                # Инкремент кол-ва пикселей фиолетового цвета
                countViolet = countViolet + 1
            # Если был найден грязно-фиолетовый цвет
            elif colorName == "dirtyViolet":
                # Инкремент кол-ва пикселей грязно-фиолетового цвета
                countDirtyViolet = countDirtyViolet + 1
            # Если был найден красный цвет
            elif colorName == "red":
                # Инкремент кол-ва пикселей красного цвета
                countRed = countRed + 1
            # Если был найден оранжевый цвет
            elif colorName == "orange":
                # Инкремент кол-ва пикселей оранжевого цвета
                countOrange = countOrange + 1
            # Если был найден жёлтый цвет
            elif colorName == "yellow":
                # Инкремент кол-ва пикселей жёлтого цвета
                countYellow = countYellow + 1
            # Если был найден грязно-жёлтый цвет
            elif colorName == "dirtyYellow":
                # Инкремент кол-ва пикселей грязно-жёлтого цвета
                countDirtyYellow = countDirtyYellow + 1
            # Если был найден малиновый цвет
            elif colorName == "crimson":
                # Инкремент кол-ва пикселей малинового цвета
                countCrimson = countCrimson + 1
            # Если был найден розовый цвет
            elif colorName == "pink":
                # Инкремент кол-ва пикселей розового цвета
                countPink = countPink + 1
            # Если был найден грязно-розовый цвет
            elif colorName == "dirtyPink":
                # Инкремент кол-ва пикселей грязно-розового цвета
                countDirtyPink = countDirtyPink + 1
            # Если был найден серый цвет
            elif colorName == "gray":
                # Инкремент кол-ва пикселей серого цвета
                countGray = countGray + 1
            # Если был найден чёрный цвет
            elif colorName == "black":
                # Инкремент кол-ва пикселей чёрного цвета
                countBlack = countBlack + 1
            # Если был найден неопределённый цвет
            else:
                # Инкремент кол-ва пикселей неопределённого цвета
                undefColor = undefColor + 1


        # Вывод процентов для каждого цвета
        print("\n\nБыли найдены следующие цвета блюда:")

        # Процент зелёного цвета
        greenPersent = round(countGreen / setLength * 100, 5)
        # Вывод процента зелёного цвета
        print("Зелёный\t" + str(greenPersent))

        # Процент синего цвета
        bluePersent = round(countBlue / setLength * 100, 5)
        # Вывод процента синего цвета
        print("Синий\t" + str(bluePersent))

        # Процент бирюзового цвета
        turquoisePersent = round(countTurquoise / setLength * 100, 5)
        # Вывод процента бирюзового цвета
        print("Бирюзовый\t" + str(turquoisePersent))

        # Процент сине-зелёного цвета
        blue_GreenPersent = round(countBlue_Green / setLength * 100, 5)
        # Вывод процента сине-зелёного цвета
        print("Сине-зелёный\t" + str(blue_GreenPersent))

        # Процент фиолетового цвета
        violetPersent = round(countViolet / setLength * 100, 5)
        # Вывод процента фиолетового цвета
        print("Фиолетовый\t" + str(violetPersent))

        # Процент грязно-фиолетового цвета
        dirtyVioletPersent = round(countDirtyViolet / setLength * 100, 5)
        # Вывод процента грязно-фиолетового цвета
        print("Грязно-фиолетовый\t" + str(dirtyVioletPersent))

        # Процент красного цвета
        redPersent = round(countRed / setLength * 100, 5)
        # Вывод процента красного цвета
        print("Красный\t" + str(redPersent))

        # Процент оранжевого цвета
        orangePersent = round(countOrange / setLength * 100, 5)
        # Вывод процента оранжевого цвета
        print("Оранжевый\t" + str(orangePersent))

        # Процент жёлтого цвета
        yellowPersent = round(countYellow / setLength * 100, 5)
        # Вывод процента жёлтого цвета
        print("Жёлтый\t" + str(yellowPersent))

        # Процент грязно-жёлтого цвета
        dirtyYellowPersent = round(countDirtyYellow / setLength * 100, 5)
        # Вывод процента грязно-жёлтого цвета
        print("Грязно-жёлтый\t" + str(dirtyYellowPersent))

        # Процент малинового цвета
        crimsonPersent = round(countCrimson / setLength * 100, 5)
        # Вывод процента малинового цвета
        print("Малиновый\t" + str(crimsonPersent))

        # Процент розового цвета
        pinkPersent = round(countPink / setLength * 100, 5)
        # Вывод процента розового цвета
        print("Розовый\t" + str(pinkPersent))

        # Процент грязно-розового цвета
        dirtyPinkPersent = round(countDirtyPink / setLength * 100, 5)
        # Вывод процента грязно-розового цвета
        print("Грязно-розовый\t" + str(dirtyPinkPersent ))

        # Процент серого цвета
        grayPersent = round(countGray / setLength * 100, 5)
        # Вывод процента серого цвета
        print("Серый\t" + str(grayPersent))

        # Процент чёрного цвета
        blackPersent = round(countBlack / setLength * 100, 5)
        # Вывод процента чёрного цвета
        print("Чёрный\t" + str(blackPersent))

        # Процент других цветов
        extraColorsPersent = round(undefColor / setLength * 100, 5)
        # Вывод процента других цветов
        print("Другие цвета\t" + str(extraColorsPersent))


        # Если процент других цветов > 3%
        if(extraColorsPersent > 3):
            print("Не все возможные цвета были найдены!!!")
            print("Изображение может быть не распознано!!!")


        # Добавление в список процентов найденных цветов процента зелёного цвета
        colors_Persents.append(greenPersent)
        # Добавление в список процентов найденных цветов процента синего цвета
        colors_Persents.append(bluePersent)
        # Добавление в список процентов найденных цветов процента бирюзового цвета
        colors_Persents.append(turquoisePersent)
        # Добавление в список процентов найденных цветов процента сине-зелёного цвета
        colors_Persents.append(blue_GreenPersent)
        # Добавление в список процентов найденных цветов процента фиолетового цвета
        colors_Persents.append(violetPersent)
        # Добавление в список процентов найденных цветов процента грязно-фиолетового
        colors_Persents.append(dirtyVioletPersent)
        # Добавление в список процентов найденных цветов процента красного цвета
        colors_Persents.append(redPersent)
        # Добавление в список процентов найденных цветов процента оранжевого цвета
        colors_Persents.append(orangePersent)
        # Добавление в список процентов найденных цветов процента жёлтого
        colors_Persents.append(yellowPersent)
        # Добавление в список процентов найденных цветов процента грязно-жёлтого цвета
        colors_Persents.append(dirtyYellowPersent)
        # Добавление в список процентов найденных цветов процента малинового цвета
        colors_Persents.append(crimsonPersent)
        # Добавление в список процентов найденных цветов процента розового цвета
        colors_Persents.append(pinkPersent)
        # Добавление в список процентов найденных цветов процента грязно-розового
        colors_Persents.append(dirtyPinkPersent)
        # Добавление в список процентов найденных цветов процента серого цвета
        colors_Persents.append(grayPersent)
        # Добавление в список процентов найденных цветов процента чёрного цвета
        colors_Persents.append(blackPersent)


        # Получение название найденного блюда
        foodName = classifyFood(colors_Persents)

    # Возвращаем название найденного блюда
    return foodName


# Функция классификации блюд на основе переданного набора процентов цветов.
def classifyFood(colorsPersents):
    # Начальное наименование блюда
    foodName = "Не известно"

    # Если на вход подан список из процентов 15 цветов
    if (len(colorsPersents) == 15):
        # Проценты определённого цвета в блюдах.
        # greenPersent = colorsPersents[0]
        # bluePersent = colorsPersents[1]
        # turquoisePersent = colorsPersents[2]
        # blue_GreenPersent = colorsPersents[3]
        # violetPersent = colorsPersents[4]
        # dirtyVioletPersent = colorsPersents[5]
        # redPersent = colorsPersents[6]
        # orangePersent = colorsPersents[7]
        # yellowPersent = colorsPersents[8]
        # dirtyYellowPersent = colorsPersents[9]
        # crimsonPersent = colorsPersents[10]
        # pinkPersent = colorsPersents[11]
        # dirtyPinkPersent = colorsPersents[12]
        # grayPersent = colorsPersents[13]
        # blackPersent = colorsPersents[14]


        # Находим индексы 4-ых наибольших процентов (от большего к меньшему)
        persents = findIndexes_Of4MaxPersents(colorsPersents)

        # Если 4 цвета были найдены
        if(len(persents) > 0):
            # Определение блюда "Сыр"
            #if(persents[0] == 8 and persents[1] == 13 and colorsPersents[8] - colorsPersents[13] > 15):
            if (persents[0] == 8 and persents[1] == 13 and colorsPersents[13] > 0):
                foodName = "Сыр"
            # Определение блюда "Огурцы"
            elif(persents[0] == 0 and persents[1] == 8 and colorsPersents[8] > 0 and colorsPersents[0] / colorsPersents[8] > 9):
                foodName = "Огурцы"
            # Определение блюда "Помидоры"
            elif(persents[0] == 6 and colorsPersents[6] >= 60 ):
                foodName = "Помидоры"
            # Определение блюда "Котлеты"
            elif(persents[0] == 12 and persents[1] == 9 and persents[2] == 11 and colorsPersents[11] > 0):
                foodName = "Котлеты"
            # Определение блюда "Чёрная икра"
            elif(persents[0] == 14 and colorsPersents[14] > 0):
                foodName = "Чёрная икра"
            # Определение блюда "Свекла"
            elif(persents[0] == 5 and persents[1] == 4 and colorsPersents[4] > 0):
                foodName = "Свекла"
            # Определение блюда "Малина"
            elif(persents[0] == 4 and persents[1] == 12 and persents[2] == 11 and colorsPersents[11] > 0):
                foodName = "Малина"
            # Определение блюда "Колбаса Докторская"
            elif(persents[0] == 11 and persents[1] == 13 and colorsPersents[13] > 0):
                foodName = "Колбаса Докт."
            # Определение блюда "Апельсин"
            elif(persents[0] == 7 and persents[1] == 6 and colorsPersents[6] > 0):
                foodName = "Апельсин"
            # Определение блюда "Белый хлеб"
            else:
                foodName = "Белый хлеб"

    # Возвращаем начальное наименование блюда
    return foodName


# Функция проверяющая, что 2 эллипса изображения не пересекаются.
# Входные параметры:
# ellipse1_Points - точки эллипса 1
# ellipse2_Params - параметры эллипса2 (кординатаХ_эллипса2, кординатаУ_эллипса2, осьА_эллипса2, осьВ_эллипса2).
def check_If_2Ellipses_Do_Not_Intersect(ellipse1_Points, ellipse2_Params):
    # Результат пересечения 2-х эллипсов
    intersection_Result = False

    # Проверка параметров, переданных функции
    check_Ellipse_Points = len(ellipse1_Points) and isinstance(ellipse1_Points[0], int)

    # Если кол-во параметров эллипса2 равно 4:
    if (len(ellipse2_Params) == 4):
        # кординатаХ_эллипса2
        centerX = ellipse2_Params[0]
        # Проверка того, что кординатаХ_эллипса2 является целым числом.
        check_centerX = isinstance(centerX, int)

        # кординатаУ_эллипса2
        centerY = ellipse2_Params[1]
        # Проверка того, что кординатаУ_эллипса2 является целым числом.
        check_centerY = isinstance(centerY, int)

        # осьА_эллипса2, параллельная оси Х
        axisA = ellipse2_Params[2]
        # Проверка того, что осьА_эллипса2 является целым числом.
        check_axisA = isinstance(axisA, int)

        # осьВ_эллипса2,  параллельная оси У
        axisB = ellipse2_Params[3]
        # Проверка того, что  осьВ_эллипса2 является целым числом.
        check_axisB = isinstance(axisB, int)

        # Проверка всех необходимых параметров
        if (check_Ellipse_Points and check_centerX and check_centerY and check_axisA and check_axisB):

            # Перебираем все точки эллипса1
            for point in ellipse1_Points:
                # Координата Х точки
                xCoord = point[0]
                # Координата У точки
                yCoord = point[1]

                """ Проверка того, что точка эллипса1 с координатами (xCoord, yCoord) не лежит внутри эллипса2 с параметрами
                (centerX, centerY, axisA, axisB)."""
                check_if_Point_Inside_Ellipse = check_If_Point_Inside_Ellipse(xCoord, yCoord, centerX, centerY, axisA, axisB)

                # Но если точка лежит внутри эллипса, то результат пересечения 2-х эллипсов становится равным True.
                if (check_if_Point_Inside_Ellipse):
                    intersection_Result = True
                    # И выходим из цикла.
                    break

    # Возвращаем результат пересечения 2-х эллипсов
    return intersection_Result



# Функция считающая, кол-во непересекающихся эллипсов изображения.
# Входные параметры:
# centersX - список координат Х центров эллипсов,
# centersY - список У-вых координат центров эллипсов,
# ellipse_axisesA - список осей А эллипсов,
# ellipse_axisesB - список осей В эллипсов.
def count_Not_Intersected_Ellipses(centersX, centersY, ellipse_axisesA, ellipse_axisesB):
    # Количество не пересекающихся эллипсов
    count_Not_Intersected_Ellipses = 0

    # Проверка параметров, переданных функции
    # Проверка координат Х центров эллипсов.
    check_centersX = len(centersX) > 0 and isinstance(centersX[0], int)
    # Проверка У-вых координат центров эллипсов,.
    check_centersY = len(centersX) == len(centersY) and isinstance(centersY[0], int)
    # Проверка списка осей А эллипсов.
    check_ellipse_axisesA = len(centersX) == len(ellipse_axisesA) and isinstance(ellipse_axisesA[0], int)
    # Проверка списка осей В эллипсов.
    check_ellipse_axisesB = len(centersX) == len(ellipse_axisesB) and isinstance(ellipse_axisesB[0], int)

    # Проверка параметров, проверяющих параметры переданные функции.
    if check_centersX and check_centersY and check_ellipse_axisesA and check_ellipse_axisesB:
        # Количества точек набора Эллипсов1
        resolution = 900

        # Результат пересечения 2-ух элллипсов с начальным значением False.
        intersect_Res = False

        # Набор точек всех эллипсов изображения
        ellipses_Points = []

        # Перебор всех индексов параметров эллипсов
        for index in range(len(centersX)):
            # Центр эллипса с координатой Х центра эллипса и координатой У центра эллипса.
            ellipse_Center = [centersX[index], centersY[index]]

            """ Получаем точки эллипса с параметрами: resolution, ellipse_Center, 
            ellipse_axisesA[index] - ось А эллипса, ellipse_axisesB[index] - ось В эллипса."""
            points = ellipse_points(resolution, ellipse_Center, ellipse_axisesA[index], ellipse_axisesB[index])[:-1]

            # Добавляем точки текущего эллипса к набору точек всех эллипсов изображения.
            ellipses_Points.append(points)


        # Перебор всех эллипсов изображения
        for i in range(len(centersX)):
            # Перебор всех эллипсов отличных от текущего эллипса изображения
            for j in range(i + 1, len(centersX)):
                # Набор параметров Эллипса2, передавемых в изображение.
                # params = [Координата_Х_центра_эллипса2, Координата_У_центра_эллипса2, ОсьА_эллпса2, ОсьВ_эллпса2]
                params = [centersX[j], centersY[j], ellipse_axisesA[j], ellipse_axisesB[j]]

                # Результат пересечения текущего эллипса (Эллипса1) и Эллипса2.
                intersect_Res = check_If_2Ellipses_Do_Not_Intersect(ellipses_Points[i], params)

                # И если 2 эллипса пересекаются, то заканчиваем переюирать все Эллипсы2 отличные от Эллипса1.
                if (intersect_Res == True):
                    break

            # Если ни одна пара из эллипсов изображения не пересекается,
            if (intersect_Res == False):
                # то инкрементируем число непересекающихся эллипсов
                count_Not_Intersected_Ellipses = count_Not_Intersected_Ellipses + 1
            else:# Иначе результат пересечения 2-ух элллипсов становится равным False.
                intersect_Res = False

    # Возвращаем кол-во непересекающихся эллипсов
    return count_Not_Intersected_Ellipses



# Функция проверяющая, что все параметры эллипсов (centersX, centersY, ellipse_axisesA, ellipse_axisesB) являются правильными.
# height - высота изображения,
# width - ширина изображения,
# centersX - список координат Х центров эллипсов,
# centersY - список У-вых координат центров эллипсов,
# ellipse_axisesA - список осей А эллипсов,
# ellipse_axisesB - список осей В эллипсов.
def checking_Ellipse_Paramms(height, width, centersX, centersY, ellipse_axisesA, ellipse_axisesB):
        # Кол-во эллипсов не выходящих за границы изображения
        count_Ellipses_Inside_Image = 0
        # Кол-во не пересекающихся эллипсов
        count_notIntersectedEllipses = 0


        # Проверка параметров, переданных функции
        # Проверка того, что высота изображения - целое число.
        checkHeight = isinstance(height, int)
        # Проверка того, что ширина изображения - целое число.
        checkWidth = isinstance(width, int)
        # Проверка набора координат Х.
        check_centersX = len(centersX) > 0 and isinstance(centersX[0], int)
        # Проверка набора координат У.
        check_centersY = len(centersX) == len(centersY) and isinstance(centersY[0], int)
        # Проверка осей А эллипсов.
        check_ellipse_axisesA = len(centersX) == len(ellipse_axisesA) and isinstance(ellipse_axisesA[0], int)
        # Проверка осей В эллипсов.
        check_ellipse_axisesB = len(centersX) == len(ellipse_axisesB) and isinstance(ellipse_axisesB[0], int)

        # Проверка параметров, осуществляющих проверку входных параметров функции
        if checkHeight and checkWidth and check_centersX and check_centersY and check_ellipse_axisesA and check_ellipse_axisesB:
            # Перебираем все индексы координат Х эллипсов изображения
            for index in range(len(centersX)):
                # Получаем 4 крайние координаты точек эллипса, находящихся на осях эллипса
                # На вход функции передаются параметры текущего эллипса ()centersX[index], centersY[index], ellipse_axisesA[index], ellipse_axisesB[index].
                extreme_Coords = get_4Extreme_Coords_Of_Ellipse(centersX[index], centersY[index], ellipse_axisesA[index], ellipse_axisesB[index])

                # minX
                coord_centerX_Minus_axisA_Div2 = extreme_Coords[0]
                # maxX
                coord_centerX_Plus_axisA_Div2 = extreme_Coords[1]
                # minY
                coord_centerY_Minus_axisB_Div2 = extreme_Coords[2]
                # maxY
                coord_centerY_Plus_axisB_Div2 = extreme_Coords[3]

                # Проверка крайних координат Х эллипса, т.е. minX >= 0 и maхX <= width.
                check_XCoords = (coord_centerX_Minus_axisA_Div2 >= 0) and (coord_centerX_Plus_axisA_Div2 <= width)
                # Проверка крайних координат У эллипса, т.е. minY>= 0 и maxY <= height
                check_YCoords = (coord_centerY_Minus_axisB_Div2 >= 0) and (coord_centerY_Plus_axisB_Div2 <= height)

                # Если проверка координат Х и координат У эллипса прошла успешно,
                if check_XCoords and check_YCoords:
                    # то инкрементируем число эллипсов не выходящих за границы изображения
                    count_Ellipses_Inside_Image = count_Ellipses_Inside_Image + 1

            # Если число эллипсов не выходящих за границы изображения больше 1,
            if count_Ellipses_Inside_Image > 1:
                # то находим число непересекающихся эллипсов при помощи функции, считающей непересекающиеся эллипсы изображения.
                count_notIntersectedEllipses = count_Not_Intersected_Ellipses(centersX, centersY, ellipse_axisesA, ellipse_axisesB)

        # Возвращаем кол-во эллипсов не выходящих за границы изображения и кол-во непересекающихся эллипсов изображения.
        return [count_Ellipses_Inside_Image, count_notIntersectedEllipses]


""" Функция возвращающая набор найденных блюд, которая полоучает на вход изображение в цветовом пространстве YUV 
и параметры эллпсов данного изображения."""
def get_Food_from_Image(imageYUV, ellipse_Params):
    # Набор найденных блюд на изображении
    result_foodSet = []

    # Проверка входных параметров эллипсов изображения
    if(len(ellipse_Params) > 0 and len(ellipse_Params[:0]) == 0):
        # Набор координат Х эллипсов
        centersX = []
        # Набор координат У эллипсов
        centersY = []
        # Набор осей А эллипсов
        ellipse_axisesA = []
        # Набор осей В эллипсов
        ellipse_axisesB = []

        # Перебор всех параметров эллипсов изображения
        for param in ellipse_Params:
            # Проверка того, что каждому эллпсу соотносится 4 параметра.
            if(len(param) == 4):
                # Добавим координату х эллипса к набору координат Х эллипсов.
                centersX.append(param[1])
                # Добавим координату у эллипса к набору координат У эллипсов.
                centersY.append(param[0])
                # Добавим осьА эллипса к набору осей А эллипсов.
                ellipse_axisesA.append(param[3])
                # Добавим осьВ эллипса к набору осей В эллипсов.
                ellipse_axisesB.append(param[2])

        # Высота изображения
        imgHeight = imageYUV.shape[0]
        # Ширина изображения
        imgWidth = imageYUV.shape[1]


        """ Результаты проверки всех параметров эллипсов изображения с помощью вызова функции проверки правильности 
        всех параметров эллипсов. """
        checkResults = checking_Ellipse_Paramms(imgHeight, imgWidth, centersX, centersY, ellipse_axisesA, ellipse_axisesB)
        # Число эллипсов внутри изображения и число не пересекающихся эллипсов
        [count_Ellipses_Inside_Image, notIntersectedEllipses] = checkResults


        """ Если число эллипсов внутри изображения и число не пересекающихся эллипсов равно, 
        а также число непересекающихся эллипсов равно числу эллипсов, ограничивающих блюда для данного изображения."""
        if(count_Ellipses_Inside_Image == notIntersectedEllipses) and (notIntersectedEllipses == len(ellipse_Params)):
            # Если существует, хотя бы 1 эллипс, который ограничивает какое-то блюдо.
            if(len(centersX) > 0):
                # Перебор всех эллипсов, относящихся к данному изображению
                for index in range(len(centersX)):
                    # Параметры текущего эллипса изображения
                    ellipse_Pars = [centersX[index], centersY[index], ellipse_axisesA[index], ellipse_axisesB[index]]

                    """ Набор пикселей изображения imageYUV, находящихся внутри элллипса ограниченного эллипсом 
                    со следующими параметрами - ellipse_Pars. """
                    pixel_insideEllipse_Set = get_imagePixels_Inside_Ellipse(imageYUV, ellipse_Pars)


                    # Получение названия блюда при помощи анализа значений всех пикселей из выборки pixel_insideEllipse_Set.
                    classified_Food = findFood_andShowAll_FoodColors(pixel_insideEllipse_Set)
                    # Выводим название найденного блюда.
                    print("\nОбнаружено блюдо: " + classified_Food)

                    # Добавляем название найденного блюда к набору найденных блюд на изображении.
                    result_foodSet.append(classified_Food)
        else:
            print("Не корректные значение центров и осей эллипсов!!!!!!!")

    # Возвращаем набор найденных блюд на изображении
    return result_foodSet


# Функция прорисовки эллипсов, ограничивающих блюда на изображении содержащем блюда.
# На вход подаются изображение и коэффициенты эллипсов текущего изображения.
def show_Classifying_Objects(image, koeffs):
    # Если на вход подано изображение
    if (len(image.shape) == 3):
        # Кол-во точек эллипса
        razresh = 200

        # Получаем контейнер изображения и область графика (ax)
        fig, ax = image_show(image)

        # Перебираем все эллипсы, относящиеся к данному изображению.
        for koeffs_img in koeffs:
            # Исключаем последнюю точку, потому что замкнутый путь не должен иметь повторяющихся точек.
            # Получаем точки эллипса на основе координат центра эллипса и осей эллипса.
            points = ellipse_points(razresh, [koeffs_img[0], koeffs_img[1]], koeffs_img[2], koeffs_img[3])[:-1]

            # Прорисовка эллипса в области ax.
            ax.plot(points[:, 0], points[:, 1], '--b', lw=3)

        # Печать изображения и эллипсов.
        plt.show()


""" Функция обработки изображений, получающая на вход директорию содержащую файлы изображения (directory), 
набор названий файлов изображений (images_Set) и флаг прорисовки эллипсов и изображений (visualization).
Данная функция возвращает набор неправильно классифицированных блюд из всех изображений и кол-во всех классифицированных блюд."""
def processing_Images(directory, images_Set, visualization):
    # Набор неправильно классифицированных блюд из всех изображений.
    notGuessed_foodSet_from_All_Directory = []

    # Индекс набора параметров эллипсов для текущего изображения.
    index = 0

    # Кол-во всех классифицированных блюд
    countFoodSet = 0


    print("Ведётся тестирование системы... ")

    # Перебор списка имен всех изображений из обучающей выборки
    for file_Of_Image in images_Set:

        # Считывание файла изображения из директории.
        image = io.imread(directory + file_Of_Image)

        # Вывод текущего имени файла-изображения.
        print("\n\n\nВ файле " + file_Of_Image)

        # Проверка того, что на вход подаётся изображение
        if (len(image.shape) == 3):
            # Преобразование изображения RGB в цветовое пространство YUV.
            image_yuv = color.rgb2yuv(image)


            """ Получаем набор найденных блюд, передавая в функцию для получения набора блюд изображение 
            в цветовом пространстве YUV, а также параметры эллипсов для текущего изоьражения (params[index])."""
            foodSet_FromImage = get_Food_from_Image(image_yuv, params[index])

            # Получаем набор правильно и неправильно угаданных блюд.
            guessed_foodSet_FromImage = check_Food(foodSet_FromImage, idealFoodSet[index])

            # Получаем набор неправильно найденных блюд на основе списка блюд guessed_foodSet_FromImage
            notGuessed_foodSet_FromImage = incorrectGuessedFood(guessed_foodSet_FromImage)

            # Добавляем набор неправильно найденных блюд для данного изображения к общему набору неправильно найденных блюд во всей директории.
            notGuessed_foodSet_from_All_Directory = notGuessed_foodSet_from_All_Directory + notGuessed_foodSet_FromImage


            # Считаем сколько было найдено блюд суммарно во всех обработанных изображениях.
            countFoodSet = countFoodSet + len(foodSet_FromImage)

            # Если флаг прорисовки эллипсов равен True.
            if visualization == True:
                # Прорисовывем эллипсы, заданные параметрами (params[index]) на изображении image.
                show_Classifying_Objects(image, params[index])

            """ Инкрементируем индекс набора параметров эллипсов изображения, т.е. переходим к набору параметров 
            эллипсов для следующего изображения."""
            index = index + 1

    # Возвращаем набор неправильно классифицированных блюд из всех изображений и кол-во всех классифицированных блюд
    return [notGuessed_foodSet_from_All_Directory, countFoodSet]


# Функция, возвращающая набор правильно угаданных блюд и набор неправильно угаданных блюд, отмеченных словом "Не правильно".
# На вход подаются найденный набор блюд и набор блюд, который на самом деле присутствует на изображении.
def check_Food(foodSet, idealFoodSet):
    # Набор правильно и неправильно угаданных блюд
    guessedFoodSet = []

    # Если длины найденного набора блюд и набора идеальных блюд, которые в действительности присутствуют на изображении, совпадают.
    if(len(foodSet) == len(idealFoodSet)):
        # Перебор всех найденных блюд и идеальных блюд.
        for index in range(len(foodSet)):
            # Если найденное блюдо и идеальное блюдо являются строками.
            if(isinstance(foodSet[index], str) and isinstance(idealFoodSet[index], str)):
                # Если найденное блюдо и идеальное блюдо совпадают,
                if(foodSet[index] == idealFoodSet[index]):
                    # то добавляем идеальное блюдо в набор правильно и неправильно угаданных блюд.
                    guessedFoodSet.append(idealFoodSet[index])
                else:# иначе
                    """ Добавляем следующую строку ("Не правильно" + foodSet[index] + "_" + idealFoodSet[index]) 
                    в набор правильно и неправильно угаданных блюд."""
                    guessedFoodSet.append("Не правильно" + foodSet[index] + "_" + idealFoodSet[index])

    # Возвращаем набор правильно и неправильно угаданных блюд.
    return guessedFoodSet


# Функция нахождения неправильно классифицированным блюд на основе списка угаданных блюд.
def incorrectGuessedFood(foodSet):
    # Набор неправильно классифицированным блюд
    resultFoodSet = []

    # Перебираем все блюда из набора классифицированным блюд.
    for food in foodSet:
        # Проверяем, что блюдо является строкой
        if (isinstance(food, str)):
            # Проверяем, что блюдо является неправильно классифицированным.
            res = food.startswith("Не правильно")

            # И если это так,
            if (res):
               # То удаляем из названия блюда строку "Не правильно".
               food.replace("Не правильно", "")

               # И добавляем его к набору неправильно классифицированных блюд.
               resultFoodSet.append(food)

    # Возвращаем набор неправильно классифицированных блюд
    return resultFoodSet


# Функция получения набора ошибок 1-го и 2-го рода на основе набора неправильно отгаданных блюд и общего кол-ва всех блюд.
def getErrorSet(notGuessed_foodSet_from_Directory, foodNumber):
    # Набор ошибок 1-го и 2-го рода
    errorSet = []

    # Проверка входных параметров функции
    if (len(notGuessed_foodSet_from_Directory) > 0) and isinstance(foodNumber, int):
        # Кол-во других блюд, распознанных как данное блюдо.
        sthOtherAsFood = 0

        # Кол-во данных блюд, распознанных как другие блюда.
        foodAsSthOther = 0


        # Создаём словарь неправильно распознанных блюд с помощью счётчика Counter, подсчитывающего кол-во уникальных неправильно распознанных блюд.
        dictionaryFoodSet = dict(Counter(notGuessed_foodSet_from_Directory))

        # Набор названий неправильно распознанных блюд.
        foodKeys = dictionaryFoodSet.keys()


        # Кол-во названий неправильно распознанных блюд.
        length_FoodKeys = len(foodKeys)

        # Перебираем блюда из списка наименований уникальных блюд.
        for food in different_typesOfFood:
            # Перебираем неправильно распознанные блюда
            for foodKey in foodKeys:
                # Если неправильно распознанное блюдо является строкой
                if isinstance(foodKey, str):
                    # Проверяем не является уникальное блюдо началом неправильно распознанного блюда.
                    check_Begin = foodKey.startswith(food)
                    # Если да,
                    if (check_Begin):
                        # то добавляем к кол-ву данных блюд, распознанных как другие блюда, число блюд данного типа из словаря dictionaryFoodSet.
                        foodAsSthOther = foodAsSthOther + dictionaryFoodSet[foodKey]

                        # Декрементируем кол-во названий неправильно распознанных блюд.
                        length_FoodKeys = length_FoodKeys - 1
                    else:
                        # Проверяем не является уникальное блюдо концом неправильно распознанного блюда.
                        check_End = foodKey.endswith(food)
                        # Если да,
                        if (check_End):
                            # то добавляем к кол-ву других блюд, распознанных как данное блюдо, число блюд данного типа из словаря dictionaryFoodSet.
                            sthOtherAsFood = sthOtherAsFood + dictionaryFoodSet[foodKey]

                            # Декрементируем кол-во названий неправильно распознанных блюд.
                            length_FoodKeys = length_FoodKeys - 1


            # Добавляем в набор ошибок 1-го и 2-го рода ошибку 1-го рода и ошибку 2-го рода для данного блюда.
            errorSet.append([round(foodAsSthOther / foodNumber * 100, 3), round(sthOtherAsFood / foodNumber * 100, 3)])

            # Обнуляем кол-во данных блюд, распознанных как другие блюда.
            foodAsSthOther = 0.
            # Обнуляем кол-во других блюд, распознанных как данное блюдо.
            sthOtherAsFood = 0


            # Если перебрали все блюда из словаря dictionaryFoodSet, то завершаем цикл перебора уникальных блюд (food).
            if (length_FoodKeys == 0):
                break
    else: # Когда нет неправильно распознанных блюд
        errorSet = [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]

    # Возвращаем набор ошибок 1-го и 2-го рода.
    return errorSet


# Функция прорисовки диаграммы ошибок
def draw_Diagramms(error_Set):
    # Список ошибок 1-го рода
    error1 = []
    # Список ошибок 2-го рода
    error2 = []

    # Подписи к диаграмме1
    params_Labels = different_typesOfFood

    # Перебираем все пары ошибок 1-го и 2-го рода для каждого из блюд
    for error1AndError2 in error_Set:
        # Добавление ошибки 1-го рода из пары, содержащей ошибку 1-го рода и ощибку 2-го рода для каждого из блюд.
        error1.append(error1AndError2[0])
        # Добавление ошибки 2-го рода из пары, содержащей ошибку 1-го рода и ощибку 2-го рода для каждого из блюд.
        error2.append(error1AndError2[1])

    # Ширина 2-ух столбцов диаграммы, где width/2 - ширина столбца ошибки 1-го рода или ширина столбца ошибки 2-го рода.
    width = 0.35

    # Количество столбцов в диаграмме
    x = np.arange(len(different_typesOfFood))

    # Диаграмма (fig) и оси (ax).
    fig, ax = plt.subplots(figsize=(45, 8))

    # Столбцы ошибки 1-го рода
    """ Синтаксис: ax.bar(X, height, width, label), где X - размер столбца вдоль оси абсцисс, 
    а height - размер столбца вдоль оси ординат"""
    ax.bar(x - width / 2, error1, width, label='Ошибка 1-го рода')

    # Столбцы ошибки 2-го рода
    ax.bar(x + width / 2, error2, width, label='Ошибка 2-го рода')
    # Заголовок
    ax.set_title('Диаграмма ошибок')
    # Подписи оси У
    ax.set_ylabel('%')
    # Установки отметок из массива x со списком отметок из столбцов
    ax.set_xticks(x)
    # Подписи к столбцам
    ax.set_xticklabels(params_Labels)
    # Добавление легенды
    ax.legend()
    # Отображение изображения
    plt.show()



#Основная программа
#Относительный путь к директории, содержащей изображения блюд
training_Directory = os.getcwd() + '/Блюда Итоговые/'

#Список блюд из директории
images_Set = get_List_of_Images(training_Directory)

# Флаг вывода на экран изображений с выделенными контурами блюд.
# Если значение флага равно True, то изображения отображаются на экране.
# Если значение флага равно False, то изображения отображаются на экране.
visualization = True

""" Получаем кол-ва неправильно классифицированных блюд и кол-ва всех блюд в изображениях,
 передавая в функцию директорию, список блюд из директории и флаг вывода на экран изображений."""
[notGuessed_foodSet_from_Directory, foodNumber] = processing_Images(training_Directory, images_Set, visualization)

# Получение набора пар ошибок 1-го и второго рода
errorSet = getErrorSet(notGuessed_foodSet_from_Directory, foodNumber)
# Вывод набора пар ошибок 1-го и 2-го рода
print("\nОшибки 1-го и 2-го рода для продуктов в выборке:")
print(errorSet)
# Вызов функции отображения ошибок 1-го и 2-го рода
draw_Diagramms(errorSet)