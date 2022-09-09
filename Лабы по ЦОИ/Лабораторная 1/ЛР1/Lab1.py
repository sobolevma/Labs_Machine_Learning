from skimage import filters
import os
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from skimage import io
import skimage.color as color


# Собственная функция сортировки на основе длины каждого элемента
def keyFunc_Item_Length(item):
   return len(item)


# Получить список изображений из директории.
def get_List_of_Images(directory):
    # Получаем список файлов-изображений в переменную images_files
    images_files = [file for file in os.listdir(directory) if file.endswith('.jpg')]

    # Получаем список файлов-изображений, начинающихся с цифры
    images_files_part1 = [image_file for image_file in images_files if
                          (image_file[0] >= str(0) and image_file[0] <= str(9))]
    # Получаем список файлов-изображений, начинающихся с последовательности "r_".
    images_files_part2 = [image_file for image_file in images_files if (image_file[0] == "r" and image_file[1] == "_")]

    # Получаем список файлов-изображений, начинающихся с "r", где следующий символ за "r" не равен "_".
    images_files_part3 = [image_file for image_file in images_files if (image_file[0] == "r" and image_file[1] != "_")]

    # Сортировка списка файлов-изображений, начинающихся с цифры
    images_files_part1.sort(key=keyFunc_Item_Length)
    # Сортировка списка файлов-изображений, начинающихся с последовательности "r_".
    images_files_part2.sort(key=keyFunc_Item_Length)
    # Сортировка списка файлов-изображений, начинающихся с "r", где следующий символ за "r" не равен "_".
    images_files_part3.sort(key=keyFunc_Item_Length)

    # Итоговый список из списков: images_files_part1, images_files_part2, images_files_part3.
    images_files = images_files_part1 + images_files_part2 + images_files_part3

    # Возвращаем итоговый список
    return images_files

"""def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax
"""

# Функция порогового преобразования изображения.
def image_thresholding(image):
    """Numpy indexing"""
    # Яркость
    brightness = 0.3
    # Скопировать в переменную img_thres изображение image
    img_thres = image.copy()
    # Заменяем в новом изображении все пиксели, яркость которых меньше brightness.
    img_thres[image < brightness] = 0
    # Заменяем в новом изображении все остальные пиксели значением 255.
    img_thres[image >= brightness] = 255

    # Возвращаем новое изображение (чёрно-белое).
    return img_thres


# Функция получения результата воздействия фильтра по ширине и по высоте.
def result_Filter(image_h, image_v):

    image = []
    # Если размеры изображения image_h и image_v одинаковы.
    if image_h.shape == image_v.shape:
        # Высота Изображения1 или Изображения 2.
        height = image_h.shape[0]
        # Ширина Изображения1 или Изображения 2.
        width = image_h.shape[1]

        # Ширина изображение черного цвета размера height х width.
        image = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                # Получени значения пикселя нового изображения в соответствии с формулой: G = sqrt(pow(G_h, 2) + pow(G_v, 2)).
                image[i][j] = sqrt(pow(image_h[i][j], 2) + pow(image_v[i][j], 2))

    # Возвращаем результ применения фильтров выделения контура к изображению по ширине и по высоте.
    return image


# Функция получения контура изображения в оттенках серого с порогом
def get_image_Gray_Threshold(image):
    # Получение изображения в оттенках серого
    image_gray = color.rgb2gray(image)

    # Применение фильтра Щарра по высоте к изображению в оттенках серого и получение контура изображения по высоте.
    edges_h = filters.scharr_h(image_gray)

    # Применение фильтра Щарра по ширине к изображению в оттенках серого и получение контура изображения по ширине.
    edges_v = filters.scharr_v(image_gray)

    # Итоговый контур, полученный из контуров по высоте и по ширине.
    edges = result_Filter(edges_h, edges_v)

    # Применяем порого к изображению итогового контура
    image_gray_threshold = image_thresholding(edges)

    # Возвращаем контур изображения в оттенках серого с порогом.
    return image_gray_threshold


# Функция объединения двух изображений
def concatenate_images(image1, image2):
    # Скопировать в переменную img_thres изображение image1
    img_res = image1.copy()

    """ Если размеры изображения image1 и image2 одинаковы, то добавляем в изобрадение img_res пиксели белого цвета 
        только тогда, когда яркость пискселей image2 больше нуля. """
    if image1.shape == image2.shape:
        img_res[image2 > 0] = 255

    return img_res

# Функция совпадения изображений по белым пикселям
def match_image_percentage(image1, image2):
    # Число совпадающих белых пикселей двух изображений, изначально равное нулю.
    num = 0

    # Процент совпадения двух изображений на основе пикселей белого цвета, изначально равный нулю.
    persent = 0

    # Число белых пикселей во 2-ом изображении, изначально равное нулю.
    count_white_from_image2 = 0

    # Если размеры 2-ух изображений совпадают.
    if image1.shape == image2.shape:
        # Высота Изображения1 или Изображения 2.
        height = image1.shape[0]
        # Ширина Изображения1 или Изображения 2.
        width = image1.shape[1]

        # Перебор всех пикселей 2-ух изображений
        for i in range(height):
            for j in range(width):
                # Если текущий пиксель 2-го изображения является белым.
                if image2[i][j] == 255:
                    # Инкрементируем число белых пикселей во 2-ом изображении.
                    count_white_from_image2 = count_white_from_image2 + 1

                    # И если цвет текущего пикселя Изображения1 совпадает с цветом текущего пикселя Изображения 2, то
                    if image1[i][j] == image2[i][j]:
                        # Инкрементируем число совпадающих белых пикселей двух изображений.
                        num = num + 1

        # Вычисляем процент совпадения двух изображений на основе пикселей белого цвета.
        persent = num / count_white_from_image2 * 100
    # Возвращаем процент совпадения 2-ух изображений.
    return persent

# Функция получения списка образцовых изображений для фрукта.
def get_Sample_Image_Set(directory, images_Training_Set, set_Name):
    #  Пустой список образцовых изображений для фрукта
    list_Of_Sample_Images = []

    #  Текущее изображение-образец
    sample_image = []

    # Вывод имени обучающей выборки.
    print("Training " + set_Name + " set...")

    # Перебор списка имен всех изображений из обучающей выборки
    for file_Of_Image in images_Training_Set:
        # Считывание файла изображения из директории.
        image = io.imread(directory + file_Of_Image)

        # Вывод текущего имени файла-изображения для обучения.
        print("On " + file_Of_Image + " from " + set_Name + " set.")

        # Проверка того, что на вход подаётся изображение
        if (len(image) > 0 and len(image[0:])):
            # Получение текущего контура изображения в оттенках серого с порогом.
            image_gray_threshold = get_image_Gray_Threshold(image)

            # Если изображение-образец существует
            if (len(sample_image) > 0 and len(sample_image[0:])):
                """ Вычисляем процент совпадения изображения-образца с текущим контуром изображения 
                в оттенках серого с порогом."""
                persent = match_image_percentage(sample_image, image_gray_threshold)

                 # Если процент совпадения больше 73%, т.е. изображения похожи с вероятностью 73%.
                if persent > 73:
                    # И если процент совпадения меньше 73%.
                    if (persent < 85):
                        # Объединяем контура изображения в оттенках серого с порогом вместе с изображением-образцом
                        img_res = concatenate_images(image_gray_threshold, sample_image)
                        # Копируем плученное изображение в изображение-образец
                        sample_image = img_res.copy()
                else:  # Иначе, если процент совпадения не больше 73%,то
                    # Добавляем изображение-образец в список образцовых изображений для фрукта.
                    list_Of_Sample_Images.append(sample_image)
                    # Очищаем изображение-образец.
                    sample_image = image_gray_threshold.copy()
            else: # Иначе копируем в изображение-образец текущий контур изображения в оттенках серого с порогом.
                sample_image = image_gray_threshold.copy()

    """Добавляем изображение-образец в список образцовых изображений для фрукта, если образец не был добавлен после выполнения всех итераций. """
    if(len(sample_image) > 0 and len(sample_image[0:])):
        list_Of_Sample_Images.append(sample_image)

    #  Возвращаем список образцовых изображений для фрукта
    return list_Of_Sample_Images


# Функция классификации изображений фруктов
def gessed_fruits(directory, images_Testing_Set, sample_Fruit1_Set, sample_Fruit2_Set):
    # Промежуточное значение для хранения процента совпадений
    val = 0

    # Кол-во изображений, распознанных как Фрукт1
    count_fruit1 = 0
    # Кол-во изображений, распознанных как Фрукт2
    count_fruit2 = 0

    # Показывает начало работы функции классификации изображений фруктов
    print("Trying to find images...")
    for file_Of_Image in images_Testing_Set:
        # Считывание файла изображения из директории.
        image = io.imread(directory + file_Of_Image)
        # Вывод текущего имени файла-изображения для обучения.
        print("Checking " + file_Of_Image)

        # Проверка того, что на вход подаётся изображение
        if (len(image) > 0 and len(image[0:])):
            # Получение текущего контура изображения в оттенках серого с порогом.
            image_gray_threshold = get_image_Gray_Threshold(image)

            """ Проверяем совпадение текущего контура изображения в оттенках серого с порогом 
            с изображением-образцом из списка образцовых изображений для Фрукта1."""
            for sample_Fruit1 in sample_Fruit1_Set:
                """ Вычисляем процент совпадения контура изображения 
                в оттенках серого с порогом c обрзцовым изображением из выборки для Фрукта1."""
                persent1 = match_image_percentage(sample_Fruit1, image_gray_threshold)

                #""" Если процент совпадения изображений больше 50% и промежуточное значение для хранения процента совпадений равно нулю."""
                if (persent1 > 50) and (val == 0):
                    # Инкремент кол-ва изображений, распознанных как Фрукт1
                    count_fruit1 = count_fruit1 + 1
                    # Поместить в промежуточное значение для хранения процента persent1.
                    val = persent1
                #""" Если процент совпадения изображений больше 50% и промежуточное значение для хранения процента совпадений меньше, чем persent1."""
                elif (persent1 > 50) and (val < persent1):
                    # Поместить в промежуточное значение для хранения процента persent1.
                    val = persent1


            """ Проверяем совпадение текущего контура изображения в оттенках серого с порогом 
                   с изображением-образцом из списка образцовых изображений для Фрукта2."""
            for sample_Fruit2 in sample_Fruit2_Set:
                """ Вычисляем процент совпадения контура изображения 
                в оттенках серого с порогом c обрзцовым изображением из выборки для Фрукта2."""
                persent2 = match_image_percentage(sample_Fruit2, image_gray_threshold)
                # break присутствует в if, т.к. если изображение распознано как Фрукт1, то больше не нужно просматривать всю образцовую выборку для Фрукта2.
                # """ Если процент совпадения изображений больше 50% и промежуточное значение для хранения процента совпадений равно нулю."""
                if (persent2 > 50) and (val == 0):
                    # Инкремент кол-ва изображений, распознанных как Фрукт2
                    count_fruit2 = count_fruit2 + 1
                    break
                # """ Если процент совпадения изображений больше 50% и промежуточное значение для хранения процента совпадений меньше, чем persent2."""
                elif (persent2 > 50) and (val < persent2):
                    # Декремент кол-ва изображений, распознанных как Фрукт1
                    count_fruit1 = count_fruit1 - 1
                    # Инкремент кол-ва изображений, распознанных как Фрукт1
                    count_fruit2 = count_fruit2 + 1
                    break
        # Обнуляем промежуточное значение для хранения процента совпадений
        val = 0

    # Возвращаем кол-во изображений, распознанных как Фрукт1 и кол-во изображений, распознанных как Фрукт2.
    return [count_fruit1, count_fruit2]

# Функция прорисовки диаграммы ошибок
def draw_diagramms(error_Fruit1_training, error_Fruit1_testing, error_Fruit2_training, error_Fruit2_testing):
    # Подписи к диаграмме1
    params_Labels = ["Обучающие выборки", "", "", "", "", "Тестовые выборки"]
    # Ошибка 1-го рода
    error1 = [error_Fruit1_training, 0., 0., 0., 0., error_Fruit1_testing]
    # Ошибка 2-го рода
    error2 = [error_Fruit2_training, 0., 0., 0., 0., error_Fruit2_testing]

    # Ширина 2-ух столбцов диаграммы, где width/2 - ширина столбца ошибки 1-го рода или ширина столбца ошибки 2-го рода.
    width = 0.35
    # Количество столбцов в диаграмме
    x = np.arange(len(error1))

    # Диаграмма (fig) и оси (ax).
    fig, ax = plt.subplots(figsize=(10, 5))

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



"""Основная программа"""
# Полный путь папки ...\fruits
fruits = r"C:\Users\Максим\Desktop\Цифровая обработка изображений\Лабораторные работы\Лабораторная 1\fruits"
# Полный путь папки ...\fruits\training\Chestnut
training_Chestnut_directory = fruits + r"\training\Chestnut\\"
# Полный путь папки ...\fruits\training\Rambutan
training_Rambutan_directory = fruits + r"\training\Rambutan\\"

# Полный путь папки ...\fruits\testing\Chestnut
testing_Chestnut_directory = fruits + r"\testing\Chestnut\\"
# Полный путь папки ...\fruits\testing\Rambutan
testing_Rambutan_directory = fruits + r"\testing\Rambutan\\"

# Получим список упорядоченных изображений из папки ...\fruits\training\Chestnut
images_Training_Chestnut_Set = get_List_of_Images(training_Chestnut_directory)
# Получим список упорядоченных изображений из папки ...\fruits\training\Rambutan
images_Training_Rambutan_Set = get_List_of_Images(training_Rambutan_directory)
# Получим список упорядоченных изображений из папки ...\fruits\testing\Chestnut
images_Testing_Chestnut_Set = get_List_of_Images(testing_Chestnut_directory)
# Получим список упорядоченных изображений из папки ...\fruits\testing\Rambutan
images_Testing_Rambutan_Set = get_List_of_Images(testing_Rambutan_directory)

# Набор образцовых фруктов Chestnuts (каштанов).
sample_Chestnuts = get_Sample_Image_Set(training_Chestnut_directory, images_Training_Chestnut_Set, "Chestnut")
# Набор образцовых фруктов Rambutans.
sample_Rambutans = get_Sample_Image_Set(training_Rambutan_directory, images_Training_Rambutan_Set, "Rambutan")

print("\n\nFrom training_Chestnut_directory")
# Синтакисис: gessed_fruits(images_directory, images_Fruit_Set, sample_Fruit1_Set, sample_Fruit2_Set)
# Возвращает ф-ия gessed_fruits 2 числа: [число_Правильно_Отгаданных_Фруктов, число_Неправильно_Отгаданных_Фруктов]

# Получаем список:[число_Правильно_Отгаданных_Chestnut, число_Неправильно_Отгаданных_Chestnut] из обучающей выборки.
gessed_nums_Of_Training_Fruits1 = gessed_fruits(training_Chestnut_directory, images_Training_Chestnut_Set,  sample_Chestnuts, sample_Rambutans)


print("\n\nFrom training_Rambutan_directory")
# Получаем список:[число_Правильно_Отгаданных_Rambutan, число_Неправильно_Отгаданных_Rambutan] из обучающей выборки.
gessed_nums_Of_Training_Fruits2 = gessed_fruits(training_Rambutan_directory, images_Training_Rambutan_Set, sample_Chestnuts, sample_Rambutans)

print("\n\nFrom testing_Chestnut_directory")
# Получаем список:[число_Правильно_Отгаданных_Chestnut, число_Неправильно_Отгаданных_Chestnut] из тестовой выборки.
gessed_nums_Of_Testing_Fruits1 = gessed_fruits(testing_Chestnut_directory, images_Testing_Chestnut_Set, sample_Chestnuts, sample_Rambutans)

print("\n\nFrom testing_Rambutan_directory")
# Получаем список:[число_Правильно_Отгаданных_Rambutan, число_Неправильно_Отгаданных_Rambutan] из тестовой выборки.
gessed_nums_Of_Testing_Fruits2 = gessed_fruits(testing_Rambutan_directory, images_Testing_Rambutan_Set, sample_Chestnuts, sample_Rambutans)


# Число образцовых фруктов
# Количество фруктов-образцов Chestnut
print("\nLength of Sample of Fruits1 is: " + str(len(sample_Chestnuts)))
# Количество фруктов-образцов Rambutan
print("\nLength of Sample of Fruits2 is: " + str(len(sample_Rambutans)))


# Отгадывание фруктов
# Вывод чисел правильно и неправильно отгаданных фруктов Chestnut из обучающей выборки.
print("\nFor Chestnut Training:")
print("As Chestnut: " + str(gessed_nums_Of_Training_Fruits1[0]) + ", as Rambutan: " + str(gessed_nums_Of_Training_Fruits1[1]) + ", Sum:"  + str(len(images_Training_Chestnut_Set)))

# Вывод чисел правильно и неправильно отгаданных фруктов Rambutan из обучающей выборки.
print("\nFor Rambutan Training:")
print("As Chestnut: " + str(gessed_nums_Of_Training_Fruits2[0]) + ", as Rambutan: " + str(gessed_nums_Of_Training_Fruits2[1]) + ", Sum:" + str(len(images_Training_Rambutan_Set)))

# Вывод чисел правильно и неправильно отгаданных фруктов Chestnut из тестовой выборки.
print("\nFor Chestnut Testing:")
print("As Chestnut: " + str(gessed_nums_Of_Testing_Fruits1[0]) + ", as Rambutan: " + str(gessed_nums_Of_Testing_Fruits1[1]) + ", Sum:" + str(len(images_Testing_Chestnut_Set)))

# Вывод чисел правильно и неправильно отгаданных фруктов Rambutan из тестовой выборки.
print("\nFor Rambutan Testing:")
print("As Chestnut: " + str(gessed_nums_Of_Testing_Fruits2[0]) + ", as Rambutan: " + str(gessed_nums_Of_Testing_Fruits2[1]) + ", Sum:" + str(len(images_Testing_Rambutan_Set)))


# Ошибки 1-го и 2-го рода.

# Ошибка 2-го рода для обучаающих выборок.
print("\n\nОшибка 2-го рода для обучающих выборок:")
error_Fruit2_training = gessed_nums_Of_Training_Fruits1[1] / len(images_Training_Chestnut_Set) * 100
error_Fruit2_training = round(error_Fruit2_training, 3)
print(str(error_Fruit2_training) + "%")

# Ошибка 1-го рода для обучаающих выборок.
print("\nОшибка 1-го рода для обучающих выборок:")
error_Fruit1_training = gessed_nums_Of_Training_Fruits2[0] / len(images_Training_Rambutan_Set) * 100
error_Fruit1_training = round(error_Fruit1_training, 3)
print(str(error_Fruit1_training) + "%")

# Ошибка 2-го рода для тестовых выборок.
print("\nОшибка 2-го рода для тестирующих выборок:")
error_Fruit2_testing = gessed_nums_Of_Testing_Fruits1[1] / len(images_Testing_Chestnut_Set) * 100
error_Fruit2_testing = round(error_Fruit2_testing, 3)
print(str(error_Fruit2_testing) + "%")

# Ошибка 1-го рода для тестовых выборок.
print("\nОшибка 1-го рода для тестирующих выборок:")
error_Fruit1_testing = gessed_nums_Of_Testing_Fruits2[0] / len(images_Testing_Rambutan_Set) * 100
error_Fruit1_testing = round(error_Fruit1_testing, 3)
print(str(error_Fruit1_testing) + "%")

# График ошибок 1-го и 2-го рода
draw_diagramms(error_Fruit1_training, error_Fruit1_testing, error_Fruit2_training, error_Fruit2_testing)
