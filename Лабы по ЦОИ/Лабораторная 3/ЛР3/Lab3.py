from skimage import draw, transform, io, color, exposure
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math


BRIGHT_RECTANGLE = 255
DARK_RECTANGLE = 50


def draw_haar_feature(w1, w2, w3, h, angle):
    img = np.zeros((h, w1 + w2 + w3), dtype=np.uint8)
    rr, cc = draw.rectangle((0, 0), extent=(h, w1), shape=img.shape)
    img[rr, cc] = BRIGHT_RECTANGLE
    rr, cc = draw.rectangle((0, w1), extent=(h, w2), shape=img.shape)
    img[rr, cc] = DARK_RECTANGLE
    if w3:
        rr, cc = draw.rectangle((0, w1 + w2), extent=(h, w3), shape=img.shape)
        img[rr, cc] = BRIGHT_RECTANGLE

    img = transform.rotate(img, 360 - angle, resize=True, preserve_range=False)

    return img


def create_Image_Dataset(directory):
    dataset = []

    image_list = [file for file in os.listdir(directory) if file.endswith('.png')]
    for img in image_list:
        if os.path.isdir(img):
            continue
        image = io.imread(directory + img)

        image_gray = color.rgb2gray(image)
        print(image[1][0][0])
        image_contrast = exposure.adjust_sigmoid(image_gray, cutoff = 0.5, gain = 25, inv=False)

        dataset.append((image, image_contrast))
    #os.chdir(os.getcwd() + "/../")
    return dataset


def feature_detection(img, haar_feature_img, haar_feature_img2, haar_feature_img3, integeral_Image,  haar_feature_Params, angle):
    coords = []
    truck_values = []
    start_value = 190
    maxVal11 = -1

    cur_value2 = -1
    cur_value3 = -1
    size = haar_feature_img.shape
    size2 = haar_feature_img2.shape
    size3 = haar_feature_img3.shape
    for x in range(img.shape[0] - size[0]):
        for y in range(img.shape[1] - size[1]):
            if x == 0 and y == 0:
                haar_Integral_Image_Val = integeral_Image[x + size[0]][y + size[1]]

                #print("Integral_ImageVal" + str(haar_Integral_Image_Val) + '\n')
            elif x > 0 and y == 0:
                haar_Integral_Image_Val = integeral_Image[x + size[0]][y + size[1]] - integeral_Image[x - 1][y + size[1]]

            elif x == 0 and y > 0:
                haar_Integral_Image_Val = integeral_Image[x + size[0]][y + size[1]] - integeral_Image[x + size[0]][y - 1]


            else:
                haar_Integral_Image_Val = integeral_Image[x + size[0]][y + size[1]] - integeral_Image[x + size[0]][y - 1] + integeral_Image[x - 1][y - 1] - integeral_Image[x - 1][y + size[1]]


            if(check_haar_IntegralImage(haar_Integral_Image_Val, haar_feature_Params, angle) > 0 ):
                cur_value = detect(img[x:x + size[0], y:y + size[1]], haar_feature_img)
                if x + size2[0] < img.shape[0] and y + size2[1] < img.shape[1] and cur_value > 0:
                    # Фильтр-ограничение
                    cur_value2 = detect(img[x:x + size2[0], y:y + size2[1]], haar_feature_img2)
                if x + size3[0] < img.shape[0] and y + size3[1] < img.shape[1] and cur_value2 < 0:
                    cur_value3 = detect(img[x:x + size3[0], y:y + size3[1]], haar_feature_img3)
                if cur_value > start_value and cur_value2 < 0 and cur_value3 < 0:
                    truck_values.append(cur_value)
                    coords.append((x, y))
    return coords, truck_values


def detect(img, haar_feature_img):
    if img.shape != haar_feature_img.shape:
        print("ERROR: SIZES NOT EQUAL")
        raise IndexError

    # этот параметр подбирается. есть граница, выше которой ничего определятся не будет
    # если он слишком мал, то будет находится ненужное
    threshold = 96 * haar_feature_img.shape[0] * haar_feature_img.shape[1] / 255  # ~17.5
    bright, dark = 0, 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if haar_feature_img[x][y] == BRIGHT_RECTANGLE / 255:
                bright += img[x][y]
            elif haar_feature_img[x][y] == DARK_RECTANGLE / 255:
                dark += img[x][y]
    if bright - dark > threshold:
            return bright - dark
    return -1


def draw_angled_rec(x0, y0, width, height, angle, img, color):
    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, color, 1)
    cv2.line(img, pt1, pt2, color, 1)
    cv2.line(img, pt2, pt3, color, 1)
    cv2.line(img, pt3, pt0, color, 1)


def generate_IntegralImage(img):
    height = img.shape[0]
    width = img.shape[1]
    # print(img.shape)
    summ = 0
    integralImage = np.zeros((height, width), dtype=np.int16)
    # Заполнение отрицательными числвми
    for x in range(height):
        for y in range(width):
            integralImage[x][y] = -1
    # print("Integral_ImageVal" + str(integralImage) + '\n')
    # Расчёт интегрального изображения
    for x in range(height):
        for y in range(width):
            if x == 0 and y == 0:
                integralImage[x][y] = img[x][y]
                print(str(img[x][y]) + '\n')

            elif x == 0 and y > 0:
                if integralImage[x][y - 1] >= 0:
                    integralImage[x][y] = integralImage[x][y - 1] + img[x][y]
                '''else:
                    for k in range(0, y):
                        summ += img[x][k]
                    integralImage[x][y] = summ
                    summ = 0'''
            elif x > 0 and y == 0:
                if integralImage[x - 1][y] >= 0:
                    integralImage[x][y] = integralImage[x - 1][y] + img[x][y]
                '''else:
                    for k in range(0, x):
                        summ += img[k][y]
                    integralImage[x][y] = summ
                    summ = 0'''

            else:
                if (integralImage[x][y - 1] >= 0) and (integralImage[x - 1][y] >= 0):
                    integralImage[x][y] = integralImage[x][y - 1] - integralImage[x - 1][y - 1] + integralImage[x - 1][y] + img[x][y]
                '''else:
                    for k in range(0, x):
                        for m in range(0, y):
                            summ += img[k][m]
                        integralImage[x][y] = summ
                        summ = 0'''

        return integralImage



def check_haar_IntegralImage(haar_IntegralImage_Val, filter_params, angle):

    w1 = filter_params[0]
    w2 = filter_params[1]
    w3 = filter_params[2]
    h = filter_params[3]

    result_ckecking = -1
    haarSum = BRIGHT_RECTANGLE * h * (w1 + w3) + DARK_RECTANGLE * h * w2
    if angle == -270 or angle == -180 or angle == -90 or angle == 0 or angle == 90 or angle == 180 or angle == 270:
        # print("HaarSum is "+str(haarSum)+"\t Integral_ImageVal"+str(haar_IntegralImage_Val)+'\n')
        if haarSum >= 0.1 * haar_IntegralImage_Val and haarSum <= 1.9 * haar_IntegralImage_Val:
            result_ckecking = haarSum

    else:
        if (haar_IntegralImage_Val - haarSum) < 0.6 * haar_IntegralImage_Val:
            result_ckecking = haarSum

    if(result_ckecking >0):
        print(str(result_ckecking) +'\n')
    return result_ckecking

def highlight_feature(img, c_img, coords, size, angle, haar_feature, n):
    c_img = color.gray2rgb(c_img)

    for i, coord in enumerate(coords):

        if coord != (-1, -1):
            if size[0] > size[1]:
                x0 = coord[1] + size[1] / 2
                y0 = coord[0] + size[0] / 2
                draw_angled_rec(x0, y0, size[0], size[1], angle, img, (255, 0, 0))
                draw_angled_rec(x0, y0, size[0], size[1], angle, c_img, (1, 0, 0))
            else:
                x0 = coord[1] + size[0] / 2
                y0 = coord[0] + size[1] / 2
                draw_angled_rec(x0, y0, size[1], size[0], angle, img, (255, 0, 0))
                draw_angled_rec(x0, y0, size[1], size[0], angle, c_img, (1, 0, 0))
    '''
    label = "Truck detected" if coord[0] != -1 else "No truck detected"
    fig, ax = plt.subplots(1, 3, squeeze=False)
    ax[0][0].set(title='Original', xlabel='Found in total')
    ax[0][0].imshow(img)
    ax[0][1].set(title='High contrast', xlabel=label)
    ax[0][1].imshow(c_img)
    ax[0][2].set(title='Haar-like feature', xlabel="Angle: {}".format(angle))
    ax[0][2].imshow(haar_feature)
    io.show()'''

    os.chdir("Truck")
    os.chdir("Detected")
    img = img[:, :, ::-1]

    cv2.imwrite(str(n) + " " + str(angle) + ".jpg", img)
    os.chdir(os.getcwd() + "/../../")



# Полный путь папки ...\fruits
images = r"C:\Users\Максим\Desktop\Цифровая обработка изображений\Лабораторные работы\Лабораторная 3\images"
# Полный путь папки ...\training
training_directory = images + r"\training\\"
# Полный путь папки ...\testing
testing_directory = images + r"\testing\\"

truck_training_dataset = create_Image_Dataset(training_directory)
truck_testing_dataset = create_Image_Dataset(testing_directory)
print("Continue")



# диапазоны углов для каждой фотки, чтобы не проверять все углы (очень долго)
for n, image in enumerate(truck_training_dataset):
    # Интегральное изображение
    int_img = transform.integral_image(img)
    print(image[1])
    if n == 0:
        start, end = -90, 91
        step = 45
        # w1=8, w2=10, w3=8, h=20
    elif n == 1:
        start, end = -90, 91
        step = 90
    elif n == 2:#2
        start, end = -90, 91
        step = 30
    elif n == 3:#3
        start, end = 0, 181
        step = 45
    elif n == 4:
        start, end = -90, 91
        step = 90
    elif n == 5:
        start, end = 45, 91
        step = 45


    elif n == 6:
        start, end = -80, 90
        step = 10

    elif n == 7:
        start, end = -80, 90
        step = 10

    elif n == 8:
        start, end = -240, 20
        step = 10

    elif n == 9:
        start, end = -180, 0
        step = 10
    elif n == 10:
        start, end = -180, 0
        step = 10
    elif n == 11:
        start, end = -180, -30
        step = 10
    else:
        start, end = -90, 90
        step = 90
        '''elif n == 12:
            start, end = -240, -30
            step = 10'''


    for i in range(start, end, step):

        print('\nImage', n, ', Angle: ', i)
        haar_feature = draw_haar_feature(w1=8, w2=10, w3=8,  h=20, angle=i)
        haar_feature1 = draw_haar_feature(w1=8, w2=10, w3=8,  h=21, angle=i)
        haar_feature2 = draw_haar_feature(w1=43, w2=10, w3=8, h=20, angle=i)
        haar_feature_Params = [8, 10, 8, 20]

        coordinates, values = feature_detection(image[1], haar_feature, haar_feature1, haar_feature2, integeral_Image,  haar_feature_Params, i)


        for id, value in enumerate(values):
            print('Coord is', coordinates[id], 'and Value is', value)
        # print(coordinates, value)
        if len(values) > 0:
            highlight_feature(image[0], image[1], coordinates, haar_feature.shape, i, haar_feature, n)
        else:
            print("Грузовиков не обнаружено.") #No trucks detected
