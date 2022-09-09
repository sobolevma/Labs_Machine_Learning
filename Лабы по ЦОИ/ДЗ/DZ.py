import numpy as np
import cv2 as cv

# Флаг включения эффекта
enable_Effect = False

# video_path = r"C:\Users\Максим\Desktop\Цифровая обработка изображений\Лабораторные работы\ДЗ\Video\Boat_Pixels_Video.mp4"
video_path = r"C:\Users\Максим\Desktop\Цифровая обработка изображений\Лабораторные работы\ДЗ\Video\Project_MY_NEWEST_effect12.avi"
if video_path.find('effect') > 0:
   enable_Effect = True

# Открытие видео-файла
cap = cv.VideoCapture(video_path)

# Параметры для обнаружения углов ШиТомаси
feature_params = dict( maxCorners = 25,
                       qualityLevel = 0.3,
                       minDistance = 5,
                       blockSize = 7)


# Параметры для оптического потока Лукаса-Канаде
lk_params = dict( winSize  = (15,15), #
                  maxLevel = 1,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Создадим случайные цвета
color = np.random.randint(0,255,(100,3))

# Возьмём первый кадр и найдём в нём углы
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Создание маски изображения для рисования
mask = np.zeros_like(old_frame)

while 1:

    # Захватывает, декодирует и возвращает следующий видеокадр.
    ret,frame = cap.read()

    if not ret:
        break

    # Преобразуем пространство BGR в полутоновый формат.
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Вычисляем оптический поток для нового и старого изображений находим новые точки
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Выбираем хорошие точки со статусом 1, как настоящие, так и предыдущие.
    if p1 is not None:
        next_Good_Feature_Points = p1[st==1]
        old_Good_Feature_Points = p0[st==1]

    # Расстояние на которое сдвигаются все точки
    total_Shift = 0

    # Флаг, указывающий на то, что были созданы новые точки.
    wrong_Feature_Number = 1

    # Когда флаг эффекта выставлен.
    if enable_Effect:
        # Счетчик хороших точек - которые прошли через преграду
        good_Feature_Points_Count = 0
        # Список плохих точек, непрошедших преграду.
        bad_Point_Index = []
        # Счетчик плохих точек - которые не прошли через преграду
        bad_Point_Count = 0

        # Проходим по всем найденным настоящим точкам
        for i in range(len(next_Good_Feature_Points)):
            ''' Ограничиваем область поиска плохих и хорощих точек. '''
            if next_Good_Feature_Points[i][1] > 639 and next_Good_Feature_Points[i][1] < 816 and next_Good_Feature_Points[i][0] > 930 and next_Good_Feature_Points[i][0] < 962:

                # Если х-овая кооордината текущей точки минус х-овая координата предыдущей меньше порога 0.2.
                if next_Good_Feature_Points[i][0] - p0[i][0][0] < 0.2:
                    # Запоминаем номер данной точки
                    bad_Point_Index.append(i)
                    # Считаем количество плохих точек, т.е потерянных точек.
                    bad_Point_Count += 1
                else:
                    # Считаем количество хороших точек, т.е. прошедших преграду точек.
                    good_Feature_Points_Count += 1
                    # Находим расстояние до точки, т.е. ширину преграды.
                    total_Shift += next_Good_Feature_Points[i][0] - p0[i][0][0]

            # Если количество плохих точек лежит в промежутке от 0 до кол всех точек
            if 0 < bad_Point_Count < len(next_Good_Feature_Points):
                if good_Feature_Points_Count == 0:
                    good_Feature_Points_Count = 1
                    if total_Shift > 0:
                        print('Общий_сдвиг', total_Shift)
                # Находим среднее расстояние на которое смесились хорошие точки
                average_Shift = total_Shift / good_Feature_Points_Count
                if total_Shift > 0:
                    print('Средний_сдвиг = ', 2 * average_Shift)

                # Для плохих точек записываем новое значение х
                for i in bad_Point_Index:
                    ''' Средний слвиг берём с отрицательным знаком, т.к. двигаемся вправо.'''
                    # Подбираем коэффициент
                    next_Good_Feature_Points[i][0] -= 0.34 * average_Shift


            # Если новых точек меньше чем старых на 1, тогда создаем новые точки
            if (len(next_Good_Feature_Points) < len(p0) - 1):
                feature_params = dict(maxCorners=len(p0) - len(next_Good_Feature_Points) + 1,
                                      qualityLevel=0.3,
                                      minDistance=10,
                                      blockSize=10)
                # Находим углы
                new_Feature_Corners = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                # Объединение старых и созданных только что точек
                new_points = np.append(next_Good_Feature_Points, new_Feature_Corners)
                # Сбрасываем флаг создания новых точек.
                wrong_Feature_Number = 0

    # Рисуем пути
    for i,(new,old) in enumerate(zip(next_Good_Feature_Points, old_Good_Feature_Points)):
        # Начальные координаты линии
        a,b = new.ravel()
        # Конечные координаты линии
        c,d = old.ravel()
        # Чертим линию
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        # Чертим точку
        frame = cv.circle(frame,(int(a), int(b)),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)

    #cv.rectangle(img, (0, 640), (1000, 815), color[0].tolist(), thickness = 2, lineType = 8, shift = 0)
    # Чертим область в которой происходит подсчёт плохих и хороших точек.
    cv.rectangle(img, (930, 640), (962, 815), (0, 0, 255), thickness=2, lineType=8, shift=0)


    cv.namedWindow("frame", cv.WINDOW_NORMAL)
    #cv.resizeWindow("frame", 1366, 768) # Вводим размеры окна
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break


    # Теперь обновим предыдущий кадр и предыдущие точки
    old_gray = frame_gray.copy()
    # Флаг создания новых точек установлен
    if wrong_Feature_Number == 1:
        p0 = next_Good_Feature_Points.reshape(-1,1,2)
    else:
        p0 = new_points.reshape(-1, 1, 2)
# Уничтожает все окна HighGUI.
cv.destroyAllWindows()
# Релиз программного ресурса
cap.release()
