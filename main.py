import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, shutil
from mask_bar import run_mask_bar
from model import img_preprocessing, recognize_letter, recognize_digit


# константы
BLACK = (0, 40, 0)
BORDER_WIDTH = 3
MIN_ALLOWED_AREA = 0.1
RES_SHIFT = 0
RES_SIZE = (28, 28)
WIDTH_CROP = 56+4
HEIGHT_CROP = 13+7
FILTER_KERNEL = 9
EROSION_KERNEL = (5, 5)

# HSV limit values denoted with mask_bar.py
# Лимиты значений цветового формата HSV, определенные с помощью файла mask_bar.py 
# LIMITS = [([85, 60, 0], [100, 255, 255]), ([0, 81, 0], [130, 255, 255]), ([120, 50, 0], [179, 255, 255]), ([0, 40, 0], [165, 255, 255]), ([7, 0, 0], [179, 51, 255]), ([0, 139, 0], [179, 255, 255]), ([0, 25, 0], [179, 255, 255]), ([70, 170, 0], [90, 255, 255]), ([116, 115, 0], [130, 255, 255]), ([0, 0, 0], [179, 255, 140])]
# BLUE_LIMITS = [([85, 60, 0], [100, 255, 255]), ([0, 81, 0], [130, 255, 255])]
# RED_LIMITS = [([120, 50, 0], [179, 255, 255]), ([0, 40, 0], [165, 255, 255]), ([7, 0, 0], [179, 51, 255]), ([0, 139, 0], [179, 255, 255])]
# BLACK_LIMITS = [([0, 25, 0], [179, 255, 255]), ([0, 0, 0], [179, 255, 140])]
# GREEN_LIMITS = [([70, 170, 0], [90, 255, 255])]
# PURPLE_LIMITS = [([116, 115, 0], [130, 255, 255])]

# Returns rectangle contour area on an image
# Находит площадь прямоугольного контура на изображении
def area_part_on_image(image, contour):
    image_area = image.shape[0] * image.shape[1]
    cnt_area = cv2.contourArea(contour)
    
    area_part = (cnt_area / image_area) * 100
    return area_part

# Denotes found contours on an image and returns a list with contours coords
# Размечат найденные контура на изображении и возвращает список найденных координат контуров
def cnts_mark_out(image, contours, hierarchy, cnts_color, brd_w, min_area=MIN_ALLOWED_AREA):
    cnt_rects = []

    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour) 
        if hierarchy[0][idx][3] == 0 and area_part_on_image(image, contour) > min_area: #const (arg)
            cv2.rectangle(image, (x, y), (x + w, y + h), cnts_color, brd_w) # const (arg)
            cnt_rects.append((x, y, w, h))
    # cv2.imshow('contours', image)
    # cv2.waitKey(0)
    return cnt_rects

# Sorts found rectangle contours by width or other
# Сортирует найденные прямоугольные контура по ширине или другому param
def sort_cnts(cnts_rects, param, reverse):
    sorted_cnts = sorted(cnts_rects, key=lambda rect: rect[param], reverse=reverse)
    return sorted_cnts

# Counts amount of symbols contours and width step for norm_found_cnts function
# Подсчёт количества контуров и шага ширины для функции norm_found_cnts
def add_cnts(total_width, *widths):
    width_steps = []
    cnts_in = []

    for width in widths:
        width_step = width // round(5 * (width / total_width))
        width_steps.append(width_step)
        cnt_in = round(width / width_step)
        cnts_in.append(cnt_in)
    
    return width_steps, cnts_in

# Norms found contours to get rid of together written symbols
# Нормирует найденные контура (разделяет на части, чтобы избавиться от слитно написанных букв)
def norm_found_cnts(symb_cnts):
    cnts_rects_norm = []
    
    if len(symb_cnts) == 1:
        x, y, w, h = symb_cnts[0]
        width_step = w // 5
        for cnt_add_i in range(5):
            step_cnt_rect = x + width_step * cnt_add_i, y, width_step, h
            cnts_rects_norm.append(step_cnt_rect)

    if len(symb_cnts) == 2:
        sorted_cnts = sort_cnts(symb_cnts, param=2, reverse=True)
        total_width = 0

        for width in [cnt[2] for cnt in sorted_cnts]:
            total_width += width

        width1, width2 = sorted_cnts[0][2], sorted_cnts[1][2]
        (width_step1, width_step2), \
        (add_range1, add_range2) = add_cnts(total_width, width1, width2)
        
        x1, y1, w1, h1 = sorted_cnts[0]
        for cnt_add_i1 in range(add_range1):
            step_cnt_rect1 = x1 + width_step1 * cnt_add_i1, y1, width_step1, h1
            cnts_rects_norm.append(step_cnt_rect1)

        if add_range2 != 1:
            x2, y2, w2, h2 = sorted_cnts[1]
            for cnt_add_i2 in range(add_range2):
                step_cnt_rect2 = x2 + width_step2 * cnt_add_i2, y2, width_step2, h2
                cnts_rects_norm.append(step_cnt_rect2)
        else:
            cnts_rects_norm.append(sorted_cnts[1])

    if len(symb_cnts) == 3:
        sorted_cnts = sort_cnts(symb_cnts, param=2, reverse=True)
        total_width = 0

        for width in [cnt[2] for cnt in sorted_cnts]:
            total_width += width

        width1, width2, width3 = sorted_cnts[0][2], sorted_cnts[1][2], sorted_cnts[2][2]
        
        (width_step1, width_step2, width_step3), \
        (add_range1, add_range2, add_range3) = add_cnts(total_width, width1, width2, width3)

        x1, y1, w1, h1 = sorted_cnts[0]
        for cnt_add_i1 in range(add_range1):
            step_cnt_rect1 = x1 + width_step1 * cnt_add_i1, y1, width_step1, h1
            cnts_rects_norm.append(step_cnt_rect1)

        if add_range2 != 1:
            x2, y2, w2, h2 = sorted_cnts[1]
            for cnt_add_i2 in range(add_range2):
                step_cnt_rect2 = x2 + width_step2 * cnt_add_i2, y2, width_step2, h2
                cnts_rects_norm.append(step_cnt_rect2)
        else:
            cnts_rects_norm.append(sorted_cnts[1])

        if add_range3 != 1:
            x3, y3, w3, h3 = sorted_cnts[2]
            for cnt_add_i3 in range(add_range3):
                step_cnt_rect3 = x3 + width_step3 * cnt_add_i3, y3, width_step3, h3
                cnts_rects_norm.append(step_cnt_rect3)
        else:
            cnts_rects_norm.append(sorted_cnts[2])

    if len(symb_cnts) == 4:
        sorted_cnts = sort_cnts(symb_cnts, param=2, reverse=True)
        x, y, w, h = sorted_cnts[0]

        cnt_rect1 = x, y, w // 2, h
        cnt_rect2 = x+(w // 2), y, w // 2, h

        cnts_lst = [cnt_rect1, cnt_rect2]
        cnts_rects_norm = [*cnts_lst, *sorted_cnts[1::]]

    if len(symb_cnts) >= 5:
        cnts_rects_norm = symb_cnts

    return sort_cnts(cnts_rects_norm, param=0, reverse=False)

# Shows an image with denoted contours
# Показывает картинку с выделенными контурами
def show_cnts_rects(image_name, image, cnts_rects, cnts_color, brd_w):
    for (x, y, w, h) in cnts_rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), cnts_color, brd_w)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)

# Shows an image part 
# Показвает какой-то прямоугольный кусок изображения
def show_img_part(image, x, y, w, h):
    img_part = image[y: y+h, x: x+w]
    cv2.imshow('img_part', img_part)
    cv2.waitKey(0)

# Resizes an image into 28x28 resolution to work with NN and justifies symbol if RES_SHIFT
# Изменяет разрешение изображения на 28x28, чтобы работать с нейронной сетью и выравнивает букву в квадрате если указать RES_SHIFT
def resize_img(image, cnt_rect, out_size=RES_SIZE, symbol_shift=RES_SHIFT):
    x, y, w, h = cnt_rect
    x, y = x - symbol_shift, y - symbol_shift
    w, h = w + (2 * symbol_shift), h + (2 * symbol_shift)
    symbol_crop = image[y: y+h, x: x+w]

    max_size = max(w, h)

    symbol_square = 0 * np.ones([max_size, max_size], np.uint8)
    if w > h:
        y_pos = max_size//2 - h//2
        symbol_square[y_pos:y_pos + h, 0:w] = symbol_crop
    elif w < h:
        x_pos = max_size//2 - w//2
        symbol_square[0:h, x_pos:x_pos + w] = symbol_crop
    else:
        symbol_square = symbol_crop
    # plt.imshow(letter_square, cmap=plt.cm.binary)
    # plt.show()
    resized = cv2.resize(symbol_square, out_size, cv2.INTER_AREA)

    return resized
        
# Finds all scans in folder with file format .jpg
# Находит все сканы из папки с форматом .jpg
def scans_from_folder(folder_path):
    scans = []
    for filename in os.listdir(folder_path):

        if os.path.isdir(f'{folder_path}/{filename}'):
            for scan in os.listdir(f'{folder_path}/{filename}'):

                if scan.endswith('.jpg'):
                    scans.append(f'{folder_path}/{filename}/{scan}')

        elif filename.endswith('.jpg'):
            scans.append(f'{folder_path}/{filename}'.replace('\\', '/'))
    
    return scans

# Crops only students number from an image
# Оставляет только верхнюю правую часть с номером поступающего
def student_num_rect(scan, w_crop=WIDTH_CROP, h_crop=HEIGHT_CROP):
    width = scan.shape[0]
    height = scan.shape[1]

    s_num_w = int(width / 100) * w_crop #constant (arg)
    s_num_h = int(height / 100) * h_crop #constant (arg)
    student_num_img = scan[0: s_num_h, width - s_num_w: width]

    return student_num_img


# Removes all noise from image with blur
# Избавляет от шума на изображении (плохо пропечатанные клетки) с помощью блюра
def filter_img(img, f_ker=FILTER_KERNEL):
    filtered_img = cv2.medianBlur(img, f_ker) #constant (arg)
    return filtered_img


# Finds all pixels in gray range
# Находит пиксели в заданных "серых" пределах (по дефолту для оставления только букв)
def img_gray_mask(img, gray_range=(0, 220)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lower, upper = gray_range
    gray_mask = cv2.inRange(gray, lower, upper)
    # plt.imshow(bright, cmap='gray')
    # plt.show()

    return gray_mask

# Collects total mask using HSV values
# Сбор маски со всеми значениями
def img_mask(img, limits):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_mask = 0

    for limit in limits:
        lower = np.array(limit[0])
        upper = np.array(limit[1])

        mask = cv2.inRange(hsv_img, lower, upper)
        total_mask += mask

    return total_mask


# Returns binary eroded image
# Возвращает бинарную картинку
def erode_thresh_img(mask, er_ker=EROSION_KERNEL):
    kernel = np.ones(er_ker, np.uint8)

    erosion = cv2.erode(mask, kernel, iterations=1) #const arg
    ret, thresh = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY_INV)
    # plt.imshow(thresh, cmap=plt.cm.binary)
    # plt.show()

    return erosion, thresh

# Saving source image with symbols info
# Сохранения изображения с информацией о распознанных символах
def save_img_data(img, img_path, dir_name, symbs_data, scans_dir='scans'):
    
    if not os.path.exists(scans_dir):
        os.mkdir(scans_dir)
    os.chdir(scans_dir)

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)

    img_name = img_path.split('/')[-1]
    cv2.imwrite(img_name, img)

    data_dir_name = 'symbols {}'.format(img_name.split('.')[-2])
    if not os.path.exists(data_dir_name):
        os.mkdir(data_dir_name)
    os.chdir(data_dir_name)

    if symbs_data:
        for i in range(len(symbs_data)):
            cv2.imwrite(f'({i}) {dir_name[i]}.jpg', symbs_data[i])
    
    os.chdir('../../..')


# Recongnizing an image (main function)
# Распознавание номера поступающего и сохранение изображений в файл (основная функция)
def recognize_scan(scan_path, mask_range, gray_range=(0, 220)):
    scans_dir = scan_path.split('/')[-2]

    scan = cv2.imread(scan_path)
    student_num_img = student_num_rect(scan)
    filtered_img = filter_img(student_num_img)
    
    if mask_range:
        mask = img_mask(filtered_img, limits=mask_range)
    else:
        mask = img_gray_mask(filtered_img, gray_range=gray_range)

    erosion, thresh = erode_thresh_img(mask=mask)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts_rects = cnts_mark_out(student_num_img, contours, hierarchy, cnts_color=BLACK, brd_w=BORDER_WIDTH)

    if not(cnts_rects):
        dir_name = scan_path.split('/')[-1].split('.')[-2]
        save_img_data(scan, scan_path, dir_name, None, scans_dir=scans_dir)
        return 'Контуров не найдено'
    
    print(f'\nScanning {scan_path}')
    print('Found countours (rectangles coords)\n', cnts_rects)
    cnts_rects_norm = norm_found_cnts(cnts_rects)
    print('Normalized rectangle contours\n', cnts_rects_norm)

    symbs_data = []
    letter_img = resize_img(erosion, cnts_rects_norm[0])
    symbs_data.append(letter_img)
    prep_img_arr = img_preprocessing(letter_img)
    
    letter, acc = recognize_letter(prep_img_arr, 'models/letters_model.h5')
    print(f'Predicted letter: ==> {letter}, Accuracy:', acc)
    result = letter

    for cnt_rect_i in range(1, len(cnts_rects_norm)):
        digit_img = resize_img(erosion, cnts_rects_norm[cnt_rect_i])
        symbs_data.append(digit_img)
        prep_img_arr = img_preprocessing(digit_img)

        digit, acc = recognize_digit(prep_img_arr, 'models/digits_model.h5')
        print(f'Predicted digit: ==> {digit}, Accuracy:', acc)
        result += str(digit)

    save_img_data(scan, scan_path, result, symbs_data, scans_dir=scans_dir)

    return result    

# Recongnizing scans from directory with folders and images
# распознавание из папки с классами
def recognize_folder_scans(folder_path, mask_range):
    scans = scans_from_folder(folder_path)

    for scan_path in scans:
        recognize_scan(scan_path, mask_range=mask_range)


# Denotes config file with HSV values
# Размечает конфиг файл (указать для каждой папки значения HSV)
def denote_dirs_config(folder, config_file):
    for file_name in os.listdir(folder):

        if os.path.isdir(f'{folder}/' + file_name):
            samples = os.listdir(f'{folder}/' + file_name)

            hsv_values = 0
            if len(samples) > 0:
                test_sample = f'{folder}/{file_name}/{samples[0]}'
            
                img = cv2.imread(test_sample)
                sample_rect = student_num_rect(img)
                
                hsv_values = run_mask_bar(sample_rect)
                cv2.destroyAllWindows()

            with open(config_file, 'a') as config_file:
                config_file.write(f'{folder}/{file_name}: {hsv_values}\n')


# Normalizing config to dictionary
# Нормализует конфиг в словарь, состоящий из ключа пути к папке и значения в виде массива значений HSV
def config_norm(config_file):

    try:
        with open(config_file) as config:
            text = config.read()
            lines = text.split('\n')
            
            dic = {}
            for line in lines:
                info = line.split(': ')
                folder = info[0]
                values = info[-1]

                if len(values) != 1:
                    values = values.split(', ')
                    values = [int(num) for num in values]
                else:
                    values = [int(values)]

                dic[folder] = values

    except Exception:
        print('Error might be in the config... Check it up!')
    
    return dic


# Recognizing scans with configuration file
# Распознает изображения с конфигурацией
def recognition_with_config(config_file):
    config_dic = config_norm(config_file)

    for folder_path, hsv_values in config_dic.items():

        if len(hsv_values) != 1:
            lower = np.array(hsv_values[:3])
            upper = np.array(hsv_values[3:])

            recognize_folder_scans(folder_path, mask_range=[(lower, upper)])

        else:
            recognize_folder_scans(folder_path, None)


# Normalize results (move all symbols files in 1 folder)
# Нормализует результат, а именно убирает все временные файлы с символами в одну папку
def norm_edited_results(folder):
    for filename in os.listdir(folder):
        path = f'{folder}/{filename}'

        if not(len(os.listdir(path))):
            os.rmdir(path)

        else:
            for info in os.listdir(path):
                if info.split()[0] == 'symbols':

                    rec_symbs_path = f'{folder}/rec_symbs'
                    if not os.path.exists(rec_symbs_path):
                        os.mkdir(rec_symbs_path)

                    shutil.move(f'{path}/{info}', rec_symbs_path)


# Recongnition with config file example
# Пример распознавания с конфиг файлом, но можно и без него, указав limits самостоятельно
# file_path = 'scans_26_08_13_30/200/200001.jpg'
# folder_path = '/'.join(file_path.split('/')[:2])
# image = cv2.imread(file_path)
# rect = student_num_rect(image)
# limits = run_mask_bar(rect)

# with open('config_f.txt', 'w') as test_file:
#     values = ', '.join([str(i) for i in limits])
#     test_file.write(f'{folder_path}: {values}')

# recognition_with_config('config_f.txt')
# norm_edited_results(folder_path.split('/')[-1])