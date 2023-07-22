import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, shutil
from mask_bar import run_mask_bar
from model import NNModel


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



class CVMethods:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

    @property
    def image(self):
        return self._image
    
    @image.setter
    def image(self, image):
        if not(isinstance(image, np.ndarray)):
            raise ValueError('Image is not numpy.ndarray.')
        self._image = image

    @property
    def image_path(self):
        return self._image_path
    
    @image_path.setter
    def image_path(self, image_path):
        if not(isinstance(image_path, str)):
            raise ValueError('Image path type can only be str.')
        self._image_path = image_path


    @staticmethod
    def sort_cnts(cnts_rects, param, reverse):
        sorted_cnts = sorted(cnts_rects, key=lambda rect: rect[param], reverse=reverse)
        return sorted_cnts


    @staticmethod
    def norm_found_cnts(symb_cnts):
        cnts_rects_norm = []
        total_width = sum([cnt[2] for cnt in symb_cnts])
        if len(symb_cnts) >= 5:
            return CVMethods.sort_cnts(symb_cnts, param=0, reverse=False)

        for contour in symb_cnts:
            x_pos = contour[0]
            width = contour[2]
            width_part = width / total_width
            division_parts = round(width_part * 5) if width_part >= (1 / 5) else 1
            width_step = width // division_parts

            for step in range(division_parts):
                x_pos += width_step * step
                symb_cnt = x_pos, contour[1], width_step, contour[3]
                cnts_rects_norm.append(symb_cnt)

        return CVMethods.sort_cnts(cnts_rects_norm, param=0, reverse=False)


    def area_part_on_image(self, image, contour):
        image_area = image.shape[0] * image.shape[1]
        cnt_area = cv2.contourArea(contour)
        
        area_part = (cnt_area / image_area) * 100
        return area_part


    def cnts_mark_out(self, image, contours, hierarchy, cnts_color, brd_w, min_area=MIN_ALLOWED_AREA):
        cnt_rects = []

        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour) 
            if hierarchy[0][idx][3] == 0 and self.area_part_on_image(image, contour) > min_area: #const (arg)
                cv2.rectangle(image, (x, y), (x + w, y + h), cnts_color, brd_w) # const (arg)
                cnt_rects.append((x, y, w, h))
        # cv2.imshow('contours', image)
        # cv2.waitKey(0)
        return cnt_rects
    

    def show_cnts_rects(self, image_name, image, cnts_rects, cnts_color, brd_w):
        for (x, y, w, h) in cnts_rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), cnts_color, brd_w)
        cv2.imshow(image_name, image)
        cv2.waitKey(0)


    def show_img_part(self, image, x, y, w, h):
        img_part = image[y: y+h, x: x+w]
        cv2.imshow('img_part', img_part)
        cv2.waitKey(0)


    def resize_img(self, image, cnt_rect, out_size=RES_SIZE, symbol_shift=RES_SHIFT):
        x, y, w, h = cnt_rect
        x, y = x - symbol_shift, y - symbol_shift
        w, h = w + (2 * symbol_shift), h + (2 * symbol_shift)
        symbol_crop = image[y: y+h, x: x+w]

        max_size = max(w, h)

        symbol_square = 0 * np.ones([max_size, max_size], np.uint8)

        try:
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
            
        except ValueError:
            print('Error in symbol image. No predicted symbol: ==> _')
            return None


    def student_num_rect(self, w_crop=WIDTH_CROP, h_crop=HEIGHT_CROP):
        scan = self.image
        width = scan.shape[0]
        height = scan.shape[1]

        s_num_w = int(width / 100) * w_crop #constant (arg)
        s_num_h = int(height / 100) * h_crop #constant (arg)
        student_num_img = scan[0: s_num_h, width - s_num_w: width]

        return student_num_img


    def filter_img(self, image, f_ker=FILTER_KERNEL):
        filtered_img = cv2.medianBlur(image, f_ker) #constant (arg)
        return filtered_img


    def img_gray_mask(self, image, gray_range=(0, 220)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        lower, upper = gray_range
        gray_mask = cv2.inRange(gray, lower, upper)
        # plt.imshow(bright, cmap='gray')
        # plt.show()
        return gray_mask


    def img_mask(self, image, limits):
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        total_mask = 0

        for limit in limits:
            lower = np.array(limit[0])
            upper = np.array(limit[1])

            mask = cv2.inRange(hsv_img, lower, upper)
            total_mask += mask

        return total_mask


    def erode_thresh_img(self, mask, er_ker=EROSION_KERNEL):
        kernel = np.ones(er_ker, np.uint8)

        erosion = cv2.erode(mask, kernel, iterations=1) #const arg
        ret, thresh = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY_INV)
        # plt.imshow(thresh, cmap=plt.cm.binary)
        # plt.show()

        return erosion, thresh



class SymbolsRecognizer:
    def __init__(self, scan_path, mask_range, gray_range=(0, 220)):
        self.scan_path = scan_path
        self.mask_range = mask_range
        self.gray_range = gray_range

    @property
    def scan_path(self):
        return self._scan_path
    
    @scan_path.setter
    def scan_path(self, scan_path):
        if not(isinstance(scan_path, str)):
            raise ValueError('Scan path can only be str.')
        self._scan_path = scan_path

    @property
    def mask_range(self):
        return self._mask_range
    
    @mask_range.setter
    def mask_range(self, mask_range):
        if not(isinstance(mask_range, list)):
            raise ValueError('Mask range can only be list[tuple[np.array, np.array]].')
        self._mask_range = mask_range

    @property
    def gray_range(self):
        return self._gray_range
    
    @gray_range.setter
    def gray_range(self, gray_range):
        if not(isinstance(gray_range, tuple)):
            raise ValueError('Gray range type can only be tuple[int, int].')
        self._gray_range = gray_range


    def recognize_scan(self):
        scans_dir = self.scan_path.split('/')[-2]

        CVPreproccessor = CVMethods(self.scan_path)
        student_num_img = CVPreproccessor.student_num_rect()
        filtered_img = CVPreproccessor.filter_img(student_num_img)
        
        if self.mask_range:
            mask = CVPreproccessor.img_mask(filtered_img, limits=self.mask_range)
        else:
            mask = CVPreproccessor.img_gray_mask(filtered_img, gray_range=self.gray_range)

        erosion, thresh = CVPreproccessor.erode_thresh_img(mask=mask)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts_rects = CVPreproccessor.cnts_mark_out(student_num_img, contours, hierarchy, cnts_color=BLACK, brd_w=BORDER_WIDTH)

        if not(cnts_rects):
            dir_name = self.scan_path.split('/')[-1].split('.')[-2]
            save_img_data(CVPreproccessor.image, self.scan_path, dir_name, None, scans_dir=scans_dir)
            return 'No contours found'
        
        print(f'\nScanning {self.scan_path}')
        print('Found contours (rectangles coords)\n', cnts_rects)
        cnts_rects_norm = CVPreproccessor.norm_found_cnts(cnts_rects)
        print('Normalized rectangle contours\n', cnts_rects_norm)

        symbs_data = []
        result = ''
        LettersModel = NNModel('letters', 'models/letters_model.h5')
        DigitsModel = NNModel('digits', 'models/digits_model.h5')

        # Symbols recognition process (letters & digits)
        # Распознавание символов (букв и цифр)
        for cnt_rect_i in range(len(cnts_rects_norm)):
            symbol_img = CVPreproccessor.resize_img(erosion, cnts_rects_norm[cnt_rect_i])
            if symbol_img is None:
                result += '_'
                continue

            symbs_data.append(symbol_img)
            prep_img_arr = NNModel.img_preprocessing(symbol_img)

            Model = LettersModel
            if cnt_rect_i > 0:
                Model = DigitsModel

            symbol, accuracy = Model.recognize(prep_img_arr)
            print(f'Predicted symbol: ==> {symbol}, Accuracy:', accuracy)
            result += str(symbol)

        save_img_data(CVPreproccessor.image, self.scan_path, result, symbs_data, scans_dir=scans_dir)

        return result    



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


# Recongnizing scans from directory with folders and images
# распознавание из папки с классами
def recognize_folder_scans(folder_path, mask_range):
    scans = scans_from_folder(folder_path)

    for scan_path in scans:
        Recognizer = SymbolsRecognizer(scan_path, mask_range=mask_range)
        Recognizer.recognize_scan()


# Denotes config file with HSV values
# Размечает конфиг файл (указать для каждой папки значения HSV)
def denote_dirs_config(folder, config_file):
    for file_name in os.listdir(folder):

        if os.path.isdir(f'{folder}/' + file_name):
            samples = os.listdir(f'{folder}/' + file_name)

            hsv_values = 0
            if len(samples) > 0:
                test_sample = f'{folder}/{file_name}/{samples[0]}'
            
                CVPreproccessor = CVMethods(test_sample)
                sample_rect = CVPreproccessor.student_num_rect()
                
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
        print('Error in config.')
    
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

                    rec_symbs_path = f'{folder}/recognized symbols'
                    if not os.path.exists(rec_symbs_path):
                        os.mkdir(rec_symbs_path)

                    shutil.move(f'{path}/{info}', rec_symbs_path)



# Recongnition with config file example
# Пример распознавания с конфиг файлом, но можно и без него, указав limits самостоятельно
def main():
    file_path = 'scans_26_08_13_30/200/200001.jpg'
    folder_path = '/'.join(file_path.split('/')[:2])
    # image = cv2.imread(file_path)
    # rect = student_num_rect(image)
    # limits = run_mask_bar(rect)

    # with open('config_f.txt', 'w') as test_file:
    #     values = ', '.join([str(i) for i in limits])
    #     test_file.write(f'{folder_path}: {values}')

    recognition_with_config('config_f.txt')
    norm_edited_results(folder_path.split('/')[-1])



if __name__ == "__main__":
    main()