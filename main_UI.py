from PyQt5 import uic, QtGui
from PyQt5.QtWidgets import QFileDialog, QColorDialog, QDialog, QMessageBox
from PyQt5.QtWidgets import *
import numpy as np
import cv2
from PIL import Image
import math
from keras.models import load_model
from setting_UI import setting
import os
from keras import *
import urllib.request
from glob import glob
import json

# design 연결
from_class = uic.loadUiType('./design2.ui')[0]

class main_ui(QDialog, from_class):
    def __init__(self):
        super(main_ui, self).__init__()
        self.setupUi(self)
        
        # 이미지 합성 버튼 클릭 이벤트
        self.image_change_btn.clicked.connect(self.image_change)
        # 이미지 경로 버튼
        self.image_path_btn.clicked.connect(self.image_path)
        
        # 다른 색 선택 버튼
        self.another_color_btn.clicked.connect(self.another_color)
        
        # 설정장 버튼
        self.setting_btn.clicked.connect(self.setting_window)
        
        # 업로드 버튼
        self.upload_btn.clicked.connect(self.image_upload)
        
        # 색 추가 버튼
        self.add_color_btn.clicked.connect(self.add_color)
        
        self.another_color = ''
        self.select_color()
        
        self.init_ui()
        
        # 모델 설정
        # self.model = load_model('./save_model/03-25-2024_01-49-24/checkpoint.h5')
        # # faces 설정
        # self.faces_detection = self.img_cascade(self.image)
        
        # 이미지 저장 경로
        self.image_save_path = './image_save/'
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)   
            
        # 이미지 다운로드 경로
        self.download_path = os.path.join(self.image_save_path,'download_img/')
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)
    
    def init_ui(self):
        self.show()
    
    # 이미지 합성 버튼 클릭
    def image_change(self):
        print('image_chage 클릭')
        
        
        
        with open('./setting_files/settings.json', 'r') as f:
            setting_data = json.load(f)
            
        self.image_save_path = setting_data['image_path']
        self.model_path = setting_data['model_path']
        
        if self.model_path.endswith('.h5') or self.model_path.endswith('.hdf5'):
            self.model = load_model(self.model_path)
        else:
            QMessageBox.about(self,'Error','모델 파일 오류')
            return
        
        # 콤보박스에 있는 색 가져오기
        color_text = self.color_list.currentText()
        color = self.color_dict[self.color_list.currentText()]
        print(color_text, color)
        
        self.color_rgb.setText(str(color))
        
        # 얼굴 좌표 가져오기
        # self.faces_detection = self.img_cascade(self.im)
        
        if self.image_path_text.text().startswith('http'):
            image_path = sorted(glob(self.download_path+'*'))[-1]
            # print(sorted(glob(self.download_path+'*')))
            img = cv2.imread(image_path)
        else:
            img = cv2.imread(self.image_path_text.text())
        
        # 내가 원하는 색을 적을 시
        if color_text == 'other':
            # print(self.color_rgb.text())
            # another_color = self.another_color()
            change_color_img = self.Change_hair_color(img, self.another_color)
        else:
            # print(print(color))
            change_color_img = self.Change_hair_color(img, color)
        
        self.out_image(change_color_img)
        
    # 색 선택하기
    def select_color(self):
        self.color_dict = {}
        with open('./setting_files/color_list.json', 'r') as f:
            color_data = json.load(f)
            
        for color_name, color_code in color_data.items():
            self.color_list.addItem(color_name)
            self.color_dict[color_name] = tuple(color_code)
            
        # # 콤보박스 연결
        # self.color_list.addItem('Red')
        # self.color_list.addItem('Yellow')
        # self.color_list.addItem('Cobalt Bule')
        # self.color_list.addItem('other')
        
        # #BGR
        # self.color_dict = {
        #     'Cobalt Bule': (140, 73, 0),
        #     'other': 'other',
        #     'Red': (0, 0, 255),
        #     'Yellow': (0, 255, 255)
        # }
        
    # 비율만큼 늘리기
    def ratio_size(self, h, w):
        print(f'가로: {w}, 세로: {h}')
        
        max_size = 700
        if h > w:
            flag = 'h'
        else:
            flag = 'w'
            
        count = 1
        while True:
            if flag == 'h':
                if h * count < max_size:
                    count += 1
                else:
                    break
            else:
                if w * count < max_size:
                    count += 1
                else:
                    break
               
        new_width = count * w
        new_height = count * h
        
        if count > 2:
            if (new_width > max_size) or (new_height > max_size):
                new_width = (count - 1) * w
                new_height = (count - 1) * h
                
        return (new_width, new_height)

    # 비율 구하는 함수
    def simplify_ratio(self, width, height):
        gcd_value = math.gcd(width, height)
        simplified_width = width // gcd_value
        simplified_height = height // gcd_value
        
        return simplified_width, simplified_height
       
    # 이미지 출력
    def out_image(self, img):
        
        h, w, c = img.shape
        
        print(f'이미지 크기: h={h}, w={w}')
        
        ratio_w, ratio_h = self.simplify_ratio(w, h)
        print(f'비율= {ratio_w}:{ratio_h}')
        

        new_size = self.ratio_size(h=ratio_h, w=ratio_w)
        
        print(f'바꾼 이미지 크기: h={new_size[0]}, w={new_size[1]}')

        img = cv2.resize(img, new_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(img)
        
        canvas = Image.new('RGB', (700, 700), color='white')
        
            # 600 - 464 = 136 = 68
        x = (700 - new_size[1]) // 2    # width
            # 600 - 570 = 30 = 15
        y = (700 - new_size[0]) // 2    # height
        
        canvas.paste(img, (y,x))
        
        canvas = np.array(canvas)
        
        h, w, c = canvas.shape
        qImg = QtGui.QImage(canvas.data, w, h, w*c, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.image.setPixmap(pixmap)
        self.image.resize(pixmap.width(), pixmap.height())
        
    # 이미지 경로 입력
    def image_path(self):
        file_path = QFileDialog.getOpenFileName(self, '파일 선택', '', 'ALL Files(*);; Image File(*.jpg *.png)')[0]
        # 이미지 경로 텍스트
        self.image_path_text.setText(file_path)
        # img = cv2.imread(file_path[0])
        
        # self.out_image(img=img)
        
    # 마스크 추출
    def predict(self, image, height=224, width=224):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # im = image

        """Preprocess the input image before prediction"""
        im = im / 255
        im = cv2.resize(im, (height, width))
        im = im.reshape((1,) + im.shape)
        
        pred = self.model.predict(im)
        
        mask = pred.reshape((224, 224))

        return mask
    
    # 얼굴 위치 특정
    def img_cascade(self, image):
        face_casecade = cv2.CascadeClassifier('./save_model/haarcascade_frontalface_default.xml')
        faces = face_casecade.detectMultiScale(image, 1.1, 5)

        for (left, top, right, bottom) in faces:
        
            # 영역 키우기
            size = 200
            bottom = min(top + bottom + size, image.shape[0])
            right = min(left + right + size, image.shape[1])
            top = max(top - size, 0)
            left = max(left - size, 0)
            
            detection_img = image[top:bottom, left:right]
            
            return detection_img
    
    # 머리 색 바꾸기
    def Change_hair_color(self, image, color):
        # image = cv2.imread(image_path)
        mask = self.predict(image)
        thresh = 0.7  # Threshold used on mask pixels

        """Create 3 copies of the mask, one for each color channel"""
        blue_mask = mask.copy()
        blue_mask[mask > thresh] = color[0]
        blue_mask[mask <= thresh] = 0

        green_mask = mask.copy()
        green_mask[mask > thresh] = color[1]
        green_mask[mask <= thresh] = 0

        red_mask = mask.copy()
        red_mask[mask > thresh] = color[2]
        red_mask[mask <= thresh] = 0

        blue_mask = cv2.resize(blue_mask, (image.shape[1], image.shape[0]))
        green_mask = cv2.resize(green_mask, (image.shape[1], image.shape[0]))
        red_mask = cv2.resize(red_mask, (image.shape[1], image.shape[0]))

        """Create an rgb mask to superimpose on the image"""
        mask_n = np.zeros_like(image)
        mask_n[:, :, 0] = blue_mask
        mask_n[:, :, 1] = green_mask
        mask_n[:, :, 2] = red_mask
        
        # alpha = self.belnding_slider.value() * 0.01
        alpha = 0.9
        # alpha = 0.85
        beta = (1.0 - alpha)
        out = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)

        # name = 'test/results/' + img.split('/', 1)[0]
        
        image_count = len(os.listdir(self.image_save_path))
        file_path = os.path.join(self.image_save_path, str(image_count+1).zfill(10)+'.png')
        print(file_path)
        cv2.imwrite(file_path, out)
        
        return out
    
    # 다른 색 선택
    def another_color(self):
        col = QColorDialog.getColor()
        rgb_color = self.hax_to_rgb(col.name())
        print(rgb_color)
        
        self.color_rgb.setText(str(rgb_color))
        self.another_color = (rgb_color[2], rgb_color[1], rgb_color[0])
        # return rgb_color
        
    # hax값 -> rgb값으로 변
    def hax_to_rgb(self, color):
        # 16진수 값에서 '#' 기호를 제거합니다.
        hex_color = color.lstrip('#')
        
        # 16진수 값을 정수로 변환합니다.
        rgb_int = int(hex_color, 16)
        
        # 각 RGB 값을 계산합니다.
        r = (rgb_int >> 16) & 0xFF
        g = (rgb_int >> 8) & 0xFF
        b = rgb_int & 0xFF
        
        return (r, g, b)
    
    # 설정창 열기
    def setting_window(self):
        self.setting = setting()
        self.make_connection(self.setting)
        self.setting.exec_()

    # 연결하기  
    def make_connection(self, setting_dialog):
        # # 이미지 경로 가져오기
        # setting_dialog.image_path.connect(self.change_image_path)
        # # 모델 경로 가져오기
        # setting_dialog.model_path.connect(self.change_model_path)
        pass
  
    # 이미지 저장 경로 설정하기
    # @pyqtSlot(str)
    # def change_image_path(self, val):
    #     self.image_save_path = val
    #     # print(val)
        
    # # 모델 경로 설정하기
    # @pyqtSlot(str)
    # def change_model_path(self, val):
    #     self.model = load_model(val)
    #     print(val)
        
    # 업로드 실행
    def image_upload(self):
        # http로 시작할 때
        if self.image_path_text.text().startswith('http'):
            # 이미지 확장자 판별
            if any([self.image_path_text.text().endswith(ext) for ext in ['webp', 'gif']]):
                QMessageBox.about(self,'Error','이미지 확장자 오류')
                return
            file_path = self.download_path+str(len(os.listdir(self.download_path))).zfill(10)+'.png'
            
            urllib.request.urlretrieve(self.image_path_text.text(), file_path)
            image = Image.open(file_path)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            self.out_image(img=image)
        # 이미지가 로컬에 있을 때
        elif self.image_path_text.text().startswith('/home/'):
            # 이미지 확장자 판별
            if any([self.image_path_text.text().endswith(ext) for ext in ['webp', 'gif']]):
                QMessageBox.about(self,'Error','이미지 확장자 오류')
            img = cv2.imread(self.image_path_text.text())
            self.out_image(img=img)
        else:
            QMessageBox.about(self,'Error','이미지 경로 오류')
        
    # 색 추가하기
    def add_color(self):
        color_name = self.add_color_name.text()
        color_value = self.add_color_value.text()        
        color_rgb_value = color_value.replace(' ', '').split(',')
        
        if any([i.isalpha() for i in color_rgb_value]):
            QMessageBox.about(self,'Error','색상 코드 오류')
            return
        
        color_bgr = tuple(int(c) for c in reversed(color_rgb_value))
        
        
        with open('./setting_files/color_list.json', 'r') as f:
            setting_data = json.load(f)
            
            color_list = []
            for c_name, c_value in setting_data.items():
                color_list.append([c_name, tuple(c_value)])
        color_list = color_list[:-1]
        
        color_list.append([color_name, color_bgr])
        color_list.append(['other', (0, 0, 0)])
        
        color_dict = {}
        for c in color_list:
            color_dict[c[0]] = c[1]

        with open('./setting_files/color_list.json', 'w', encoding='utf-8') as f:
            json.dump(color_dict, f, ensure_ascii=False, indent='\t')
            
        self.color_list.clear()
        self.select_color()
        
        
# class main_ui(QMainWindow, from_class):
#     def __init__(self):
#         super().__init__()
#         self.setupUi(self)
#         # 이미지 합성 버튼 클릭 이벤트
#         self.image_change_btn.clicked.connect(self.image_change)
#         # 이미지 경로 버튼
#         self.image_path_btn.clicked.connect(self.image_path)
        
#         # 다른 색 선택 버튼
#         self.another_color_btn.clicked.connect(self.another_color)
        
#         # 설정장 버튼
#         self.setting_btn.clicked.connect(self.setting_window)
        
#         self.another_color = ''
#         self.select_color()
        
#         # self.model = load_model('./save_model/03-25-2024_01-49-24/checkpoint.h5')
#         # self.faces_detection = self.img_cascade(self.image)
        
#     # 이미지 합성 버튼 클릭
#     def image_change(self):
#         print('image_chage 클릭')
        
#         # 콤보박스에 있는 색 가져오기
#         color_text = self.color_list.currentText()
#         color = self.color_dict[self.color_list.currentText()]
        
#         # 얼굴 좌표 가져오기
#         # self.faces_detection = self.img_cascade(self.im)
        
#         # 내가 원하는 색을 적을 시
#         if color_text == 'other':
#             # print(self.color_rgb.text())
#             # another_color = self.another_color()
#             change_color_img = self.Change_hair_color(self.image_path_text.text(), self.another_color)
#         else:
#             # print(print(color))
#             change_color_img = self.Change_hair_color(self.image_path_text.text(), color)
        
#         self.out_image(change_color_img)
        
#     # 이미지 저장 버튼 클릭
#     # def image_save(self):
#     #     print('image_save 버튼 클릭')
        
#     def select_color(self):
#         # 콤보박스 연결
#         self.color_list.addItem('Red')
#         self.color_list.addItem('Yellow')
#         self.color_list.addItem('Cobalt Bule')
#         self.color_list.addItem('other')
        
#         #BGR
#         self.color_dict = {
#             'Cobalt Bule': (140, 73, 0),
#             'other': 'other',
#             'Red': (0, 0, 255),
#             'Yellow': (0, 255, 255)
#         }
        
#     # 비율만큼 늘리기
#     def ratio_size(self, h, w):
#         print(f'가로: {w}, 세로: {h}')
        
#         max_size = 700
#         if h > w:
#             flag = 'h'
#         else:
#             flag = 'w'
            
#         count = 1
#         while True:
#             if flag == 'h':
#                 if h * count < max_size:
#                     count += 1
#                 else:
#                     break
#             else:
#                 if w * count < max_size:
#                     count += 1
#                 else:
#                     break
               
#         new_width = count * w
#         new_height = count * h
        
#         if count > 2:
#             if (new_width > max_size) or (new_height > max_size):
#                 new_width = (count - 1) * w
#                 new_height = (count - 1) * h
                
#         return (new_width, new_height)

#     # 비율 구하는 함수
#     def simplify_ratio(self, width, height):
#         gcd_value = math.gcd(width, height)
#         simplified_width = width // gcd_value
#         simplified_height = height // gcd_value
        
#         return simplified_width, simplified_height
       
#     # 이미지 출력
#     def out_image(self, img):
        
#         h, w, c = img.shape
        
#         print(f'이미지 크기: h={h}, w={w}')
        
#         ratio_w, ratio_h = self.simplify_ratio(w, h)
#         print(f'비율= {ratio_w}:{ratio_h}')
        

#         new_size = self.ratio_size(h=ratio_h, w=ratio_w)
        
#         print(f'바꾼 이미지 크기: h={new_size[0]}, w={new_size[1]}')

#         img = cv2.resize(img, new_size)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         img = Image.fromarray(img)
        
#         canvas = Image.new('RGB', (700, 700), color='white')
        
#             # 600 - 464 = 136 = 68
#         x = (700 - new_size[1]) // 2    # width
#             # 600 - 570 = 30 = 15
#         y = (700 - new_size[0]) // 2    # height
        
#         canvas.paste(img, (y,x))
        
#         canvas = np.array(canvas)
        
#         h, w, c = canvas.shape
#         qImg = QtGui.QImage(canvas.data, w, h, w*c, QtGui.QImage.Format_RGB888)
#         pixmap = QtGui.QPixmap.fromImage(qImg)
#         self.image.setPixmap(pixmap)
#         self.image.resize(pixmap.width(), pixmap.height())
       
#     # 이미지 경로 입력
#     def image_path(self):
#         file_path = QFileDialog.getOpenFileName(self, '파일 선택', '', 'ALL Files(*.jpg)')
#         # 이미지 경로 텍스트
#         self.image_path_text.setText(file_path[0])
        
#         img = cv2.imread(file_path[0])
        
#         self.out_image(img=img)
        
#     # 마스크 추출
#     def predict(self, image, height=224, width=224):
#         im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # im = image

#         """Preprocess the input image before prediction"""
#         im = im / 255
#         im = cv2.resize(im, (height, width))
#         im = im.reshape((1,) + im.shape)
        
#         pred = self.model.predict(im)
        
#         mask = pred.reshape((224, 224))

#         return mask
    
#     # 얼굴 위치 특정
#     def img_cascade(self, image):
#         face_casecade = cv2.CascadeClassifier('./save_model/haarcascade_frontalface_default.xml')
#         faces = face_casecade.detectMultiScale(image, 1.1, 5)

#         for (left, top, right, bottom) in faces:
        
#             # 영역 키우기
#             size = 200
#             bottom = min(top + bottom + size, image.shape[0])
#             right = min(left + right + size, image.shape[1])
#             top = max(top - size, 0)
#             left = max(left - size, 0)
            
#             detection_img = image[top:bottom, left:right]
            
#             return detection_img
    
#     # 머리 색 바꾸기
#     def Change_hair_color(self, image_path, color):
#         image = cv2.imread(image_path)
#         mask = self.predict(image)
#         thresh = 0.7  # Threshold used on mask pixels

#         """Create 3 copies of the mask, one for each color channel"""
#         blue_mask = mask.copy()
#         blue_mask[mask > thresh] = color[0]
#         blue_mask[mask <= thresh] = 0

#         green_mask = mask.copy()
#         green_mask[mask > thresh] = color[1]
#         green_mask[mask <= thresh] = 0

#         red_mask = mask.copy()
#         red_mask[mask > thresh] = color[2]
#         red_mask[mask <= thresh] = 0

#         blue_mask = cv2.resize(blue_mask, (image.shape[1], image.shape[0]))
#         green_mask = cv2.resize(green_mask, (image.shape[1], image.shape[0]))
#         red_mask = cv2.resize(red_mask, (image.shape[1], image.shape[0]))

#         """Create an rgb mask to superimpose on the image"""
#         mask_n = np.zeros_like(image)
#         mask_n[:, :, 0] = blue_mask
#         mask_n[:, :, 1] = green_mask
#         mask_n[:, :, 2] = red_mask
        
#         # alpha = self.belnding_slider.value() * 0.01
#         alpha = 0.9
#         # alpha = 0.85
#         beta = (1.0 - alpha)
#         out = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)

#         # name = 'test/results/' + img.split('/', 1)[0]
        
#         return out
    
#     # 다른 색 선택
#     def another_color(self):
#         col = QColorDialog.getColor()
#         rgb_color = self.hax_to_rgb(col.name())
#         # print(rgb_color)
        
#         self.color_rgb.setText(str(rgb_color))
#         self.another_color = (rgb_color[2], rgb_color[1], rgb_color[0])
#         # return rgb_color
        
#     # hax값 -> rgb값으로 변
#     def hax_to_rgb(self, color):
#         # 16진수 값에서 '#' 기호를 제거합니다.
#         hex_color = color.lstrip('#')
        
#         # 16진수 값을 정수로 변환합니다.
#         rgb_int = int(hex_color, 16)
        
#         # 각 RGB 값을 계산합니다.
#         r = (rgb_int >> 16) & 0xFF
#         g = (rgb_int >> 8) & 0xFF
#         b = rgb_int & 0xFF
        
#         return (r, g, b)
    
#     # 설정창 열기
#     def setting_window(self):
#         self.second = secondWindow()
#         self.second.exec()
#         self.show()
#         pass


    
# class secondWindow(QDialog, QWidget, from_setting):
#     def __init__(self):
#         super(secondWindow,self).__init__()
#         self.initUi()
#         self.show()
        
#         # 설정 저장 버튼
#         self.setting_save_btn.clicked.connect(self.setting_save)
#         # 이미지 저장 위치 경로
#         self.image_save_path_btn.clicked.connect(self.change_image_path)
#         # 모델 경로 버튼
#         self.model_path_btn.clicked.connect(self.change_model_path)

#     def initUi(self):
#         self.setupUi(self)
#         self.show()
        
#     # 설정 저장하기
#     def setting_save(self):
        
#         self.close()
#         pass
    
#     # 이미지 경로 변경
#     def change_image_path(self):
#         folder_name = QFileDialog.getExistingDirectory(self, '폴더 선택', '')[0]
#         self.image_path_label.setText(folder_name)
    
#     # 모델 변경
#     def change_model_path(self):
#         model_name = QFileDialog.getOpenFileName(self, '파일 선택', '', 'ALL Files(*.h5)')[0]
#         self.model_path_label.setText(model_name)
    
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     myWindow = main_ui()
#     myWindow.show()
#     app.exec_()


