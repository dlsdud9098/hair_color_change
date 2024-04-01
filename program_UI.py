import sys
from PyQt5 import uic, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import cv2
from PIL import Image
import math
from keras.models import load_model

# design 연결
from_class = uic.loadUiType('./design.ui')[0]

class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # 이미지 합성 버튼 클릭 이벤트
        self.image_change_btn.clicked.connect(self.image_chage)
        # 이미지 저장 버튼 클릭 이벤트
        self.image_save_btn.clicked.connect(self.image_save)
        # 이미지 경로 버튼
        self.image_path_btn.clicked.connect(self.image_path)
        
        
        # belnding 스크롤바 기존 수치 값 설정
        self.belnding_slider.setValue(85)
        # belnding 스크롤바 연결
        self.belnding_slider.valueChanged.connect(self.belnding)
        
        # 채도 스크롤바 기존 수치 값 설정
        self.saturation_slider.setValue(50)
        # 채도 스크롤바 연결
        self.saturation_slider.valueChanged.connect(self.saturation)
        
        # 콤보박스 연결
        self.color_list.addItem('Red')
        self.color_list.addItem('Yellow')
        self.color_list.addItem('Cobalt Bule')
        self.color_list.addItem('other')
        
        self.color_dict = {
            'Cobalt Bule': (140, 73, 0),
            'other': 'other',
            'Red': (0, 0, 255),
            'Yellow': (0, 255, 255)
        }
        
        self.model = load_model('./save_model/03-25-2024_01-49-24/checkpoint.h5')
        # self.faces_detection = self.img_cascade(self.image)
        
        
        
    # 이미지 합성 버튼 클릭
    def image_chage(self):
        print('image_chage 클릭')
        
        # 콤보박스에 있는 색 가져오기
        color_text = self.color_list.currentText()
        color = self.color_dict[self.color_list.currentText()]
        
        # 얼굴 좌표 가져오기
        # self.faces_detection = self.img_cascade(self.im)
        
        change_color_img = self.Change_hair_color(self.image_path_text.text(), color)
        
        self.out_image(change_color_img)
        
        # 내가 원하는 색을 적을 시
        if color_text == 'other':
            print(self.color_RGB.text())
        else:
            print(print(color))
        
    # 이미지 저장 버튼 클릭
    def image_save(self):
        print('image_save 버튼 클릭')
        
    # 블랜딩 슬라이더 연결
    def belnding(self, value):
        print('블랜딩 슬라이더 값', value)
        alpha = self.belnding_slider.value() // 100
        beta = (1.0 - alpha)
        # out = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)
        
    # 채도 슬라이더 연결
    def saturation(self, value):
        print('saturation 슬라이더 값', value)
        
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
        file_path = QFileDialog.getOpenFileName(self, '파일 선택', '', 'ALL Files(*.jpg)')
        # 이미지 경로 텍스트
        self.image_path_text.setText(file_path[0])
        
        img = cv2.imread(file_path[0])
        
        self.out_image(img=img)
        
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
    def Change_hair_color(self, image_path, color):
        image = cv2.imread(image_path)
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
        
        alpha = self.belnding_slider.value() * 0.01
        # alpha = 0.85
        beta = (1.0 - alpha)
        out = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)

        # name = 'test/results/' + img.split('/', 1)[0]
        
        return out
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()