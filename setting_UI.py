from PyQt5.QtWidgets import QSlider, QDialog, QLabel, QFileDialog
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5 import uic
import json
from collections import OrderedDict

from_setting = uic.loadUiType('./setting.ui')[0]

class setting(QDialog, from_setting): 
    image_path = pyqtSignal(str)
    model_path = pyqtSignal(str)     # 발생하는 int type 시그널을 저장하는 시그널 객체 

    def __init__(self): 
        super(setting, self).__init__() 
        self.setupUi(self)
        
        # 설정 저장 버튼
        self.setting_save_btn.clicked.connect(self.setting_save)
        # 이미지 저장 위치 경로
        self.image_save_path_btn.clicked.connect(self.change_image_path)
        # 모델 경로 버튼
        self.model_path_btn.clicked.connect(self.change_model_path)
        
        # self.image_path_label.setText('./image_save/')
        # self.model_path_label.setText('./save_model/03-25-2024_01-49-24/checkpoint.h5')
        self.folder_name = ''
        self.model_name = ''
        
        with open('./setting_files/settings.json', 'r') as f:
            setting_data = json.load(f)
            
        self.folder_path = setting_data['image_path']
        self.model_path = setting_data['model_path']
        self.image_path_label.setText(self.folder_path)
        self.model_path_label.setText(self.model_path)
        
        self.folder_name = self.folder_path
        self.model_name = self.model_path
        
        self.init_ui() 

    def init_ui(self): 
        self.show()
        
    # 설정 저장하기
    def setting_save(self):
        # 해당 설정 보내기
        # self.image_path.emit(self.folder_name)
        # self.model_path.emit(self.model_name)
        
        # 설정 json 파일로 저장하기
        file_data = OrderedDict()
        file_data['name'] = 'settings'
        file_data['image_path'] = self.folder_name
        file_data['model_path'] = self.model_name
        
        with open('./setting_files/settings.json', 'w', encoding='utf-8') as make_file:
            json.dump(file_data, make_file, ensure_ascii=False, indent='\t')
        self.close()
        
    # 이미지 경로 변경
    def change_image_path(self):
        self.folder_name = QFileDialog.getExistingDirectory(self, '폴더 선택', './')
        # self.folder_name  = '.' + self.folder_name
        if self.folder_name:
            self.image_path_label.setText(self.folder_name)
        else:
            self.image_path_label.setText('./hair_color_change/image_save')
            self.folder_name = self.image_path
    
    # 모델 변경
    def change_model_path(self):
        self.model_name = QFileDialog.getOpenFileName(self, '파일 선택', '', 'ALL Files(*);; tensorflow file(*.h5 *.hdf5)')[0]
        print(self.model_name)
        if self.model_name:
            self.model_path_label.setText(self.model_name)
        else:
            self.model_path_label.setText('./hair_color_change/save_model/03-25-2024_01-49-24/checkpoint.h5')
            self.model_name = self.model_path