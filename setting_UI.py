from PyQt5.QtWidgets import QSlider, QDialog, QLabel, QFileDialog
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5 import uic

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
        
        self.init_ui() 

    def init_ui(self): 
        self.show()

    # def on_changed_value(self, val):     # 슬라이더를 움직일시 발생하는 시그널을    
    #     self.changedValue.emit(val)      # 준비된 시그널 객체에 보냅니다
        
    # 설정 저장하기
    def setting_save(self):
        self.image_path.emit(self.folder_name)
        self.model_path.emit(self.model_name)
        self.close()
        pass
    
    # 이미지 경로 변경
    def change_image_path(self):
        self.folder_name = QFileDialog.getExistingDirectory(self, '폴더 선택', '')[0]
        self.image_path_label.setText(self.folder_name)
    
    # 모델 변경
    def change_model_path(self):
        self.model_name = QFileDialog.getOpenFileName(self, '파일 선택', '', 'ALL Files(*.h5)')[0]
        self.model_path_label.setText(self.model_name)