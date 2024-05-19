import sys 
from PyQt5.QtWidgets import QApplication
# from setting_UI import setting
from main_UI import main_ui

if __name__ == '__main__': 
   app = QApplication(sys.argv) 
   sd = main_ui() 
#    pb = setting() 
#    sd.make_connection(pb)

   sys.exit(app.exec_())