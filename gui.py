import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
# from model import *
import pickle
import joblib
from celebrity_profiling_ngram_baseline import _preprocess_feed, inv_g_dict, inv_o_dict
age_model = pickle.load(open("./pretrained-models/age-model", 'rb'))
gender_model = pickle.load(open("./pretrained-models/gender-model", 'rb'))
occ_model = pickle.load(open("./pretrained-models/occ-model", 'rb'))
vec = joblib.load("./data/celeb-word-vectorizer.joblib")

def predict_age_gender_occupation(data):
    data = " <eotweet> ".join([ch for ch in data])
    data_featurized = vec.transform([data])
    age = age_model.predict(data_featurized)[0]
    gender = gender_model.predict(data_featurized)[0]
    occ = occ_model.predict(data_featurized)[0]

    return [age, inv_g_dict[gender], inv_o_dict[occ]]


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Celebrity profiling'
        self.left = 100
        self.top = 100
        self.width = 600
        self.height = 400
        self.initUI()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # create text
        self.static_textbox = QLabel(self)
        self.static_textbox.setText("Please enter a tweet")
        # self.static_textbox.setAlignment(Qt.AlignLeft)
        self.static_textbox.move(20, 20)
        self.static_textbox.resize(380,40)

        # Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 80)
        self.textbox.resize(380,40)

        # Create a button in the window
        self.button = QPushButton('Predict celebrity\'s gender, age and profession', self)
        self.button.move(20,140)
        self.button.resize(380,40)

        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.show()

    @pyqtSlot()
    def on_click(self):
        textboxValue = self.textbox.text()
        # gender, age, occupation = ["male", 1960, "sports"]
        age, gender, occupation = predict_age_gender_occupation(textboxValue)
        # print(age, gender, occupation)
        print_msg = "Age: {}  | Gender: {} | Profession: {}".format(age, gender, occupation)
        QMessageBox.question(self, 'Predicted traits', print_msg, QMessageBox.Ok, QMessageBox.Ok)
        self.textbox.setText("")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())