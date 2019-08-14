import TestLoad
import Vectorizer
from sklearn.externals import joblib
import Training
from torch.utils.data import DataLoader, Dataset
import AnswerGen
import PyQt5
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.ui = uic.loadUi('panel.ui',self)
        self.Listener_set()
        # self.ui.label2.setText("loading...")
        #querys = DataLoader(Vectorizer.qTrainSet(querys),batch_size = 1,
            #shuffle = False, num_workers = 8)
        #docs = DataLoader(Vectorizer.docTrainSet(docs),batch_size = 1,
            #shuffle = False, num_workers = 8)
        # self.ui.label2.setText("load complete")
    def Listener_set(self):
        self.ui.probBtn.clicked.connect(self.randProb)
        self.ui.searchBtn.clicked.connect(self.search)
        self.ui.StaBtn.clicked.connect(self.showstatus)
        self.ui.valBtn.clicked.connect(self.val)
        self.ui.reloadBtn.clicked.connect(self.reload_)
        self.ui.trainBtn.clicked.connect(self.train)
    def randProb(self):
        query = querys[np.random.randint(0,len(querys))]['q']
        self.ui.queryArea.setText(query)
    def search(self):
        ans , s = ansGen(self.ui.queryArea.toPlainText())
        self.ui.ansArea.setPlainText(ans)
    def showstatus(self):
        ans = str(len(querys))
        self.ui.label2.setText("QueryNumber:" + ans)
    def val(self):
        num = np.random.randint(0,len(querys))
        query = querys[num]['q']
        self.ui.queryArea.setText(query)
        ans, s = ansGen(self.ui.queryArea.toPlainText())
        self.ui.ansArea.setPlainText(ans + "\n" + "score:" + str(s) + "\n" + "CorrectAns: " + docs[num]['ans']['text'] + " (" + docs[num]['plainText'])
    def reload_(self):
        loadTrainingData1()
        self.ui.label2.setText("Load Complete, QueryNumber:" + str(len(querys)))
    def train(self):
        Training.training(querys,docs)


def loadTrainingData1():
    global querys, docs
    querys , docs = TestLoad.testLoad()
    querys = Vectorizer.batchQVec(querys)
    docs = Vectorizer.batchDVec(docs)
    Vectorizer.saveVecs(querys,docs)
    return querys, docs

def loadDataFromTestFile():
    querys = joblib.load("TestQuerys.pkl")
    docs = joblib.load("TestDocs.pkl")
    return querys,docs

def training():
    Training.training(querys,docs)

def ansGen(text):
    return AnswerGen.answerGen(text,docs)

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mWin = MainWindow()
    mWin.show()
    sys.exit(app.exec_())

querys,docs = loadDataFromTestFile()

if __name__ == "__main__":
    main()