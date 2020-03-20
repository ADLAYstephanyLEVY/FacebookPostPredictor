#importing libraries
import os
import numpy as np
import pandas as pd
import flask
import pickle
#import waitress /// waitress.serve(app, host='0.0.0.0', port=8890)
from flask import Flask, render_template, request, url_for
from werkzeug import secure_filename
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.externals import joblib
from decimal import *

app = Flask(__name__)

@app.route('/')
@app.route('/index')


def index():
    return flask.render_template('index.html')
 
def LogisticReg(xTrain, xTest, yTrain, yTest):
   #################### Logistic Regression ########################
   logReg = LogisticRegression()
   fited = logReg.fit(xTrain, yTrain)
   score = logReg.score(xTrain, yTrain)
   LRpredict = logReg.predict(xTest)
   return("Logistic Regression Accuracy:", metrics.accuracy_score(yTest, LRpredict))

def KNNeighbors(xTrain, xTest, yTrain, yTest):
   #################### KNN ########################
   knn = KNeighborsClassifier(n_neighbors = 3)
   fited = knn.fit(xTrain, yTrain)
   score = knn.score(xTrain, yTrain)
   KNNpredict = knn.predict(xTest)
   return("KNN Accuracy:", metrics.accuracy_score(yTest, KNNpredict))

def SuperVectorMachine(xTrain, xTest, yTrain, yTest):
   #################### Support Vector Machines ########################
   svmachine = SVC()
   fited = svmachine.fit(xTrain, yTrain)
   score = svmachine.score(xTrain, yTrain)
   SVMpredict = svmachine.predict(xTest)
   return("SVM Accuracy:", metrics.accuracy_score(yTest, SVMpredict))

def DecisionTree(xTrain, xTest, yTrain, yTest):
   #################### Decision Tree ########################
   dTree = tree.DecisionTreeClassifier()
   fited = dTree.fit(xTrain, yTrain)
   score = dTree.score(xTrain, yTrain)
   DTpredict = dTree.predict(xTest)
   return("Decision Tree Accuracy:", metrics.accuracy_score(yTest, DTpredict))
       
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      ###CARGAR ARCHIVO
      fileLoaded = request.files['file']
      fileLoaded.save(secure_filename(fileLoaded.filename))
      ###OBTENER INFO DEL ARCHIVO
      fileSaved = fileLoaded.filename
      data = pd.read_csv(fileSaved)
      ###MATRICES
      arrX = data[data.columns[:-1]].values
      arrY = data[data.columns[-1]].values
      X_train, X_test, Y_train, Y_test = train_test_split(arrX, arrY, test_size=0.2, random_state=None)
      ###PREDICTION
      LRpred = LogisticReg(X_train, X_test, Y_train, Y_test)
      KNNpred = KNNeighbors(X_train, X_test, Y_train, Y_test)
      SVMpred = SuperVectorMachine(X_train, X_test, Y_train, Y_test)
      DTpred = DecisionTree(X_train, X_test, Y_train, Y_test)
      ###CHOOSE THE BEST PREDICTOR
      print('LR:',LRpred, 'KNN:',KNNpred, 'SVM:',SVMpred, 'DT:',DTpred)
      largest = [LRpred, KNNpred, SVMpred, DTpred]
      sor = sorted(largest)
      bestPredictor = max(sor)
      print(bestPredictor)
      
      ###SAVE MODEL WITH PICKLE
      pickle.dump(bestPredictor, open('predictor.sav', 'wb'))
      joblib.dump(bestPredictor, 'model.pkl') 
      ### se the function that makes the prediction
      return flask.render_template('uploaded.html', bestPredictor=bestPredictor)

@app.route('/formulario', methods = ['GET','POST'])
def prediction():
   if request.method == 'POST':
      to_predict_list = request.form.to_dict()
      to_predict_list = list(to_predict_list.values())
      to_predict_list = list(map(int, to_predict_list))
      print("Predict list: ", to_predict_list)
      to_predict = np.array(to_predict_list).reshape(1,6)
      print("Reshaped list: ", to_predict)
      loaded_model = pickle.load(open("predictor.sav", "rb"))
      print("Loaded model: ", loaded_model)
      #PREDICTION WITH PICKLE 
      #result = loaded_model.predict(to_predict)
      #print("Result: ", result)
      #PREDICTION WITH JOBLIB
      pipe = joblib.load('model.pkl')
      print("PIPE: ", pipe)
      pred = pd.Series(pipe.predict(to_predict_list))
      print("JOBLIB: ", pred)
      # print("to_predict_list:",to_predict_list)
      # result = valuePredictor(to_predict_list)
      # print("result:",result)
      # if int(result)==0:
      #        post_predict = 'Es bajo'
      # else:
      #        post_predict = 'No es bajo'
             
   return render_template('predict.html')

if __name__ == '__main__':
   app.run(debug = True)





