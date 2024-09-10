import pickle
from flask import Flask, render_template, request,jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # type: ignore

application = Flask(__name__)
app = application

#import ridge regressor and standardscaler
ridge_model = pickle.load(open('models/ridge_Algeria.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler_Algeria.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data = standard_scaler.transform([[Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data)
        return render_template('home.html',results = result[0])
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(debug=True)