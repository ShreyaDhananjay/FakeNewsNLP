from flask import render_template, url_for, request, redirect
from fakenews import app
import pandas as pd
from fakenews.clean import preprocess
from fakenews.prediction import predict

@app.route("/", methods = ['POST', 'GET'])
@app.route("/home",methods = ['POST', 'GET'])
def home():
    result = ""
    if request.method == 'POST':
        result = request.form
        l1 = []
        user_ex = []
        for key, value in result.items():
            if key != 's1':
                l1.append(value)
        user_ex.append(l1)
        user_ex = pd.DataFrame(user_ex)
        user_ex.rename(columns={0: 'title', 1: 'text'}, inplace=True)
        user_ex = preprocess(user_ex)
        user_ex.to_csv("fakenews/example4.csv")
        result = predict(user_ex)
    return render_template('home.html', result=result)
    
