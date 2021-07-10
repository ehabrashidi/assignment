import joblib
from helpers.dummies import *
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load('models/model.h5')
scaler = joblib.load('models/scaler.h5')



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/predict', methods=['GET'])
def predict():
    all_data = request.args
    Age = int(all_data['Age'])
    job = job_dummies[all_data['job']]
    education = education_dummies[all_data['education']]
    duration = int(all_data['duration'])
    previous = float(all_data['previous'])
    poutcome = float(all_data['poutcome'])
    cons = float(all_data['cons.conf.idx'])
    marital = marital_dummies[all_data['marital']]
    credit = int(all_data['credit'])
    housing = int(all_data['housing'])
    loan = int(all_data['loan'])
    
    #weekday = weekdays_dummies[all_data['weekday']]
    #pod = pod_dummies[all_data['peroid_of_day']]
    # humidity = float(all_data['humidity'])
    #month = int(all_data['datetime'].split('-')[1])
    #hour = int(all_data['datetime'].split('T')[1].split(':')[0])
    #is_rush_hour = int(all_data['is_rush_hour'])
    x = [Age, loan, housing, marital,duration,previous,poutcome,cons,credit,housing,loan]
    x += job + education 


    x = scaler.transform([x])
    bikes_count = round(model.predict(x)[0])

    return render_template('prediction.html', bikes_count=bikes_count)


if __name__ == "__main__":
    app.run()