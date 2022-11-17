from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
import os

modeldir = os.path.join(os.path.dirname(__file__), 'model/lr_model_predict.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'model/scaler.pkl')

model = pickle.load(open(modeldir,'rb'))
scaler = joblib.load(scaler_path)

app = Flask(__name__, static_folder='static')

@app.route('/')
def main():
    return(render_template('main.html'))

@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]

    x = scaler.transform(final_features)

    prediction = model.predict(x)

    output = {0.0:'Tidak Akan Hujan', 1.0:'Akan Hujan'}
    return render_template('main.html',prediction_text='Kemungkinan Besok {}'.format(output[prediction[0]]))

if __name__=='__main__':
    app.run()