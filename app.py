from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
import joblib
import os

modelfile = 'static/diabmodel/random_forst_model_Nirav.pkl'
label_dict = {"No":0,"Yes":1}
gender_map = {"Female":0,"Male":1}
target_label_map = {"Negative":0,"Positive":1}

def get_fvalue(val):
	feature_dict = {"No":0,"Yes":1}
	for key,value in feature_dict.items():
		if val == key:
			return value

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value

'''Loaded machine learning Model'''
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model




app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/api', methods=['POST'])
def DiabPredict():
    encoded_array = []

    DiabModle = load_model(modelfile)
    data = request.get_json()
    for i in data.values():
        if type(i) == int:
            encoded_array.append(i)
        elif i in ["Female", "Male"]:
            res = get_value(i, gender_map)
            encoded_array.append(res)
        else:
            encoded_array.append(get_fvalue(i))
        # debug info
    print(encoded_array)
    ml_input = np.array(encoded_array).reshape(1, -1)
    print(ml_input)
    print(type(ml_input))
    prediction = DiabModle.predict(ml_input)
    print(prediction)
    predict_probability = DiabModle.predict_proba(ml_input)
    print(predict_probability)
    if prediction == 1:
        print("Positive Risk-{}".format(prediction[0]))
        pred_probability_score = {"Negative DM": predict_probability[0][0] * 100,"Positive DM": predict_probability[0][1] * 100}
        print(pred_probability_score)
        return  jsonify(pred_probability_score)
    else:
        print("Negative Risk-{}".format(prediction[0]))
        pred_probability_score = {"Negative DM": predict_probability[0][0] * 100,"Positive DM": predict_probability[0][1] * 100}
        print(pred_probability_score)
        return jsonify(pred_probability_score)




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5005)
