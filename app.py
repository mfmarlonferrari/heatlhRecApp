import pickle
import flask
import os
import numpy as np

app = flask.Flask(__name__)

model = pickle.load(open("xgb_card_80auc.pkl","rb"))
   
@app.route('/predict', methods=['POST'])
def predict():

   features = flask.request.get_json(force=True)['features']
   prediction = model.predict([features])
   probab = model.predict_proba([features])
   response = {'prediction': prediction.tolist(), 'probability':probab.tolist()}

   return flask.jsonify(response)

if __name__ == "__main__":
   app.debug = True
    app.run()
