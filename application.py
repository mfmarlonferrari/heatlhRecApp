import pickle
import flask
import os
import numpy as np

application = flask.Flask(__name__)

model = pickle.load(open("xgb_card_80auc.pkl","rb"))
   
@application.route('/predict', methods=['POST'])
def predict():

   features = flask.request.get_json(force=True)['features']
   data =  np.array([features])
   prediction = model.predict(data)
   probab = model.predict_proba(data)
   response = {'prediction': prediction.tolist(), 'probability':probab.tolist()}

   return flask.jsonify(response)

if __name__ == "__main__":
   application.run()
