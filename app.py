import json
import os
from flask import Flask, jsonify, request
from model import classify

app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

@app.route('/')
def index():
    return "Intelligent Paint Shop Object Detection"

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
  data = json.loads(request.get_data())
  #print(data)
  if data == None:
    return 'Got None'
  else:
    image = eval(data["image"])
    prediction = classify.predict(image)

  return jsonify(prediction)

if __name__ == '__main__':
  app.run(port=port, host='0.0.0.0')
