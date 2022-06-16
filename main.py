import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

from flask import Flask, jsonify, make_response

app = Flask(__name__)

#load file
df = pd.read_csv("assets/ML_user_data_transformed.csv", engine="python", index_col=[0])
model = keras.models.load_model("assets/collaborative_model.h5")

# Create label encoder for translating value for model inputs
user_enc = LabelEncoder()
user_enc.fit(df['User_id'].values)

item_enc = LabelEncoder()
item_enc.fit(df['Item_id'].values)

def convertListfromInteger(identifier, list_int):
    converted = []
    for n in list_int:
        converted.append(identifier+'{0:06}'.format(n))
    return converted

@app.route("/", methods=["GET"])
def main():
    message = jsonify({
        "message": "Flask ML Deploy Main Route"
        })
    return make_response(message, 200)

@app.route("/<user_id>", methods=["GET"])
def predict(user_id):
    user_input = user_id
    
    # mengubah item id unique menjadi array hotel untuk input model 
    arr_hotel = item_enc.transform(df['Item_id'].unique())
    # membuat array user sejumlah hotel berdasarkan input dari BE
    arr_user = np.full(shape=len(arr_hotel), fill_value=(int(user_input[1:]) - 1), dtype=np.int64)

    preds = model.predict([tf.constant(arr_user),tf.constant(arr_hotel)])
    predictions = np.array([a[0] for a in preds])
    recommended_hotel_id = (-predictions).argsort()

    arr_output = []
    for item in recommended_hotel_id:
        arr_output.append(arr_hotel[item])

    output = convertListfromInteger('H', arr_output)

    message = jsonify({
        "message": "Predict success", 
        "userId": user_input,
        "output": output
        })
    return make_response(message, 200)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
