from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
df=pd.read_csv("Water Quality Testing (1).csv")
df=df.dropna()
x=df.iloc[:,0:5]
y=df.iloc[:,5:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            int_features = [float(x) for x in request.form.values()]
            final = [np.array(int_features)]
            user_input = scaler.transform(final)
            print(int_features)
            print(final)

            prediction = model.predict(user_input)
            print(prediction)

            if prediction[0] == 1:
                return render_template('index.html', pred='The water is safe for consumption.')
            else:
                return render_template('index.html', pred='The water is unsafe for consumption.')
        except Exception as e:
            return render_template('index.html', pred=f'An error occurred: {str(e)}')
    else:
        return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)
