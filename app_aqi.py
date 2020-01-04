import pickle
from flask import Flask,render_template,url_for,request
import pandas as pd


#load the pickle file
loaded_model= pickle.load(open('Randomforest_regressor.pkl', 'rb'))
## initialise flask class 
app = Flask(__name__)

#Function for what we get on the home page of the app
# here we will start the app
@app.route('/')  # this is the default home page of the app
def home():
    return render_template('home.html')
@app.route('/predict', methods =['POST'])  
# this is a post functionality, like what will happen when you will click the predict button
def predict():
    df=pd.read_csv('final_2018.csv')
    prediction= loaded_model.predict(df.iloc[:,:-1].values)  #.values to convert the df into array as predict works on arrays
    my_prediction = prediction.tolist() #converting results into a list
    return render_template('result.html', prediction = my_prediction)  # returning the results to the results file

if __name__ == '__main__':
    app.run(debug=True)