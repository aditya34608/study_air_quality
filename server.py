        
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:53:46 2019

@author: Aditya Kumar
"""

from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np


with open('Picklefile.pkl','rb') as f:
    reg = pickle.load(f)

app = Flask(__name__)

@app.route("/home")
def home_page():
  return "<h1>Welcome to Walmart</h1>"

@app.route("/")
def view_template():
    return render_template("index.html")

@app.route("/data4", methods=["GET","POST"])
def form_data():
    if request.method == "GET":
        return "<h1>Sorry, You mistaken somewhere</h1>"
    else:
        user_data = request.form
        
        date = user_data['calender_date']
        
        date="2019-06-19"
        date_lst = list(str(date).split('-'))
        
        
        date_lst = [int(i) for i in date_lst ]
     
        month = date_lst[1]
        year = date_lst[2]
        date = date_lst[0]
#        store=12
#        item=10
#        weather=0
        #output = {"Store":store,"Item":item,"Weather":weather}
        
        input_data = [month,year,date]
        input_data = np.array(input_data)
        input_data = input_data.reshape(1,3)
        
        
        
        
        input_pred = reg.predict(input_data)
        input_pred = input_pred.round(2)
        result = input_pred[0][0]
        result_1 = input_pred[0][1]
        result_2 = input_pred[0][2] 
            
        #print("units :",input_pred[0])
        

        return jsonify(msg=str(result),msg2=str(result_1),msg3=str(result_2))
#        return jsonify(msg=str(result))
        #return render_template("user_info.html", info=result)



if __name__ == "__main__": 
    app.run(debug=True,port=8000)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    