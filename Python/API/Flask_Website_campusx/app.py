from flask import Flask, jsonify, render_template, request,redirect, session
from db_json import Database
from questions import QuestionAnswer
import json


dbo = Database()

## Create a Simple flask Application
app = Flask(__name__)


## Routing to Welcome (Main) Page http://127.0.0.1:5000
@app.route("/")
def welcome():
    return render_template('welcome.html') ## Routing to Main Page



## Routing to login Page http://127.0.0.1:5000/login
@app.route('/login') 
def login(): 
    return render_template('login.html')



## Routing to register Page http://127.0.0.1:5000/register
@app.route('/register')
def register():
    return render_template('register.html')


## Routing to register Page http://127.0.0.1:5000/perform_registration
@app.route('/perform_registration',methods=['post'])
def perform_registration():
    name = request.form.get('user_ka_name')
    email = request.form.get('user_ka_email')
    password = request.form.get('user_ka_password')
    
    response = dbo.insert(name, email, password)

    if response:
        return render_template('login.html',message="Registration Successful. Kindly login to proceed")
    else:
        return render_template('register.html',message="Email already exists")
    


## Routing to register Page http://127.0.0.1:5000/perform_login
@app.route('/perform_login', methods=['post'])
def perform_login():
    email = request.form.get('user_ka_email')
    password = request.form.get('user_ka_password')
    response = dbo.authenticate(email,password)
    if response:
        return redirect('/profile')
    else:
        return render_template ('login.html', message= "Incorrect Email / Password")
    

## Routing to register Page http://127.0.0.1:5000/profile
@app.route('/profile')
def profile():
    return render_template('profile.html')


## Routing to register Page http://127.0.0.1:5000/ner
@app.route('/ner')
def ner():
    return render_template ('/ner.html')

## Routing to register Page http://127.0.0.1:5000/perform_ner
@app.route('/perform_ner')
def perform_ner():
    pass


@app.route('/category_selection')
def category_selection():
    return render_template ('category.html')


@app.route('/display') 
def display():
    return render_template('display.html' )
    # return 'this is display'

## Routing to register Page http://127.0.0.1:5000/display_question 
@app.route('/get_question_answer', methods=['post'])
def get_question_answer():
    category = request.form.get('category')
    qa = QuestionAnswer()
    qa_dict = qa.get_question_answer(category)

    test = " "
    for k, v in qa_dict.items():
        test += f"{k} --> {v}<br/><br/>"
    return test
    # return render_template ('display.html', message = category)
 



@app.route('/calendar')
def calender():
    return 'this is calendar page'


if __name__=="__main__":
    app.run(debug=True)