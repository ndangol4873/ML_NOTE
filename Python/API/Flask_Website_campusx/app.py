from flask import Flask, render_template, request
from db_json import Database


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

    user = f"{name} {email} {password}"
    return user


if __name__=="__main__":
    app.run(debug=True)