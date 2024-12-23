from flask import Flask, render_template, request, redirect, url_for



## Create a Simple flask Application
app = Flask(__name__)

## Flask app routing 

## Routing to Welcome (Main) Page http://127.0.0.1:5000
@app.route("/", methods=['GET'])
def welcome():
    return render_template('welcome.html') ## Routing to Main Page

## Routing to Index Page http://127.0.0.1:5000/index
@app.route("/index", methods=['GET']) 
def index():
    return "This is Index Page "

## Routing with (Parameterization) also known as Variable Rule

## Routing to success page http://127.0.0.1:5000/success/55
@app.route("/success/<int:score>")   
def success(score):
    return f"The person has passed and score is: {score}" 

## Routing to fail page http://127.0.0.1:5000/fail/55
@app.route("/fail/<int:score>")    ## Routing to fail page 
def fail(score):
    return f"The person has failed and score is: {score}"   



## GET AND POST EXAMPLE

## Renders HTML page and performs GET AND POST Operation
@app.route('/calculate',methods=["GET","POST"])
def calculate():
    if request.method =="GET":
        return render_template ('calculate.html')
    else:
        maths = float(request.form['maths'])
        science = float(request.form['science'])
        history = float(request.form['history'])

        average_marks = (maths+science+history)/3
        result = ""
        if average_marks >=250:
            result = "success"
        else:
            result = "fail"
        return redirect(url_for(result,score=average_marks))

        # return render_template('calculate.html', score = round(average_marks,2))


if __name__=="__main__":
    app.run(debug=True)