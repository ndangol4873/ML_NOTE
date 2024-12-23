from flask import Flask, request, jsonify

## Create a Simple flask Application
app = Flask(__name__)

## Creating FALSK API 
## http://127.0.0.1:5000/api
@app.route('/api',methods=["POST"])
def welcome_to_flask():
    data = request.get_json()
    name = data['name']
    greet = data['greet']
    age = data['age']
    return jsonify(f"{greet} {name} and i am {age} years old")

if __name__=='__main__':
    app.run(debug=True)