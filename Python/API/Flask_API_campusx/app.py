from flask import Flask, jsonify, request 
import ipl


app = Flask(__name__)


@app.route('/')
def home():
    return 'Hello World'


@app.route('/api/teams_name')
def teams_name():
    teams = ipl.teamsNameAPI()
    return jsonify(teams)


@app.route('/api/teams_record')
def teams_record():
    team_name = request.args.get('team')
    response = ipl.teamsRecordsAPI(team_name)
    return response


@app.route('/api/batting_record')
def batting_record():
    batsman_name = request.args.get('batsman')
    response = ipl.batsmanAPI(batsman_name)
    return response
    

## /api/teamvteam?team1=Rajasthan Royals&team2=Royal Challengers Bangalore
@app.route('/api/teamvteam')
def teamvteam():
    team1 = request.args.get('team1')
    team2 = request.args.get('team2')

    response = ipl.teamVteamAPI(team1,team2)
    print(response)
    return jsonify(response)


## Balls API Test
@app.route('/api/balls')
def balls():
    response = ipl.ball_record_count()
    if response:
        return response
    

# @app.route('/api/team-records')


app.run(debug = True)