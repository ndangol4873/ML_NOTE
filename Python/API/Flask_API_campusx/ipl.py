import numpy as np
import pandas as pd
from flask import jsonify
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    


matches = pd.read_csv('d:/ML_NOTE_DATASET/ipl-matches.csv')
balls = pd.read_csv('d:/ML_NOTE_DATASET/IPL_Ball_by_Ball_2008_2022.csv')



ball_withmatch = balls.merge(matches, on='ID', how='inner').copy()

bowling_team = []
for index, (team1, battingteam) in enumerate(zip(ball_withmatch['Team1'], ball_withmatch['BattingTeam'])):
    if team1 == battingteam:
        bowling_team.append(np.nan)
    else:
        bowling_team.append(team1)

ball_withmatch['BowlingTeam'] = bowling_team
ball_withmatch['BowlingTeam'] = ball_withmatch['BowlingTeam'].fillna(ball_withmatch['Team2'])
batter_data = ball_withmatch[['BowlingTeam','BattingTeam','Player_of_Match']]




def team1vsteam2(team, team2):
    df = matches[((matches['Team1'] == team) & (matches['Team2'] == team2)) | (
                (matches['Team2'] == team) & (matches['Team1'] == team2))].copy()
    mp = df.shape[0]
    won = df[df.WinningTeam == team].shape[0]
    nr = df[df.WinningTeam.isnull()].shape[0]
    loss = mp - won - nr

    return {'matchesplayed': mp,
            'won': won,
            'loss': loss,
            'noResult': nr}




def ball_record_count():
    row = {
        'Total Bowling Data : ':balls.shape[0]
    }
    return jsonify(row)




def teamsNameAPI():
    '''Return  Teams Name '''
    teams = list(set(list(matches['Team1']) + list(matches['Team2'])))
    teams_dict ={
        'teams':teams
    }
    return teams_dict


def allRecord(team):
    df = matches[(matches['Team1'] == team) | (matches['Team2'] == team)].copy()
    mp = df.shape[0]
    won = df[df.WinningTeam == team].shape[0]
    nr = df[df.WinningTeam.isnull()].shape[0]
    loss = mp - won - nr
    nt = df[(df.MatchNumber == 'Final') & (df.WinningTeam == team)].shape[0]
    return {'matchesplayed': mp,
            'won': won,
            'loss': loss,
            'noResult': nr,
            'title': nt}



def teamsRecordsAPI(team):
    df = matches[(matches['Team1'] == team) | (matches['Team2'] == team)].copy()
    self_record = allRecord(team)
    TEAMS = matches.Team1.unique()
    against = {team2: team1vsteam2(team, team2) for team2 in TEAMS}
    data = {team: {'overall': self_record,
                   'against': against}}
    return json.dumps(data, cls=NpEncoder)


def batsmanVsTeam(batsman, team, df):
    df = df[df.BowlingTeam == team].copy()
    return batsmanRecord(batsman, df)

def batsmanAPI(batsman, balls=batter_data):
    df = balls[balls.innings.isin([1, 2])]  # Excluding Super overs
    self_record = batsmanRecord(batsman, df=df)
    TEAMS = matches.Team1.unique()
    against = {team: batsmanVsTeam(batsman, team, df) for team in TEAMS}
    data = {
        batsman: {'all': self_record,
                  'against': against}
    }
    return json.dumps(data, cls=NpEncoder)

bowler_data = batter_data.copy()


def batsmanRecord(batsman, df):
    if df.empty:
        return np.nan
    out = df[df.player_out == batsman].shape[0]
    df = df[df['batter'] == batsman]
    inngs = df.ID.unique().shape[0]
    runs = df.batsman_run.sum()
    fours = df[(df.batsman_run == 4) & (df.non_boundary == 0)].shape[0]
    sixes = df[(df.batsman_run == 6) & (df.non_boundary == 0)].shape[0]
    if out:
        avg = runs / out
    else:
        avg = np.inf

    nballs = df[~(df.extra_type == 'wides')].shape[0]
    if nballs:
        strike_rate = runs / nballs * 100
    else:
        strike_rate = 0
    gb = df.groupby('ID').sum()
    fifties = gb[(gb.batsman_run >= 50) & (gb.batsman_run < 100)].shape[0]
    hundreds = gb[gb.batsman_run >= 100].shape[0]
    try:
        highest_score = gb.batsman_run.sort_values(ascending=False).head(1).values[0]
        hsindex = gb.batsman_run.sort_values(ascending=False).head(1).index[0]
        if (df[df.ID == hsindex].player_out == batsman).any():
            highest_score = str(highest_score)
        else:
            highest_score = str(highest_score) + '*'
    except:
        highest_score = gb.batsman_run.max()

    not_out = inngs - out
    mom = df[df.Player_of_Match == batsman].drop_duplicates('ID', keep='first').shape[0]
    data = {
        'innings': inngs,
        'runs': runs,
        'fours': fours,
        'sixes': sixes,
        'avg': avg,
        'strikeRate': strike_rate,
        'fifties': fifties,
        'hundreds': hundreds,
        'highestScore': highest_score,
        'notOut': not_out,
        'mom': mom
    }

    return data




def teamVteamAPI(team1,team2):
    '''Returns the match details played between two teams'''
    valid_team = list(set(list(matches['Team1']) + list(matches['Team2'])))

    if team1 in valid_team and team2 in valid_team:
        temp_df = matches.loc[(matches['Team1'] == team1) & (matches['Team2'] == team2) | (matches['Team1'] == team2) & (matches['Team2'] == team1)]
        total_matches = temp_df.shape[0]

        matches_won_team1 = temp_df['WinningTeam'].value_counts()[team1]
        matches_won_team2 = temp_df['WinningTeam'].value_counts()[team2]

        draws = total_matches - (matches_won_team1 + matches_won_team2)

        response = {
            'total_matches': str(total_matches),
              team1: str(matches_won_team1),
              team2: str(matches_won_team2),
              'draws': str(draws)
        }
        return response
    else:
        return {'message' : 'Invalid Team Name'}

    