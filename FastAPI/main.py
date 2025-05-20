from fastapi import FastAPI, Path, HTTPException, Query
import json

app = FastAPI()


## Helper Function 
def load_data():
    try:
        with open('patients.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error loading data: {str(e)}')
    

### --------------------This the Examples of GET request------------------------------------------------------------------------------

@app.get("/") ## Route to main 
def hello():
    return {'message': 'hello World'}


@app.get('/about') ## Route to about 
def about():
    return {'message': 'A fully functional API to manage your patient records'}


@app.get('/view') ## Route View Data 
def view_data():
    return load_data()


## Path parameter example
@app.get('/patient/{patient_id}') ## Route Patient Details
def view_patient(patient_id:str = Path(..., description='ID of Patient in the DB', example='P001')): ## Adding Readability enhancement
    data = load_data()
    if patient_id in data:
        return data.get(patient_id)
    raise HTTPException(status_code=404, detail='Patient not Found') ## HTTP excaption Handling 


## Query Parameter example
@app.get('/sort')
def sort_patients(sort_by:str = Query(..., description= 'Sort on basis of height, weight or bmi'),
                  order:str = Query('asc', description= 'Sort in accending or decending order')):
    valid_fields = ['height', 'weight', 'bmi']

    ## Arguement Data validation logic
    if sort_by not in valid_fields:
        raise HTTPException(status_code=400, detail= f"Invalid fields select from {valid_fields}")
    
    if order not in ['asc', 'desc']:
        raise HTTPException(status_code=400, detail=f"Invalid order select between asc and desc '{order}'" )
    
    ## Aassigning the boolean variable for reverse paramenter below
    sort_order = True if order=='desc' else False
    
    data = load_data()
    sorted_data = sorted (data.values(), key= lambda x: x.get(sort_by,0), reverse = sort_order)
    return sorted_data
