from fastapi import FastAPI, Path, HTTPException, Query

import json


app = FastAPI()


# ------------------------- Helper Function -------------------------
def load_data():
    try:
        with open('patients.json', 'r') as f:
            return json.load(f)  # Load patient data from JSON file
    except Exception as e:
        # Raise a server-side error if file reading fails
        raise HTTPException(status_code=500, detail=f'Error loading data: {str(e)}')

# ------------------------- GET Endpoints ----------------------------

@app.get("/")  # Root route
def hello():
    return {'message': 'hello World'}  # Simple greeting

@app.get('/about')  # API information route
def about():
    return {'message': 'A fully functional API to manage your patient records'}

@app.get('/view')  # View all patients' data
def view_data():
    return load_data()

# -------------------- Path Parameter Example ------------------------

@app.get('/patient/{patient_id}')
def view_patient(
    patient_id: str = Path(..., description='ID of Patient in the DB', example='P001')
):
    data = load_data()
    if patient_id in data:
        return data.get(patient_id)  # Return patient record if ID exists
    raise HTTPException(status_code=404, detail='Patient not Found')  # 404 error if not found

# -------------------- Query Parameter Example -----------------------

@app.get('/sort')
def sort_patients(
    sort_by: str = Query(..., description='Sort on basis of height, weight or bmi'),
    order: str = Query('asc', description='Sort in ascending or descending order')
):
    valid_fields = ['height', 'weight', 'bmi']

    # Input validation for sorting field
    if sort_by not in valid_fields:
        raise HTTPException(status_code=400, detail=f"Invalid field. Choose from {valid_fields}")
    
    # Input validation for sorting order
    if order not in ['asc', 'desc']:
        raise HTTPException(status_code=400, detail=f"Invalid order. Choose either 'asc' or 'desc'. Got '{order}'")

    # Determine sorting direction
    sort_order = True if order == 'desc' else False

    data = load_data()
    # Sort data based on specified key and order
    sorted_data = sorted(data.values(), key=lambda x: x.get(sort_by, 0), reverse=sort_order)
    return sorted_data




