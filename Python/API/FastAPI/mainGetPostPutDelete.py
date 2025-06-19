from fastapi import FastAPI, Path, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field  
from typing import Annotated, Literal, Optional
import json


## Pydantic Model For Create New Patient
class Patient(BaseModel):
    id : Annotated[str, Field(..., description= 'Patient ID', example = 'P001')]
    name: Annotated[str, Field(..., description= 'Patient Name')]
    city: Annotated[str, Field(..., description= 'City')]
    age: Annotated[int, Field(...,gt=0, lt=120, description= 'Patient Age')]
    gender:Annotated[Literal['male', 'female','other'], Field(..., description= 'Patient Gender')]
    height:Annotated[float, Field(..., gt=0, description='Patient Height in mtrs')]
    weight: Annotated[float, Field(..., gt=0, description = 'Patient Weight in kgs')]

    # Computed property: Body Mass Index (BMI)
    @computed_field  
    @property
    def bmi(self) -> float:
        bmi = self.weight / (self.height ** 2)  # BMI formula: weight (kg) / height² (m²)
        return round(bmi, 2)  # Round BMI to 2 decimal places

    # Computed property: Health verdict based on BMI
    @computed_field
    @property
    def verdict(self) -> str:
        # Categorize BMI result into health status
        if self.bmi < 18.5:
            return 'Underweight'
        elif self.bmi < 25:
            return 'Normal'
        elif self.bmi < 30:
            return 'Overweight' 
        else:
            return 'Obese'
        

## Pydantic Model For Update Existing Patient (no ID parameter because, IDs are not meant to be updated.)
class PatientUpdate(BaseModel):
    name: Annotated[Optional[str], Field(default=None)]
    city: Annotated[Optional[str], Field(default=None)]
    age: Annotated[Optional[int], Field(default=None, gt=0)]
    gender: Annotated[Optional[Literal['male', 'female']], Field(default=None)]
    height: Annotated[Optional[float], Field(default=None, gt=0)]
    weight: Annotated[Optional[float], Field(default=None, gt=0)]
        

## CURD 
# CREATE : POST
# UPDATER : PUT
# RETREVAL : GET
# DELETE : 


## Fast API
app = FastAPI()


# ------------------------- Helper Function -------------------------
# Function to load data from the JSON file
def load_data():
    try:
        with open('patients.json', 'r') as f:
            data = json.load(f)  # Read and parse JSON data from file
            return data  # Return the loaded patient data
    except Exception as e:
        # Raise a server-side error if file reading fails
        raise HTTPException(status_code=500, detail=f'Error loading data: {str(e)}')
    

# Function to save data to the JSON file
def save_data(data):
    with open('patients.json', 'w') as f:
        return json.dump(data, f)  # Write patient data back to JSON file
    




# ------------------------- GET Endpoints (RETRIVE)----------------------------

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






# ------------------------- POST Endpoints (CREATE) ----------------------------

@app.post('/create')
def create_patient(patient:Patient):  # Create Patient Function(uses Patient model defined above to parse and validate incoming request data)

    data = load_data()     # Load existing patient records from the database (e.g., a JSON file)

    if patient.id in data: # Check Patient already exist in data base
        raise HTTPException(status_code=400, detail = 'Patient already exists')
    
    # New Patient logic
    key = patient.id ## key variable = patient.id
    value = patient.model_dump(exclude=['id']) # value variable = New Patient record from patient pydantic object into dictionary excluding "id"
   
    data[key] = value # Adding New Patient record into the database

    # save into the json file
    save_data(data)

    return JSONResponse(status_code=201, content={'message':'patient created successfully'})







# ------------------------- PUT Endpoint (UPDATE) ----------------------------
@app.put('/edit/{patient_id}')
def update_patient(patient_id: str, patient_update: PatientUpdate):
    
    # Load existing patient records from the JSON file
    data = load_data()

    # Check if the patient with the given ID exists in the database
    if patient_id not in data:
        raise HTTPException(status_code=404, detail='Patient does not exist')
    
    # Fetch the current data of the patient to be updated
    existing_patient_info = data[patient_id]

    # Convert the incoming update object to a dictionary
    # Only include fields that are explicitly set in the request (exclude unset fields)
    updated_patient_info = patient_update.model_dump(exclude_unset=True)

    # Update the existing patient record with the new values
    for k, v in updated_patient_info.items():
        existing_patient_info[k] = v
    
    # Reassign the ID to ensure it stays intact in the patient record
    existing_patient_info['id'] = patient_id 
    
    # Convert updated dictionary into a Patient Pydantic object
    # This ensures BMI and Verdict computed fields are recalculated correctly
    existing_patient_info_pydantic = Patient(**existing_patient_info)

    # Convert the updated Pydantic object back to a dictionary (excluding 'id')
    # Because patient ID is used as the dictionary key, we don’t need to store it inside the object again
    existing_patient_info = existing_patient_info_pydantic.model_dump(exclude="id")

    # Update the patient record in the full data set
    data[patient_id] = existing_patient_info

    # Save the updated patient data back to the JSON file
    save_data(data)

    # Return the full updated patient object (includes computed fields)
    return JSONResponse(status_code=200, content={'message': 'Patient Updated....'})




# ------------------------- PUT Endpoint (UPDATE) ----------------------------
@app.delete('/delete/{patient_id}')
def delete_patient(patient_id: str):

    # Load existing patient records from the JSON file
    data = load_data()

    # Check if the patient with the given ID exists in the database
    if patient_id not in data:
        raise HTTPException(status_code=404, detail='Patient does not exist')
    
    del data[patient_id]

    # Save the updated patient data back to the JSON file
    save_data(data)

    # Return the full updated patient object 
    return JSONResponse(status_code=200, content={'message': 'Patient Deleted....'})