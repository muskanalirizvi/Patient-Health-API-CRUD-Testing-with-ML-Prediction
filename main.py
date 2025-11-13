from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, computed_field, ValidationError
from typing import Literal, List, Optional
import joblib
import json
import os
import numpy as np
import random
from datetime import datetime
import logging

# ==============================
# Logging Setup
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Load model and scaler
# ==============================
try:
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model/scaler: {e}")
    model = None
    scaler = None

# ==============================
# Initialize FastAPI
# ==============================
app = FastAPI(
    title="Patient Record & Disease Prediction API",
    description="Manage patient records and predict diabetes using ML model.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Allow CORS (for frontend apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# JSON Database File
# ==============================
DATA_FILE = "patients.json"
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f, indent=4)
    logger.info(f"Created empty database: {DATA_FILE}")


# ==============================
# Helper Functions
# ==============================
def load_data():
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_data(data):
    try:
        # Backup existing file
        if os.path.exists(DATA_FILE):
            backup_path = f"{DATA_FILE}.backup.{datetime.now().strftime('%Y%m%d%H%M%S')}"
            os.replace(DATA_FILE, backup_path)
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        raise HTTPException(status_code=500, detail="Failed to save patient data")


# ==============================
# Patient Schema (Pydantic Model)
# ==============================
class Patient(BaseModel):
    id: str = Field(..., description="Unique patient ID like P001, P002", example="P001")
    name: str = Field(..., description="Full name of the patient", example="Ali Raza")
    email: EmailStr = Field(..., description="Valid email address", example="ali@gmail.com")
    gender: Literal["male", "female"] = Field(..., description="Gender of the patient", example="male")
    contact_number: str = Field(..., description="Phone number", example="03121234567")
    address: str = Field(..., description="Home address", example="Lahore, Pakistan")
    age: int = Field(..., gt=0, description="Age in years", example=35)
    height: float = Field(..., gt=0, description="Height in centimeters", example=170)
    weight: float = Field(..., gt=0, description="Weight in kilograms", example=70)
    pregnancies: int = Field(..., ge=0, description="Number of pregnancies (0 for males)", example=2)
    family_history: bool = Field(..., description="True if family has diabetes history", example=True)
    activity_level: Literal["low", "medium", "high"] = Field(..., description="Physical activity level", example="medium")

    date_of_visit: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    outcome: str = "Pending"

    # ==========================
    # Computed Fields
    # ==========================
    @computed_field
    @property
    def bmi(self) -> float:
        """Compute Body Mass Index (BMI)"""
        return round(self.weight / ((self.height / 100) ** 2), 2)

    @computed_field
    @property
    def glucose(self) -> float:
        """Estimate glucose based on BMI & family history"""
        base = random.uniform(70, 140)
        if self.family_history:
            base += 10
        if self.bmi > 30:
            base += 15
        return round(base, 1)

    @computed_field
    @property
    def blood_pressure(self) -> float:
        """Estimate blood pressure based on age & activity"""
        base = random.uniform(60, 90)
        if self.age > 40:
            base += 10
        if self.activity_level == "low":
            base += 5
        return round(base, 1)

    @computed_field
    @property
    def skin_thickness(self) -> float:
        """Estimate skin thickness based on BMI"""
        return round(random.uniform(10, 40) + (self.bmi - 20) * 0.5, 1)

    @computed_field
    @property
    def insulin(self) -> float:
        """Estimate insulin from glucose"""
        return round(self.glucose * random.uniform(0.5, 1.5), 1)

    @computed_field
    @property
    def diabetes_pedigree_function(self) -> float:
        """Simulated genetic diabetes risk factor"""
        return round(random.uniform(0.6, 1.0) if self.family_history else random.uniform(0.2, 0.6), 3)


# ==============================
# Update Model (partial updates)
# ==============================
class PatientUpdate(BaseModel):
    # All fields optional so that client can send partial updates
    id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    gender: Optional[Literal["male", "female"]] = None
    contact_number: Optional[str] = None
    address: Optional[str] = None
    age: Optional[int] = Field(None, gt=0)
    height: Optional[float] = Field(None, gt=0)
    weight: Optional[float] = Field(None, gt=0)
    pregnancies: Optional[int] = Field(None, ge=0)
    family_history: Optional[bool] = None
    activity_level: Optional[Literal["low", "medium", "high"]] = None

    # date_of_visit and outcome typically shouldn't be updated by client; if required add explicitly


# ==============================
# Response Models
# ==============================
class PatientResponse(Patient):
    class Config:
        from_attributes = True


class PredictionResponse(BaseModel):
    id: str
    name: str
    email: EmailStr
    prediction: str
    probability: float
    date_of_visit: str


# ==============================
# NEW ENDPOINTS: Home & Health
# ==============================

@app.get(
    "/",
    summary="API Welcome Page",
    description="Returns a friendly greeting and API overview.",
    tags=["Info"]
)
def home():
    """Home endpoint with API info and quick links."""
    return {
        "message": "Welcome to the Patient Record & Disease Prediction API!",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "Add Patient": "POST /patients",
            "Get All Patients": "GET /patients",
            "Get Patient": "GET /patients/{id}",
            "Update Patient": "PUT /patients/{id}",
            "Delete Patient": "DELETE /patients/{id}",
            "Predict Diabetes": "POST /predict_disease/{id}",
            "Health Check": "GET /health"
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get(
    "/health",
    summary="Health Check",
    description="Verifies model, scaler, and database status.",
    tags=["Info"],
    responses={
        200: {"description": "All healthy"},
        503: {"description": "Service unavailable"}
    }
)
def health_check():
    issues = []

    # Check model
    if model is None:
        issues.append("Model failed to load")
    else:
        try:
            _ = joblib.load("diabetes_model.pkl")
        except Exception as e:
            issues.append(f"Model load error: {e}")

    # Check scaler
    if scaler is None:
        issues.append("Scaler failed to load")
    else:
        try:
            _ = joblib.load("scaler.pkl")
        except Exception as e:
            issues.append(f"Scaler load error: {e}")

    # Check database
    try:
        data = load_data()
        patient_count = len(data)
    except Exception as e:
        issues.append(f"Database error: {e}")
        patient_count = 0

    if not issues:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy",
                "model": "loaded",
                "scaler": "loaded",
                "database": "accessible",
                "patients_count": patient_count,
                "checked_at": datetime.utcnow().isoformat() + "Z"
            }
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "issues": issues,
                "checked_at": datetime.utcnow().isoformat() + "Z"
            }
        )


# ==============================
# CRUD Endpoints
# ==============================

@app.post("/patients", response_model=PatientResponse, tags=["Patients"])
def add_patient(patient: Patient):
    logger.info(f"Adding patient: {patient.name} ({patient.id})")
    data = load_data()
    if any(p["id"] == patient.id for p in data):
        raise HTTPException(status_code=400, detail="Patient ID already exists")
    # store patient as dict (keeps computed values)
    data.append(patient.dict())
    save_data(data)
    return patient


@app.get("/patients", response_model=List[PatientResponse], tags=["Patients"])
def get_all_patients(skip: int = 0, limit: int = 100, search: Optional[str] = None):
    data = load_data()
    if search:
        data = [p for p in data if search.lower() in p["name"].lower() or search in p["id"]]
    return data[skip:skip + limit]


@app.get("/patients/{id}", response_model=PatientResponse, tags=["Patients"])
def get_patient(id: str):
    data = load_data()
    for patient in data:
        if patient["id"] == id:
            return patient
    raise HTTPException(status_code=404, detail="Patient not found")


@app.put("/patients/{id}", response_model=PatientResponse, tags=["Patients"])
def update_patient(id: str, updated_patient: PatientUpdate):
    """
    Partial update endpoint:
    - accepts only the fields you want to update
    - merges them into the stored patient
    - re-validates and recalculates computed fields by instantiating Patient(...)
    """
    data = load_data()
    for i, patient in enumerate(data):
        if patient["id"] == id:
            # If caller included id in body, ensure it matches the path id
            if updated_patient.id is not None and updated_patient.id != id:
                raise HTTPException(
                    status_code=400,
                    detail="Patient ID in URL and body must match"
                )

            # Merge only provided fields
            incoming = updated_patient.dict(exclude_unset=True)
            merged = {**patient, **incoming}

            # Re-validate using Patient model (this re-computes computed fields)
            try:
                validated = Patient(**merged)
            except ValidationError as e:
                raise HTTPException(status_code=400, detail=str(e))

            # Save back to DB (store as dict to preserve same format)
            data[i] = validated.dict()
            save_data(data)
            logger.info(f"Updated patient: {id}")
            return validated

    raise HTTPException(status_code=404, detail="Patient not found")


@app.delete("/patients/{id}", tags=["Patients"])
def delete_patient(id: str):
    data = load_data()
    new_data = [p for p in data if p["id"] != id]
    if len(new_data) == len(data):
        raise HTTPException(status_code=404, detail="Patient not found")
    save_data(new_data)
    logger.info(f"Deleted patient: {id}")
    return {"message": "Patient deleted successfully"}


# ==============================
# Disease Prediction Endpoint
# ==============================

@app.post("/predict_disease/{id}", response_model=PredictionResponse, tags=["Prediction"])
def predict_disease(id: str):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="ML model not available")

    data = load_data()
    patient = None
    for p in data:
        if p["id"] == id:
            patient = p
            break

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Required features
    try:
        features = np.array([[
            patient["pregnancies"],
            patient["glucose"],
            patient["blood_pressure"],
            patient["skin_thickness"],
            patient["insulin"],
            patient["bmi"],
            patient["diabetes_pedigree_function"],
            patient["age"]
        ]])
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")

    try:
        features_scaled = scaler.transform(features)
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    outcome = "Positive" if prediction == 1 else "Negative"
    patient["outcome"] = outcome
    save_data(data)

    logger.info(f"Prediction for {id}: {outcome} (Prob: {probability:.3f})")

    return PredictionResponse(
        id=id,
        name=patient["name"],
        email=patient["email"],
        prediction=outcome,
        probability=round(probability, 3),
        date_of_visit=patient["date_of_visit"]
    )