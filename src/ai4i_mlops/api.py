
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ai4i_mlops.predict import predict, load_model_info


app = FastAPI(
    title="AI4I Predictive Maintenance API",
    version="0.1.0"
)


class MachineInput(BaseModel):
    type: Literal["L", "M", "H"]
    air_temperature: float = Field(gt=0)
    process_temperature: float = Field(gt=0)
    rotational_speed: int = Field(gt=0)
    torque: float = Field(ge=0)
    tool_wear: int = Field(ge=0)


@app.get("/")
def home():
    return {"message": "AI4I Predictive Maintenance API"}


@app.get("/model-info")
def model_info():
    return load_model_info()


@app.post("/predict")
def predict_failure(data: MachineInput):
    features = {
        "Type": data.type,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear
    }

    return predict(features)
