from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
import pickle

# Створюємо FastAPI додаток
app = FastAPI(
    title="ML Service",
    description="API для прогнозування якості яйцеклітин",
    version="1.0",
    docs_url="/docs",  # Увімкнути Swagger UI
    redoc_url="/redoc",  # Увімкнути ReDoc
)

@app.get("/")
def root():
    return {"message": "ML Service is running!"}

# Завантажуємо модель
with open(r"C:\Users\yuliia.sokolova\PycharmProjects\MLService\ml_service\model_pipline", "rb") as f:
    model_pipline = pickle.load(f)


# Визначення структури вхідних даних
class InputData(BaseModel):
    I_beta_HCG_mIU_mL: float
    Age_yrs: int
    BMI: float
    Cycle_length_days: int
    Cycle_R_I: int
    Fast_food_Y_N: int
    Hair_loss_Y_N: int
    Hb_g_dl: float
    Height_Cm: float
    Hip_inch: float
    Marraige_Status_Yrs: int
    PRG_ng_mL: float
    Pimples_Y_N: int
    Pulse_rate_bpm: float
    RBS_mg_dl: float
    Reg_Exercise_Y_N: int
    Skin_darkening_Y_N: int
    Vit_D3_ng_mL: float
    Waist_inch: float
    Weight_Kg: float
    Weight_gain_Y_N: int
    Hair_growth_Y_N: int

# Відповідність між очікуваними назвами та назвами у `MinMaxScaler`
rename_mapping = {
    "I_beta_HCG_mIU_mL": "  I   beta-HCG(mIU/mL)",
    "Age_yrs": " Age (yrs)",
    "BMI": "BMI",
    "Cycle_length_days": "Cycle length(days)",
    "Cycle_R_I": "Cycle(R/I)",
    "Fast_food_Y_N": "Fast food (Y/N)",
    "Hair_loss_Y_N": "Hair loss(Y/N)",
    "Hb_g_dl": "Hb(g/dl)",
    "Height_Cm": "Height(Cm) ",
    "Hip_inch": "Hip(inch)",
    "Marraige_Status_Yrs": "Marraige Status (Yrs)",
    "PRG_ng_mL": "PRG(ng/mL)",
    "Pimples_Y_N": "Pimples(Y/N)",
    "Pulse_rate_bpm": "Pulse rate(bpm) ",
    "RBS_mg_dl": "RBS(mg/dl)",
    "Reg_Exercise_Y_N": "Reg.Exercise(Y/N)",
    "Skin_darkening_Y_N": "Skin darkening (Y/N)",
    "Vit_D3_ng_mL": "Vit D3 (ng/mL)",
    "Waist_inch": "Waist(inch)",
    "Weight_Kg": "Weight (Kg)",
    "Weight_gain_Y_N": "Weight gain(Y/N)",
    "Hair_growth_Y_N": "hair growth(Y/N)"
}

# Ендпойнт для передбачення
@app.post("/predict")
def predict(data: InputData):
    try:
        # Перетворюємо вхідні дані у словник з правильними назвами
        formatted_data = {rename_mapping[key]: getattr(data, key) for key in data.__annotations__}

        # Перетворюємо у DataFrame з правильними назвами колонок
        input_data_df = pd.DataFrame([formatted_data])

        # Масштабування (якщо є scaler)
        scaler = model_pipline.get("scaler", None)
        if scaler:
            input_data = scaler.transform(input_data_df)
        else:
            input_data = input_data_df.values  # Якщо scaler немає, передаємо просто значення у numpy-масив

        # Прогнозування
        model = model_pipline["model"]
        prediction = model.predict(input_data)

        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Щоб запустити сервіс, введи в терміналі:
# uvicorn ml_service:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
# ReDoc: http://127.0.0.1:8000/redoc
