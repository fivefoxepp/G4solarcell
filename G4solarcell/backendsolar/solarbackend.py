from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from flask_cors import CORS
import os
import sys
import pickle # Import pickle to load the scaler

# ====================================================================
# *** คำเตือนสำคัญสำหรับการนำไปใช้งานจริง (Production) ***
#
# โมเดล AI ถูกฝึกด้วย MinMax Scaler ดังนั้นคุณต้องนำเข้า Scaler (.pkl)
# ที่บันทึกไว้ และใช้มันเพื่อแปลง (transform) ข้อมูล Voltage, Temp,
# Dust, และ Irradiance ก่อนที่จะนำเข้าโมเดลเพื่อทำนาย
# ====================================================================

# กำหนดขอบเขตทางกายภาพของเซ็นเซอร์ (Input Validation)
# TEMP_MAX = 60.0 ถูกตั้งไว้เพื่อให้อยู่ภายใต้การควบคุมของ AI
VOLT_MIN, VOLT_MAX = -5.0, 20.0
TEMP_MIN, TEMP_MAX = -50.0, 60.0 
DUST_MIN, DUST_MAX = 0.0, 1000.0
LIGHT_MIN, LIGHT_MAX = -10.0, 20000.0


def validate_sensor_inputs(volt, temp, dust, light):
    """
    ตรวจสอบความสมเหตุสมผลของค่าเซ็นเซอร์ที่รับจากผู้ใช้ (เพื่อป้องกัน Input Error)
    Returns: ข้อความ Error (String) ถ้ามี, หรือ None ถ้าค่าถูกต้อง
    """
    # ตรวจสอบไปทีละค่าอย่างละเอียด
    if not (VOLT_MIN <= volt <= VOLT_MAX):
        return f"Input Error: Voltage ({volt:.2f} V) อยู่นอกขอบเขตที่รับได้ ({VOLT_MIN} V ถึง {VOLT_MAX} V)"
    
    if not (TEMP_MIN <= temp <= TEMP_MAX):
        return f"Input Error: Temperature ({temp:.2f}°C) อยู่นอกขอบเขตที่รับได้ ({TEMP_MIN}°C ถึง {TEMP_MAX}°C)"
    
    if not (DUST_MIN <= dust <= DUST_MAX):
        return f"Input Error: Dust ({dust:.2f} µg/m³) อยู่นอกขอบเขตที่รับได้ ({DUST_MIN} µg/m³ ถึง {DUST_MAX} µg/m³)"
    
    if not (LIGHT_MIN <= light <= LIGHT_MAX):
        return f"Input Error: Irradiance ({light:.2f} lx) อยู่นอกขอบเขตที่รับได้ ({LIGHT_MIN} lx ถึง {LIGHT_MAX} lx)"
    
    return None # ค่าถูกต้องทั้งหมด

# ====================================================================
# Flask App Setup
# ====================================================================
app = Flask(__name__)
CORS(app, supports_credentials=True)

# 1. โหลด Model AI (.h5)
# *** กรุณาตรวจสอบชื่อไฟล์ model ให้ตรงกับที่บันทึกไว้ใน Jupyter Notebook (solar_panel_ai_10_classes.h5)
MODEL_FILE_NAME = "solar_panel_ai_10_classes.h5" 
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", MODEL_FILE_NAME)
model = None
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}", file=sys.stderr)
except Exception as e:
    print(f"❌ Error loading model: {e}", file=sys.stderr)
    model = None

# 2. โหลด Scaler (.pkl)
SCALER_FILE_NAME = "scaler_10_classes.pkl"
SCALER_PATH = os.path.join(os.path.dirname(__file__), "model", SCALER_FILE_NAME)
scaler = None
try:
    with open(SCALER_PATH, 'rb') as file:
        scaler = pickle.load(file)
    print(f"✅ Scaler loaded successfully from: {SCALER_PATH}", file=sys.stderr)
except Exception as e:
    print(f"❌ Error loading scaler: {e}", file=sys.stderr)
    scaler = None

# Mapping Labels: (อัปเดตเป็น 10 คลาส)
LABEL_MAP = {
    0: "ปกติ (Normal)",
    1: "ฝุ่น (Soiling)",
    2: "อุณหภูมิสูง (High Temp)",
    3: "เงาบัง (Shading)",
    4: "Sensor Error / ค่าผิดปกติเกินขอบเขต",
    5: "ร้อน+เงา (Temp + Shade)",
    6: "ฝุ่น+ร้อน (Dust + Temp)",
    7: "ฝุ่น+เงา (Dust + Shade)",
    8: "ฝุ่น+ร้อน+เงา (Worst Case)",
    9: "อุณหภูมิต่ำ (Low Temp)" # **Class 9 ถูกเพิ่มเข้ามา**
}


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Solar Fault Analyzer API is running"}), 200

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or Scaler not loaded. Service unavailable."}), 503
    
    # Handle OPTIONS preflight request (for CORS)
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json()

    # ตรวจสอบว่าข้อมูลครบหรือไม่
    required_keys = ["voltage", "temperature", "dust", "irradiance"]
    if not all(k in data for k in required_keys):
        return jsonify({
            "error": f"Missing one or more required keys: {required_keys}"
        }), 400
        
    try:
        # ดึงค่าและแปลงเป็น float
        voltage = float(data["voltage"])
        temperature = float(data["temperature"])
        dust = float(data["dust"])
        irradiance = float(data["irradiance"])

        # 1. ตรวจสอบ Input Validation
        validation_error = validate_sensor_inputs(voltage, temperature, dust, irradiance)
        if validation_error:
            # หากมี Error จากการกรอกค่าไม่สมจริง ให้ตอบกลับ Error ทันที (ซึ่งสอดคล้องกับ Class 4)
            return jsonify({
                "result": LABEL_MAP[4], # ส่งผลลัพธ์เป็น Class 4 โดยตรง
                "error": validation_error
            }), 400

        # 2. ถ้าไม่มี Input Error จึงนำเข้าโมเดล AI เพื่อทำนาย
        # ลำดับข้อมูลต้องเป็นไปตามที่ใช้ในการฝึก: [Voltage, Dust, Temperature, Irradiance]
        # (แก้ไขลำดับเพื่อให้ตรงกับ features ที่ใช้ใน Jupyter Notebook: Voltage, Dust, Temperature, Irradiance)
        input_array = np.array([[voltage, dust, temperature, irradiance]])
        
        # 3. **ขั้นตอนสำคัญ**: Scale ข้อมูลด้วย Scaler ที่โหลดมา
        scaled_input = scaler.transform(input_array)
        
        # 4. ทำนาย
        prediction_array = model.predict(scaled_input, verbose=0)[0] 

        # 5. แปลงผลลัพธ์ให้อ่านง่าย
        predicted_class_index = np.argmax(prediction_array)
        predicted_label = LABEL_MAP.get(predicted_class_index, "Unknown Class")

        # สร้าง object ความน่าจะเป็น
        probabilities = {
            # ใช้ LABEL_MAP[i] เป็น Key เพื่อให้ Frontend ไม่ต้อง Map ซ้ำ
            LABEL_MAP[i]: float(prob) for i, prob in enumerate(prediction_array)
        }
        
        # ส่งผลลัพธ์กลับ
        return jsonify({
            "result": predicted_label,
            "probabilities": probabilities
        }), 200
        
    except ValueError:
        return jsonify({"error": "Invalid data format. All inputs must be numerical."}), 400
    except Exception as e:
        print(f"Prediction Error (Internal Server): {e}", file=sys.stderr)
        return jsonify({"error": f"Internal server error during prediction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4300, debug=True)