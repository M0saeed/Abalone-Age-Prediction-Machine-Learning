import os
import pickle
from flask import Flask, request, render_template
import numpy as np

# تحميل النموذج المحفوظ
model_path = 'dtr.pkl' if os.path.exists('dtr.pkl') else ('model.pkl' if os.path.exists('model.pkl') else None)
model = None

if model_path:
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Warning: Model file (dtr.pkl or model.pkl) not found.")

app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # استخراج البيانات من النموذج وتأكيد أنواعها
            sex = int(request.form['sex'])
            length = float(request.form['length'])
            diameter = float(request.form['diameter'])
            height = float(request.form['height'])
            whole_weight = float(request.form['wholeWeight'])
            shucked_weight = float(request.form['Shuckedweight'])
            viscera_weight = float(request.form['Visceraweight'])
            shell_weight = float(request.form['Shellweight'])

            # تجميع المدخلات في مصفوفة ثنائية الأبعاد (1 row, 8 features)
            features = np.array([[sex, length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight]])

            # إجراء التوقع
            prediction = model.predict(features)[0]
            
            # تقريب النتيجة لرقمين عشريين
            predicted_age = round(float(prediction), 2)

            return render_template('index.html', age=predicted_age)

        except ValueError:
            return render_template('index.html', error_message="الرجاء أدخل أرقاماً صالحة في جميع الحقول.")
        except Exception as e:
            return render_template('index.html', error_message=f"حدث خطأ: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)