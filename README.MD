# Crop Yield Prediction Web Application

## Overview
This project is a web application for predicting crop yield based on input features such as year, rainfall, and irrigation area. It leverages Flask for the backend and integrates a machine learning model trained with Random Forest Regression.

---

## Features
- **Input Form**: Users can input year, rainfall (mm), and irrigation area (ha) through an intuitive form on the web page.
- **Prediction**: The application predicts the crop yield (in kg/ha) using a pre-trained model.
- **Interactive UI**: The interface is responsive, user-friendly, and includes an appealing design.
- **Error Handling**: Basic error handling ensures the application provides meaningful feedback for invalid inputs.

---

## Project Structure
```
project_root/
|│-- static/
|   ├── bk.jpg                 # Background image for the web page
|   └── styles.css            # Additional styles (if needed)
|│-- templates/
|   └── index.html           # Main HTML file for the application
|│-- final_rf_model_v2.pkl             # Pre-trained Random Forest model
|│-- app.py                            # Flask application file
|│-- README.md                         # Project documentation
```

---

## Prerequisites
1. **Python**: Ensure Python 3.7+ is installed.
2. **Flask**: Install Flask using pip:
   ```bash
   pip install flask
   ```
3. **Joblib**: Install joblib for model loading:
   ```bash
   pip install joblib
   ```
4. **Numpy**: Ensure numpy is installed for numerical computations:
   ```bash
   pip install numpy
   ```

---

## Usage Instructions

1. **Clone or Download the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Place the Model File**
   Ensure `final_rf_model_v2.pkl` is located in the root directory of the project. If the file is missing, the application will not run.

3. **Run the Application**
   Execute the Flask application:
   ```bash
   python app.py
   ```
   The application will start on `http://127.0.0.1:5000` by default.

4. **Access the Web Application**
   Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

5. **Use the Application**
   - Input values for the year, rainfall, and irrigation area.
   - Click the "Predict Yield" button to see the prediction.

---

## Code Breakdown

### `app.py`
The main Flask application file includes the following components:

1. **Model Loading**:
   ```python
   try:
       model = joblib.load("final_rf_model_v2.pkl")
       print("Model loaded successfully.")
   except FileNotFoundError:
       print("Error: 'final_rf_model.pkl' not found.")
       exit()
   ```
   - Loads the pre-trained Random Forest model.
   - Exits gracefully if the model file is missing.

2. **Routes**:
   - `"/"`: Renders the `index.html` file (home page).
   - `"/predict"`: Handles form submission, processes input data, and returns predictions.

3. **Prediction Logic**:
   ```python
   features = np.array([[year, rainfall, irrigation_area]])
   prediction = model.predict(features)[0]
   ```
   - Converts input values into the format expected by the model.
   - Fetches the prediction from the model.

4. **Error Handling**:
   Ensures the application can handle invalid inputs and display meaningful feedback.

---

## HTML (Frontend)
The `index.html` file includes:
- **Form for Input**: Fields for entering year, rainfall, and irrigation area.
- **Result Display**: Dynamic section for showing predictions.
- **Footer**: Information about copyright, privacy, and terms of service.

---

## Background Image
The background image is loaded dynamically using Flask's `url_for`:
```html
background-image: url("{{ url_for('static', filename='bk.jpg') }}");
```
Ensure the `bk.jpg` image is placed in the `static` directory.

---

## Example Prediction
### Input:
- Year: `2025`
- Rainfall: `300`
- Irrigation Area: `500`

### Output:
```
Prediction: 5045.87 kg/ha
```

---

## Future Improvements
1. **Error Handling Enhancements**:
   - Provide detailed feedback for missing or invalid inputs.
2. **Model Updates**:
   - Support more crops and additional features.
3. **UI Enhancements**:
   - Add responsive design for mobile devices.
4. **Deployment**:
   - Host the application using cloud services (e.g., AWS, Heroku).

---

## License
This project is open-source and available for educational purposes. Contributions are welcome!

