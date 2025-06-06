# DiseasePredictionApp
app.py
Building the interactive UI (form inputs, buttons, visualizations)
Loading the trained model and pre-processing tools
Processing user inputs and generating predictions
Displaying top predictions, charts, and insights
Providing PDF and CSV export features
Supporting batch prediction via CSV upload

random_forest_model.pkl
This is the trained Random Forest classification model, saved as a binary .pkl file using joblib. It is loaded inside app.py and used to perform predictions based on the user’s input.
The input dimensions and preprocessing pipeline must match those used during training.

scaler.pkl
This file contains the scaling object (e.g., StandardScaler or MinMaxScaler) trained on the original data. It normalizes numerical features like temperature, humidity, and wind speed.
During deployment, user input is first transformed using this scaler before being passed to the model.

label_encoder.pkl
This file stores the LabelEncoder, which maps between numeric class labels and actual disease names. The model’s numeric output is decoded back into readable class names for display to users.

requirements.txt
This file lists all the Python dependencies required to run the application. It is used by platforms like Streamlit Cloud to install the environment automatically.
