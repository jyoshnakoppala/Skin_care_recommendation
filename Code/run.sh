#!/bin/bash

echo ">> Installing Python dependencies..."
pip install -r requirements.txt

echo ">> Testing Skin Type Model..."
python Skin_Metrics/Skin_Type/Skin_Type_Model.py

# Step 2: Train Skin Tone Model
echo ">> Testing Skin Tone Model..."
python Skin_Metrics/Skin_Tone/Skin_Tone_Model.py

# Step 3: Train Acne Severity Model
echo ">> Testing Acne Severity Model..."
python Skin_Metrics/Acne/Acne_Model.py

# Step 4: Launch the Web App
echo ">> Launching Streamlit Web App..."
streamlit run app.py