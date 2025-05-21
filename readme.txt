How to Run the Facial Skincare Recommendation System
-----------------------------------------------------

Prerequisites:
Make sure you have Python 3.8+ installed, and install the required packages below:

Install dependencies:
> pip install torch torchvision torchaudio
> pip install scikit-learn
> pip install pandas numpy matplotlib
> pip install pillow
> pip install streamlit

(Optional: If needed)
> pip install opencv-python

Folder Structure & File Descriptions
------------------------------------------------------------

Root Directory:
├── app.py
    - Main Streamlit web application that integrates all predictions and displays skincare recommendations.

├── readme.txt
    - Instructions on how to install dependencies, run the training scripts, and launch the app.

├── Recommendation.ipynb
    - Jupyter notebook version of the recommendation logic (TF-IDF + KNN based).

├── Recommender/
│   ├── cosmetics.csv
│       - Skincare product metadata including product names and ingredient lists for recommendation engine.

Skin_Metrics/
├── Acne/
│   ├── Dataset/
│       - Folder containing acne severity classification training/testing images.
│   ├── Acne_Model.py
│       - Script to train the acne severity classification model.
│   ├── acne.ipynb
│       - Jupyter version of the acne training + evaluation process.
│   ├── acne.pt
│       - Trained PyTorch model file for acne severity classification.
│   ├── test_image.jpg
│       - Sample image used for testing acne classification model.

├── Skin_Tone/
│   ├── Dataset/
│       - Contains images of various skin tones (subfolders: Black, Brown, White).
│   ├── Skin_Tone_Model.py
│       - Script to train the skin tone classification model.
│   ├── skin_tone.ipynb
│       - Jupyter notebook version of skin tone model training and testing.
│   ├── skin_tone.pt
│       - Trained PyTorch model file for skin tone classification.
│   ├── test_image.jpg
│       - Sample image for testing skin tone classification.

├── Skin_Type/
│   ├── Dataset/
│       - Folder with images used to classify skin type (oily, dry, normal).
│   ├── Skin_Type_Model.py
│       - Script to train the skin type classifier.
│   ├── skin_type.ipynb
│       - Jupyter notebook with training and validation code for skin type.
│   ├── skin_type.pt
│       - Trained model weights for skin type prediction.
│   ├── test_image.jpg
│       - Image used to validate/test skin type prediction pipeline.


Run Each Component:

1. Train Skin Type Model:
> python Skin_Type_Model.py

2. Train Skin Tone Model:
> python Skin_Tone_Model.py

3. Train Acne Severity Model:
> python Acne_Model.py

4. Launch the Web App:
> streamlit run app.py

Note:
- Make sure your data is placed in the correct folders as expected by the scripts.
- Model weights and product CSVs should be in the working directory or updated in the code.
