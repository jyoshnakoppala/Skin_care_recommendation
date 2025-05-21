# Facial Skincare Recommendation System

This project is a deep learning-powered system designed to analyze facial images and provide personalized skincare product recommendations. It classifies a user's **skin tone**, **skin type**, and **acne severity** using fine-tuned CNN models and combines **content-based filtering** with **TF-IDF + KNN** collaborative-style recommendations to suggest suitable products.

##  Features

- **Image-based Classification**:
  - Skin Tone: Fair, Medium, Dark
  - Skin Type: Oily, Dry, Normal
  - Acne Severity: Mild, Moderate, Severe

- **Personalized Recommendations**:
  - Filters products based on skin type, tone, and acne severity
  - Recommends products with similar ingredients using TF-IDF + KNN

- **Web Interface**:
  - Developed using **Streamlit** for easy user interaction

---

##  Models Used

Fine-tuned pre-trained CNNs using **PyTorch**:

- ResNet18
- ResNet50
- MobileNetV2
- EfficientNetB0 (Best performing model)

All models were trained with:
- CrossEntropyLoss
- Adam Optimizer
- Epochs: 20
- Batch Size: 32
---

##  Evaluation Metrics

- **Accuracy**
- **Precision / Recall / F1-score**
- **Confusion Matrix**

### Best Performance (EfficientNetB0)
| Task          | Test Accuracy |
|---------------|---------------|
| Skin Tone     | 86.22%        |
| Skin Type     | 88.43%        |
| Acne Severity | 85.00%        |

---

##  Tech Stack

- Python
- PyTorch
- Scikit-learn
- Streamlit
- NumPy / Pandas / Matplotlib
- PIL.Image

---
