import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cosmetics data
cosmetics_df = pd.read_csv("Recommender/cosmetics.csv")

# Helper: classify price
def price_category(price):
    if price < 50:
        return 'Low'
    elif 50 <= price < 150:
        return 'Mid-range'
    else:
        return 'High'

cosmetics_df['Price_category'] = cosmetics_df['Price'].apply(price_category)

# Constants
SKIN_TYPES = ['Oily', 'Dry', 'Normal']
SKIN_TONES = ['Black', 'Brown', 'White']
ACNE_SEVERITIES = ['Mild', 'Moderate', 'Severe']
PRODUCT_TYPES = cosmetics_df['Label'].unique().tolist()
PRICE_CATEGORIES = ['Low', 'Mid-range', 'High']

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Model loading
def load_model(path):
    model = torch.load(path, map_location=torch.device('cpu'))
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    return model

# Predict
@st.cache_resource
def get_prediction(_model, _image_tensor, class_names):
    with torch.no_grad():
        outputs = _model(_image_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]


def content_based_filter(df, skin_type, skin_tone, acne_severity):
    if skin_type in df.columns:
        df = df[df[skin_type] == 1]

    if skin_tone == 'Black':
        df = df[~df['Ingredients'].str.contains('Titanium Dioxide|Zinc Oxide|White Cast', case=False, na=False)]
    elif skin_tone == 'White':
        df = df[df['Ingredients'].str.contains('SPF|Sunscreen|Titanium Dioxide', case=False, na=False)]
    elif skin_tone == 'Brown':
        df = df[~df['Ingredients'].str.contains('Alcohol Denat|Fragrance', case=False, na=False)]

    if acne_severity == 'Severe':
        df = df[~df['Ingredients'].str.contains('Coconut Oil|Isopropyl|Lanolin|Myristate|Fragrance|Petrolatum', case=False, na=False)]
    elif acne_severity == 'Moderate':
        df = df[~df['Ingredients'].str.contains('Coconut Oil|Lanolin', case=False, na=False)]
    elif acne_severity == 'Mild':
        df = df[df['Ingredients'].str.contains('Niacinamide|Salicylic Acid|Zinc|Glycerin|Butylene Glycol|Phenoxyethanol|Sodium Hyaluronate', case=False, na=False)]

    return df

# Collaborative filtering
def collaborative_filtering(df, top_n=5):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Ingredients'].fillna(""))
    knn = NearestNeighbors(n_neighbors=min(top_n, len(df)), metric='cosine')
    knn.fit(tfidf_matrix)
    distances, indices = knn.kneighbors(tfidf_matrix)
    return df.iloc[indices[0]]

# Recommendation pipeline
def get_recommendations(image, product_type, price_category):
    tensor = preprocess_image(image)
    skin_type = get_prediction(load_model("Skin_Metrics/Skin_Type/skin_type.pt"), tensor, SKIN_TYPES)
    skin_tone = get_prediction(load_model("Skin_Metrics/Skin_Tone/skin_tone.pt"), tensor, SKIN_TONES)
    acne_severity = get_prediction(load_model("Skin_Metrics/Acne/acne.pt"), tensor, ACNE_SEVERITIES)

    df = cosmetics_df[(cosmetics_df['Label'].str.lower() == product_type.lower()) &
                      (cosmetics_df['Price_category'].str.lower() == price_category.lower())].copy()

    filtered = content_based_filter(df, skin_type, skin_tone, acne_severity)
    recommended = collaborative_filtering(filtered)
    return recommended[['Label', 'Brand', 'Name', 'Price', 'Rank']]

# Streamlit UI
st.title("Skincare Product Recommender")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
product_type = st.selectbox("Select product type", PRODUCT_TYPES)
price_category = st.selectbox("Select price category", PRICE_CATEGORIES)

if uploaded_file:
    original_image = Image.open(uploaded_file)
    st.subheader("Uploaded Image")
    st.image(original_image, caption="Uploaded Face Image", width=300)  

    image = original_image.convert("RGB")

    if st.button("Recommend Products"):
        recommendations = get_recommendations(image, product_type, price_category)
        st.subheader("Top Product Recommendations")
        st.dataframe(recommendations.reset_index(drop=True))

