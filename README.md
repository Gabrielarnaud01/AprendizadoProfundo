# Bodyfat Vision AI Project
	Course: Deep Learning
	Semester: 2025.1
	Professor: PATRICK CESAR ALVES TERREMATTE
	Class: T01

# Group Members
	GABRIEL ARNAUD PAIVA TORRES (20210093332)
	DAVI VIEIRA DE CARVALHO LIMA (20220077619)

# 🧠 Description — Body Fat Estimation via Computer Vision
    This repository implements a **Deep Learning** system for body composition analysis, using two photos (front and side) to estimate measurements and body fat percentage. The project integrates:

    📸 **Computer Vision** with a ResNet18 Backbone for body measurement extraction.
    🧠 **Tabular Neural Network** for final body fat prediction.
    📊 **Streamlit Dashboard** for image upload and results visualization.
    📏 Automatic **Feature Engineering** (BMI, WHR, WHtR calculation).
    🚀 **Hybrid Pipeline** (Image + Demographic Data).

# 🚀 How to Install and Run the Project

    **Prerequisite:** Python 3.11.9 (Recommended version)

    1) Create and activate a virtual environment (Optional, but recommended)
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate

    2) Install dependencies
    pip install streamlit torch torchvision pandas numpy joblib Pillow scikit-learn

    3) Verify model files
    Make sure the following files are in the root folder or inside 'dados_processados/':
    - modelo_medidas_visao.pth
    - modelo_bodyfat_avancado.pth
    - dados_processados/scaler.pkl
    - dados_processados/sex_encoder.pkl

    4) Run the Streamlit Dashboard
    streamlit run app.py
    
    The application will automatically open at:
    http://localhost:8501

# 🧩 Main Files and Classes

### 📸 DualViewBodyModel (Computer Vision)
PyTorch class responsible for processing the images.
Implements:
- Pre-trained **ResNet18** Backbone.
- Feature fusion from two views (Front + Side).
- Regression layer to estimate 9 body measurements (Chest, Waist, Hip, etc.).

The core method in the `forward` pass concatenates the feature vectors:
```python
    combined = torch.cat((f_front, f_side), dim=1)
    return self.regressor(combined)
