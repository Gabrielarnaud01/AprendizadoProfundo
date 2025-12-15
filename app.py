import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import cv2
import mediapipe as mp
from PIL import Image
import math


# 1. DEFINIÇÃO DA MLP (CÉREBRO MATEMÁTICO)

class BodyFatPredictorSimple(nn.Module):
    def __init__(self, input_dim):
        super(BodyFatPredictorSimple, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(64, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.relu(self.layer1(x)))
        x = self.dropout2(self.relu(self.layer2(x)))
        x = self.relu(self.layer3(x))
        x = self.output(x)
        return x


# 2. CONFIGURAÇÕES E MEDIAPIPE

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

@st.cache_resource
def load_ml_resources():
    device = torch.device("cpu") 
    try:
        # Carregar Scaler e Encoder
        scaler = joblib.load('dados_processados/scaler.pkl')
        encoder = joblib.load('dados_processados/sex_encoder.pkl') 
        
        # Carregar Modelo MLP
        n_features = scaler.n_features_in_
        fat_model = BodyFatPredictorSimple(input_dim=n_features).to(device)
       
        fat_model.load_state_dict(torch.load('modelo_bodyfat_avancado.pth', map_location=device))
        fat_model.eval()
        
        return fat_model, scaler, encoder
    except FileNotFoundError:
        return None, None, None


# 3. LÓGICA DE VISÃO 

def processar_medidas_mediapipe(image_file, altura_real_cm):
    # Converter upload do Streamlit (Bytes) para Imagem OpenCV
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Processar com MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None, image # Retorna imagem original se falhar

    # Desenhar esqueleto na imagem (Para mostrar ao usuário)
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB) 

    lm = results.pose_landmarks.landmark
    h, w, _ = image.shape

    # --- A. CÁLCULO DA ESCALA ---
    # Pegar pontos chaves
    nose = lm[mp_pose.PoseLandmark.NOSE]
    l_heel = lm[mp_pose.PoseLandmark.LEFT_HEEL]
    r_heel = lm[mp_pose.PoseLandmark.RIGHT_HEEL]

    # Altura em Pixels (Nariz até média dos pés)
    feet_x = (l_heel.x + r_heel.x) / 2
    feet_y = (l_heel.y + r_heel.y) / 2
    
    # Distância Euclidiana (em pixels reais)
    px_height = math.sqrt(((nose.x - feet_x) * w)**2 + ((nose.y - feet_y) * h)**2)
    
    # Correção: O nariz está a ~88% da altura (Topo da cabeça é mais alto)
    px_height_corrected = px_height * 1.15
    
    # Fator de Conversão: Quantos CM vale 1 Pixel?
    cm_per_px = altura_real_cm / px_height_corrected

    # --- B. EXTRAÇÃO DE LARGURAS ---
    def get_width(p1_enum, p2_enum):
        p1 = lm[p1_enum]
        p2 = lm[p2_enum]
        dist_px = math.sqrt(((p1.x - p2.x) * w)**2 + ((p1.y - p2.y) * h)**2)
        return dist_px * cm_per_px

    w_shoulder = get_width(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    w_hip = get_width(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP)
    
    # --- C. APLICAÇÃO DOS FATORES CALIBRADOS ---
    # Aqui usamos os valores que descobrimos na análise de dados anterior
    medidas = {
        'Neck': w_shoulder * 1.2,
        'Chest': w_shoulder * 2.8, 
        'Abdomen': w_hip * 3.5,    # Calibrado (era 2.9)
        'Hip': w_hip * 3.8,        # Calibrado (era 3.1)
        'Thigh': w_hip * 1.6
    }
    
    return medidas, annotated_image

# ==============================================================================
# 4. INTERFACE STREAMLIT
# ==============================================================================
st.set_page_config(page_title="BodyScan AI", layout="centered", page_icon="💪")

st.title("💪 BodyScan AI: Avaliação Física Inteligente")
st.markdown("""
Esta aplicação utiliza **Visão Computacional Geométrica (MediaPipe)** para extrair medidas corporais 
e uma **Rede Neural (MLP)** para calcular a gordura corporal com precisão clínica.
""")

# Carregar recursos
fat_model, scaler, encoder = load_ml_resources()

if fat_model is None:
    st.error("Erro Crítico: Modelos (.pth/.pkl) não encontrados na pasta 'dados_processados/'.")
    st.stop()

# --- SIDEBAR: DADOS DO USUÁRIO ---
st.sidebar.header("1. Dados Biométricos")
sexo = st.sidebar.selectbox("Sexo", ["M", "F"])
idade = st.sidebar.slider("Idade", 18, 80, 25)
altura_cm = st.sidebar.number_input("Altura (cm)", 100, 230, 175)
peso_kg = st.sidebar.number_input("Peso (kg)", 40.0, 180.0, 80.0)

# --- ÁREA PRINCIPAL: UPLOAD ---
st.header("2. Escaneamento Corporal")
st.info("Envie uma foto de corpo inteiro, de frente, com roupas justas.")

uploaded_file = st.file_uploader("Carregar Foto Frontal", type=['jpg', 'jpeg', 'png'])

# Inicializar estado
if 'medidas_ia' not in st.session_state:
    st.session_state['medidas_ia'] = {}

if uploaded_file is not None:
    if st.button("🔍 Escanear Medidas"):
        with st.spinner("O MediaPipe está mapeando seu esqueleto..."):
            medidas, img_esqueleto = processar_medidas_mediapipe(uploaded_file, altura_cm)
            
            if medidas:
                st.session_state['medidas_ia'] = medidas
                st.image(img_esqueleto, caption="Esqueleto Identificado pela IA", use_column_width=True)
                st.success("Medidas extraídas com sucesso!")
            else:
                st.error("Não foi possível detectar uma pessoa na imagem. Tente uma foto mais clara.")

# --- FORMULÁRIO DE CONFIRMAÇÃO ---
if st.session_state['medidas_ia']:
    st.divider()
    st.header("3. Conferência de Medidas")
    
    # Valores padrões (Heurística baseada na altura para o que a IA não vê)
    defaults = st.session_state['medidas_ia']
    
    with st.form("form_calculo"):
        c1, c2 = st.columns(2)
        
        # Medidas Principais (Vindas da IA)
        c1.markdown("### 📏 Medidas Detectadas")
        abdomen = c1.number_input("Abdômen (Cintura)", value=float(defaults.get('Abdomen', 90.0)))
        hip = c1.number_input("Quadril", value=float(defaults.get('Hip', 100.0)))
        neck = c1.number_input("Pescoço", value=float(defaults.get('Neck', 38.0)))
        chest = c1.number_input("Peitoral", value=float(defaults.get('Chest', 100.0)))
        
        # Medidas Secundárias (Estimadas por proporção)
        c2.markdown("### 🧬 Estimativas Proporcionais")
        thigh = c2.number_input("Coxa", value=float(defaults.get('Thigh', 55.0)))
        knee = c2.number_input("Joelho", value=altura_cm * 0.22)
        ankle = c2.number_input("Tornozelo", value=altura_cm * 0.13)
        biceps = c2.number_input("Bíceps", value=altura_cm * 0.18)
        forearm = c2.number_input("Antebraço", value=altura_cm * 0.16)
        wrist = c2.number_input("Pulso", value=altura_cm * 0.10)
        
        submitted = st.form_submit_button("🚀 Calcular Gordura Corporal")
        
    if submitted:
        # Preparar dados para a MLP
        dados = {
            'Sex': 1 if sexo == 'M' else 0, # Assumindo encoder binário simples, ajuste se usar LabelEncoder
            'Age': idade,
            'Weight': peso_kg,
            'Height': altura_cm / 100.0, # Metros
            'Neck': neck, 'Chest': chest, 'Abdomen': abdomen, 'Hip': hip,
            'Thigh': thigh, 'Knee': knee, 'Ankle': ankle,
            'Biceps': biceps, 'Forearm': forearm, 'Wrist': wrist
        }
        
        # Engenharia de Atributos (CRUCIAL: Tem que ser igual ao treino)
        df = pd.DataFrame([dados])
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        df['WHR'] = df['Abdomen'] / df['Hip']
        df['WHtR'] = df['Abdomen'] / (df['Height'] * 100)
        
        # Predição
        try:
            # Reordenar colunas
            df = df[scaler.feature_names_in_]
            
            # Escalar
            X = scaler.transform(df)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            
            # Inferência
            gordura = fat_model(X_tensor).item()
            
            # Resultado
            st.markdown("---")
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric(label="Gordura Corporal", value=f"{gordura:.1f}%")
                
            with col_res2:
                # Classificação básica
                if sexo == 'M':
                    if gordura < 6: status = "Gordura Essencial"
                    elif gordura < 14: status = "Atleta"
                    elif gordura < 25: status = "Fitness/Médio"
                    else: status = "Obeso"
                else:
                    if gordura < 14: status = "Gordura Essencial"
                    elif gordura < 21: status = "Atleta"
                    elif gordura < 32: status = "Fitness/Médio"
                    else: status = "Obeso"
                
                st.info(f"Classificação Estimada: **{status}**")
                st.progress(min(gordura/50, 1.0))
                
        except Exception as e:
            st.error(f"Erro no cálculo: {e}")
            st.warning("Verifique se as colunas do Scaler batem com os dados gerados.")