import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import joblib

# ==============================================================================
# 1. DEFINIÇÃO DAS CLASSES DOS MODELOS (Exatamente como foram treinadas)
# ==============================================================================

# --- Modelo 1: Visão Computacional (DualViewBodyModel) ---
class DualViewBodyModel(nn.Module):
    def __init__(self, num_targets):
        super(DualViewBodyModel, self).__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.regressor = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_targets)
        )
    
    def forward(self, img_front, img_side):
        f_front = self.feature_extractor(img_front)
        f_front = f_front.view(f_front.size(0), -1)
        f_side = self.feature_extractor(img_side)
        f_side = f_side.view(f_side.size(0), -1)
        combined = torch.cat((f_front, f_side), dim=1)
        return self.regressor(combined)

# --- Modelo 2: Preditor de Gordura (BodyFatPredictorSimple) ---
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
        return self.output(x)

# ==============================================================================
# 2. CONFIGURAÇÕES E CARREGAMENTO
# ==============================================================================

# Chaves que o Modelo de Visão prevê (na ordem exata do treinamento)
VISION_KEYS = [
    "arm_length_cm", "chest_circumference_cm", "front_build_cm", 
    "hips_circumference_cm", "leg_length_cm", "neck_circumference_cm",
    "shoulder_width_cm", "thigh_circumference_cm", "waist_circumference_cm"
]

@st.cache_resource
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Carregar Scaler e Encoder
        scaler = joblib.load('dados_processados/scaler.pkl')
        encoder = joblib.load('dados_processados/sex_encoder.pkl')
        
        # Carregar Modelo de Visão (Modelo 1)
        # Nota: Ajuste o num_targets se você treinou com ou sem peso/altura. 
        # Assumindo 9 targets (sem peso/altura) conforme nossa última conversa.
        vision_model = DualViewBodyModel(num_targets=len(VISION_KEYS)).to(device)
        vision_model.load_state_dict(torch.load('modelo_medidas_visao.pth', map_location=device)) # Descomente quando tiver o arquivo
        vision_model.eval()

        # Carregar Modelo de Gordura (Modelo 2)
        n_features = scaler.n_features_in_
        fat_model = BodyFatPredictorSimple(input_dim=n_features).to(device)
        fat_model.load_state_dict(torch.load('modelo_bodyfat_avancado.pth', map_location=device))
        fat_model.eval()
        
        return vision_model, fat_model, scaler, encoder, device
    
    except FileNotFoundError as e:
        st.error(f"Arquivo faltando: {e}. Verifique se todos os .pth e .pkl estão na pasta.")
        return None, None, None, None, device

# Configuração da Página
st.set_page_config(page_title="AI Body Fat Estimator", layout="centered")
st.title("💪 AI Body Composition Analyzer")
st.markdown("Estime suas medidas e gordura corporal usando apenas duas fotos.")

# Carrega tudo
vision_model, fat_model, scaler, encoder, device = load_resources()

if vision_model is None:
    st.stop() # Para se não carregar

# ==============================================================================
# 3. INTERFACE - PASSO 0: DADOS DO USUÁRIO
# ==============================================================================

st.sidebar.header("1. Seus Dados")
sexo = st.sidebar.selectbox("Sexo Biológico", ["M", "F"])
idade = st.sidebar.number_input("Idade", 18, 100, 30)
altura_cm = st.sidebar.number_input("Altura (cm)", 100, 250, 175)
peso_kg = st.sidebar.number_input("Peso (kg)", 40.0, 200.0, 75.0)

# Converter altura para metros para cálculo interno
altura_m = altura_cm / 100.0

# ==============================================================================
# 4. INTERFACE - PASSO 1: UPLOAD E VISÃO COMPUTACIONAL
# ==============================================================================

st.header("Passo 1: Upload das Fotos")
col1, col2 = st.columns(2)
with col1:
    front_file = st.file_uploader("Foto Frontal", type=['jpg', 'png', 'jpeg'])
with col2:
    side_file = st.file_uploader("Foto Lateral", type=['jpg', 'png', 'jpeg'])

# Inicializa sessão para guardar medidas
if 'medidas_estimadas' not in st.session_state:
    st.session_state['medidas_estimadas'] = {}

def process_image(image_file):
    img = Image.open(image_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(device)

if front_file and side_file:
    if st.button("🔍 Analisar Medidas com IA"):
        with st.spinner("A IA está medindo você..."):
            img_f = process_image(front_file)
            img_s = process_image(side_file)
            
            with torch.no_grad():
                preds = vision_model(img_f, img_s).cpu().numpy()[0]
            
            # Mapear vetor de predição para dicionário
            resultados = {k: float(v) for k, v in zip(VISION_KEYS, preds)}
            
            # --- MAPEAR VISÃO -> TABULAR ---
            # O modelo tabular precisa de nomes específicos (Abdomen, Neck, etc.)
            # A visão dá nomes técnicos (waist_circumference_cm). Vamos traduzir.
            
            mapeamento = {
                'neck_circumference_cm': 'Neck',
                'chest_circumference_cm': 'Chest',
                'waist_circumference_cm': 'Abdomen', # Cintura vira Abdomen
                'hips_circumference_cm': 'Hip',
                'thigh_circumference_cm': 'Thigh'
                # Note que Biceps, Joelho, etc, a IA não vê. Vamos preencher depois.
            }
            
            # Salva no Session State
            st.session_state['medidas_estimadas'] = {}
            for k_vis, k_tab in mapeamento.items():
                st.session_state['medidas_estimadas'][k_tab] = resultados.get(k_vis, 0.0)
            
            st.success("Medidas extraídas com sucesso!")

# ==============================================================================
# 5. INTERFACE - PASSO 2: CONFERÊNCIA E CÁLCULO FINAL
# ==============================================================================

if st.session_state['medidas_estimadas']:
    st.divider()
    st.header("Passo 2: Conferência e Resultado Final")
    st.info("Abaixo estão as medidas que a IA encontrou. As que faltam foram estimadas por média. Você pode corrigir qualquer valor.")

    # Cria um formulário para o usuário confirmar os dados
    with st.form("form_final"):
        c1, c2, c3 = st.columns(3)
        
        # Recupera valores ou usa padrão
        vals = st.session_state['medidas_estimadas']
        
        # --- Estimativa Inteligente para Medidas Faltantes ---
        # Baseado em proporções corporais médias
        # Pulso ~ 10.5% da altura
        # Tornozelo ~ 13% da altura
        # Bíceps ~ 40% a 50% da Coxa (grosso modo)
        
        estimativa_pulso = altura_cm * 0.10
        estimativa_tornozelo = altura_cm * 0.13
        estimativa_joelho = altura_cm * 0.22
        estimativa_biceps = float(vals.get('Thigh', 55.0)) * 0.6 # Chute: biceps é 60% da coxa
        estimativa_antebraco = estimativa_biceps * 0.85

        st.markdown("---")
        st.caption("Medidas complementares (Pré-preenchidas por proporção, ajuste se necessário):")
        cc1, cc2, cc3, cc4 = st.columns(4)
        
        knee = cc1.number_input("Joelho", value=float(f"{estimativa_joelho:.1f}"))
        ankle = cc2.number_input("Tornozelo", value=float(f"{estimativa_tornozelo:.1f}"))
        biceps = cc3.number_input("Bíceps", value=float(f"{estimativa_biceps:.1f}"))
        forearm = cc4.number_input("Antebraço", value=float(f"{estimativa_antebraco:.1f}"))
        wrist = cc1.number_input("Pulso", value=float(f"{estimativa_pulso:.1f}"))
        
        calcular_btn = st.form_submit_button("🚀 Calcular % de Gordura Corporal")

    if calcular_btn:
        # 1. Montar Dicionário Completo
        dados_input = {
            'Sex': sexo,
            'Age': idade,
            'Weight': peso_kg,
            'Height': altura_m, # Modelo tabular usa metros (baseado no código Part 5)
            'Neck': neck,
            'Chest': chest,
            'Abdomen': abdomen,
            'Hip': hip,
            'Thigh': thigh,
            'Knee': knee,
            'Ankle': ankle,
            'Biceps': biceps,
            'Forearm': forearm,
            'Wrist': wrist
        }
        
        # 2. Feature Engineering (O Pulo do Gato - Igual ao Treino)
        df_input = pd.DataFrame([dados_input])
        
        df_input['BMI'] = df_input['Weight'] / (df_input['Height'] ** 2)
        df_input['WHR'] = df_input['Abdomen'] / df_input['Hip']
        df_input['WHtR'] = df_input['Abdomen'] / (df_input['Height'] * 100)
        
        # 3. Encoding e Scaling
        try:
            # Ordenar colunas conforme o Scaler aprendeu
            cols_treino = scaler.feature_names_in_
            
            # Encode Sex
            df_input['Sex'] = encoder.transform(df_input['Sex'])
            
            # Garantir ordem
            df_input = df_input[cols_treino]
            
            # Escalar
            X_final = scaler.transform(df_input)
            X_tensor = torch.tensor(X_final, dtype=torch.float32).to(device)
            
            # 4. Predição Final
            with torch.no_grad():
                bf_pred = fat_model(X_tensor).item()
            
            # 5. Exibição Bonita
            st.balloons()
            st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{bf_pred:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>de Gordura Corporal Estimada</p>", unsafe_allow_html=True)
            
            # Classificação Simples
            if sexo == 'M':
                ref = "14% (Atleta) - 24% (Médio)"
            else:
                ref = "21% (Atleta) - 31% (Médio)"
            st.info(f"Referência para seu sexo: {ref}")
            
        except Exception as e:
            st.error(f"Erro no processamento dos dados: {e}")
