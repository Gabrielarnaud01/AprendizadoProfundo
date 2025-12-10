import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("🔄 Recriando arquivos PKL para a versão local do Scikit-Learn...")

# 1. Cria a pasta se não existir
if not os.path.exists('dados_processados'):
    os.makedirs('dados_processados')

try:
    # 2. Carrega e prepara os dados
    df = pd.read_csv('BodyFat - Extended.csv')
    df = df.drop(columns=['Original', 'Source'], errors='ignore')

    # Engenharia de Features (Igual ao App)
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['WHR'] = df['Abdomen'] / df['Hip']
    df['WHtR'] = df['Abdomen'] / (df['Height'] * 100)
    
    # 3. Encoder (Sexo)
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])

    # 4. Scaler (Treino)
    X = df.drop('BodyFat', axis=1)
    y = df['BodyFat']
    
    # IMPORTANTE: random_state=42 para manter a mesma matemática do treino da IA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    # 5. Salva (Sobrepondo os antigos)
    joblib.dump(scaler, 'dados_processados/scaler.pkl')
    joblib.dump(le, 'dados_processados/sex_encoder.pkl')

    print("✅ Sucesso! Arquivos atualizados.")
    print("Agora pare o Streamlit (Ctrl+C) e rode 'streamlit run app.py' novamente.")

except FileNotFoundError:
    print("❌ Erro: O arquivo 'BodyFat - Extended.csv' não está na pasta.")
