# Projeto Bodyfat Vision AI
	Disciplina: Aprendizado Profundo (Deep Learning)
	Semestre: 2025.1
	Professor: [Nome do Professor]
	Turma: [Sua Turma]

# Integrantes do Grupo
	GABRIEL ARNAUD PAIVA TORRES (20210093332)
	DAVI VIEIRA DE CARVALHO LIMA (20220077619)

# 🧠 Descrição — Estimativa de Gordura via Visão Computacional
    Este repositório implementa um sistema de **Deep Learning** para análise de composição corporal, utilizando duas fotos (frontal e lateral) para estimar medidas e percentual de gordura. O projeto integra:

    📸 **Visão Computacional** com Backbone ResNet18 para extração de medidas corporais.
    🧠 **Rede Neural Tabular** para predição final de gordura corporal.
    📊 **Dashboard Streamlit** para upload de imagens e visualização de resultados.
    📏 **Engenharia de Features** automática (Cálculo de IMC, WHR, WHtR).
    🚀 **Pipeline Híbrido** (Imagem + Dados Demográficos).

# 🚀 Como Instalar e Executar o Projeto

    **Pré-requisito:** Python 3.11.9 (Versão recomendada)

    1) Criar e ativar ambiente virtual (Opcional, mas recomendado)
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate

    2) Instalar dependências
    pip install streamlit torch torchvision pandas numpy joblib Pillow scikit-learn

    3) Verificar arquivos de modelo
    Certifique-se de que os seguintes arquivos estão na pasta raiz ou em 'dados_processados/':
    - modelo_medidas_visao.pth
    - modelo_bodyfat_avancado.pth
    - dados_processados/scaler.pkl
    - dados_processados/sex_encoder.pkl

    4) Rodar o Dashboard Streamlit
    streamlit run app.py
    
    A aplicação abrirá automaticamente em:
    http://localhost:8501

# 🧩 Arquivos e Classes Principais

### 📸 DualViewBodyModel (Visão Computacional)
Classe PyTorch responsável por processar as imagens.
Implementa:
- Backbone **ResNet18** pré-treinada.
- Fusão de características de duas visões (Frontal + Lateral).
- Camada de regressão para estimar 9 medidas corporais (Peito, Cintura, Quadril, etc.).

O método central no `forward` concatena os vetores de características:
```python
    combined = torch.cat((f_front, f_side), dim=1)
    return self.regressor(combined)
