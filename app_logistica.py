import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

class ModeloRegressaoLogistica:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.features = []
        self.target = ''
        self.is_fitted = False
    
    def tratamento_dados(self, df, features, target):
        """Tratamento dos dados para o treinamento"""
        self.features = features
        self.target = target

        X = df[features]
        y = df[target]

        if y.dtype == 'object':
            y = y.map({'W':1, 'L':0, 'Vitória':1, 'Derrota':0})
        
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y
    
    def treinamento(self, X, y):
        """Treinando o modelo"""
        self.model.fit(X, y)
        self.is_fitted = True

    def calcular_probabilidade(self, X):
        """Predição em probabilidade"""
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado ainda")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predicao(self, X):
        """Previsões binárias"""
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado ainda")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


st.set_page_config(
    page_title="Celtics Stats Analyzer - Regressão Logística",
    page_icon="🏀",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #007A33;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .celtics-green {
        background-color: #007A33;
        color: white;
        padding: 10px;
        border-radius: 10px;
    }
    .stats-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #007A33;
        margin: 10px 0px;
    }
    .stButton>button {
        background-color: #007A33;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #005A25;
        color: white;
    }
    .spacing-large {
        margin-bottom: 3rem;
    }
    .spacing-medium {
        margin-bottom: 2rem;
    }
    .spacing-small {
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def carregar_dados():
    df = pd.read_csv("celtics_2024_25.csv")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE")

   
    df = df.rename(columns={
        "GAME_DATE": "Data do Jogo",
        "MATCHUP": "Confronto",
        "WL": "Vitória/Derrota",
        "PTS": "Pontos",
        "REB": "Rebotes",
        "AST": "Assistências",
        "FGM": "Arremessos Convertidos",
        "FGA": "Arremessos Tentados",
        "FG_PCT": "Percentual de Arremesso",
        "FG3M": "Cestas de 3 Convertidas",
        "FG3A": "Cestas de 3 Tentativas",
        "FG3_PCT": "Percentual de 3 Pontos",
        "FTM": "Lances Livres Convertidos",
        "FTA": "Lances Livres Tentados",
        "FT_PCT": "Percentual de Lances Livres",
        "OREB": "Rebotes Ofensivos",
        "DREB": "Rebotes Defensivos",
        "STL": "Roubos de Bola",
        "BLK": "Tocos",
        "TOV": "Erros (Turnovers)",
        "PF": "Faltas",
        "PLUS_MINUS": "+/-"
    })
    return df

def criar_variavel_target(df):
    """Criando variável binária para vitória/derrota"""
    df_copy = df.copy()

  
    if 'Vitória/Derrota' in df_copy.columns:
        df_copy['VITORIA'] = df_copy['Vitória/Derrota'].map({'W':1, 'L':0})
    else:
        df_copy['VITORIA'] = (df_copy['Pontos'] > df_copy['Pontos'].median()).astype(int)

    return df_copy


st.markdown('<h1 class="main-header">🏀 Celtics Stats Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<div class="celtics-green"><h3 style="margin:0; text-align:center;">Regressão Logística - Previsão de Vitórias</h3></div>', unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div style='text-align: center; margin: 20px 0;'>
            <p style='font-size: 1.2rem;'>Preveja a probabilidade de vitória dos Celtics usando Regressão Logística.</p>
        </div>
        """, unsafe_allow_html=True)


df = carregar_dados()


with st.sidebar:
    st.markdown("### ☘️ Configurações do Modelo")
    st.markdown("---")
    
   
    st.markdown("**Filtro por Data**")
    min_date = df["Data do Jogo"].min()
    max_date = df["Data do Jogo"].max()
    date_range = st.date_input(
        "Selecione o período:",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    

    if len(date_range) == 2:
        mask = (df["Data do Jogo"] >= pd.to_datetime(date_range[0])) & (df["Data do Jogo"] <= pd.to_datetime(date_range[1]))
        df = df[mask]

    st.markdown("---")
    st.markdown("### 📈 Configuração Avançada")
    
    st.markdown("**Filtro para Modelo Logístico**")
    use_all_data = st.checkbox("Usar todos os dados para treinamento", value=True)


st.markdown("---")
st.markdown("### ☘️ Visualização dos Dados")

with st.expander("Clique para ver os dados da temporada", expanded=False):
    col1, col2 = st.columns([3,1])
    
    with col1:
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.markdown("#### Estatísticas Gerais")
        st.metric("🍀 Total de Jogos", len(df))
        st.metric("🏆 Vitórias", len(df[df["Vitória/Derrota"] == "W"]))
        st.metric("💔 Derrotas", len(df[df["Vitória/Derrota"] == "L"]))
        st.metric("💚 Pontos por Jogo", f"{df['Pontos'].mean():.1f}")


df_logistica = criar_variavel_target(df)


st.markdown("---")
st.markdown("### 📊 Regressão Logística - Previsão de Vitórias")

st.markdown("""
<div class="stats-card">
    <h4 style="margin:0; color: #007A33;">📊 Sobre a Regressão Logística</h4>
    <p style="margin:5px 0; font-size: 0.9rem;">
    • Produz uma curva em forma de "S" (Sigmoide) entre 0 e 1<br>
    • O eixo Y representa a <b>probabilidade</b> do evento ocorrer<br>
    • Se probabilidade > 0.5: previsão de VITÓRIA (1)<br>
    • Se probabilidade < 0.5: previsão de DERROTA (0)
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown("---")
st.markdown("### 🎯 Configuração do Modelo")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Variável Target (Y)**")
    st.info("Modelo prevê: **Probabilidade de Vitória**")
    st.metric("Total de Vitórias", f"{df_logistica['VITORIA'].sum()}")
    st.metric("Taxa de Vitórias", f"{(df_logistica['VITORIA'].sum() / len(df_logistica)) * 100:.1f}%")

with col2:
    st.markdown("**Variáveis Preditoras (X)**")
    st.markdown("*Selecione as estatísticas que influenciam a vitória:*")
    
    
    vars_nao_permitidas = ["SEASON_ID", "TEAM_ID", "GAME_ID", "Data do Jogo", "Confronto", "Vitória/Derrota"]
    vars_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    vars_permitidas = [v for v in vars_numericas if v not in vars_nao_permitidas]
    
    
    vars_logistic = [v for v in vars_permitidas if v != 'VITORIA' and v not in ['Pontos']]
    
    selected_logistic_vars = st.multiselect(
        "Selecione as variáveis:",
        vars_logistic,
        default=['Percentual de Arremesso', 'Rebotes', 'Assistências', 'Roubos de Bola'],
        key="logistic_vars"
    )

if len(selected_logistic_vars) == 0:
    st.warning("Selecione ao menos uma variável preditora para a Regressão Logística.")
    st.stop()


try:
   
    modelo_logistico = ModeloRegressaoLogistica()
    X_log, y_log = modelo_logistico.tratamento_dados(df_logistica, selected_logistic_vars, 'VITORIA')
    
    
    modelo_logistico.treinamento(X_log, y_log)
    
    
    y_pred_log = modelo_logistico.predicao(df_logistica[selected_logistic_vars])
    y_pred_proba = modelo_logistico.calcular_probabilidade(df_logistica[selected_logistic_vars])
    
    
    accuracy = accuracy_score(y_log, y_pred_log)
    auc_score = roc_auc_score(y_log, y_pred_proba[:, 1])

    
    st.markdown("---")
    st.markdown("### 📊 Resultados da Regressão Logística")
    
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <h4 style="margin:0; color: #007A33;">Acurácia</h4>
            <h2 style="margin:0; color: #007A33;">{accuracy:.1%}</h2>
            <p style="margin:0; font-size: 0.8rem;">Precisão do modelo</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <h4 style="margin:0; color: #007A33;">AUC-ROC</h4>
            <h2 style="margin:0; color: #007A33;">{auc_score:.3f}</h2>
            <p style="margin:0; font-size: 0.8rem;">Capacidade de discriminação</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        vitorias_corretas = np.sum((y_pred_log == 1) & (y_log == 1))
        total_vitorias = np.sum(y_log == 1)
        recall = vitorias_corretas / total_vitorias if total_vitorias > 0 else 0
        st.markdown(f"""
        <div class="stats-card">
            <h4 style="margin:0; color: #007A33;">Recall</h4>
            <h2 style="margin:0; color: #007A33;">{recall:.1%}</h2>
            <p style="margin:0; font-size: 0.8rem;">Vitórias identificadas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        previsoes_vitoria = np.sum(y_pred_log == 1)
        st.markdown(f"""
        <div class="stats-card">
            <h4 style="margin:0; color: #007A33;">Previsões</h4>
            <h2 style="margin:0; color: #007A33;">{previsoes_vitoria}/{len(y_pred_log)}</h2>
            <p style="margin:0; font-size: 0.8rem;">Vitórias previstas</p>
        </div>
        """, unsafe_allow_html=True)
    

    st.markdown("#### 📈 Coeficientes do Modelo")

    coef_df = pd.DataFrame({
        'Variável': selected_logistic_vars,
        'Coeficiente': modelo_logistico.model.coef_[0],
        'Odds Ratio': np.exp(modelo_logistico.model.coef_[0]),
        'Impacto': np.abs(modelo_logistico.model.coef_[0])
    }).sort_values('Impacto', ascending=False)

    coef_df['Efeito'] = coef_df['Coeficiente'].apply(
        lambda x: '🟢 Aumenta chance' if x > 0 else '🔴 Diminui chance' if x < 0 else '⚪ Neutro'
    )

    coef_display = coef_df[['Variável', 'Coeficiente', 'Odds Ratio', 'Efeito']].copy()
    coef_display['Coeficiente'] = coef_display['Coeficiente'].apply(lambda x: f'{x:.4f}')
    coef_display['Odds Ratio'] = coef_display['Odds Ratio'].apply(lambda x: f'{x:.4f}')
    
    st.dataframe(coef_display, use_container_width=True, hide_index=True)
    
   
    st.markdown("#### 💡 Interpretação dos Coeficientes")

    with st.expander("Clique para entender a interpretação:", expanded=False):
        st.markdown("""
        - **Coeficiente Positivo**: Aumenta a probabilidade de vitória
        - **Coeficiente Negativo**: Diminui a probabilidade de vitória  
        - **Odds Ratio > 1**: Aumenta as chances de vitória
        - **Odds Ratio < 1**: Diminui as chances de vitória
        - **Exemplo**: Odds Ratio = 1.5 significa 50% mais chances de vitória
        """)

    st.markdown("---")
    st.markdown("### 📊 Visualizações da Regressão Logística")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Curva ROC", "Matriz Confusão", "Probabilidades", "Equação"])

    with tab1:
      
        fpr, tpr, thresholds = roc_curve(y_log, y_pred_proba[:, 1])
        
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        ax_roc.plot(fpr, tpr, color='#007A33', linewidth=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
        ax_roc.set_xlabel('Taxa de Falsos Positivos')
        ax_roc.set_ylabel('Taxa de Verdadeiros Positivos')
        ax_roc.set_title('Curva ROC - Desempenho do Modelo', fontweight='bold')
        ax_roc.legend()
        ax_roc.grid(True, alpha=0.3)
        st.pyplot(fig_roc)
    
    with tab2:
       
        cm = confusion_matrix(y_log, y_pred_log)
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax_cm,
                   xticklabels=['Derrota Prev', 'Vitória Prev'],
                   yticklabels=['Derrota Real', 'Vitória Real'])
        ax_cm.set_title('Matriz de Confusão', fontweight='bold')
        st.pyplot(fig_cm)

        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        - **Verdadeiros Positivos**: {tp} vitórias corretamente previstas
        - **Verdadeiros Negativos**: {tn} derrotas corretamente previstas  
        - **Falsos Positivos**: {fp} derrotas previstas como vitórias
        - **Falsos Negativos**: {fn} vitórias previstas como derrotas
        """)
    
    with tab3:
       
        fig_prob, ax_prob = plt.subplots(figsize=(10, 6))
        
       
        prob_vitorias = y_pred_proba[y_log == 1, 1]
        prob_derrotas = y_pred_proba[y_log == 0, 1]
        
        ax_prob.hist(prob_vitorias, bins=20, alpha=0.7, color='#007A33', label='Vitórias Reais', density=True)
        ax_prob.hist(prob_derrotas, bins=20, alpha=0.7, color='#BA9653', label='Derrotas Reais', density=True)
        ax_prob.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Limite de Decisão (0.5)')
        ax_prob.set_xlabel('Probabilidade Prevista de Vitória')
        ax_prob.set_ylabel('Densidade')
        ax_prob.set_title('Distribuição das Probabilidades Previstas', fontweight='bold')
        ax_prob.legend()
        ax_prob.grid(True, alpha=0.3)
        st.pyplot(fig_prob)
    
    with tab4:
    
        st.markdown("#### 🧮 Equação da Regressão Logística")
        
        eq_parts = [f"{modelo_logistico.model.intercept_[0]:.4f}"]
        for coef, var in zip(modelo_logistico.model.coef_[0], selected_logistic_vars):
            eq_parts.append(f"{coef:+.4f} × {var}")
        
        eq_linear = " + ".join(eq_parts)
        eq_final = f"p = 1 / [1 + e^(-({eq_linear}))]"
        
        st.code(eq_final, language="latex")

        st.markdown("#### 🎯 Exemplo de Cálculo")
        
       
        sample_idx = 0
        sample_data = df_logistica[selected_logistic_vars].iloc[sample_idx]
        
        st.markdown(f"**Dados do jogo {sample_idx + 1}:**")
        st.write(sample_data)
        
      
        z = modelo_logistico.model.intercept_[0]
        for coef, var in zip(modelo_logistico.model.coef_[0], selected_logistic_vars):
            z += coef * sample_data[var]
        
        prob_manual = 1 / (1 + np.exp(-z))
        prob_model = y_pred_proba[sample_idx, 1]
        
        st.markdown(f"""
        - **Cálculo linear (z)**: {z:.4f}
        - **Probabilidade calculada**: {prob_manual:.1%}
        - **Probabilidade do modelo**: {prob_model:.1%}
        - **Previsão**: {'🏆 VITÓRIA' if prob_model > 0.5 else '💔 DERROTA'}
        - **Resultado real**: {'🏆 VITÓRIA' if y_log.iloc[sample_idx] == 1 else '💔 DERROTA'}
        """)

    
    st.markdown("---")
    st.markdown("### 🔮 Simulador de Previsões")
    
    st.markdown("**Insira valores para prever a probabilidade de vitória:**")
    
    col1, col2 = st.columns(2)
    
    input_values = {}

    with col1:
        for i, var in enumerate(selected_logistic_vars[:len(selected_logistic_vars)//2]):
            min_val = float(df_logistica[var].min())
            max_val = float(df_logistica[var].max())
            mean_val = float(df_logistica[var].mean())
            
            input_values[var] = st.slider(
                f"{var}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                key=f"sim_{var}"
            )
    
    with col2:
        for i, var in enumerate(selected_logistic_vars[len(selected_logistic_vars)//2:]):
            min_val = float(df_logistica[var].min())
            max_val = float(df_logistica[var].max())
            mean_val = float(df_logistica[var].mean())
            
            input_values[var] = st.slider(
                f"{var}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                key=f"sim2_{var}"
            )
    
   
    if st.button("🎯 Calcular Probabilidade de Vitória", type="primary"):
        input_df = pd.DataFrame([input_values])
        proba = modelo_logistico.calcular_probabilidade(input_df)[0, 1]
        
        st.markdown("---")
        if proba > 0.5:
            st.success(f"## 🏆 Probabilidade de Vitória: {proba:.1%}")
            st.balloons()
        else:
            st.error(f"## 💔 Probabilidade de Vitória: {proba:.1%}")
        
        st.markdown(f"""
        <div class="stats-card">
            <h4>📊 Detalhes da Previsão:</h4>
            <p>• <b>Probabilidade</b>: {proba:.1%}</p>
            <p>• <b>Limite de decisão</b>: 50%</p>
            <p>• <b>Previsão</b>: {'VITÓRIA' if proba > 0.5 else 'DERROTA'}</p>
            <p>• <b>Confiança</b>: {'Alta' if proba > 0.7 or proba < 0.3 else 'Média' if proba > 0.6 or proba < 0.4 else 'Baixa'}</p>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Erro ao treinar o modelo de regressão logística: {str(e)}")
    st.info("Verifique se as variáveis selecionadas são adequadas e se há dados suficientes.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏀 Boston Celtics Stats Analyzer - Regressão Logística | Temporada 2024-25</p>
</div>
""", unsafe_allow_html=True)