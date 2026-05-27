import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🏦",
    layout="wide"
)

BASE = os.path.dirname(os.path.abspath(__file__))

# ─── SIDEBAR ───────────────────────────────────────────────
st.sidebar.title("🏦 Navigation")
page = st.sidebar.radio("Aller vers :", [
    "🏠 Accueil",
    "🤖 CNN",
    "🤖 LSTM",
    "🤖 GRU",
    "🤖 CNN-LSTM-GRU",
    "📊 Comparaison",
    "🎯 Simulateur"
])

# ─── FONCTION CHARGEMENT ───────────────────────────────────
@st.cache_data
def load_algo(algo):
    path = os.path.join(BASE, algo)
    with open(os.path.join(path, "metriques.json")) as f:
        metriques = json.load(f)
    cm       = np.load(os.path.join(path, "confusion_matrix.npy"))
    fpr      = np.load(os.path.join(path, "fpr.npy"))
    tpr      = np.load(os.path.join(path, "tpr.npy"))
    loss     = np.load(os.path.join(path, "history_loss.npy"))
    val_loss = np.load(os.path.join(path, "history_val_loss.npy"))
    acc      = np.load(os.path.join(path, "history_acc.npy"))
    val_acc  = np.load(os.path.join(path, "history_val_acc.npy"))
    return metriques, cm, fpr, tpr, loss, val_loss, acc, val_acc

# ─── FONCTION AFFICHAGE ALGO ───────────────────────────────
def show_algo(algo):
    metriques, cm, fpr, tpr, loss, val_loss, acc, val_acc = load_algo(algo)

    st.title(f"🤖 Modèle {algo}")
    st.divider()

    st.header("📊 Performance")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Accuracy",  f"{metriques['accuracy']:.2%}")
    col2.metric("F1 Score",  f"{metriques['f1']:.2%}")
    col3.metric("ROC AUC",   f"{metriques['roc_auc']:.2%}")
    col4.metric("Recall",    f"{metriques['recall']:.2%}")
    col5,col6,col7 = st.columns(3)
    col5.metric("Précision",   f"{metriques['precision']:.2%}")
    col6.metric("Spécificité", f"{metriques['specificity']:.2%}")
    col7.metric("MCC",         f"{metriques['mcc']:.4f}")
    st.divider()

    st.header("🔥 Matrice de Confusion")
    fig1, ax1 = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=["Normal","Fraude"],
                yticklabels=["Normal","Fraude"])
    ax1.set_ylabel("Réel"); ax1.set_xlabel("Prédit")
    st.pyplot(fig1)
    st.divider()

    st.header("📈 Courbes ROC & Entraînement")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC={metriques['roc_auc']:.3f}")
        ax2.plot([0,1],[0,1],'k--')
        ax2.set_title("Courbe ROC")
        ax2.legend()
        st.pyplot(fig2)
    with c2:
        fig3, ax3 = plt.subplots()
        ax3.plot(loss, label="Train")
        ax3.plot(val_loss, label="Val")
        ax3.set_title("Loss")
        ax3.legend()
        st.pyplot(fig3)
    with c3:
        fig4, ax4 = plt.subplots()
        ax4.plot(acc, label="Train")
        ax4.plot(val_acc, label="Val")
        ax4.set_title("Accuracy")
        ax4.legend()
        st.pyplot(fig4)

# ─── PAGE ACCUEIL ──────────────────────────────────────────
if page == "🏠 Accueil":
    st.title("🏦 Détection de Fraude Bancaire par Deep Learning")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 À propos du Dataset")
        st.markdown("""
        Le dataset utilisé est le célèbre **Credit Card Fraud Detection**
        disponible sur Kaggle.

        Il contient les transactions bancaires par carte de crédit
        effectuées en **septembre 2013** par des titulaires européens.
        """)
        st.subheader("📊 Caractéristiques")
        st.markdown("""
        - **Total transactions :** 284 807
        - **Fraudes détectées :** 492 (0.17%)
        - **Variables :** 30 (V1 à V28 + Time + Amount)
        - **Cible :** Class (0 = Normal, 1 = Fraude)
        - **Déséquilibre :** Dataset fortement déséquilibré
        """)

    with col2:
        st.subheader("🤖 Modèles étudiés")
        st.markdown("""
        | Modèle | Description |
        |--------|-------------|
        | **CNN** | Réseau de neurones convolutif |
        | **LSTM** | Long Short-Term Memory |
        | **GRU** | Gated Recurrent Unit |
        | **CNN-LSTM-GRU** | Modèle hybride combiné |
        """)
        st.subheader("🎯 Objectif")
        st.markdown("""
        Comparer les performances de 4 algorithmes de **Deep Learning**
        pour la détection automatique de transactions frauduleuses,
        en maximisant le **Recall** et le **F1 Score**.
        """)

    st.divider()
    st.info("👈 Utilise le menu à gauche pour explorer chaque modèle !")

# ─── PAGES ALGORITHMES ─────────────────────────────────────
elif page == "🤖 CNN":
    show_algo("CNN")

elif page == "🤖 LSTM":
    show_algo("LSTM")

elif page == "🤖 GRU":
    show_algo("GRU")

elif page == "🤖 CNN-LSTM-GRU":
    show_algo("CNN-LSTM-GRU")

# ─── PAGE COMPARAISON ──────────────────────────────────────
elif page == "📊 Comparaison":
    st.title("📊 Comparaison des Modèles")
    st.divider()

    algos = ["CNN", "LSTM", "GRU", "CNN-LSTM-GRU"]
    data = {}
    for algo in algos:
        m, _, _, _, _, _, _, _ = load_algo(algo)
        data[algo] = m

    st.header("📋 Tableau Récapitulatif")
    rows = []
    for algo in algos:
        m = data[algo]
        rows.append({
            "Modèle":      algo,
            "Accuracy":    f"{m['accuracy']:.2%}",
            "F1 Score":    f"{m['f1']:.2%}",
            "ROC AUC":     f"{m['roc_auc']:.2%}",
            "Recall":      f"{m['recall']:.2%}",
            "Précision":   f"{m['precision']:.2%}",
            "Spécificité": f"{m['specificity']:.2%}",
            "MCC":         f"{m['mcc']:.4f}",
        })
    st.dataframe(rows, use_container_width=True)
    st.divider()

    st.header("📈 Comparaison Visuelle")
    metriques_noms = ["accuracy", "f1", "roc_auc", "recall", "precision"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metriques_noms))
    width = 0.2
    for i, algo in enumerate(algos):
        vals = [data[algo][m] for m in metriques_noms]
        ax.bar(x + i*width, vals, width, label=algo)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(["Accuracy","F1","AUC","Recall","Précision"])
    ax.set_ylim(0, 1.1)
    ax.set_title("Comparaison des métriques par modèle")
    ax.legend()
    st.pyplot(fig)

# ─── PAGE SIMULATEUR ───────────────────────────────────────
elif page == "🎯 Simulateur":
    st.title("🎯 Simulateur de Prédiction — CNN-LSTM-GRU")
    st.markdown("Entre les valeurs d'une transaction pour savoir si elle est **frauduleuse ou normale**.")
    st.divider()

    import onnxruntime as ort
    import joblib

    @st.cache_resource
    def load_onnx():
        path = os.path.join(BASE, "CNN-LSTM-GRU", "model.onnx")
        return ort.InferenceSession(path)

    @st.cache_resource
    def load_scaler():
        path = os.path.join(BASE, "CNN-LSTM-GRU", "scaler.pkl")
        return joblib.load(path)

    session = load_onnx()
    scaler  = load_scaler()

    feature_names = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]

    st.subheader("1️⃣ Entre les valeurs de la transaction")
    cols = st.columns(5)
    values = []
    for i, fname in enumerate(feature_names):
        val = cols[i%5].number_input(fname, value=0.0, format="%.4f")
        values.append(val)

    st.divider()

    if st.button("🔍 Analyser la transaction"):
        # Normalisation
        inp = np.array(values).reshape(1, -1)
        inp_scaled = scaler.transform(inp)

        # Reshape pour CNN-LSTM-GRU (1, 1, 30)
        inp_reshaped = inp_scaled.reshape(1, 1, 30).astype(np.float32)

        # Prédiction ONNX
        input_name = session.get_inputs()[0].name
        proba = session.run(None, {input_name: inp_reshaped})[0][0][0]

        # Résultat
        st.subheader("2️⃣ Résultat")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilité de fraude", f"{proba:.2%}")
            st.progress(float(proba))
        with col2:
            if proba > 0.5:
                st.error(f"🚨 FRAUDE DÉTECTÉE — Probabilité : {proba:.2%}")
            else:
                st.success(f"✅ Transaction NORMALE — Probabilité : {proba:.2%}")
