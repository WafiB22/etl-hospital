"""
dashboard_hopital.py
--------------------
Dashboard Streamlit — ETL Hospital (version pandas)
Reproduit les agrégations Gold sans PySpark ni PostgreSQL.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Hôpital",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Style CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0f1117; }

    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2a3a5c;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #4fc3f7;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 6px;
    }
    .kpi-delta {
        font-size: 0.85rem;
        color: #66bb6a;
        margin-top: 4px;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #cdd6f4;
        border-left: 3px solid #4fc3f7;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }
    .badge {
        display: inline-block;
        background: #1e3a5f;
        color: #4fc3f7;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Chargement & ETL pandas (Bronze → Silver → Gold)
# ──────────────────────────────────────────────
@st.cache_data
def load_and_transform():
    """Pipeline ETL complet en pandas — Bronze → Silver → Gold."""

    # ── BRONZE : lecture CSV bruts ──
    BASE = "data/raw/"
    patients    = pd.read_csv(BASE + "patients.csv")
    admissions  = pd.read_csv(BASE + "admissions.csv")
    diagnoses   = pd.read_csv(BASE + "diagnoses.csv")
    medications = pd.read_csv(BASE + "medications.csv")
    billing     = pd.read_csv(BASE + "billing.csv")

    # ── SILVER : nettoyage ──

    # Patients
    patients = patients.drop_duplicates("id_patient")
    patients = patients[patients["age"].between(1, 119)]

    # Admissions — normalisation dates
    for col in ["date_entree", "date_sortie"]:
        admissions[col] = pd.to_datetime(admissions[col], errors="coerce")
    admissions = admissions.drop_duplicates("id_admission")
    admissions["duree_sejour_jours"] = (
        admissions["date_sortie"] - admissions["date_entree"]
    ).dt.days
    admissions = admissions[admissions["duree_sejour_jours"].isna() | (admissions["duree_sejour_jours"] >= 0)]

    # Diagnoses
    diagnoses["code_icd"]    = diagnoses["code_icd"].str.strip().str.upper()
    diagnoses["description"] = diagnoses["description"].str.strip()
    diagnoses = diagnoses[diagnoses["id_admission"].notna()]

    # Medications
    medications["medicament"] = medications["medicament"].str.strip().str.upper()
    medications["duree"]      = pd.to_numeric(medications["duree"], errors="coerce")
    medications = medications[medications["id_admission"].notna() & medications["medicament"].notna()]

    # Billing
    billing["montant_total"] = pd.to_numeric(billing["montant_total"], errors="coerce")
    billing = billing[billing["montant_total"] > 0]

    # Diagnostic principal par admission
    diag_principal = (
        diagnoses.groupby("id_admission")
        .agg(
            code_icd_principal=("code_icd", "first"),
            description_pathologie=("description", "first"),
            nb_diagnostics=("code_icd", "count")
        )
        .reset_index()
    )

    # Silver master (jointure centrale)
    silver = (
        admissions
        .merge(patients[["id_patient", "age", "sexe", "ville"]], on="id_patient", how="left")
        .merge(diag_principal, on="id_admission", how="left")
        .merge(billing[["id_admission", "montant_total", "mode_paiement"]], on="id_admission", how="left")
    )
    silver["annee"] = silver["date_entree"].dt.year
    silver["mois"]  = silver["date_entree"].dt.month

    # ── GOLD : agrégations ──

    # gold_sejour_stats
    sejour_stats = (
        silver.groupby(["annee", "mois", "service", "code_icd_principal", "description_pathologie"])
        .agg(
            nb_admissions=("id_admission", "count"),
            duree_sejour_moy=("duree_sejour_jours", "mean"),
            duree_sejour_min=("duree_sejour_jours", "min"),
            duree_sejour_max=("duree_sejour_jours", "max"),
            cout_moyen=("montant_total", "mean"),
            cout_total=("montant_total", "sum"),
        )
        .round(2)
        .reset_index()
    )

    # gold_frequentation
    frequentation = (
        silver.groupby(["annee", "mois", "service"])
        .agg(
            nb_admissions=("id_admission", "count"),
            nb_patients_uniques=("id_patient", "nunique"),
            age_moyen=("age", "mean"),
            pct_femmes=("sexe", lambda x: (x == "F").mean()),
        )
        .round(2)
        .reset_index()
    )

    # gold_medication_stats
    med_enriched = medications.merge(
        silver[["id_admission", "service", "annee", "mois", "id_patient"]],
        on="id_admission", how="left"
    ).dropna(subset=["service"])
    med_stats = (
        med_enriched.groupby(["annee", "mois", "service", "medicament"])
        .agg(
            nb_prescriptions=("id_admission", "count"),
            nb_patients_uniques=("id_patient", "nunique"),
            duree_moy_jours=("duree", "mean"),
        )
        .round(1)
        .reset_index()
    )

    return silver, sejour_stats, frequentation, med_stats, medications, patients

# ──────────────────────────────────────────────
# Chargement
# ──────────────────────────────────────────────
try:
    silver, sejour_stats, frequentation, med_stats, medications, patients = load_and_transform()
    data_ok = True
except FileNotFoundError:
    data_ok = False

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Dashboard Hôpital")
    st.markdown("<span class='badge'>ETL Pipeline — pandas</span>", unsafe_allow_html=True)
    st.markdown("---")

    if data_ok:
        services = ["Tous"] + sorted(silver["service"].dropna().unique().tolist())
        service_filtre = st.selectbox("🏨 Service", services)

        annees = sorted(silver["annee"].dropna().unique().astype(int).tolist())
        annee_filtre = st.selectbox("📅 Année", ["Toutes"] + annees)

        st.markdown("---")
        st.markdown("**Pipeline ETL**")
        st.markdown("🟤 Bronze → CSV bruts")
        st.markdown("⚪ Silver → Nettoyage & jointures")
        st.markdown("🟡 Gold → Agrégations & KPIs")

    onglet = st.radio("Navigation", ["📊 KPIs", "🏨 Fréquentation", "💊 Médicaments", "🔬 Pathologies", "👥 Patients"])

# ──────────────────────────────────────────────
# Filtrage
# ──────────────────────────────────────────────
if data_ok:
    df = silver.copy()
    if service_filtre != "Tous":
        df = df[df["service"] == service_filtre]
    if annee_filtre != "Toutes":
        df = df[df["annee"] == int(annee_filtre)]

    freq_f = frequentation.copy()
    sejour_f = sejour_stats.copy()
    med_f = med_stats.copy()
    if service_filtre != "Tous":
        freq_f  = freq_f[freq_f["service"] == service_filtre]
        sejour_f = sejour_f[sejour_f["service"] == service_filtre]
        med_f   = med_f[med_f["service"] == service_filtre]
    if annee_filtre != "Toutes":
        freq_f  = freq_f[freq_f["annee"] == int(annee_filtre)]
        sejour_f = sejour_f[sejour_f["annee"] == int(annee_filtre)]
        med_f   = med_f[med_f["annee"] == int(annee_filtre)]

# ──────────────────────────────────────────────
# Couleurs matplotlib dark
# ──────────────────────────────────────────────
PALETTE = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ce93d8", "#80cbc4"]
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1f2e",
    "axes.edgecolor":   "#2a3a5c",
    "axes.labelcolor":  "#8899aa",
    "xtick.color":      "#8899aa",
    "ytick.color":      "#8899aa",
    "text.color":       "#cdd6f4",
    "grid.color":       "#1e2d4a",
    "grid.alpha":       0.5,
})

# ──────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────
if not data_ok:
    st.error("❌ Fichiers CSV introuvables. Assure-toi que le dossier `data/raw/` est présent avec les 5 CSV.")
    st.stop()

# ── PAGE KPIs ──
if onglet == "📊 KPIs":
    st.markdown("# 📊 Tableau de bord — KPIs")
    st.markdown(f"*Données filtrées · {len(df):,} admissions*")

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, f"{len(df):,}", "Admissions"),
        (c2, f"{df['id_patient'].nunique():,}", "Patients uniques"),
        (c3, f"{df['duree_sejour_jours'].mean():.1f}j", "Durée moy. séjour"),
        (c4, f"{df['montant_total'].mean():,.0f}€", "Coût moyen"),
        (c5, f"{df['montant_total'].sum():,.0f}€", "Revenu total"),
    ]
    for col, val, label in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Admissions par service</div>", unsafe_allow_html=True)
        svc = df.groupby("service")["id_admission"].count().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(svc.index, svc.values, color=PALETTE[:len(svc)], edgecolor="none", height=0.6)
        for bar, val in zip(bars, svc.values):
            ax.text(val + 5, bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", fontsize=9, color="#cdd6f4")
        ax.set_xlabel("Nombre d'admissions")
        ax.grid(axis="x")
        ax.spines[["top", "right", "left"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<div class='section-title'>Modes de paiement</div>", unsafe_allow_html=True)
        paiement = df["mode_paiement"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        wedges, texts, autotexts = ax.pie(
            paiement.values, labels=paiement.index,
            colors=PALETTE[:len(paiement)], autopct="%1.1f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops={"edgecolor": "#0f1117", "linewidth": 2}
        )
        for t in autotexts:
            t.set_color("#0f1117"); t.set_fontsize(9); t.set_fontweight("bold")
        ax.set_facecolor("#0f1117")
        st.pyplot(fig)
        plt.close()

    st.markdown("<div class='section-title'>Évolution mensuelle des admissions</div>", unsafe_allow_html=True)
    monthly = df.groupby(["annee", "mois"])["id_admission"].count().reset_index()
    monthly["periode"] = monthly["annee"].astype(str) + "-" + monthly["mois"].astype(str).str.zfill(2)
    monthly = monthly.sort_values("periode")
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.fill_between(monthly["periode"], monthly["id_admission"], alpha=0.2, color="#4fc3f7")
    ax.plot(monthly["periode"], monthly["id_admission"], color="#4fc3f7", linewidth=2, marker="o", markersize=4)
    ax.set_ylabel("Admissions")
    ax.grid(axis="y")
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    st.pyplot(fig)
    plt.close()

# ── PAGE FRÉQUENTATION ──
elif onglet == "🏨 Fréquentation":
    st.markdown("# 🏨 Fréquentation par service")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Durée moyenne de séjour par service</div>", unsafe_allow_html=True)
        duree = df.groupby("service")["duree_sejour_jours"].mean().sort_values(ascending=False).round(1)
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(duree.index, duree.values, color=PALETTE[:len(duree)], edgecolor="none", width=0.6)
        for bar, val in zip(bars, duree.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{val}j", ha="center", fontsize=9, color="#cdd6f4")
        ax.set_ylabel("Jours")
        ax.grid(axis="y")
        ax.spines[["top", "right", "left"]].set_visible(False)
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<div class='section-title'>Coût moyen par service</div>", unsafe_allow_html=True)
        cout = df.groupby("service")["montant_total"].mean().sort_values(ascending=False).round(0)
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(cout.index, cout.values, color=PALETTE[1:len(cout)+1], edgecolor="none", width=0.6)
        for bar, val in zip(bars, cout.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f"{val:,.0f}€", ha="center", fontsize=8, color="#cdd6f4")
        ax.set_ylabel("Montant (€)")
        ax.grid(axis="y")
        ax.spines[["top", "right", "left"]].set_visible(False)
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)
        plt.close()

    st.markdown("<div class='section-title'>Répartition Homme / Femme par service</div>", unsafe_allow_html=True)
    sexe_svc = df.groupby(["service", "sexe"])["id_admission"].count().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(sexe_svc))
    w = 0.35
    if "F" in sexe_svc.columns:
        ax.bar(x - w/2, sexe_svc["F"], w, label="Femmes", color="#f06292", edgecolor="none")
    if "M" in sexe_svc.columns:
        ax.bar(x + w/2, sexe_svc["M"], w, label="Hommes", color="#4fc3f7", edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(sexe_svc.index, rotation=30, ha="right")
    ax.set_ylabel("Admissions"); ax.legend()
    ax.grid(axis="y"); ax.spines[["top", "right", "left"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.markdown("<div class='section-title'>Distribution de l'âge des patients</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.hist(df["age"].dropna(), bins=30, color="#4fc3f7", edgecolor="#0f1117", alpha=0.85)
    ax.set_xlabel("Âge"); ax.set_ylabel("Nombre de patients")
    ax.grid(axis="y"); ax.spines[["top", "right", "left"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

# ── PAGE MÉDICAMENTS ──
elif onglet == "💊 Médicaments":
    st.markdown("# 💊 Statistiques médicaments")

    top_med = (
        med_f.groupby("medicament")["nb_prescriptions"].sum()
        .sort_values(ascending=False).head(15)
    )

    st.markdown("<div class='section-title'>Top 15 médicaments les plus prescrits</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.barh(top_med.index[::-1], top_med.values[::-1], color=PALETTE[0], edgecolor="none", height=0.6)
    for bar, val in zip(bars, top_med.values[::-1]):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", fontsize=9, color="#cdd6f4")
    ax.set_xlabel("Nombre de prescriptions")
    ax.grid(axis="x"); ax.spines[["top", "right", "left"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Prescriptions par service</div>", unsafe_allow_html=True)
        med_svc = med_f.groupby("service")["nb_prescriptions"].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(med_svc.index, med_svc.values, color=PALETTE[2], edgecolor="none", height=0.6)
        ax.set_xlabel("Prescriptions")
        ax.grid(axis="x"); ax.spines[["top", "right", "left"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<div class='section-title'>Durée moyenne de traitement (jours)</div>", unsafe_allow_html=True)
        duree_med = med_f.groupby("medicament")["duree_moy_jours"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(duree_med.index[::-1], duree_med.values[::-1], color=PALETTE[3], edgecolor="none", height=0.6)
        ax.set_xlabel("Jours")
        ax.grid(axis="x"); ax.spines[["top", "right", "left"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

# ── PAGE PATHOLOGIES ──
elif onglet == "🔬 Pathologies":
    st.markdown("# 🔬 Analyse des pathologies")

    top_diag = df["description_pathologie"].value_counts().head(15)
    st.markdown("<div class='section-title'>Top 15 pathologies les plus fréquentes</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.barh(top_diag.index[::-1], top_diag.values[::-1], color=PALETTE[4], edgecolor="none", height=0.6)
    ax.set_xlabel("Nombre d'admissions")
    ax.grid(axis="x"); ax.spines[["top", "right", "left"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Coût moyen par pathologie (Top 10)</div>", unsafe_allow_html=True)
        cout_diag = df.groupby("description_pathologie")["montant_total"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(cout_diag.index[::-1], cout_diag.values[::-1], color=PALETTE[1], edgecolor="none", height=0.6)
        ax.set_xlabel("Coût moyen (€)")
        ax.grid(axis="x"); ax.spines[["top", "right", "left"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<div class='section-title'>Durée de séjour par pathologie (Top 10)</div>", unsafe_allow_html=True)
        duree_diag = df.groupby("description_pathologie")["duree_sejour_jours"].mean().sort_values(ascending=False).head(10).round(1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(duree_diag.index[::-1], duree_diag.values[::-1], color=PALETTE[0], edgecolor="none", height=0.6)
        ax.set_xlabel("Durée moyenne (jours)")
        ax.grid(axis="x"); ax.spines[["top", "right", "left"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("<div class='section-title'>Pathologies par service (heatmap)</div>", unsafe_allow_html=True)
    top10_diag = df["description_pathologie"].value_counts().head(10).index
    pivot = df[df["description_pathologie"].isin(top10_diag)].pivot_table(
        index="description_pathologie", columns="service",
        values="id_admission", aggfunc="count", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, ax=ax, cmap="Blues", linewidths=0.5,
                linecolor="#0f1117", annot=True, fmt="d",
                cbar_kws={"shrink": 0.8})
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=8)
    st.pyplot(fig)
    plt.close()

# ── PAGE PATIENTS ──
elif onglet == "👥 Patients":
    st.markdown("# 👥 Profil des patients")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{patients['age'].mean():.1f}</div>
            <div class="kpi-label">Âge moyen</div></div>""", unsafe_allow_html=True)
    with c2:
        pct_f = (patients["sexe"] == "F").mean() * 100
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{pct_f:.1f}%</div>
            <div class="kpi-label">% Femmes</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{patients['ville'].nunique()}</div>
            <div class="kpi-label">Villes représentées</div></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Top 10 villes</div>", unsafe_allow_html=True)
        villes = patients["ville"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(villes.index[::-1], villes.values[::-1], color=PALETTE[0], edgecolor="none", height=0.6)
        ax.set_xlabel("Nombre de patients")
        ax.grid(axis="x"); ax.spines[["top", "right", "left"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<div class='section-title'>Répartition par sexe</div>", unsafe_allow_html=True)
        sexe = patients["sexe"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(sexe.values, labels=["Femmes" if x=="F" else "Hommes" for x in sexe.index],
               colors=["#f06292", "#4fc3f7"], autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor": "#0f1117", "linewidth": 2})
        st.pyplot(fig)
        plt.close()

    st.markdown("<div class='section-title'>Pyramide des âges</div>", unsafe_allow_html=True)
    bins = [0,10,20,30,40,50,60,70,80,90,120]
    labels = ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90+"]
    patients["tranche"] = pd.cut(patients["age"], bins=bins, labels=labels, right=False)
    pyramid = patients.groupby(["tranche","sexe"], observed=True).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 4))
    if "F" in pyramid.columns:
        ax.barh(pyramid.index.astype(str), pyramid["F"], color="#f06292", label="Femmes", edgecolor="none")
    if "M" in pyramid.columns:
        ax.barh(pyramid.index.astype(str), -pyramid["M"], color="#4fc3f7", label="Hommes", edgecolor="none")
    ax.axvline(0, color="#cdd6f4", linewidth=0.8)
    ax.set_xlabel("Nombre de patients")
    ax.legend(); ax.grid(axis="x"); ax.spines[["top","right","left"]].set_visible(False)
    st.pyplot(fig)
    plt.close()
