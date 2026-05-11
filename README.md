# 🏥 Pipeline ETL Hospitalier — Groupe 3

**OUALI Leticia & BOUIMEDJ Wafi** · Avril 2026  
Dataset : [Hospital Patient Records — Kaggle](https://www.kaggle.com/datasets/blueblushed/hospital-dataset-for-practice)

---

## 📋 Vue d'ensemble

Pipeline ETL complet avec Apache Spark et Airflow pour analyser les hospitalisations d'un réseau de cliniques.

```
CSV Kaggle (Raw)
    ↓
PySpark Extract → Bronze (Parquet, copie fidèle)
    ↓
PySpark Transform → Silver (Parquet, nettoyé + jointures)
    ↓
PySpark Aggregate → Gold (PostgreSQL, tables analytiques)
    ↓
Dashboard (Metabase / Streamlit)
```

---

## 🗂️ Structure du projet

```
etl_hospital/
├── docker-compose.yml          # Infrastructure complète
├── .env.example                # Variables d'environnement (à copier en .env)
├── .gitignore
│
├── dags/
│   └── dag_hospital_etl.py     # DAG Airflow (@daily)
│
├── jobs/
│   ├── job_bronze.py           # Extraction CSV → Parquet
│   ├── job_silver.py           # Nettoyage + jointures
│   ├── job_gold.py             # Agrégations → PostgreSQL
│   └── utils/
│       ├── spark_session.py    # SparkSession partagée
│       └── logger.py           # Logging unifié
│
├── sql/
│   ├── init_db.sql             # Schéma PostgreSQL (Gold)
│   └── create_multiple_databases.sh
│
├── config/
│   └── generate_dataset.py     # Génération dataset synthétique
│
├── notebooks/
│   └── exploration_pipeline.ipynb
│
└── data/                       # Monté en volume Docker (non versionné)
    ├── raw/                    # CSV sources
    ├── bronze/                 # Parquet bruts
    ├── silver/                 # Parquet nettoyés
    ├── gold/                   # (optionnel, backup Parquet)
    └── quarantine/             # Lignes invalides
```

---

## 🚀 Démarrage rapide

### 1. Prérequis

- Docker Desktop ≥ 24.0
- Docker Compose ≥ 2.20
- 8 Go de RAM recommandés

### 2. Configuration

```bash
# Cloner le projet
git clone <url-du-repo>
cd etl_hospital

# Créer le fichier d'environnement
cp .env.example .env
# Éditez .env et changez le mot de passe PostgreSQL
nano .env
```

### 3. Données sources

**Option A — Dataset Kaggle** (recommandé pour la production)
```bash
# Installez la CLI Kaggle : pip install kaggle
# Configurez votre token API Kaggle dans ~/.kaggle/kaggle.json
kaggle datasets download -d blueblushed/hospital-dataset-for-practice
unzip hospital-dataset-for-practice.zip -d data/raw/
```

**Option B — Dataset synthétique** (pour les tests locaux)
```bash
pip install faker  # si nécessaire
python config/generate_dataset.py --rows 50000 --output data/raw/
```

### 4. Lancement de l'infrastructure

```bash
# Créer les répertoires de données
mkdir -p data/{raw,bronze,silver,gold,quarantine} logs

# Démarrer tous les services
docker-compose up -d

# Attendre 30 secondes et vérifier
docker-compose ps
```

### 5. Accès aux interfaces

| Service          | URL                        | Credentials         |
|------------------|----------------------------|---------------------|
| Airflow UI       | http://localhost:8080       | admin / admin       |
| Spark Master UI  | http://localhost:8082       | —                   |
| Jupyter Notebook | http://localhost:8888       | (token dans logs)   |
| PostgreSQL       | localhost:5432              | voir `.env`         |

### 6. Exécution du pipeline

**Via Airflow (recommandé) :**
1. Ouvrez http://localhost:8080
2. Activez le DAG `hospital_etl_pipeline`
3. Cliquez "Trigger DAG" ▶

**Via ligne de commande (développement) :**
```bash
# Bronze
docker-compose exec airflow-scheduler python /opt/airflow/jobs/job_bronze.py

# Silver
docker-compose exec airflow-scheduler python /opt/airflow/jobs/job_silver.py

# Gold
docker-compose exec airflow-scheduler python /opt/airflow/jobs/job_gold.py
```

---

## 🏗️ Architecture des données

### Couche Bronze
- **Format** : Parquet (Snappy) dans `data/bronze/`
- **Transformations** : Aucune — copie fidèle des CSV
- **Tables** : `patients/`, `admissions/`, `diagnoses/`, `medications/`, `billing/`

### Couche Silver
- **Format** : Parquet dans `data/silver/`
- **Transformations** :
  - Suppression des doublons sur `id_admission`
  - Normalisation des dates (MM/DD/YYYY et YYYY-MM-DD → ISO 8601)
  - Cast des colonnes numériques
  - Normalisation des codes ICD en majuscules
  - Filtrage des lignes invalides → `data/quarantine/`
  - Jointure : admissions ↔ patients ↔ diagnoses ↔ billing
  - Calcul de la durée de séjour en jours
- **Tables** : `silver_master/`, `silver_medications/`

### Couche Gold (PostgreSQL)
| Table | Description |
|-------|-------------|
| `gold_sejour_stats` | Durée de séjour, coût, taux de réadmission par service/pathologie/mois |
| `gold_medication_stats` | Médicaments les plus prescrits par service |
| `gold_frequentation` | Fréquentation mensuelle + démographie |
| `gold_kpi_dashboard` | KPIs synthétiques pour la direction |
| `v_dashboard_principal` | Vue consolidée pour le dashboard |

---

## ✅ Contrôles qualité

| Contrôle | Couche | Action si échec |
|----------|--------|-----------------|
| Unicité `id_admission` | Silver | Dédoublonnage |
| `id_patient` non null | Silver | Quarantaine |
| Dates valides | Silver | Quarantaine si impossible à parser |
| `date_sortie` ≥ `date_entree` | Silver | Quarantaine |
| `montant_total` > 0 | Silver | Quarantaine |
| Codes ICD normalisés | Silver | Correction automatique (uppercase) |

Les lignes rejetées sont stockées dans `data/quarantine/` avec une colonne `_raison_rejet`.

---

## 🔒 Sécurité

- Credentials PostgreSQL dans `.env` (non versionné, dans `.gitignore`)
- `.env.example` commité avec des valeurs fictives
- Dataset 100% synthétique — conformité RGPD par nature
- Les jobs Spark ne loguent jamais les valeurs individuelles

---

## 📊 Connexion Metabase

```bash
# Lancer Metabase (optionnel, non inclus dans docker-compose)
docker run -d -p 3000:3000 \
  -e MB_DB_TYPE=postgres \
  -e MB_DB_HOST=localhost \
  -e MB_DB_PORT=5432 \
  -e MB_DB_DBNAME=hospital_gold \
  -e MB_DB_USER=hospital_admin \
  -e MB_DB_PASS=<votre_password> \
  --name metabase metabase/metabase
```

Puis connectez-vous à http://localhost:3000 et utilisez la vue `v_dashboard_principal`.

---

## ⚡ Scalabilité

Le pipeline est conçu pour s'exécuter en local (mode `local[*]`).

Pour passer à l'échelle :
1. Remplacer `SPARK_MASTER=local[*]` par l'URL du cluster Spark/Databricks/EMR
2. Remplacer PostgreSQL par BigQuery ou Redshift pour la couche Gold
3. Ajouter le partitionnement Silver par mois pour les gros volumes
