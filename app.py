#"C:\Users\Dell\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run "03_Streamlit_App/app.py"
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "original_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"
METADATA_PATH = Path(__file__).resolve().parent / "model_metadata.json"

st.set_page_config(page_title="Egypt Property Intelligence", page_icon="🏙️", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_assets():
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    model = joblib.load(MODEL_PATH)
    return metadata, model

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

meta, model = load_assets()
raw = load_data()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#f4f7fb;}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#142943 0%,#1d4564 100%);}
section[data-testid="stSidebar"] * {color:#f5f8fc !important;}
.hero {padding:2rem 2.2rem;border-radius:22px;background:linear-gradient(120deg,#142943,#285b78);color:white;margin-bottom:1.2rem;box-shadow:0 14px 35px rgba(20,39,68,.16);}
.hero h1 {font-family:'Manrope',sans-serif;font-size:2.15rem;font-weight:800;margin:0 0 .45rem 0;color:white;}
.hero p {color:#d6e4f4;margin:0;font-size:1rem;}
.metric {background:white;border:1px solid #e4eaf2;padding:1rem 1.1rem;border-radius:16px;box-shadow:0 5px 18px rgba(30,50,80,.05);min-height:105px;}
.metric small {color:#718096;font-weight:600;}
.metric h2 {color:#172b4d;margin:.25rem 0 0;font-family:'Manrope',sans-serif;font-size:1.35rem;overflow-wrap:anywhere;}
.panel {background:white;border:1px solid #e4eaf2;padding:1.2rem 1.35rem;border-radius:18px;box-shadow:0 5px 18px rgba(30,50,80,.04);}
.stButton>button, .stFormSubmitButton>button {background:#1e5278;color:white;border:0;border-radius:11px;padding:.65rem 1.4rem;font-weight:700;}
.stButton>button:hover, .stFormSubmitButton>button:hover {background:#163d5a;color:white;}
div[data-testid="stMetric"] {background:white;padding:1rem;border:1px solid #e4eaf2;border-radius:14px;}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(f'<div class="metric"><small>{label}</small><h2>{value}</h2></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# 🏙️ EGYPT")
    st.markdown("## PROPERTY INTELLIGENCE")
    st.caption("Machine Learning · Regression")
    page = st.radio("NAVIGATION", ["Overview", "Data Insights", "Price Prediction", "Model Performance"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Selected model**")
    st.write(meta["best_model"])
    st.caption("Predictions are estimates based on historical asking-price listings.")

if page == "Overview":
    st.markdown('<div class="hero"><h1>Egyptian Real Estate · Price Intelligence</h1><p>Explore listing patterns and estimate property asking prices with a trained regression pipeline.</p></div>', unsafe_allow_html=True)
    st.markdown("### Project snapshot")
    cols = st.columns(4)
    values = [("Original listings", f"{meta['rows_original']:,}"), ("Input features", str(len(meta["features"]))), ("Modeling records", f"{meta['rows_modeling']:,}"), ("Selected model", meta["best_model"])]
    for column, (label, value) in zip(cols, values):
        with column:
            metric_card(label, value)
    st.markdown("### Project objective")
    st.markdown('<div class="panel">The project estimates Egyptian property asking prices from property characteristics such as area, rooms, property type, city, compound, furnishing, and delivery details. The saved pipeline includes preprocessing, so prediction inputs receive the same transformations used during training.</div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown("""**Workflow**

1. Explore the original listing data
2. Clean missing and inconsistent values in a modeling copy
3. Impute, scale, and encode within leakage-safe pipelines
4. Tune regression models using training-only cross-validation
5. Compare final predictions on one held-out test set""")
    with right:
        st.markdown("**Selected model test performance**")
        metrics = meta["metrics"]
        for metric_name in ["MAE", "RMSE", "R2", "Adjusted_R2"]:
            st.write(f"{metric_name}: {metrics[metric_name]:,.4f}")

elif page == "Data Insights":
    st.markdown('<div class="hero"><h1>Data Insights</h1><p>Understand the structure, distribution, and geography of the property listings.</p></div>', unsafe_allow_html=True)
    data = raw.copy()
    for column in data.columns:
        if data[column].dtype == "object":
            data[column] = data[column].astype("string").str.strip().replace(r"^Unknown$", pd.NA, regex=True)
    data["Price"] = pd.to_numeric(data["Price"].astype("string").str.replace(",", "", regex=False), errors="coerce")
    first, second = st.columns(2)
    with first:
        st.markdown("#### Asking price distribution")
        figure = px.histogram(data.dropna(subset=["Price"]), x="Price", nbins=55, marginal="box", color_discrete_sequence=["#2d668e"])
        figure.update_layout(template="plotly_white", height=390)
        st.plotly_chart(figure, use_container_width=True)
    with second:
        st.markdown("#### Listings by property type")
        counts = data["Type"].fillna("Missing").value_counts().head(12).rename_axis("Type").reset_index(name="Listings")
        figure = px.bar(counts, x="Listings", y="Type", orientation="h", color_discrete_sequence=["#4f8b83"])
        figure.update_layout(template="plotly_white", height=390, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(figure, use_container_width=True)
    st.markdown("#### Median asking price by city")
    city = data.dropna(subset=["Price", "City"]).groupby("City")["Price"].agg(["median", "count"]).query("count >= 30").sort_values("median", ascending=False).head(15).reset_index()
    figure = px.bar(city, x="median", y="City", orientation="h", hover_data=["count"], labels={"median": "Median asking price", "City": "City"}, color_discrete_sequence=["#bc8953"])
    figure.update_layout(template="plotly_white", height=500, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(figure, use_container_width=True)
    st.markdown("#### Source data quality")
    quality_cols = st.columns(3)
    with quality_cols[0]:
        metric_card("Exact duplicate rows", f"{int(raw.duplicated().sum()):,}")
    with quality_cols[1]:
        metric_card("Missing target values", f"{int(data['Price'].isna().sum()):,}")
    with quality_cols[2]:
        metric_card("Distinct cities", f"{raw['City'].nunique(dropna=True):,}")

elif page == "Price Prediction":
    st.markdown('<div class="hero"><h1>Estimate a Property Price</h1><p>Enter the property details below. The saved pipeline automatically handles imputation and encoding.</p></div>', unsafe_allow_html=True)
    st.markdown("#### Input features")
    with st.form("prediction_form"):
        values = {}
        columns = st.columns(2)
        for index, feature in enumerate(meta["features"]):
            with columns[index % 2]:
                if feature in meta["numeric_features"]:
                    default = float(meta["numeric_defaults"].get(feature, 0.0))
                    values[feature] = st.number_input(feature.replace("_", " "), value=default, min_value=0.0, step=1.0, help="Enter a non-negative value.")
                else:
                    options = meta["categorical_options"].get(feature, [])
                    options = [str(item) for item in options if str(item).strip()]
                    options = ["Not specified"] + options
                    values[feature] = st.selectbox(feature.replace("_", " "), options, index=0)
        submitted = st.form_submit_button("✨ Predict property price", use_container_width=True)
    if submitted:
        try:
            sample = pd.DataFrame([values], columns=meta["features"])
            for feature in meta["categorical_features"]:
                sample[feature] = sample[feature].replace("Not specified", None)
            prediction = float(model.predict(sample)[0])
            st.success("Prediction generated successfully")
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("### Estimated asking price")
            st.markdown(f"# {prediction:,.0f}")
            st.caption("This is a machine-learning estimate, not a formal valuation or guaranteed market price.")
            st.download_button("Download prediction as CSV", sample.assign(Predicted_Price=prediction).to_csv(index=False).encode("utf-8"), file_name="property_price_prediction.csv", mime="text/csv")
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as error:
            st.error(f"Unable to generate a prediction: {error}")

else:
    st.markdown('<div class="hero"><h1>Model Performance</h1><p>Held-out test-set evaluation and training-only hyperparameter search.</p></div>', unsafe_allow_html=True)
    st.markdown(f"### Selected model: {meta['best_model']}")
    metrics = meta["metrics"]
    metric_columns = st.columns(4)
    for column, (label, key) in zip(metric_columns, [("MAE", "MAE"), ("RMSE", "RMSE"), ("R²", "R2"), ("Adjusted R²", "Adjusted_R2")]):
        with column:
            metric_card(label, f"{metrics[key]:,.4f}")
    results = pd.DataFrame(meta["results"])
    st.markdown("#### Default versus tuned model comparison")
    st.dataframe(results.style.format({"MAE": "{:,.2f}", "MSE": "{:,.2f}", "RMSE": "{:,.2f}", "R2": "{:.4f}", "Adjusted_R2": "{:.4f}"}), use_container_width=True, hide_index=True)
    st.caption(meta.get("benchmark_results_note", ""))
    metric = st.selectbox("Compare models by", ["RMSE", "MAE", "R2", "Adjusted_R2"])
    figure = px.bar(results, x="Model", y=metric, color="Stage", barmode="group", title=f"Model comparison · {metric}", color_discrete_sequence=["#92a8bc", "#2d668e"])
    figure.update_layout(template="plotly_white", xaxis_tickangle=-35)
    st.plotly_chart(figure, use_container_width=True)
    st.markdown("#### GridSearchCV summary")
    st.metric("Best cross-validation RMSE", f"{meta['best_cv_rmse']:,.2f}")
    search_table = pd.DataFrame(meta["search_results"])
    st.dataframe(search_table, use_container_width=True, hide_index=True)
    st.markdown("#### Final selected model configuration")
    st.json(meta["selected_parameters"])
    st.markdown("#### GridSearchCV best parameters for the selected model")
    st.json(meta["best_params"])
    st.caption(meta.get("selected_gridsearch_note", ""))
    st.info("A traditional confusion matrix is not appropriate for continuous regression targets. Actual-versus-predicted and residual analysis are used instead.")
