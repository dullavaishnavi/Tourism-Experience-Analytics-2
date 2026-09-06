import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Tourism Experience Analytics", page_icon="🌍", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "cleaned_tourism_dataset.csv"
REG_MODEL = ROOT / "models" / "r.joblib"
CLF_MODEL = ROOT / "models" / "c.joblib"

@st.cache_data
def load_data():
    d = pd.read_csv(DATA)
    for c in ["VisitYear","VisitMonth","VisitMode","AttractionId","Rating","UserId",
              "ContinentId","RegionId","CountryId","CityId","AttractionCityId","AttractionTypeId"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

@st.cache_resource
def load_models():
    return joblib.load(REG_MODEL), joblib.load(CLF_MODEL)

df = load_data()
reg, clf = load_models()

st.title("🌍 Tourism Experience Analytics")
st.caption("Classification • Rating Prediction • Personalized Attraction Recommendations")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Dashboard", "Predictions", "Recommendations"])
    st.divider()
    st.write(f"**Transactions:** {len(df):,}")
    st.write(f"**Users:** {df.UserId.nunique():,}")
    st.write(f"**Attractions:** {df.AttractionId.nunique():,}")

if page == "Dashboard":
    st.subheader("Tourism analytics dashboard")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Users", f"{df.UserId.nunique():,}")
    c3.metric("Average Rating", f"{df.Rating.mean():.2f}/5")
    c4.metric("Attractions", f"{df.AttractionId.nunique():,}")

    left,right = st.columns(2)
    with left:
        st.markdown("### Visit modes")
        mode_counts = df["VisitModeName"].value_counts().rename_axis("Visit Mode").reset_index(name="Visits")
        st.bar_chart(mode_counts.set_index("Visit Mode"))
    with right:
        st.markdown("### Top attractions by visits")
        top = df.groupby("Attraction").agg(Visits=("TransactionId","count"), Avg_Rating=("Rating","mean")).sort_values("Visits", ascending=False).head(10)
        st.dataframe(top.round(2), use_container_width=True)

    st.markdown("### Rating distribution")
    st.bar_chart(df["Rating"].value_counts().sort_index())

    st.markdown("### Top regions")
    reg_counts = df["Region"].replace("-", np.nan).dropna().value_counts().head(10)
    st.bar_chart(reg_counts)

elif page == "Predictions":
    st.subheader("AI predictions")
    st.info("Enter a traveller profile and attraction context. The app predicts the likely visit mode and expected attraction rating.")

    # compact option lists
    def opts(col):
        return sorted([str(x) for x in df[col].dropna().unique() if str(x) != "-"])

    col1,col2,col3 = st.columns(3)
    with col1:
        continent = st.selectbox("Continent", opts("Continent"))
        regions = opts("Region")
        region = st.selectbox("Region", regions)
    with col2:
        countries = opts("Country")
        country = st.selectbox("Country", countries)
        cities = opts("UserCity")
        user_city = st.selectbox("User city", cities)
    with col3:
        attraction_type = st.selectbox("Attraction type", opts("AttractionType"))
        attractions = opts("Attraction")
        attraction = st.selectbox("Attraction", attractions)
        visit_month = st.slider("Visit month", 1, 12, 6)

    visit_year = st.number_input("Visit year", min_value=2000, max_value=2100, value=2026)

    row = pd.DataFrame([{
        "VisitYear": visit_year, "VisitMonth": visit_month,
        "Continent": continent, "Region": region, "Country": country,
        "UserCity": user_city, "AttractionType": attraction_type,
        "AttractionCity": str(df.loc[df.Attraction.eq(attraction), "AttractionCity"].mode().iloc[0]) if (df.Attraction.eq(attraction)).any() else "",
        "Attraction": attraction
    }])

    if st.button("Predict", type="primary"):
        predicted_mode = clf.predict(row)[0]
        predicted_rating = float(np.clip(reg.predict(row)[0], 1, 5))
        a,b = st.columns(2)
        a.metric("Predicted visit mode", predicted_mode)
        b.metric("Predicted rating", f"{predicted_rating:.2f} / 5")

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(row)[0]
            classes = clf.classes_
            p = pd.DataFrame({"Visit Mode": classes, "Probability": probs}).sort_values("Probability", ascending=False)
            st.markdown("#### Visit-mode probabilities")
            st.dataframe(p.style.format({"Probability":"{:.1%}"}), use_container_width=True)

else:
    st.subheader("🎯 Personalized attraction recommendations")
    st.write("Recommendations combine attraction rating, popularity, type preference, and—when a valid User ID is supplied—past user behavior.")

    user_id = st.number_input("User ID (optional)", min_value=0, value=0, step=1)
    preferred_type = st.selectbox("Preferred attraction type (optional)", ["All"] + sorted(df["AttractionType"].dropna().unique().tolist()))
    preferred_city = st.selectbox("Destination city (optional)", ["All"] + sorted(df["AttractionCity"].dropna().unique().tolist()))
    n = st.slider("Number of recommendations", 5, 15, 10)

    if st.button("Generate recommendations", type="primary"):
        agg = df.groupby(["AttractionId","Attraction","AttractionAddress","AttractionCity","AttractionType"], dropna=False).agg(
            Avg_Rating=("Rating","mean"), Visits=("TransactionId","count")
        ).reset_index()
        agg["score"] = agg["Avg_Rating"] * 0.70 + np.log1p(agg["Visits"]) * 0.20

        if preferred_type != "All":
            agg["score"] += np.where(agg["AttractionType"].eq(preferred_type), 0.75, 0)
        if preferred_city != "All":
            agg["score"] += np.where(agg["AttractionCity"].eq(preferred_city), 0.50, 0)

        if user_id > 0 and (df.UserId == user_id).any():
            hist = df[df.UserId == user_id]
            visited = set(hist.AttractionId.dropna().astype(int))
            type_pref = hist.groupby("AttractionType")["Rating"].mean().to_dict()
            agg["score"] += agg["AttractionType"].map(type_pref).fillna(hist.Rating.mean() if len(hist) else df.Rating.mean()) * 0.15
            agg = agg[~agg.AttractionId.astype(int).isin(visited)]
            st.success(f"Personalized using {len(hist)} historical visits. Previously visited attractions were excluded.")
        elif user_id > 0:
            st.warning("User ID not found; showing profile-based recommendations.")

        result = agg.sort_values(["score","Avg_Rating"], ascending=False).head(n).copy()
        result["Avg_Rating"] = result["Avg_Rating"].round(2)
        result["score"] = result["score"].round(2)
        st.dataframe(result[["Attraction","AttractionType","AttractionCity","AttractionAddress","Avg_Rating","Visits","score"]],
                     use_container_width=True, hide_index=True)
