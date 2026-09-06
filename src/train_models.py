from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "models"

def main():
    t = pd.read_excel(DATA / "Transaction.xlsx")
    u = pd.read_excel(DATA / "User.xlsx")
    city = pd.read_excel(DATA / "City.xlsx")
    country = pd.read_excel(DATA / "Country.xlsx")
    region = pd.read_excel(DATA / "Region.xlsx")
    continent = pd.read_excel(DATA / "Continent.xlsx")
    item = pd.read_excel(DATA / "Item.xlsx")
    typ = pd.read_excel(DATA / "Type.xlsx")
    mode = pd.read_excel(DATA / "Mode.xlsx")

    d = t.merge(u, on="UserId", how="left")
    d = d.merge(city[["CityId","CityName"]].rename(columns={"CityName":"UserCity"}), on="CityId", how="left")
    d = d.merge(country[["CountryId","Country"]], on="CountryId", how="left")
    d = d.merge(region[["RegionId","Region"]], on="RegionId", how="left")
    d = d.merge(continent[["ContinentId","Continent"]], on="ContinentId", how="left")
    d = d.merge(item, on="AttractionId", how="left")
    d = d.merge(typ, on="AttractionTypeId", how="left")
    d = d.merge(mode.rename(columns={"VisitModeId":"VisitMode","VisitMode":"VisitModeName"}), on="VisitMode", how="left")
    d = d.merge(city[["CityId","CityName"]].rename(columns={"CityId":"AttractionCityId","CityName":"AttractionCity"}),
                on="AttractionCityId", how="left")

    features = ["VisitYear","VisitMonth","Continent","Region","Country","UserCity",
                "AttractionType","AttractionCity","Attraction"]
    X = d[features]
    cats = [c for c in features if X[c].dtype == "object"]
    nums = [c for c in features if c not in cats]
    prep = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
                              ("num","passthrough",nums)])

    xr, xt, yr, yt = train_test_split(X, d.Rating.astype(float), test_size=.2, random_state=42)
    reg = Pipeline([("prep",prep),("model",RandomForestRegressor(n_estimators=45,max_depth=10,min_samples_leaf=3,n_jobs=-1,random_state=42))])
    reg.fit(xr,yr)

    xc, xv, yc, yv = train_test_split(X, d.VisitModeName.astype(str), test_size=.2, random_state=42, stratify=d.VisitModeName)
    prep2 = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
                               ("num","passthrough",nums)])
    clf = Pipeline([("prep",prep2),("model",RandomForestClassifier(n_estimators=50,max_depth=10,min_samples_leaf=3,n_jobs=-1,random_state=42,class_weight="balanced_subsample"))])
    clf.fit(xc,yc)

    OUT.mkdir(exist_ok=True)
    joblib.dump(reg, OUT/"r.joblib")
    joblib.dump(clf, OUT/"c.joblib")
    print("Models saved to", OUT)

if __name__ == "__main__":
    main()
