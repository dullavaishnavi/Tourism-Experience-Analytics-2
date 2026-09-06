# Tourism Experience Analytics

A Streamlit-ready machine learning project for **classification, rating prediction, and personalized attraction recommendations** using the supplied Tourism Dataset.

## Project objectives

1. **Regression:** predict an attraction rating (1–5).
2. **Classification:** predict the user's visit mode (Business, Couples, Family, Friends, or Solo).
3. **Recommendation:** rank attractions using rating, popularity, attraction type, destination city, and optional user history.
4. **Analytics:** explore visit modes, ratings, popular attractions, and regions.

## Repository structure

```text
Tourism-Experience-Analytics/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── DEPLOYMENT_CHECKLIST.md
├── data/
│   ├── cleaned_tourism_dataset.csv
│   └── source Excel files
├── models/
│   ├── r.joblib
│   ├── c.joblib
│   └── metrics.json
├── notebooks/
│   └── Tourism_Experience_Analytics_Google_Colab.ipynb
├── src/
│   └── train_models.py
└── .streamlit/
    └── config.toml
```

## Dataset

The project uses the supplied Tourism Dataset tables: Transaction, User, City, Country, Region, Continent, Item, Type, and Mode.

The consolidated dataset contains **52,930 transaction records**.

## Model results

The included models were trained with a fixed random seed and a 20% test split.

See `models/metrics.json` for the exact recorded metrics.

> The model artifacts are tied to the scikit-learn version in `requirements.txt`. Keep that version unchanged when deploying.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository named `Tourism-Experience-Analytics`.
2. Upload/push all files in this folder.
3. In Streamlit Community Cloud, choose **Deploy an app**.
4. Select the GitHub repository and the `main` branch.
5. Set the main file to `app.py`.
6. Deploy.

No secrets are required for this version.

## Re-train models

If the source data changes:

```bash
python src/train_models.py
```

Then replace the model artifacts in `models/`.

For Google Colab, open the notebook in `notebooks/` and run the cells.

## Streamlit features

- Interactive tourism dashboard
- Visit-mode prediction with class probabilities
- Attraction rating prediction
- User-history-aware recommendations
- Destination and attraction-type filters
- Previously visited attraction exclusion when a valid User ID is entered
