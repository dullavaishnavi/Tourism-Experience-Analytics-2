# Deployment Checklist

- [ ] Push the repository to GitHub.
- [ ] Confirm `app.py` is in the repository root.
- [ ] Confirm `requirements.txt` is in the repository root.
- [ ] Confirm `data/cleaned_tourism_dataset.csv` exists.
- [ ] Confirm `models/r.joblib` and `models/c.joblib` exist.
- [ ] Keep `scikit-learn==1.8.0` unchanged unless models are retrained.
- [ ] Test with `streamlit run app.py`.
- [ ] Deploy the `main` branch on Streamlit Community Cloud.
