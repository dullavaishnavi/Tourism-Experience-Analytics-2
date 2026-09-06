# Data

This folder contains the Tourism Dataset source Excel files and the consolidated `cleaned_tourism_dataset.csv`.

The consolidated file is built by joining:
- Transaction
- User
- City
- Country
- Region
- Continent
- Item (attraction)
- Type (attraction type)
- Mode (visit mode)

For GitHub/Streamlit deployment, the app reads the consolidated CSV. The original Excel files are retained for reproducibility.
