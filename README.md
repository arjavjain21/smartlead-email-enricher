# Smartlead Email Account Enricher

Streamlit app that:

1. Protects access with a simple password gate
2. Fetches all Smartlead email accounts via API and caches them
3. Lets you upload a CSV of inboxes
4. Matches your emails against Smartlead data
5. Outputs an enriched CSV with all Smartlead fields added

## Setup

1. Clone this repository

2. Create `.streamlit/secrets.toml`:

```toml
[smartlead]
bearer = "REPLACE_WITH_YOUR_SMARTLEAD_BEARER_TOKEN"
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the app

```bash
streamlit run app.py
```

5. Open the URL shown in the terminal and enter the password:

```text
h-y-e-e-r-k-e-1-2-3
```

## Notes

- The bearer token is read from Streamlit secrets.
- Smartlead email accounts are cached per Streamlit session using `st.cache_data`.
