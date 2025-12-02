import json
import requests
import pandas as pd
import streamlit as st

# ------------------------
# Config
# ------------------------

st.set_page_config(page_title="Smartlead Email Account Enricher", layout="wide")

SMARTLEAD_ENDPOINT = "https://server.smartlead.ai/api/email-account/get-total-email-accounts"
APP_PASSWORD = "hyperke123"


def check_password() -> bool:
    """Simple password gate stored only client side in session_state."""

    def password_entered():
        if st.session_state.get("password_input", "") == APP_PASSWORD:
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("Access required")
        st.text_input("Enter password", type="password", key="password_input", on_change=password_entered)
        st.stop()

    if not st.session_state["password_correct"]:
        st.title("Access required")
        st.error("Incorrect password")
        st.text_input("Enter password", type="password", key="password_input", on_change=password_entered)
        st.stop()

    return True


def get_state_df(key: str) -> pd.DataFrame | None:
    """Convenience accessor for DataFrames stored in session_state."""

    df = st.session_state.get(key)
    if isinstance(df, pd.DataFrame):
        return df
    return None


def get_smartlead_bearer() -> str:
    """Read bearer token from Streamlit secrets."""
    try:
        return st.secrets["smartlead"]["bearer"]
    except Exception:
        st.error(
            'Smartlead bearer token not found in secrets. '
            'Set it under [smartlead] bearer = "..." in .streamlit/secrets.toml'
        )
        st.stop()


@st.cache_data(show_spinner=True)
def fetch_all_email_accounts(limit: int = 500) -> pd.DataFrame:
    """Fetch all Smartlead email accounts with pagination and return as flattened DataFrame."""

    bearer = get_smartlead_bearer()
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {bearer}",
    }

    all_accounts = []
    seen_ids = set()
    offset = 0

    while True:
        params = {"offset": offset, "limit": limit}
        resp = requests.get(SMARTLEAD_ENDPOINT, headers=headers, params=params, timeout=60)
        resp.raise_for_status()

        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Smartlead API returned ok = false: {data}")

        accounts = data.get("data", {}).get("email_accounts", [])
        if not accounts:
            break

        # Deduplicate by id to avoid infinite loops if API ignores offset
        new_accounts = []
        for acc in accounts:
            acc_id = acc.get("id")
            if acc_id in seen_ids:
                continue
            seen_ids.add(acc_id)
            new_accounts.append(acc)

        if not new_accounts:
            break

        all_accounts.extend(new_accounts)

        if len(accounts) < limit:
            break

        offset += limit

    if not all_accounts:
        return pd.DataFrame()

    return flatten_email_accounts(all_accounts)


def flatten_email_accounts(accounts: list[dict]) -> pd.DataFrame:
    """Flatten Smartlead email_accounts JSON into a DataFrame with useful columns."""

    df = pd.json_normalize(accounts, sep=".")

    # Normalize emails for matching while keeping original for output
    if "from_email" in df.columns:
        df["from_email_normalized"] = df["from_email"].astype(str).str.strip().str.lower()

    rename_map = {
        "email_warmup_details.status": "warmup_status",
        "email_warmup_details.warmup_reputation": "warmup_reputation",
        "email_warmup_details.is_warmup_blocked": "warmup_blocked",
        "email_campaign_account_mappings_aggregate.aggregate.count": "campaign_count",
        "client.email": "client_email",
    }
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_renames)

    if "dns_validation_status" in df.columns:
        df["dns_validation_status_json"] = df["dns_validation_status"].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) and x else ""
        )
        df = df.drop(columns=["dns_validation_status"])

    if "email_account_tag_mappings" in df.columns:
        def extract_tag_field(tags, field):
            if not isinstance(tags, list):
                return ""
            values = []
            for item in tags:
                tag_obj = item.get("tag") if isinstance(item, dict) else None
                if isinstance(tag_obj, dict):
                    value = tag_obj.get(field)
                    if value is not None:
                        values.append(str(value))
            return ", ".join(sorted(set(values))) if values else ""

        df["tag_names"] = df["email_account_tag_mappings"].apply(lambda x: extract_tag_field(x, "name"))
        df["tag_ids"] = df["email_account_tag_mappings"].apply(lambda x: extract_tag_field(x, "id"))
        df["tag_colors"] = df["email_account_tag_mappings"].apply(lambda x: extract_tag_field(x, "color"))

        df = df.drop(columns=["email_account_tag_mappings"])

    dict_cols = [
        col for col in df.columns
        if df[col].apply(lambda x: isinstance(x, dict)).any()
    ]

    for col in dict_cols:
        expanded = pd.json_normalize(
            df[col].apply(lambda x: x if isinstance(x, dict) else {}),
            sep="."
        ).add_prefix(f"{col}.")
        expanded.index = df.index
        df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

    if "from_email" not in df.columns:
        raise RuntimeError("Expected 'from_email' field not found in Smartlead response")

    preferred_order = [
        "from_email",
        "from_name",
        "id",
        "type",
        "message_per_day",
        "daily_sent_count",
        "time_to_wait_in_mins",
        "smart_sender_flag",
        "is_smtp_success",
        "is_imap_success",
        "warmup_status",
        "warmup_reputation",
        "warmup_blocked",
        "campaign_count",
        "client_id",
        "client_email",
        "tag_names",
        "tag_ids",
        "tag_colors",
        "dns_validation_status_json",
    ]
    existing_pref = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_pref]

    df = df[existing_pref + remaining]

    return df


def show_sidebar_info(api_df: pd.DataFrame | None):
    with st.sidebar:
        st.header("Smartlead data")
        if api_df is None or api_df.empty:
            st.info("No Smartlead accounts loaded yet")
        else:
            st.success(f"Loaded {len(api_df):,} email accounts from Smartlead")

        if st.button("Refresh Smartlead cache"):
            fetch_all_email_accounts.clear()
            st.rerun()


def main():
    check_password()

    st.title("Smartlead Email Account Enricher")

    st.markdown(
        """
This tool:
1. Fetches all email accounts from Smartlead via the API and caches them for this session  
2. Lets you upload a CSV of inboxes  
3. Matches your emails against Smartlead accounts  
4. Outputs an enriched CSV with all Smartlead fields added
        """
    )

    if "api_df" not in st.session_state:
        st.session_state["api_df"] = None
    if "merged_result" not in st.session_state:
        st.session_state["merged_result"] = None
    if "matches_found" not in st.session_state:
        st.session_state["matches_found"] = 0

    api_df = get_state_df("api_df")

    st.markdown("### Step 1: Upload your CSV of inboxes")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            input_df = pd.read_csv(uploaded_file, encoding_errors="ignore")

        if input_df.empty:
            st.error("Uploaded CSV is empty")
            return

        st.write("Preview of uploaded data:")
        st.dataframe(input_df.head())

        st.markdown("### Step 2: Select the email column in your CSV")
        email_col = st.selectbox(
            "Column that contains inbox email addresses",
            options=list(input_df.columns),
        )

        st.markdown("### Step 3: Fetch Smartlead accounts")
        col1, col2 = st.columns([1, 3])

        with col1:
            fetch_button = st.button("Fetch and cache Smartlead accounts")

        if fetch_button:
            with st.spinner("Fetching all email accounts from Smartlead..."):
                try:
                    api_df = fetch_all_email_accounts()
                except Exception as e:
                    st.error(f"Error fetching Smartlead accounts: {e}")
                    return

            if api_df is None or api_df.empty:
                st.warning("Smartlead API returned no email accounts")
                return

            st.session_state["api_df"] = api_df
            st.success(f"Fetched and cached {len(api_df):,} Smartlead email accounts")
            st.write("Preview of Smartlead data:")
            st.dataframe(api_df.head())

        if api_df is None:
            try:
                api_df = fetch_all_email_accounts()
                st.session_state["api_df"] = api_df
            except Exception:
                api_df = None

        show_sidebar_info(api_df)

        if api_df is not None and not api_df.empty:
            st.markdown("### Step 4: Enrich your CSV using Smartlead data")

            if st.button("Enrich and generate output CSV"):
                with st.spinner("Matching emails and enriching CSV..."):
                    working_input = input_df.copy()
                    working_input["_normalized_email"] = (
                        working_input[email_col].astype(str).str.strip().str.lower()
                    )

                    api_df_for_merge = api_df.copy()
                    api_df_for_merge["from_email_normalized"] = api_df_for_merge[
                        "from_email"
                    ].astype(str).str.strip().str.lower()

                    merged_df = working_input.merge(
                        api_df_for_merge,
                        left_on="_normalized_email",
                        right_on="from_email_normalized",
                        how="left",
                        suffixes=("", "_smartlead"),
                    )

                merged_df = merged_df.drop(columns=["_normalized_email", "from_email_normalized"])

                matches_found = merged_df["from_email"].notna().sum()
                st.session_state["merged_result"] = merged_df
                st.session_state["matches_found"] = matches_found

        merged_result = get_state_df("merged_result")
        matches_found = st.session_state.get("matches_found", 0)

        if merged_result is not None:
            if matches_found:
                st.success(f"Enrichment complete — matched {matches_found:,} inbox(es)")
            else:
                st.warning(
                    "Enrichment complete but no matching Smartlead inboxes were found."
                )

            st.write("Preview of enriched data:")
            st.dataframe(merged_result.head())

            csv_bytes = merged_result.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="Download enriched CSV",
                data=csv_bytes,
                file_name="enriched_smartlead_accounts.csv",
                mime="text/csv",
            )
    else:
        show_sidebar_info(api_df)
        st.info("Upload a CSV to get started")


if __name__ == "__main__":
    main()
