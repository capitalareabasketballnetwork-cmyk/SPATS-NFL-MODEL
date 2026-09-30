# SPATS account setup

The application code supports Google OIDC login and persistent per-user Saved Models.

## 1. Google OIDC
Create a Google Cloud Web application OAuth client. Add this authorized redirect URI:

`https://YOUR-STREAMLIT-APP.streamlit.app/oauth2callback`

Then add these values in Streamlit Community Cloud > App > Settings > Secrets:

```toml
[auth]
redirect_uri = "https://YOUR-STREAMLIT-APP.streamlit.app/oauth2callback"
cookie_secret = "GENERATE_A_LONG_RANDOM_SECRET"
client_id = "YOUR_GOOGLE_CLIENT_ID"
client_secret = "YOUR_GOOGLE_CLIENT_SECRET"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

[supabase]
url = "YOUR_SUPABASE_PROJECT_URL"
secret_key = "YOUR_SUPABASE_SECRET_SERVICE_ROLE_KEY"
```

Never commit these values to GitHub.

## 2. Supabase
Create a Supabase project, open its SQL editor, and run `supabase_saved_models.sql`.

The app identifies the signed-in user from the OIDC subject/email and stores/fetches only rows matching that user ID.

## 3. Reboot
After saving Streamlit secrets, reboot the app. Users will be required to sign in with Google before accessing SPATS.
