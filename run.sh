VERIFICATION_COUNTRY=Ethiopia \
VERIFICATION_DATA_DIR=$(pwd)/data/diagnostics \
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES streamlit run ui/app.py -- server.port 8502
