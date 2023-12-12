# Create Streamlit configuration directory
mkdir -p ~/.streamlit

# Create Streamlit credentials file
echo "[general]" > ~/.streamlit/credentials.toml
echo "email = \"vivekgogi123vg@gmail.com\"" >> ~/.streamlit/credentials.toml

# Create Streamlit configuration file
echo "[server]" > ~/.streamlit/config.toml
echo "enableCORS = false" >> ~/.streamlit/config.toml

# Install TensorFlow
pip install tensorflow==2.6.0
