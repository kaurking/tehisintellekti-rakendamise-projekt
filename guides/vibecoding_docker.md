# 🚀 VS Code & Docker Sandbox Setup Guide

This guide will walk you through setting up a fully isolated, Docker-powered development environment (Dev Container) for building Streamlit apps with Miniconda. 

By using this setup, your code runs safely inside a secure Linux container, ensuring your local machine stays clean and your library versions never conflict.



---

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your computer:
1. **Docker Desktop** (Running in the background)
2. **Visual Studio Code (VS Code)**
3. The **Dev Containers** extension installed in VS Code.

---

## 📁 Step 1: Project Structure

Create a new empty folder for your project and set up the following exact file structure:

```text
your-project-folder/
 ├── .devcontainer/
 │    ├── devcontainer.json
 │    └── Dockerfile
 ├── environment.yml
 └── app.py
```

---

## 🛠️ Step 2: Configure the Blueprint Files

Copy and paste the following code into their respective files.

### 1. `.devcontainer/Dockerfile`
This tells Docker to pull the official Microsoft Miniconda image.
```dockerfile
FROM [mcr.microsoft.com/devcontainers/miniconda:latest](https://mcr.microsoft.com/devcontainers/miniconda:latest)
```

### 2. `.devcontainer/devcontainer.json`
This configures VS Code inside the container. It automatically installs Python support and your preferred AI coding assistant. *(Replace the placeholder with your AI extension's ID).*
```json
{
    "name": "Miniconda Sandbox",
    "build": { 
        "context": "..",
        "dockerfile": "Dockerfile"
    },
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "PASTE-YOUR-AI-EXTENSION-ID-HERE"
            ]
        }
    }
}
```

### 3. `environment.yml`
This is your Conda shopping list. Add your required libraries here.
```yaml
name: my_streamlit_env
dependencies:
  - python=3.10
  - pip
  - pip:
    - streamlit
    - pandas
```

### 4. `app.py`
A simple test file to ensure the connection works.
```python
import streamlit as st

st.title("🎉 Sandbox is LIVE!")
st.write("Your Dev Container is perfectly routing Streamlit to your browser.")
st.balloons()
```

---

## 🏗️ Step 3: Build the Sandbox

1. Open your project folder in VS Code.
2. Press `Ctrl + Shift + P` to open the Command Palette.
3. Type **Rebuild** and select **Dev Containers: Rebuild and Reopen in Container**.
4. Wait for Docker to download the image and build the container. When finished, you will see `Dev Container: Miniconda` in the bottom-left corner of VS Code.

---

## 📦 Step 4: Install Libraries & Activate

Once inside the container, open a new terminal in VS Code (`Terminal > New Terminal`) and run these commands sequentially:

**1. Create the environment from your file:**
```bash
conda env create -f environment.yml
```

**2. Initialize Conda for the terminal:**
```bash
conda init bash
```
*(After running this, click the trash can icon to close the terminal, and open a brand new one).*

**3. Activate your environment:**
```bash
conda activate my_streamlit_env
```
*(Your terminal prompt should now start with `(my_streamlit_env)`).*

---

## 🏃 Step 5: Run the App

With your environment activated, start the Streamlit server:

```bash
streamlit run app.py --server.port 8502
```
VS Code will automatically detect the port and prompt you to "Open in Browser" (or you can hold `Ctrl` and click the `http://localhost:8502` link in the terminal). 

Happy Coding!

---

## 💡 Troubleshooting & Pro Tips

* **Stuck at `EOF` or `Network Timeout` during the build?** Strict corporate antivirus software (like HP Wolf Security) or IPv6 network routing can block Docker from downloading the base image. Try temporarily pausing your antivirus threat containment or disabling IPv6 on your network adapter just until the initial `docker pull` finishes.
* **Port Conflicts:** If you already have a Streamlit app running locally, the default port `8501` will clash. Always use the `--server.port 8502` (or 8503, etc.) flag to force the container app onto a new port.
* **Missing AI Extension?** The Dev Container acts as a second brain. If your AI chat disappears, ensure its Extension ID is listed in the `devcontainer.json` file and rebuild, or install it manually via the VS Code Extensions panel under the "Dev Container" section.