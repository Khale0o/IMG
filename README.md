# AI Face Segmentation & Filters

Simple Streamlit demo that uses MediaPipe Selfie Segmentation with OpenCV filters.

## Run on Windows PowerShell

```powershell
cd C:\Users\hp\Desktop\my_python_project
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

## Fix dependencies manually

If MediaPipe is installed incorrectly, reinstall the packages inside the active virtual environment:

```powershell
python -m pip install --upgrade pip
python -m pip uninstall mediapipe -y
python -m pip install streamlit opencv-python numpy mediapipe
```

Then verify MediaPipe:

```powershell
python -c "import mediapipe as mp; print(mp.__file__); print(mp.__version__); print(hasattr(mp, 'solutions'))"
```

The last line should print:

```text
True
```

## Important

Use a normal Python installation from python.org or the Microsoft Store. The project needs a real development Python that supports `venv` and `pip`.
