# Fixed Traffic Light Tracker

## Generate the model

From the VS Code PowerShell terminal:

```powershell
cd "C:\Users\yujun\Desktop\traffic-light-static-fixed"

py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r .\tools\requirements.txt
python .\tools\export_model.py
```

Confirm this file exists:

```text
models\traffic-light.onnx
```

## Start the static site

```powershell
py -m http.server 8000
```

Open:

```text
http://localhost:8000
```

Then press `Ctrl+F5` and click **Start camera**.

## Troubleshooting

Open Chrome DevTools:

```text
F12 → Console
F12 → Network
```

The page now shows visible errors if ONNX Runtime, the model, or the camera fails.
