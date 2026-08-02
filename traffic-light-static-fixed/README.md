# Traffic Light Tracker — tiled inference update

This version requests a high-resolution rear-camera stream and scans overlapping
crops from the upper part of the frame. It is a drop-in replacement for the
previous static browser project.

## Copy the model

Copy your existing model into:

```text
models/traffic-light.onnx
```

## Test in VS Code

```powershell
cd "C:\Users\yujun\Desktop\traffic-light-static-tiled"
py -m http.server 8000
```

Open:

```text
http://localhost:8000
```

Press `Ctrl+F5` after replacing an older version.

## Azure Static Web Apps

Deploy these files from the repository root:

```text
index.html
app.js
styles.css
staticwebapp.config.json
models/traffic-light.onnx
```

Use:

```yaml
app_location: "/"
api_location: ""
output_location: ""
skip_app_build: true
```

## Scan modes

- **Upper frame · 2 tiles:** faster; good initial phone setting.
- **Upper frame · 3 tiles:** better small-object coverage; slower.
- **Full frame · faster:** behavior closest to the previous version.

Turn on **Show scan tiles** to verify what area is being processed.

## Notes

Tiling improves the number of pixels available to the generic COCO detector,
but it does not replace a traffic-light-specific model. The next major accuracy
improvement will come from training a dedicated model on distant traffic signals.
