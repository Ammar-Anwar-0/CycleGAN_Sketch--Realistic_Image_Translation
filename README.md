# CycleGAN for Sketch ↔ Realistic Image Translation 🌀🖼️

This project implements a **CycleGAN** using PyTorch to translate between **sketches and realistic face images**. It includes both training code and a Flask-based web application to demo real-time image translation.

---

## 🚀 Features

- Translate **sketch → real image** and **real image → sketch**
- PyTorch-based CycleGAN with:
  - Two generators (`G` for Sketch→Real, `F` for Real→Sketch)
  - Two discriminators (`D_X`, `D_Y`)
  - Cycle-consistency and identity loss
- Inference visualizations with Matplotlib
- Flask web app to upload images and view translated results
- Saves models: `generator_G.pth`, `generator_F.pth`, etc.

---

## 🧠 Training Highlights

- **Data**: Grayscale paired images resized to 128×128
- **Losses**: MSE (GAN), L1 (cycle, identity)
- **Optimizers**: Adam with learning rate 0.0002
- **Epochs**: 10
- Outputs include generator and discriminator loss plots

---

## 💻 Web App (Flask)

Run `app.py` to launch a web interface:
- Upload a **sketch** → get a **generated realistic image**
- Upload a **real image** → get a **generated sketch**

### Routes
- `/` — Home page
- `/upload_sketch` — Upload sketch for generation
- `/upload_real` — Upload real image for sketch generation

### Files are saved under:
- `uploads/`
- `├── your_uploaded_image.png`
- `├── generated_real_your_image.png`
- `├── generated_sketch_your_image.png`

## 🧪 How to Run

1. Train models (or use saved `.pth` files)
2. Place them in the project folder
3. Run the app:
```bash
python app.py
