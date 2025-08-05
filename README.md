
# 🩺 APTOS Diabetic Retinopathy Classifier with DenseNet121

This is a web-based application built using **Streamlit** and **PyTorch** that allows users to upload a retinal image and predicts the **severity of diabetic retinopathy** using a trained **DenseNet121 model**.

---

## 📌 Features

- Upload retinal fundus images (JPEG, PNG).
- Predict the class of diabetic retinopathy (0 to 4).
- Built with Streamlit for an interactive web experience.
- Uses a fine-tuned DenseNet121 model for high accuracy.

---

## 💻 Live Demo

*Coming soon… (e.g., Streamlit Community Cloud / Hugging Face Spaces deployment link)*

---

## 🛠️ Installation & Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/aptos-densenet-streamlit.git
cd aptos-densenet-streamlit
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Linux/Mac
venv\Scripts\activate         # On Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 🧠 Model Details

- Architecture: `DenseNet121`
- Framework: `PyTorch`
- Trained on: [APTOS 2019 Blindness Detection Dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection/)
- Output: 5 Classes  
  - `0` - No DR  
  - `1` - Mild  
  - `2` - Moderate  
  - `3` - Severe  
  - `4` - Proliferative DR

---

## 📁 File Structure

```
├── app.py                # Streamlit app
├── model.pth             # Trained DenseNet121 model weights
├── requirements.txt      # Required Python packages
├── README.md             # Project documentation
└── sample_images/        # Folder for test retinal images
```

---

## 🚀 Deployment

You can deploy this app to:
- [Streamlit Community Cloud](https://streamlit.io/cloud)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Render](https://render.com)

Let me know if you'd like a deployment guide!

---

## 🙋‍♂️ Author

**Sanjay Bn**  
[LinkedIn](https://www.linkedin.com/in/sanjaybn/)  
Healthcare IT | Deep Learning | Business Analyst

---

## 📝 License

This project is licensed under the MIT License.
