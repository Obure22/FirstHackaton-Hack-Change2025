import os
import uuid
from io import BytesIO

from flask import (
    Flask, render_template, request,
    send_file, redirect, url_for, flash
)
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# ==========================
#        НАСТРОЙКИ
# ==========================

app = Flask(__name__)
app.secret_key = "super-secret-key"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ускорения
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_grad_enabled(False)

# ==========================
#       УСТРОЙСТВО
# ==========================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("🔥 Используем GPU")
else:
    device = torch.device("cpu")
    print("⚙️ GPU нет — работаем на CPU")

# ==========================
#       ЗАГРУЗКА МОДЕЛИ
# ==========================

MODEL_PATH = "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

LABELS = {
    0: "Отрицательная",
    1: "Нейтральная",
    2: "Положительная"
}

# ==========================
#     СУПЕР-УСКОРЕННЫЙ ИНФЕРЕНС
# ==========================

def predict_batch(texts, batch_size=128):
    """
    Ультра-быстрый инференс.
    - Большие батчи (128)
    - max_length 128 (вместо 256)
    - Параллельная токенизация
    - Полный прогресс-бар
    """
    all_preds = []
    encoded_batches = []

    # -------- Токенизация --------
    print("⌛ Токенизация...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing", ncols=80):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        encoded_batches.append(enc)

    # -------- МОДЕЛЬ --------
    print("🔥 Инференс модели...")
    for enc in tqdm(encoded_batches, desc="Model", ncols=80):
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(**enc).logits
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

    return all_preds


# ==========================
#           ROUTES
# ==========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if 'file' not in request.files:
        flash("Файл не найден.")
        return redirect(url_for("index"))

    file = request.files['file']
    if file.filename == "":
        flash("Файл не выбран.")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(file)
    except Exception as e:
        flash(f"Ошибка чтения: {e}")
        return redirect(url_for("index"))

    if 'text' not in df.columns:
        flash("В CSV должна быть колонка text.")
        return redirect(url_for("index"))

    texts = df['text'].astype(str).tolist()

    print(f"📄 Загружено строк: {len(texts)}")
    print("🚀 Запускаем классификацию...")

    preds = predict_batch(texts)

    print("✅ Готово!")

    df['pred'] = preds
    df['label_name'] = df['pred'].map(LABELS)

    if 'src' not in df.columns:
        df['src'] = 'unknown'

    counts = df['label_name'].value_counts().to_dict()

    file_id = str(uuid.uuid4())
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")

    preview_df = df.head(200)

    return render_template(
        "results.html",
        file_id=file_id,
        table=preview_df.to_dict(orient="records"),
        counts=counts,
        labels_order=list(LABELS.values())
    )


@app.route("/download/<file_id>")
def download(file_id):
    path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.csv")
    if not os.path.exists(path):
        flash("Файл не найден.")
        return redirect(url_for("index"))

    return send_file(
        path,
        as_attachment=True,
        download_name="predicted.csv",
        mimetype="text/csv"
    )


@app.route("/evaluate", methods=["POST"])
def evaluate():
    if 'file' not in request.files:
        flash("Файл не найден.")
        return redirect(url_for("index"))

    file = request.files['file']
    if file.filename == "":
        flash("Файл не выбран.")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(file)
    except Exception as e:
        flash(f"Ошибка чтения: {e}")
        return redirect(url_for("index"))

    if 'text' not in df.columns or 'label' not in df.columns:
        flash("В CSV должны быть text и label.")
        return redirect(url_for("index"))

    texts = df['text'].astype(str).tolist()
    true_labels = df['label'].tolist()

    print(f"📄 Строк для оценки: {len(texts)}")
    print("🚀 Предсказания для метрики...")

    preds = predict_batch(texts)

    macro_f1 = f1_score(true_labels, preds, average="macro")

    report = classification_report(
        true_labels,
        preds,
        target_names=[LABELS[i] for i in sorted(LABELS.keys())],
        digits=4
    )

    return render_template(
        "metrics.html",
        macro_f1=round(macro_f1, 4),
        report=report
    )


if __name__ == "__main__":
    # host="0.0.0.0" — пригодится для деплоя
    app.run(debug=True)
