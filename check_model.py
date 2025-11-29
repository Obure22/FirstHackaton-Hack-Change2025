import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score

MODEL_PATH = "model"       # Папка твоей обученной модели
TRAIN_PATH = "train.csv"   # Положи сюда train.csv

# -----------------------------
# 1. Устройство
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# 2. Загрузка модели
# -----------------------------
print("\nЗагружаем модель...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print("Модель загружена.")

LABELS = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

# -----------------------------
# 3. Функция предсказания
# -----------------------------
def predict(text):
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits

    pred = torch.argmax(logits, dim=1).item()
    return pred, LABELS[pred]

# -----------------------------
# 4. Тест на ручных примерах
# -----------------------------
print("\n=== РУЧНЫЕ ПРИМЕРЫ ===")

examples = [
    "Прекрасное качество, очень доволен покупкой",
    "Ужасный товар, вообще не работает",
    "Всё отлично!"
]

for text in examples:
    pred_id, pred_name = predict(text)
    print(f"{text}  →  {pred_id} ({pred_name})")

# -----------------------------
# 5. Проверка F1 на train.csv
# -----------------------------
print("\n=== ПРОВЕРКА НА TRAIN.CSV ===")

df = pd.read_csv(TRAIN_PATH)

if "text" not in df.columns or "label" not in df.columns:
    print("train.csv должен содержать колонки text и label")
    exit()

# Берём подвыборку, чтобы быстро считать
sample = df.sample(2000, random_state=42) if len(df) > 2000 else df

texts = sample["text"].astype(str).tolist()
true_labels = sample["label"].tolist()

preds = []

BATCH_SIZE = 64
print(f"Считаем предсказания для {len(sample)} примеров...")

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i+BATCH_SIZE]

    enc = tokenizer(
        batch,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits

    batch_pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
    preds.extend(batch_pred)

macro_f1 = f1_score(true_labels, preds, average="macro")

print("\n=== РЕЗУЛЬТАТЫ ===")
print("Macro-F1 на подвыборке train:", macro_f1)
print("==============================\n")

