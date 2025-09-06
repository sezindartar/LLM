import csv
import numpy as np
import torch
import gradio as gr
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")
csv.field_size_limit(10000000)


def setup_device():
    """Cihazı ayarlar (CUDA veya CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU bulundu: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("❌ CUDA destekli GPU bulunamadı, CPU kullanılacak.")
    return device


def load_medical_dataset():
    """Tıbbi veri setini yükler ve böler"""
    print("📊 Veri seti yükleniyor...")
    try:
        dataset_dict = load_dataset("Intelligent-Internet/II-Medical-RL")

        if 'test' in dataset_dict and 'train' in dataset_dict:
            train_dataset, test_dataset = dataset_dict['train'], dataset_dict['test']
        else:
            dataset_split = dataset_dict['train'].train_test_split(test_size=0.2, seed=42)
            train_dataset, test_dataset = dataset_split['train'], dataset_split['test']

        label_names = sorted(list(set(train_dataset["label"])))
        print("🏷️ Etiketler:", label_names)
        print("📈 Sınıf Dağılımı:", Counter(train_dataset['label']))

        return train_dataset, test_dataset, label_names
    except Exception as e:
        print(f"❌ Veri seti yükleme hatası: {e}")
        return None, None, None


def setup_model(model_name, label_names, device):
    """Modeli ve tokenizer'ı yükler"""
    print(f"🤖 Model yükleniyor: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label_names)
        ).to(device)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parametre sayısı: {param_count:,}")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        return None, None


def process_datasets(train_dataset, test_dataset, tokenizer, label_names):
    """Veri setlerini işler ve tokenizasyon yapar"""
    print("🔄 Veri seti işleniyor...")

    label_map = {name: i for i, name in enumerate(label_names)}

    def encode_labels(example):
        example["labels"] = label_map[example["label"]]
        return example

    def tokenize(batch):
        return tokenizer(
            batch["question"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    train_dataset = train_dataset.map(encode_labels)
    test_dataset = test_dataset.map(encode_labels)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, test_dataset


def compute_metrics(eval_pred):
    """Değerlendirme metriklerini hesaplar"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = (predictions == labels).mean()

    num_classes = len(np.unique(labels))
    f1_scores = []
    for class_id in range(num_classes):
        true_positives = ((predictions == class_id) & (labels == class_id)).sum()
        predicted_positives = (predictions == class_id).sum()
        actual_positives = (labels == class_id).sum()

        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores.append(f1)

    return {"accuracy": float(accuracy), "f1": float(np.mean(f1_scores))}


def train_model(model, train_dataset, test_dataset):
    """Modeli eğitir"""
    print("🎯 Model eğitimi başlıyor...")

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        learning_rate=3e-5,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),  # CUDA varsa FP16 kullan
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    try:
        trainer.train()
        print("✅ Eğitim tamamlandı!")
        eval_results = trainer.evaluate()
        print(f"📊 Final Sonuçları: Accuracy={eval_results['eval_accuracy']:.4f}, F1={eval_results['eval_f1']:.4f}")
        return trainer
    except Exception as e:
        print(f"❌ Eğitim sırasında hata oluştu: {e}")
        return None


def create_inference_ui(model, tokenizer, label_names, device):
    """Gradio arayüzünü ve tahmin fonksiyonunu oluşturur"""
    model.eval()
    answer_mapping = {"A": "🅰", "B": "🅱", "C": "©", "D": "🆔"}

    def predict_label(text):
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                prediction = torch.argmax(logits, dim=1).item()

            predicted_label = label_names[prediction]
            return f"{answer_mapping.get(predicted_label, '❓')} Seçenek {predicted_label}"
        except Exception as e:
            return f"❌ Hata: {str(e)}"

    return gr.Interface(
        fn=predict_label,
        inputs=gr.Textbox(lines=3, label="🏥 Tıbbi Soru (İngilizce)",
                          placeholder="Örn: Is vaccination safe during pregnancy?"),
        outputs=gr.Textbox(label="🎯 Tahmin Sonucu"),
        title="🏥 Tıbbi Soru Sınıflandırma (BERT)",
        description="Tıbbi bir sorunun hangi kategoriye ait olduğunu tahmin eder.",
        examples=[["Are preterm twins at increased risk?"], ["What are the symptoms of diabetes?"]],
    )


def main():
    """Ana program akışı"""
    device = setup_device()
    train_ds, test_ds, labels = load_medical_dataset()
    if not labels: return

    model, tokenizer = setup_model("bert-base-uncased", labels, device)
    if not model: return

    train_ds, test_ds = process_datasets(train_ds, test_ds, tokenizer, labels)

    trainer = train_model(model, train_ds, test_ds)
    if not trainer: return

    print(f"💾 Model kaydediliyor: ./medical-bert-model")
    trainer.save_model("./medical-bert-model")
    tokenizer.save_pretrained("./medical-bert-model")

    interface = create_inference_ui(model, tokenizer, labels, device)
    print("\n🎉 Sistem Hazır! Gradio arayüzü başlatılıyor...")
    interface.launch(server_name="0.0.0.0", server_port=7861,share=True)


if __name__ == "__main__":
    main()

