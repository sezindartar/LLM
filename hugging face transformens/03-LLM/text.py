import csv
csv.field_size_limit(10000000)

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import torch
import gradio as gr
from collections import Counter
import gc
import warnings
warnings.filterwarnings("ignore")

# 🔥 RTX 4050 Optimizasyonları
def setup_rtx_4050():
    """RTX 4050 için optimal CUDA ayarları"""
    print("🚀 RTX 4050 CUDA Optimizasyonu Başlatılıyor...")
    
    # CUDA kontrol
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
        
        # CUDA optimizasyonları
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Memory management
        torch.cuda.empty_cache()
        
        return device
    else:
        print("❌ CUDA bulunamadı! CPU kullanılacak.")
        return torch.device("cpu")

# Setup
device = setup_rtx_4050()

# 🎯 Memory-Efficient Dataset Loading
def load_medical_dataset():
    """Bellek verimli dataset yükleme"""
    print("📊 Dataset yükleniyor...")
    
    try:
        dataset_dict = load_dataset("Intelligent-Internet/II-Medical-RL")
        print(f"✅ Dataset yüklendi: {dataset_dict}")
        
        # Split kontrolü
        if 'test' in dataset_dict and 'train' in dataset_dict:
            train_dataset = dataset_dict['train']
            test_dataset = dataset_dict['test']
        else:
            full_dataset = dataset_dict['train']
            dataset_split = full_dataset.train_test_split(test_size=0.2, seed=42)
            train_dataset = dataset_split['train']
            test_dataset = dataset_split['test']
        
        # Sınıf dağılımını gör
        label_counts = Counter(train_dataset['label'])
        print("📈 Sınıf Dağılımı:", label_counts)
        
        # Etiket isimlerini belirle
        label_names = sorted(list(set(train_dataset["label"])))
        print("🏷️ Etiketler:", label_names)
        
        return train_dataset, test_dataset, label_names
        
    except Exception as e:
        print(f"❌ Dataset yükleme hatası: {e}")
        return None, None, None

# 🤖 RTX 4050 Optimize Model Setup
def setup_optimized_model(label_names):
    """RTX 4050 için optimize edilmiş model"""
    print("🤖 Model yükleniyor...")
    
    # Daha küçük model RTX 4050 için
    model_name = "bert-base-uncased"  # Küçük model
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=len(label_names),
            torch_dtype=torch.float32,  # FP32 - stable training için
        )
        
        # Model'i GPU'ya taşı
        model = model.to(device)
        
        # Model boyutunu göster
        param_count = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parametreleri: {param_count:,}")
        print(f"💾 Model VRAM: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        return None, None

# 🔄 Efficient Data Processing
def process_datasets(train_dataset, test_dataset, tokenizer, label_names):
    """Bellek verimli veri işleme"""
    print("🔄 Dataset işleniyor...")
    
    # Etiketleri sayısallaştır
    def encode_labels(example):
        example["labels"] = label_names.index(example["label"])
        return example
    
    # Tokenization function
    def tokenize(batch):
        return tokenizer(
            batch["question"], 
            padding="max_length", 
            truncation=True,
            max_length=256,  # RTX 4050 için daha küçük
            return_tensors="pt"
        )
    
    # Process datasets
    train_dataset = train_dataset.map(encode_labels, num_proc=4)
    test_dataset = test_dataset.map(encode_labels, num_proc=4)
    
    train_dataset = train_dataset.map(tokenize, batched=True, num_proc=4)
    test_dataset = test_dataset.map(tokenize, batched=True, num_proc=4)
    
    # Set format
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset

# 📊 Metrics Setup - sklearn-free version
def setup_metrics():
    """Metrik hesaplama - sklearn kullanmadan"""
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Manuel accuracy hesaplama
        accuracy = (predictions == labels).mean()
        
        # Manuel F1 hesaplama (macro)
        num_classes = len(np.unique(labels))
        f1_scores = []
        
        for class_id in range(num_classes):
            # Her sınıf için precision ve recall
            true_positives = ((predictions == class_id) & (labels == class_id)).sum()
            predicted_positives = (predictions == class_id).sum()
            actual_positives = (labels == class_id).sum()
            
            precision = true_positives / (predicted_positives + 1e-8)
            recall = true_positives / (actual_positives + 1e-8)
            
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            f1_scores.append(f1)
        
        macro_f1 = np.mean(f1_scores)
        
        return {
            "accuracy": float(accuracy),
            "f1": float(macro_f1),
        }
    
    return compute_metrics

# 🎯 RTX 4050 Optimized Training
def train_model(model, tokenizer, train_dataset, test_dataset, compute_metrics):
    """RTX 4050 için optimize edilmiş eğitim"""
    print("🎯 Model eğitimi başlıyor...")
    
    # RTX 4050 için optimize edilmiş training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,    # Daha küçük batch - RTX 4050 için
        per_device_eval_batch_size=4,     # Eval için biraz daha büyük
        gradient_accumulation_steps=4,    # Effective batch size = 2*4=8
        num_train_epochs=2,               # Daha az epoch, hızlı test
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps", 
        save_steps=400,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_steps=200,
        fp16=False,                       # FP16 kapatıldı - hata giderimi için
        bf16=False,                       # BF16 de kapatıldı
        dataloader_num_workers=0,         # CUDA için
        remove_unused_columns=False,
        report_to="none",                 # Logging kapatıldı
        seed=42,
        max_grad_norm=1.0,                # Gradient clipping
        save_total_limit=2,               # Disk alanı tasarrufu
        prediction_loss_only=False,
        skip_memory_metrics=True,         # Memory metrics skip
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Memory monitoring
    def log_memory_usage():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            cached = torch.cuda.memory_reserved() / 1024**2
            print(f"🔋 VRAM - Kullanılan: {allocated:.1f}MB, Cached: {cached:.1f}MB")
    
    # Eğitim öncesi memory
    log_memory_usage()
    
    # Eğitim
    try:
        print("🚀 Eğitim başlıyor (FP32 mode - stable)...")
        trainer.train()
        print("✅ Eğitim tamamlandı!")
        
        # Final evaluation
        eval_results = trainer.evaluate()
        print(f"📊 Final Results:")
        print(f"  📈 Accuracy: {eval_results['eval_accuracy']:.4f}")
        print(f"  📈 F1 Score: {eval_results['eval_f1']:.4f}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ VRAM yetersiz! Daha küçük batch size deneyin.")
            print("💡 Çözüm: batch_size=1, gradient_accumulation=8")
            return None
        else:
            print(f"❌ Eğitim hatası: {e}")
            return None
    except Exception as e:
        print(f"❌ Genel hata: {e}")
        return None
    
    # Eğitim sonrası memory
    log_memory_usage()
    
    return trainer

# 💾 Model Saving
def save_model(model, tokenizer, save_path="./medical-bert-rtx4050"):
    """Model kaydetme"""
    print(f"💾 Model kaydediliyor: {save_path}")
    
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("✅ Model kaydedildi!")
        
    except Exception as e:
        print(f"❌ Model kaydetme hatası: {e}")

# 🎨 Gradio UI Setup
def create_inference_function(model, tokenizer, label_names, device):
    """Inference fonksiyonu"""
    
    # 4 sınıf mapping
    answer_mapping = {
        "A": "🅰 Seçenek A",
        "B": "🅱 Seçenek B", 
        "C": "© Seçenek C",
        "D": "🆔 Seçenek D"
    }
    
    model.eval()
    
    def predict_label(text):
        try:
            if len(text.strip().split()) < 3:
                return "⚠ Soru çok kısa veya anlamlı değil. Lütfen tıbbi bir soru yazın."
            
            # Tokenize
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=256
            )
            
            # GPU'ya taşı
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=1).item()
            
            predicted_label = label_names[prediction]
            turkish_answer = answer_mapping.get(predicted_label, f"❓ Tanımsız: {predicted_label}")
            
            # VRAM kullanımını göster
            vram_used = torch.cuda.memory_allocated() / 1024**2
            result = f"{turkish_answer}\n\n💾 VRAM: {vram_used:.1f}MB"
            
            return result
            
        except Exception as e:
            return f"❌ Hata oluştu: {str(e)}"
    
    return predict_label

# 🎮 Gradio Interface
def launch_gradio_interface(predict_fn):
    """Gradio arayüzü"""
    
    interface = gr.Interface(
        fn=predict_fn,
        inputs=gr.Textbox(
            lines=3, 
            label="🏥 Tıbbi Soru (İngilizce)", 
            placeholder="Örn: Are preterm twins at increased risk?"
        ),
        outputs=gr.Textbox(label="🎯 Tahmin Sonucu"),
        title="🏥 RTX 4050 Tıbbi Soru Sınıflandırma (BERT)",
        description="🚀 RTX 4050 GPU ile hızlandırılmış tıbbi soru sınıflandırma sistemi!",
        examples=[
            ["Are preterm twins at increased risk?"],
            ["Is vaccination safe during pregnancy?"],
            ["Can I take ibuprofen during breastfeeding?"],
            ["What are the symptoms of diabetes?"],
            ["How to treat high blood pressure?"],
        ],
        theme="default",
        allow_flagging="never"
    )
    
    return interface

# 🚀 Main Execution
def main():
    """Ana fonksiyon"""
    print("🚀 RTX 4050 Tıbbi BERT Sistemi Başlatılıyor...")
    print("=" * 60)
    
    # Dataset yükleme
    train_dataset, test_dataset, label_names = load_medical_dataset()
    if train_dataset is None:
        print("❌ Dataset yüklenemedi!")
        return
    
    # Model setup
    model, tokenizer = setup_optimized_model(label_names)
    if model is None:
        print("❌ Model yüklenemedi!")
        return
    
    # Dataset processing
    train_dataset, test_dataset = process_datasets(train_dataset, test_dataset, tokenizer, label_names)
    
    # Metrics
    compute_metrics = setup_metrics()
    
    # Training
    trainer = train_model(model, tokenizer, train_dataset, test_dataset, compute_metrics)
    if trainer is None:
        print("❌ Eğitim başarısız!")
        return
    
    # Model kaydetme
    save_model(model, tokenizer)
    
    # Gradio interface
    predict_fn = create_inference_function(model, tokenizer, label_names, device)
    interface = launch_gradio_interface(predict_fn)
    
    print("\n🎉 RTX 4050 Tıbbi BERT Sistemi Hazır!")
    print("🌐 Gradio arayüzü başlatılıyor...")
    
    # Launch
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )

if __name__ == "__main__":
    main()