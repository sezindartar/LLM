#TASK 1.
#----------------------------------------------------------------------------------------------------------
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "Transformers are amazing!"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("Token IDs:", token_ids)

# Verilen cümle 4 adet küçük dil birimine (token'a) ayrıldı.
# Her token, modelin anlayabileceği şekilde benzersiz sayısal ID’lere (token IDs) dönüştürüldü.


#------------------------------------------------------------------------------------------------------------

#TASK 2.
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

print(classifier("The new update made everything so much easier!"))
# [{'label': 'POSITIVE', 'score': 0.9996...}]

print(classifier("I'm confident that I'll master deep learning soon!"))
# [{'label': 'POSITIVE', 'score': 0.9997...}]

print(classifier("This interface is frustrating and hard to use."))
# [{'label': 'NEGATIVE', 'score': 0.9968...}]


#------------------------------------------------------------------------------------------------------------
#TASK 3.

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

result = generator("After spending hours exploring neural networks,", max_length=30, temperature=0.9)
print(result[0]['generated_text'])

print("------------------------------------------------------------------------------------------------------------")

result = generator("As the sun rose over the quiet city,", max_length=30, temperature=0.3)
print(result[0]['generated_text'])

print("------------------------------------------------------------------------------------------------------------")

result = generator("I wanted to focus, but my mind kept drifting", max_length=30, temperature=0.7)
print(result[0]['generated_text'])

print("------------------------------------------------------------------------------------------------------------")

# Aynı model ayarlarıyla üç farklı başlangıç cümlesi denendi. Her seferinde farklı metinler üretildi.
# Temperature değeri 0.7 olduğunda, anlam açısından dengeli fakat yer yer belirsiz ifadeler gözlendi.
# 0.3 ile denemelerde model daha tutarlı ve sade cümleler üretti, ancak içerik kısa ve tekrarlıydı.
# 0.9 ile denendiğinde ise daha yaratıcı, uzun ve açıklayıcı cümleler üretildi. Fakat bazen anlam bütünlüğü zayıflayabiliyor.
# Temperature değeri arttıkça çeşitlilik ve yaratıcılık artıyor; azalttıkça öngörülebilirlik yükseliyor, ancak içerik zayıflıyor.

#------------------------------------------------------------------------------------------------------------
#TASK 4.   
import os
from openai import OpenAI
from dotenv import load_dotenv

# .env dosyasındaki API anahtarını yükle
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prompt cümlesi
prompt = "Bir sabah uyandığında her şey değişmişti çünkü artık kimse konuşmuyor, herkes sadece bakışlarla iletişim kuruyordu."

# API üzerinden cevap oluştur
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  
    messages=[
        {"role": "system", "content": "Sen yaratıcı bir hikaye anlatıcısısın."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=256,
)

# Cevabı yazdır
print("🚀 Tamamlanan Metin:\n")
print(response.choices[0].message.content)

## Bir sabah uyandığında her şey değişmişti çünkü artık kimse konuşmuyor, herkes sadece bakışlarla iletişim kuruyordu.
# 
# Tolga, kentin yüksek güvenlikli veri merkezinde gece nöbetindeydi. Uyandığında alarm çalmıyordu. Monitörler normaldi. Ama... sessizlik anormaldi.
# 
# Dışarı çıktı, sokakta insanlar vardı ama bir tuhaflık vardı: kimse konuşmuyordu.
# 
# Göz teması... Herkesin bakışları adeta bir algoritma gibi çalışıyordu. 
# Bir bakış – “merhaba”. Uzun bir bakış – “tehlike var”. Yan bakış – “yardım et.”
# 
# Tolga, gün boyunca tek bir ses bile duymadan analiz yaptı. Sonunda fark etti: 
# dün gece şehir, deneysel bir nöral ağ tabanlı iletişim protokolüne geçirilmişti.
# 
# İnsanların beyin dalgaları, yapay zeka tarafından algılanıyor ve doğrudan aktarılıyordu.
# 
# Ancak sistem hatalıydı. Duygular kodlara karışıyor, düşünceler çarpıtılıyordu.
# 
# Tolga’nın son baktığı ekranda kırmızı bir mesaj belirdi:
# “Konuşma yetisini tamamen silmek için son 2 dakika.”
# 
# Ve sonra sistemden bir sinyal geldi. Tolga içgüdüsel olarak düşündü: “Hayır.”
# Ama sistem, bunu “evet” olarak yorumladı...



## Sonuç beni çok şaşırtmadı çünkü OpenAI’nin dil modelleri, yaratıcı ve anlamlı metin üretme konusunda oldukça başarılı.
# Belirlenen maksimum kelime sınırı nedeniyle üretilen metin nispeten kısa kaldı.
# Eğer token veya kelime sınırı daha yüksek ayarlanmış olsaydı, model çok daha detaylı ve derinlemesine bir hikaye sunabilirdi.
# Bu nedenle, hikayenin potansiyeli tam olarak ortaya çıkmamış olabilir.
