
import os
import pandas as pd
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import traceback


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXCEL_FILE_PATH =  r'C:\Users\MAC\Desktop\04-LLM\soru-cevap.xlsx'
CHROMA_COLLECTION_NAME = 'python_qa_collection'
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API anahtarı .env dosyasında bulunamadı. Lütfen kontrol edin.")

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    chroma_client = chromadb.Client()
except Exception as e:
    print(f"İstemciler başlatılırken bir hata oluştu: {e}")
    exit()



NUM_ITEMS_TO_ADD = 1

def setup_vector_db():
    """
    Excel'i okur, verileri vektörleştirir ve ChromaDB'ye yükler.
    (Böl ve Yönet Test Versiyonu)
    """
    print(">>> Vektör veritabanı kurulumu başlıyor...")
    
    try:
       
        if CHROMA_COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
            chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
        
        collection = chroma_client.create_collection(name=CHROMA_COLLECTION_NAME)
        
        df = pd.read_excel(EXCEL_FILE_PATH)
        
        
        print(f">>> Test için sadece ilk {NUM_ITEMS_TO_ADD} adet veri kullanılacak.")
        questions = df['Soru'].tolist()[:NUM_ITEMS_TO_ADD]
        answers = df['Cevap'].tolist()[:NUM_ITEMS_TO_ADD]
        

        if not questions:
            print("HATA: Test için hiç veri bulunamadı. NUM_ITEMS_TO_ADD değerini kontrol et.")
            return None

        print(f">>> {len(questions)} adet soru vektörleştiriliyor...")
        response = openai_client.embeddings.create(
            input=questions, 
            model=EMBEDDING_MODEL,
            timeout=30.0
        )
        embeddings = [item.embedding for item in response.data]
        
        print(">>> Vektörler ChromaDB'ye ekleniyor...")
        
        
        collection.add(
            embeddings=embeddings,
            documents=questions,
            metadatas=[{'cevap': answer} for answer in answers],
            ids=[f"id_{i}" for i in range(len(questions))]
        )
        
        print(">>> Vektör veritabanı başarıyla kuruldu.")
        return collection
        
    except Exception:
        print("\n!!!!!! KURULUM SIRASINDA BEKLENMEDİK BİR HATA YAKALANDI !!!!!!")
        traceback.print_exc()
        return None

def ask_question(question: str, collection):
    """
    Kullanıcı sorusunu alır, RAG sürecini işletir ve cevabı döndürür.
    """
    print("\n>>> Cevap aranıyor...")
    try:
        query_embedding = openai_client.embeddings.create(
            input=[question], model=EMBEDDING_MODEL
        ).data[0].embedding

        results = collection.query(query_embeddings=[query_embedding], n_results=1)

        if not results['ids'][0]:
            retrieved_context = "Veritabanında ilgili bir bilgi bulunamadı."
            distance = 1.0
        else:
            retrieved_question = results['documents'][0][0]
            retrieved_answer = results['metadatas'][0][0]['cevap']
            distance = results['distances'][0][0]
            retrieved_context = f"Soru: \"{retrieved_question}\"\nCevap: \"{retrieved_answer}\""

        similarity_threshold = 0.25
        if distance < similarity_threshold:
            return f"[DOKÜMANDAN DİREKT]: {retrieved_answer}"

        prompt = f"""
        Sen Python programlama dili konusunda uzman bir asistansın. Görevin, sana verilen kullanıcı sorusunu, DOKÜMAN BİLGİSİ'ni referans alarak veya genel bilginle yanıtlamaktır.

        KURALLAR:
        1. Eğer kullanıcı sorusu, sağlanan DOKÜMAN BİLGİSİ ile yakından ilgiliyse, cevabını bu bilgiye dayandırarak oluştur.
        2. Eğer DOKÜMAN BİLGİSİ yetersizse AMA soru genel olarak Python ile ilgiliyse, kendi genel bilgini kullanarak soruyu cevapla.
        3. EN ÖNEMLİ KURAL: Eğer kullanıcının sorusu Python programlama ile tamamen ALAKASIZ ise (örneğin siyaset, spor, hava durumu, kişisel sorular vb.), başka hiçbir şey yazmadan SADECE 'Bu soru benim uzmanlık alanımın dışındadır.' yanıtını ver.

        ---
        DOKÜMAN BİLGİSİ (Referans Alınacak İçerik):
        {retrieved_context}
        ---
        
        KULLANICI SORUSU:
        "{question}"
        """

        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Sen Python konusunda uzman bir asistansın ve sadece Python ile ilgili soruları cevaplarsın."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return f"[AI ÜRETİMİ]: {response.choices[0].message.content}"

    except Exception:
        traceback.print_exc()
        return "Soru işlenirken beklenmedik bir hata oluştu. Lütfen terminali kontrol edin."

def main():
    """
    Ana program akışı: Sistemi kurar ve interaktif soru-cevap döngüsünü başlatır.
    """
    collection = setup_vector_db()
    
    if collection:
        print("\n" + "="*50)
        print("      İNTERAKTİF RAG SİSTEMİ BAŞLATILDI")
        print("Sisteme Python ile ilgili sorular sorabilirsiniz.")
        print("Çıkmak için 'exit' veya 'çıkış' yazmanız yeterlidir.")
        print("="*50)

        # Kullanıcıdan sürekli girdi almak için sonsuz döngü
        while True:
            # Kullanıcıdan bir soru girmesini iste
            user_question = input("\nLütfen sorunuzu girin > ")

            # Çıkış komutlarını kontrol et
            if user_question.lower() in ['exit', 'çıkış', 'quit']:
                print("\nProgramdan çıkılıyor. Hoşça kalın!")
                break
            
            # Soru boş değilse, cevap üret
            if user_question.strip():
                answer = ask_question(user_question, collection)
                print(f"\nCEVAP:\n{answer}")
            else:
                print("Lütfen geçerli bir soru girin.")
    else:
        print("\nSistem kurulumu başarısız olduğu için program sonlandırılıyor.")

if __name__ == '__main__':
    main()
    
    
    """
    İNTERAKTİF RAG SİSTEMİ BAŞLATILDI
Sisteme Python ile ilgili sorular sorabilirsiniz.
Çıkmak için 'exit' veya 'çıkış' yazmanız yeterlidir.
==================================================

Lütfen sorunuzu girin > python nedir 

>>> Cevap aranıyor...

CEVAP:
[DOKÜMANDAN DİREKT]: Python, yorumlamalı, nesne yönelimli, üst düzey bir programlama dilidir. Dinamik yazım (dynamic typing) ve dinamik bağlama (dynamic binding) özellikleri ile bilinir.

Lütfen sorunuzu girin > bugün hava nasıl 

>>> Cevap aranıyor...

CEVAP:
[AI ÜRETİMİ]: Bu soru benim uzmanlık alanımın dışındadır.

Lütfen sorunuzu girin >
"""