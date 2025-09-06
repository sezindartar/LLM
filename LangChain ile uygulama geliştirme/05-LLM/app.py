from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("❌ OPENAI_API_KEY bulunamadı. .env dosyanızı kontrol edin.")

file_path = r"C:\Users\fb\Desktop\05-llm\ML 00 Makine Öğrenmesi Giriş.pdf"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ PDF dosyası bulunamadı: {file_path}")
loader = PyPDFLoader(file_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = splitter.split_documents(documents)

mbedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(texts, mbedding_model)

llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=openai_api_key)
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
konular = {
    "Makine öğrenmesi nedir": "Makine öğrenmesi nedir",
    "Makine öğrenmesi uygulama adımları": "Makine öğrenmesi uygulama adımları",
    "Makine öğrenmesi türleri": "Makine öğrenmesi türleri",
    "Makine öğrenmesinde performans parametreleri": "Makine öğrenmesinde performans parametreleri"
}
print("🎓 ML Mentor Asistanı Başladı (Çıkmak için 'çık' yazın)\n")
ogrenilen_konular = set()

while True:
    user_input = input("👤 Siz: ")
    if user_input.lower() in ["çık", "exit"]:
        print("Görüşmek üzere!")
        break
    
    if "bugün ne öğrendim" in user_input.lower() or "geri bildirim" in user_input.lower():
        print("\n Asistan (Geri Bildirim):")
        
        if ogrenilen_konular:
            print("Bugün öğrendiğiniz konular:")
            for konu in ogrenilen_konular:
                print(f"- {konu}")
        else: 
            print("- Henüz bir konu üzerinde konuşmadık.")
        eksik_konular = set(konular.values()) - ogrenilen_konular
        if eksik_konular:
            print("\nEksik Kalan Konular:")
            for konu in eksik_konular:
                print(f"- {konu}")
            print("\nBu konulara da göz atmanızı öneririm.")
        else:
            print("\nTebrikler! Tüm temel konuları öğrendiniz.")
        continue
        
    response = qa_chain.run(user_input)
    print("🤖 Asistan:", response)
    
    for anahtar, konu in konular.items():
        if anahtar in user_input.lower():
            ogrenilen_konular.add(konu)
    
    
