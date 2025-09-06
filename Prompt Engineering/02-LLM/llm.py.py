# GEREKLİ KÜTÜPHANELERİ YÜKLE:
# pip install -U langchain langchain-openai langchain-community python-dotenv

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 1. Ortam değişkenlerini yükle (API anahtarları için .env dosyası gerekli)
load_dotenv()

# 2. LLM Tanımla
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# 3. Few-shot örnekleri oluştur (Tarzı öğretmek için)
examples = [
    {"question": "Fransa nerede?", "answer": "Fransa Batı Avrupa'da yer alır."},
    {"question": "Almanya'nın başkenti nedir?", "answer": "Almanya'nın başkenti Berlin'dir."},
    {"question": "İtalya'da ne yenir?", "answer": "İtalya’da pizza ve makarna ünlüdür."},
]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Soru: {question}\nCevap: {answer}\n"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Aşağıda bazı örnek soru-cevaplar var. Lütfen aynı tarzda yanıt ver:",
    suffix="Soru: {user_question}\nCevap:",
    input_variables=["user_question"]
)

# 4. Few-shot promptunu kullanan Tool fonksiyonu
def qna_tool_func(user_input):
    prompt = few_shot_prompt.format(user_question=user_input)
    response = llm.invoke(prompt)
    return response.content.strip()

# 5. Tool'u tanımla
qna_tool = Tool(
    name="QnA",
    func=qna_tool_func,
    description="Şehirler, ülkeler, yemekler hakkında soruları yanıtlar"
)

tools = [qna_tool]

# 6. Konuşma belleğini oluştur (Bağlamı koruyacak)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 7. Conversational Agent oluştur
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 8. Çok Adımlı Konuşmayı Çalıştır
cevap1 = agent.invoke("Simyacı kitabı ne anlatıyor?")
print("🧠 Cevap 1:", cevap1)

cevap2 = agent.invoke("kitabın yazarı kim?")
print("🧠 Cevap 2:", cevap2)

cevap3 = agent.invoke("Bu kitabın ana mesajı ne?")
print("🧠 Cevap 3:", cevap3)


"""
    C:\Users\MAC\Desktop\projelerim\02-LLM\homework2.py:54: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
C:\Users\MAC\Desktop\projelerim\02-LLM\homework2.py:57: LangChainDeprecationWarning: LangChain agents will continue to be supported, but it is recommended for new use cases to be built with LangGraph. LangGraph offers a more flexible and full-featured framework for building agents, including support for tool-calling, persistence of state, and human-in-the-loop workflows. For details, refer to the `LangGraph documentation <https://langchain-ai.github.io/langgraph/>`_ as well as guides for `Migrating from AgentExecutor <https://python.langchain.com/docs/how_to/migrate_agent/>`_ and LangGraph's `Pre-built ReAct agent <https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/>`_.
  agent = initialize_agent(


> Entering new AgentExecutor chain...
```json
{
    "action": "Final Answer",
    "action_input": "\"Simyacı\", Paulo Coelho tarafından yazılan bir romanın adıdır. Hikaye, genç bir İspanyol çoban olan Santiago'nun kişisel efsanesini (kendi kişisel amacı veya hedefi) bulma ve gerçekleştirme arayışını anlatır. Santiago'nun yolculuğu, onu çölün ötesine, Mısır piramitlerine götürür, burada onu bekleyen bir hazine olduğuna inanmaktadır. Ancak, yolculuk sırasında, Santiago'nun asıl hazinesinin, kişisel efsanesini takip etme sürecinde kazandığı bilgelik ve içsel aydınlanma olduğunu keşfederiz. Bu kitap, hayallerimizi takip etmenin ve kişisel efsanemizi bulmanın önemini vurgular." 
}
```

> Finished chain.
🧠 Cevap 1: {'input': 'Simyacı kitabı ne anlatıyor?', 'chat_history': [HumanMessage(content='Simyacı kitabı ne anlatıyor?', additional_kwargs={}, response_metadata={}), AIMessage(content='"Simyacı", Paulo Coelho tarafından yazılan bir romanın adıdır. Hikaye, genç bir İspanyol çoban olan Santiago\'nun 
kişisel efsanesini (kendi kişisel amacı veya hedefi) bulma ve gerçekleştirme arayışını anlatır. Santiago\'nun yolculuğu, onu çölün ötesine, Mısır piramitlerine götürür, burada onu bekleyen bir hazine olduğuna inanmaktadır. Ancak, yolculuk sırasında, Santiago\'nun asıl hazinesinin, kişisel efsanesini takip etme sürecinde kazandığı bilgelik ve içsel aydınlanma olduğunu keşfederiz. Bu kitap, hayallerimizi takip etmenin ve kişisel efsanemizi bulmanın önemini vurgular.', additional_kwargs={}, response_metadata={})], 'output': '"Simyacı", Paulo Coelho tarafından yazılan bir romanın adıdır. Hikaye, genç 
bir İspanyol çoban olan Santiago\'nun kişisel efsanesini (kendi kişisel amacı veya hedefi) bulma ve gerçekleştirme arayışını anlatır. Santiago\'nun yolculuğu, onu çölün ötesine, Mısır piramitlerine götürür, burada onu bekleyen bir hazine olduğuna inanmaktadır. Ancak, yolculuk sırasında, Santiago\'nun 
asıl hazinesinin, kişisel efsanesini takip etme sürecinde kazandığı bilgelik ve içsel aydınlanma olduğunu keşfederiz. Bu kitap, hayallerimizi takip etmenin ve kişisel efsanemizi bulmanın önemini vurgular.'}


> Entering new AgentExecutor chain...
```json
{
    "action": "Final Answer",
    "action_input": "Kitabın yazarı Paulo Coelho'dur."
}
```

> Finished chain.
🧠 Cevap 2: {'input': 'kitabın yazarı kim?', 'chat_history': [HumanMessage(content='Simyacı kitabı ne anlatıyor?', additional_kwargs={}, response_metadata={}), AIMessage(content='"Simyacı", Paulo Coelho tarafından yazılan bir romanın adıdır. Hikaye, genç bir İspanyol çoban olan Santiago\'nun kişisel efsanesini (kendi kişisel amacı veya hedefi) bulma ve gerçekleştirme arayışını anlatır. Santiago\'nun yolculuğu, onu çölün ötesine, Mısır piramitlerine 
götürür, burada onu bekleyen bir hazine olduğuna inanmaktadır. Ancak, yolculuk sırasında, Santiago\'nun asıl hazinesinin, kişisel efsanesini takip etme sürecinde kazandığı bilgelik ve içsel aydınlanma olduğunu keşfederiz. Bu kitap, hayallerimizi takip etmenin ve kişisel efsanemizi bulmanın önemini vurgular.', additional_kwargs={}, response_metadata={}), HumanMessage(content='kitabın yazarı kim?', additional_kwargs={}, response_metadata={}), AIMessage(content="Kitabın yazarı Paulo Coelho'dur.", additional_kwargs={}, response_metadata={})], 'output': "Kitabın yazarı Paulo Coelho'dur."}


> Entering new AgentExecutor chain...
```json
{
    "action": "Final Answer",
    "action_input": "Paulo Coelho'nun 'Simyacı' adlı kitabının ana mesajı, kişisel efsanemizi (yani kendi hayallerimizi ve hedeflerimizi) takip etmenin ve bu yolculuk sırasında öğrendiklerimizin değerini anlamanın önemi üzerinedir. Bu kitap, maddi hazine peşinde koşmanın ötesinde, hayatın anlamını ve 
kişisel gelişimimizi keşfetme yolculuğunu vurgular."
}
```

> Finished chain.
🧠 Cevap 3: {'input': 'Bu kitabın ana mesajı ne?', 'chat_history': [HumanMessage(content='Simyacı kitabı ne anlatıyor?', additional_kwargs={}, response_metadata={}), AIMessage(content='"Simyacı", Paulo Coelho tarafından yazılan bir romanın adıdır. Hikaye, genç bir İspanyol çoban olan Santiago\'nun kişerine götürür, burada onu bekleyen bir hazine olduğuna inanmaktadır. Ancak, yolculuk sırasında, Santiago\'nun asıl hazinesinin, kişisel efsanesini takip etme sürecinde kazandığı bilgelik ve içsel aydınlanma olduğunu keşfederiz. Bu kitap, hayallerimizi takip etmenin ve kişisel efsanemizi bulmanın önemini vurgular.', additional_kwargs={}, response_metadata={}), HumanMessage(content='kitabın yazarı kim?', additional_kwargs={}, response_metadata={}), AIMessage(content="Kitabın yazarı Paulo Coelho'dur.", additional_kwargs={}, response_metadata={}), HumanMessage(content='Bu kitabın ana mesajı ne?', additional_kwargs={}, response_metadata={}), AIMessage(content="Paulo Coelho'nun 'Simyacı' adlı kitabının ana mesajı, kişisel efsanemizi (yani kendi hayallerimizi ve hedeflerimizi) takip etmenin ve bu yolculuk sırasında öğrendiklerimizin değerini anlamanın önemi üzerinedir. Bu kitap, maddi hazine peşinde 
koşmanın ötesinde, hayatın anlamını ve kişisel gelişimimizi keşfetme yolculuğunu vurgular.", additional_kwargs={}, response_metadata={})], 'output': "Paulo Coelho'nun 'Simyacı' adlı kitabının ana mesajı, kişisel efsanemizi (yani kendi hayallerimizi ve hedeflerimizi) takip etmenin ve bu yolculuk sırasında öğrendiklerimizin değerini anlamanın önemi üzerinedir. Bu kitap, maddi hazine peşinde koşmanın ötesinde, hayatın anlamını ve kişisel gelişimimizi keşfetme yolculuğunu vurgular."}
(myenv) PS C:\Users\MAC\Desktop\projelerim\02-LLM> 
(myenv) PS C:\Users\MAC\Desktop\projelerim\02-LLM> 
"""