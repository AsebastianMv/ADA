import chainlit as cl
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType,create_pandas_dataframe_agent
from dotenv import dotenv_values
from langchain.agents.agent_types import AgentType
import os
from langchain.memory import ConversationBufferWindowMemory,ConversationSummaryMemory,ConversationKGMemory,CombinedMemory
import pandas as pd
from langchain.vectorstores import Chroma

os.environ['REQUESTS_CA_BUNDLE'] = 'C:/Users/armadrid/Documents/LangChain/ROOTBANCOLOMBIACA.crt'
ENV = dotenv_values(".env")

df_proveedores = pd.read_excel("./data/Supplier.xlsx")
df_gasto = pd.read_excel("./data/Gasto.xlsx")
df_Contratos = pd.read_excel("./data/Contratos.xlsx")

@cl.on_chat_start
async def main():         
    #Se define el modelo, se configura y se consume la información del vector store
    llm = OpenAI(model_name="text-davinci-003",temperature=0,verbose=True) 
    vectorStore = Chroma(persist_directory="./vector_store",embedding_function=HuggingFaceEmbeddings())

    #Se configura imagen y mensaje de inicio
    elements = [
    cl.Image(name="image1",display="inline", path="./imagenes/Capibara.png")
    ]
    await cl.Message(content="Hola! Bienvenido, te ayudaré a encontrar información de Abastecimiento", elements=elements).send()

    #Se define la cadena para buscar la información no estructurada
    chain = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=vectorStore.as_retriever()             
    )
    #Se define el modelo para el contexto de memoria del chat y se configura el promt
    llm_context = ChatOpenAI(temperature=0, model_name="gpt-4")
    PREFIX = """
    You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc.    
    You should use the tools below to answer the question posed of you:

    Summary of the whole conversation:
    {chat_history_summary}

    Last few messages between you and user:
    {chat_history_buffer}

    Entities that the conversation is about:
    {chat_history_KG}

    """

    chat_history_buffer = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history_buffer",
    input_key="input"
    )

    chat_history_summary = ConversationSummaryMemory(
        llm=llm_context, 
        memory_key="chat_history_summary",
        input_key="input"
        )

    chat_history_KG = ConversationKGMemory(
        llm=llm_context, 
        memory_key="chat_history_KG",
        input_key="input",
        )

    memory = CombinedMemory(memories=[chat_history_buffer, chat_history_summary, chat_history_KG])

    #Se instancia el pandas dataframe, con el contexto y la información de memoria
    pandasPower = create_pandas_dataframe_agent(
    llm, 
    [df_proveedores,df_Contratos,df_gasto],
    prefix=PREFIX,    
    verbose=True, 
    agent_executor_kwargs={"memory": memory},
    input_variables=['dfs_head','num_dfs', 'input', 'agent_scratchpad', 'chat_history_buffer', 'chat_history_summary', 'chat_history_KG']
    )
    #Se definen las tools que usara el agente
    tools = [
        Tool(
           name="Desestructurados",
           func= chain.run,
           description="Usar cuando la pregunta no requiera datos numérico y sea solo cualitativa"            
        ),
        Tool(
            name="Estructurados",
            func= pandasPower.run,
            description="Usar cuando la pregunta sea numérica de cantidades, promedios o datos de cuantitativos y que no existan en el ConversationBufferMemory"            
        ),
    ]

    #Se inicializa el agente
    agent_chain = initialize_agent(
        tools, llm=llm,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True,handle_parsing_errors=True
    )
    #Se setea el agente para ser usado en las respuestas a los mensajes del usuario
    cl.user_session.set("agent", agent_chain)
   
  

@cl.on_message
async def main(message):
   #Se obtiene el agente
   agent = cl.user_session.get("agent")      
   #Se obtiene la respuesta
   res = await agent.arun(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
   #Se envia la respuesta hacia el usuario
   await cl.Message(content=res).send()