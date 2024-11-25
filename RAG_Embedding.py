import os
from langchain_community.document_loaders import FireCrawlLoader
from dotenv import load_dotenv,find_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

find_dotenv()

load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "about_data")
persistent_directory = os.path.join(data_dir, "chroma_about_data_data")



if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"The directory {data_dir} does not exist. Please check the path."
        )

    

    firecrawl_api = os.getenv("FireCrawlLoader_api")
    url = ["https://www.apple.com/"]
    loader = FireCrawlLoader(url,api_key= firecrawl_api,mode= "crawl")

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)


    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  


    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)

else:
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
    )

llm = ChatOpenAI(model="gpt-4o")



def fetch_and_extract_jobs():
    """
    Fetches the webpage content from a predefined URL and extracts job openings from it.
    """
    url = "https://www.workcohol.com/page-career"  # URL hardcoded as a source for job openings
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text_content = soup.get_text(separator="\n")

        prompt = PromptTemplate(
            input_variables=["web_content"],
            template=(
                "Given the following webpage content, extract all the job openings. "
                "Job openings are typically titles or headings indicating positions available at a company. "
                "If no job openings are found, respond with 'No job openings found.'\n\n"
                "Webpage Content:\n{web_content}\n\n"
                "Extracted Job Openings:"
            )
        )
        chain = LLMChain(llm=ChatOpenAI(model="gpt-4o"), prompt=prompt)

        result = chain.run({"web_content": text_content})

        job_openings = [line.strip() for line in result.split("\n") if line.strip()]
        return job_openings if job_openings else ["No job openings found."]
    except requests.exceptions.RequestException as e:
        return [f"Error fetching webpage: {e}"]

# ----------------Question Prompt--with Chat-History-----------------------

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question about the company's details, "
    "such as its services, history, mission, or products, reformulate the question "
    "into a standalone version that can be understood without relying on the chat history. "
    "Do NOT provide an answer to the question, only rewrite or return it as is."
)



contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    # -------------------Answer Prompt-----Gives the Contextuale Answer---------------------


qa_system_prompt = (
        "Given a visual input, such as an image of a document, logo, or other visual material "
    "related to a company, and a user question about the company's details (e.g., services, mission, products), "
    "analyze the visual content and provide a concise and accurate answer based solely on the information "
    "available in the visual input. Do not infer or assume details that are not explicitly present in the visual."
        "\n\n"
        "{context}"
    )

qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


    # ---------------Combining Questions & Answer ---------------------

rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain)





react_docstore_prompt = hub.pull("hwchase17/react")


    # --------------------Tools--------------------


tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="useful for when you need to answer questions about the context",
    ),
    Tool(
        name="Fetch and Extract Job Openings",
        func=lambda input: fetch_and_extract_jobs(),
        description=(
            "Scrapes the company's webpage for text content and uses an LLM to extract "
            "job openings based on context. Useful for retrieving a list of current job openings."
        ),
    ),
]


    # -------------------ReAct Agent with document store retriever--------------

agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_docstore_prompt,
    )

agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, handle_parsing_errors=True, verbose=True,
    )

chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke(
        {"input": query, "chat_history": chat_history})
    print(f"AI: {response['output']}")

# ----------------Update history-----------------

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))




