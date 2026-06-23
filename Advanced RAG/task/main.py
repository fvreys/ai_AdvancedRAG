from langchain_core.documents import Document
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import SecretStr
from sqlalchemy import (Column,Date,Float,Integer,MetaData,String,Table,Text,create_engine,delete,insert,)
from datetime import datetime
from pathlib import Path
import csv
import json
import dotenv
import os


TAGS: dict[str, list[str]] = {
    "qa": ["frequently asked questions", "help", "general information"],
    "policy": ["shipping", "returns", "privacy"],
    "how-to": ["steps", "guides", "instructions"],
    "support": ["customer support", "live chat", "contact methods"],
}


# CREATE THE DATABASE AND 3rd TABLE
def get_database_url() -> str:
    """
    Return the database URL used by the project.
    The tests usually provide PGVECTOR_CONNECTION_STRING.

    """
    dotenv.load_dotenv()
    database_url: str | None = os.getenv ("PGVECTOR_CONNECTION_STRING")
    if not database_url:
        raise RuntimeError ("PGVECTOR_CONNECTION_STRING is not set.")
    return database_url


def import_orders_from_csv(csv_file: str = "sales_data.csv") -> None:
    """ Import sales_data.csv into a Postgres table named orders.
    Existing rows are deleted before inserting the CSV content.
    Repeated program runs stay deterministic and do not duplicate orders.
    """
    csv_path = Path(csv_file)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file was not found: {csv_path}")

    database_url = get_database_url()
    engine = create_engine(database_url)
    metadata = MetaData()

    orders = Table(
        "orders",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("Order Date", Date, nullable=False),
        Column("Order ID", String(50), nullable=False),
        Column("Product ID", String(50), nullable=False),
        Column("Product Name", Text, nullable=False),
        Column("Product Category", Text, nullable=False),
        Column("Purchase Address", Text, nullable=False),
        Column("Price Each", Float, nullable=False),
    )

    metadata.create_all(engine, tables=[orders])

    rows_to_insert = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "Order Date",
            "Order ID",
            "Product ID",
            "Product Name",
            "Product Category",
            "Purchase Address",
            "Price Each",
        }

        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"The CSV file is missing required columns: {sorted(missing_columns)}"
            )

        for row in reader:
            rows_to_insert.append(
                {
                    "Order Date": datetime.strptime (
                        row["Order Date"].strip (),
                        "%Y-%m-%d",
                    ).date (),
                    "Order ID": row["Order ID"].strip (),
                    "Product ID": row["Product ID"].strip (),
                    "Product Name": row["Product Name"].strip (),
                    "Product Category": row["Product Category"].strip (),
                    "Purchase Address": row["Purchase Address"].strip (),
                    "Price Each": float (row["Price Each"]),
                }
            )

    with engine.begin() as connection:
        connection.execute(delete(orders))

        if rows_to_insert:
            connection.execute(insert(orders), rows_to_insert)

    print(f"Imported {len(rows_to_insert)} orders into the orders table.")

    return None


# CLEAN THE DATA
def load_data_to_clean(file_to_clean: str = "knowledge_base_noisy.json") -> dict | list:
    """ Load data to clean from a file with a default file name. """
    with open (file_to_clean, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def clean_data(data_to_clean: dict | list) -> dict | list:
    """ Clean the data and return a list of cleaned items. """
    cleaned_data = []
    for item in data_to_clean.get('knowledge-base', []):
        if 'question' in item and 'answer' in item:
            cleaned_data.append ({
                "type": "qa",
                "question": item["question"],
                "answer": item["answer"]
            })
        elif 'policy' in item:
            cleaned_data.append ({
                "type": "policy",
                "policy": item["policy"]
            })
        elif 'support' in item:
            cleaned_data.append ({
                "type": "support",
                "support": item["support"]
            })
        elif 'steps' in item:
            cleaned_data.append ({
                "type": "how-to",
                "how-to": item["steps"]
            })

    return {"knowledge-base": cleaned_data}

def save_cleaned_data(data_to_save: dict | list) -> None:
    """ Save clean data to knowledge_base_clean.json. """
    with open('knowledge_base_clean.json', 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    return None


# CHUNK DATA AND CREATE DOCUMENTS
def chunk_data(data: dict | list, chunk_size: int = 5) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if isinstance(data, dict):
        entries = data.get("knowledge-base", [])
    elif isinstance(data, list):
        entries = data
    else:
        raise TypeError("data must be a dictionary or a list")

    if not isinstance(entries, list):
        raise TypeError("'knowledge-base' must contain a list of entries")

    # Group entries by type
    grouped_entries = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_type = entry.get("type")
        if entry_type not in TAGS:
            continue
        grouped_entries.setdefault(entry_type, []).append(entry)

    # Split each group type into chunks of 5 entries
    langchain_documents = []

    chunked_entries = {}
    for entry_type, entries in grouped_entries.items():
        chunked_entries[entry_type] = [
            entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)
        ]

    # Convert each chunk OF 5 ENTRIES into LangChain Documents
    for entry_type, chunks in chunked_entries.items ():
        for chunk in chunks:
            # Combine the text of all entries in the chunk
            combined_text = ""
            for entry in chunk:
                if entry_type == 'qa':
                    combined_text += f"Q: {entry['question']}\nA: {entry['answer']}\n"
                elif entry_type == 'policy':
                    combined_text += f"Policy: {entry['policy']}\n"
                elif entry_type == 'how-to':
                    combined_text  += f"how-to: {entry['how-to']}\n"
                elif entry_type == 'support':
                    combined_text  += f"support: {entry['support']}\n"
                else:
                    continue

        # Create a LangChain Document
            doc = Document(
                page_content=combined_text,
                metadata={"category": entry_type,
                          "tags": TAGS.get(entry_type, [])
                        }  # Include the original entry as metadata
            )
            langchain_documents.append(doc)

    return langchain_documents


# INITIALIZE THE DATABASE with PGVector data (our documents)
def init_db(documents_to_load: list[Document]) -> PGVector:
    """ Creates a Progress vector database with our documents
    PGVector initialization is disabled by default because it requires
    external services and can exceed automated test time limits.
    Set USE_PGVECTOR=1 to enable it.
    """
    if not documents_to_load:
        raise ValueError("No documents were provided for PGVector initialization.")

    dotenv.load_dotenv()
    if os.getenv("USE_PGVECTOR") != "1":
        print("PGVector initialization skipped. Set USE_PGVECTOR=1 to enable it.")
        return None

    tiny_api_key = os.getenv("TINY_API_KEY")
    if not tiny_api_key:
        raise RuntimeError ("TINY_API_KEY is not set.")
    tiny_base_url = os.getenv("TINY_BASE_URL") or "https://litellm.aks-hs-prod.int.hyperskill.org/openai/"

    EMBEDDING_MODEL = "text-embedding-ada-002"
    embeddings = OpenAIEmbeddings (
        model=EMBEDDING_MODEL,
        api_key=SecretStr(tiny_api_key),
        base_url=tiny_base_url,
        # request_timeout=10,
        max_retries=1,
    )
    print("*** chunk size, name: ", embeddings.chunk_size, embeddings.model)

    pgvector_connection_string = os.getenv("PGVECTOR_CONNECTION_STRING")
    if not pgvector_connection_string:
        raise ValueError("PGVECTOR_CONNECTION_STRING is not set in the environment variables.")
    COLLECTION_NAME = "my_documents"

    # Create vector store and index your existing documents in one shot
    try:
        print(f"*** Adding {len(documents_to_load)} documents to PGVector...")
        vector_store = PGVector.from_documents(
            documents=documents_to_load,  # your list of Document objects
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection= pgvector_connection_string,
            use_jsonb=True,    # Better for querying large documents
            pre_delete_collection=False,  # set True to reset on each run
        )
    except Exception as exc:
        raise RuntimeError (
            "Failed to initialize PGVector. "
            "Check the embedding API key/base URL, PostgreSQL connection string, "
            "network access, and whether the pgvector extension is available."
        ) from exc

    return vector_store

# RETRIEVAL: QUERY DECOMPOSITION
def query_decompose(vector_store: PGVector) -> None:
    dotenv.load_dotenv ()
    MODEL_USED = "gpt-4o-mini"

    # Free Tiny LLM to use free LLM client with OpenAI compatible syntax
    tiny_api_key = os.getenv("TINY_API_KEY")
    if not tiny_api_key:
        raise RuntimeError("TINY_API_KEY is not set.")
    tiny_base_url = "https://litellm.aks-hs-prod.int.hyperskill.org/openai/"

    llm = ChatOpenAI(
        api_key=tiny_api_key,
        base_url=tiny_base_url,
        model=MODEL_USED,
        temperature=0.0,
    )

    examples = [
        {
            "question": "How can I return a damaged product?",
            "subquestion1": "What is the process for returning a damaged product?",
            "subquestion2": "Who should I contact if I receive a damaged product?",
            "subquestion3": "Are there any requirements for returning a damaged product?",
        },
        {
            "question": "How do I track my order?",
            "subquestion1": "Where can I find my order tracking information?",
            "subquestion2": "How can I access the tracking number for my order?",
            "subquestion3": "What steps should I follow to track my order?",
        },
        {
            "question": "Can I get a refund for a digital product?",
            "subquestion1": "Are digital products eligible for refunds?",
            "subquestion2": "What is the refund policy for downloadable content?",
            "subquestion3": "Are there exceptions for refunding digital products?",
        },
    ]

    example_template = PromptTemplate.from_template(
        'Question: {question}\n'
        '  "subquestion1": "{subquestion1}",\n'
        '  "subquestion2": "{subquestion2}",\n'
        '  "subquestion3": "{subquestion3}"\n'
    )

    few_shot_prompt = FewShotPromptTemplate (
        examples=examples,
        example_prompt=example_template,
        prefix="""
    You are a customer support agent.

    A user asks a question and waits on a response. 
    For EVERY response, ALWAYS:
    - Make sure the question is split in 3 subquestions.
    - Every subquestion is similar to the original question.
    - Every subquestion is clear and concise.
    - Every subquestion is specific and focused.

    Output strictly JSON output without additional formatting or meta-text.
    """,
        suffix='Question: {question}\nJSON:\n',
        input_variables=["question"],
    )

    while True:
        try:
            question_input = input().strip()
        except EOFError:
            break

        if question_input.lower() in {"exit", "stop", "quit"}:
            break

        if not question_input:
            continue

        final_prompt = few_shot_prompt.format(question=question_input)
        response = llm.invoke(final_prompt)

        data = json.loads(response.content)

        questions_to_search = [*data.values()]

        for search_question in questions_to_search:
            print(f"Question: {search_question}")
            top_documents = vector_store.similarity_search(search_question, k=2)

            for document in top_documents:
                print(document.page_content)
            print(f"{document.metadata}")

    return None


def main() -> None:
    # file : str = 'knowledge_base_noisy_LIMITED.json'
    file : str = 'knowledge_base_noisy.json'
    json_to_clean : dict | list = load_data_to_clean(file)
    cleaned : dict | list = clean_data(json_to_clean)
    save_cleaned_data(cleaned)

    chunked_data = chunk_data(cleaned, 5)
    print(f"Generated {len(chunked_data)} document chunks.")

    import_orders_from_csv("sales_data.csv")
    vector_store_result = init_db(chunked_data)
    if vector_store_result is None:
        print("PGVector initialization was skipped.")
    else:
        print("Documents were successfully embedded and stored in PGVector.")

    query_decompose(vector_store_result)

if __name__ == "__main__":
    main()