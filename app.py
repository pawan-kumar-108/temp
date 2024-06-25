import chainlit as cl
# its better than gradio and streamlit for the purpose of convo ai
from datasets import load_dataset
from haystack.document_stores import InMemoryDocumentStore

#from haystack.components.retrievers import InMemoryBM25Retriever
#from haystack.component.builders import PromptBuilder

from haystack.nodes import PromptNode, PromptTemplate, AnswerParser, BM25Retriever
from haystack.pipelines import Pipeline
from haystack.utils import print_answers
import os
from dotenv import load_dotenv

load_dotenv()

from PyPDF2 import PdfReader
import docx
import os
#file_path = "\Ayurvedic Dataset"

def read_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text= ""
        for page_num in range(len(pdf_reader.pages)):
            text+= pdf_reader.pages[page_num].extract_text()
        return text
    

def read_word(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text+= paragraph.text + "/\n"
    return text

def read_txt(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

def read_directory(directory):
    combined_text = ""
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith(".pdf"):
            combined_text += read_pdf(file_path)
        elif file_name.endswith(".docx"):
            combined_text += read_word(file_path)
        elif file_name.endswith(".txt"):
            combined_text += read_txt(file_path)
    return combined_text


from datasets import load_dataset
#read_directory('C:\Users\hp\Downloads\')
#splitted_doc= read_word(r'C:\Users\hp\Downloads\MentalHack\Ayurvedic Dataset.docx')
#dataset = load_dataset(splitted_doc, split="train")

#document = read_word(r'C:\Users\hp\Downloads\MentalHack\Ayurvedic Dataset.docx')
# Corrected usage of read functions


#dataset_content = read_word(r'C:\Users\hp\Downloads\MentalHack\Ayurvedic Dataset.docx')
dataset_content = read_txt(r'C:\Users\hp\Downloads\MentalHack\Ayurvedic Dataset.txt')


# dictionary documents format for write_documents to be used later..
documents = [{"content": dataset_content, "meta": {"name": "Ayurvedic Dataset"}}]
document_store = InMemoryDocumentStore(use_bm25=True)

# Now documents are added to the document store
document_store.write_documents(documents)

#document_store = InMemoryDocumentStore(use_bm25 = True)
#loading the database i s done

#now will add the document to the document store
#document_store.write_documents(document)

#initializing the retriever
retriever = BM25Retriever(document_store= document_store, top_k= 3)




prompt_template= PromptTemplate(
    prompt= """
    Answer the provided question based solely on the provided document. If the document does not
    have any matched answer. Say them that "No answer found"
    Documents: {join(documents)}
    Question: {query}
    Answer:
    """ , 
    output_parser = AnswerParser()

)

HF_TOKEN = os.environ.get("HF_TOKEN")

prompt_node = PromptNode(
    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.3",
    api_key = HF_TOKEN ,
    default_prompt_template = prompt_template ,

)

#creating the pipeline

generative_pipeline = Pipeline()
generative_pipeline.add_node(component= retriever, name="retriever", inputs=["Query"])
generative_pipeline.add_node(component= prompt_node, name= "prompt_node", inputs=["retriever"])

#now to runthe above code on chainlit
#a decorator
'''
@cl.on_message
async def main(message:str):
    response = await cl.make_async(generative_pipeline.run)(message)
    sentences = response['answers'][0].answer.split('\n')


    #checking if the last sentences end with . ? or !
    if sentences and not sentences[-1].strip().endswith(('.', '?', '!')):

        # Remove the last sentence
        sentences.pop()

    result = '\n'.join(sentences[1:])
    await cl.Message(author="Bot", content=result).send()
    '''

# retrieved_docs = retriever.retrieve(query = "How to cure cold?", top_k = 3)
# print(retrieved_docs)
# from chainlit import Message
# test_query = ""
# prompt_response = prompt_node.run(query = test_query, documents=retrieved_document)
# print(prompt_response)

# @cl.on_message
# async def main(message: str):
#     # Log the type and content of the message
#     print(f"Received message of type {type(message)}: {message}")
    
#     if isinstance(message, Message):
#         message = message.content  # Extract the string content if it's a Message object
    
#     response = await cl.make_async(generative_pipeline.run)(message)
    
#     # Log the response
#     print(f"Response: {response}")
    
#     await cl.Message(content=response).send()


@cl.on_message
async def main(message: cl.Message):
    question = message.content  # Extract the content of the message
    prediction = generative_pipeline.run(query=question)
    answers = prediction.get('answers', [])

    if answers:
        formatted_response = "\n".join([answer.answer for answer in answers])
    else:
        formatted_response = "I'm sorry, I couldn't find an answer to your question."

    await cl.Message(
        content=formatted_response
    ).send()