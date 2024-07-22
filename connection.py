from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models import Model
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from pymilvus import connections

load_dotenv()

api_key = os.getenv("WATSONX_APIKEY", None)
project_id = os.environ["PROJECT_ID"]
ibm_cloud_url = os.environ["IBM_CLOUD_URL"]
api_key = os.environ["WATSONX_APIKEY"]
openai_key = os.environ["OPENAI_API_KEY"]

if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }

def connect_watsonx_embedding(model_id_embedding):
    embed_params = {
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
        }
    print('connecting to watsonxembedding...')
    watsonx_embedding = WatsonxEmbeddings(
            model_id=model_id_embedding,
            url=ibm_cloud_url,
            project_id=project_id,
            params=embed_params,
        )
    return watsonx_embedding

def connect_sentencetransformer_embedding(model_id_embedding):
    print('connecting to sentencetranformerembedding...')
    sentencetransformer_embedding = SentenceTransformer(f'{model_id_embedding}')
    return sentencetransformer_embedding

def connect_watsonx_llm(model_id_llm, decoding_method, min_new_tokens, max_new_tokens, repetition_penalty):
    print('connecting to watsonxllm...')
    params = {
        'decoding_method': decoding_method,
        'min_new_tokens': min_new_tokens,
        'max_new_tokens': max_new_tokens,
        'temperature': 0.0,
        'repetition_penalty': repetition_penalty
    }
    model_llm = Model(model_id= model_id_llm,
                    params=params, credentials=creds,
                    project_id=project_id)
    
    return model_llm

def connect_to_milvus():
    print('connecting to milvus...')
    connections.connect(host= 'localhost', port='19530')


# def create_milvus_db(collection_name):
#     item_id    = FieldSchema( name="id",         dtype=DataType.INT64,    is_primary=True, auto_id=True )
#     text       = FieldSchema( name="text",       dtype=DataType.VARCHAR,  max_length= 50000             )
#     embeddings = FieldSchema( name="embeddings", dtype=DataType.FLOAT_VECTOR,    dim=768                )
#     schema     = CollectionSchema( fields=[item_id, text, embeddings], description="Inserted policy from user", enable_dynamic_field=True )
#     collection = Collection( name=collection_name, schema=schema, using='default' )
#     return collection

# def drop_milvus_collection():
#     utility.drop_collection('leavepdf_TH')
#     print('dropped milvus collection')

# def split_text_with_overlap(text, chunk_size, overlap_size):
#     chunks = []
#     start_index = 0

#     while start_index < len(text):
#         end_index = start_index + chunk_size
#         chunk = text[start_index:end_index]
#         chunks.append(chunk)
#         start_index += (chunk_size - overlap_size)

#     return chunks

# def embedding_data(chunks, model_embedding):
#     collection_leavepdf = create_milvus_db(col_name)
#     if str(type(model_embedding)) == "<class 'sentence_transformers.SentenceTransformer.SentenceTransformer'>":
#         vector  = model_embedding.encode(chunks)
#         collection_leavepdf.insert([chunks,vector.tolist()])
#     elif str(type(model_embedding)) == "<class 'langchain_ibm.embeddings.WatsonxEmbeddings'>":
#         vector  = model_embedding.embed_documents(chunks)
#         collection_leavepdf.insert([chunks,vector])
#     else:
#         raise ValueError(f"Invalid model type: embedding_model is not WatsonxEmbeddings or SentenceTransformer model.")
#     collection_leavepdf.create_index(field_name="embeddings",\
#                                 index_params={"metric_type":"IP","index_type":"IVF_FLAT","params":{"nlist":16384}})
#     return collection_leavepdf

# def find_answer_doc_from_q_df(question_df, collection, model_embedding):
#     question_list = question_df['question'].tolist()    # embedding question
#     if str(type(model_embedding)) == "<class 'sentence_transformers.SentenceTransformer.SentenceTransformer'>":
#         embedded_question_vector  = model_embedding.encode(question_list)
#     elif str(type(model_embedding)) == "<class 'langchain_ibm.embeddings.WatsonxEmbeddings'>":
#         embedded_question_vector  = model_embedding.embed_documents(question_list)
#     else:
#         raise ValueError(f"Invalid model type: embedding_model is not WatsonxEmbeddings or SentenceTransformer model.")
#     collection.load()           # query data from collection
#     hits = collection.search(data=embedded_question_vector, anns_field="embeddings", param={"metric":"IP","offset":0},
#                     output_fields=["text"], limit=5)
#     return hits

# def generate_doc(question_df, hits):
#     for i in question_df.index:
#         doc_combine = ''
#         for ii in range(len(hits[0])):
#             question_df.loc[i,f"doc_{ii}"]  = hits[i][ii].text
            
#             doc_combine += hits[i][ii].text + ', '
#         question_df.loc[i,'contexts'] = doc_combine


# def find_response(model_llm, content_df):
#     for i in content_df.index:
#         prompt = generate_prompt_en(content_df["question"][i], content_df["contexts"][i])
#         logging.info(prompt)
#         answer = model_llm.generate_text(prompt)
#         content_df.loc[i,"answer"]= answer