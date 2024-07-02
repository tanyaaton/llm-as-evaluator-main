import os
from function import ( connect_watsonx_embedding, connect_sentencetransformer_embedding,connect_watsonx_llm, connect_to_milvus, 
                      embedding_data ,find_answer_doc_from_q_df, generate_doc, find_response, drop_milvus_collection)
import pandas as pd
from utils.config import settings

# embedding model
model_embedder_source = settings.generate_answer.embedder_model.source
model_id_embedder =     settings.generate_answer.embedder_model.name

if model_embedder_source == 'watsonxai':
    model_embedder = connect_watsonx_embedding(model_id_embedder)
elif model_embedder_source == 'huggingface':
    model_embedder = connect_sentencetransformer_embedding(model_id_embedder)

chunk_size=         settings.generate_answer.embedder_model.chunk_size
overlap_size=       settings.generate_answer.embedder_model.overlap_size

# large language model
model_id_llm=       settings.generate_answer.llm_generate.name
decoding_method=    settings.generate_answer.llm_generate.decoding_method
min_new_tokens=     settings.generate_answer.llm_generate.min_new_tokens
max_new_tokens=     settings.generate_answer.llm_generate.max_new_tokens
repetition_penalty= settings.generate_answer.llm_generate.repetition_penalty

model_llm  = connect_watsonx_llm(model_id_llm, 
                                 decoding_method, min_new_tokens, max_new_tokens, repetition_penalty)

# make sure to run `milvus-server --proxy-port 19530` command in terminal to connect to milvus lite
connect_to_milvus()

# create question dataframe 
question_df = pd.read_csv('csv_files/question.csv')

# embed data
thai_text = open("text/leave_policy_TH.txt", encoding="utf8").read()
collection = embedding_data(thai_text, model_embedder, chunk_size, overlap_size)

# get hits and save doc file 
hits = find_answer_doc_from_q_df(question_df, collection, model_embedder)
generate_doc(question_df, hits)

content_df = question_df.loc[:,['question','contexts']]
find_response(model_llm, content_df)

print('exporting csv file...')
question_df.to_csv('csv_files/content.csv')
content_df.to_csv('csv_files/content.csv')

drop_milvus_collection()