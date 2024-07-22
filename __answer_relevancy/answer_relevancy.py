import pandas as pd
import numpy as np
from utils.config import settings
from __answer_relevancy.function_answer_relevancy import predict_question_from_answer_llm3_TH, npsumdot
from connection import ( connect_watsonx_embedding, connect_sentencetransformer_embedding, connect_watsonx_llm)
import datetime

# IBM embedding (en)
model_embedder_source = settings.answer_relevancy.embedder_model.source
model_id_embedder =     settings.answer_relevancy.embedder_model.name

if model_embedder_source == 'watsonxai':
    model_embedder  =     model_embedder = connect_watsonx_embedding(model_id_embedder)
elif model_embedder_source == 'huggingface':
    model_embedder  =     model_embedder = connect_sentencetransformer_embedding(model_id_embedder)
else:   raise ValueError(f"Invalid input: '{model_embedder_source}' model is not supported. Pleasechoose model from 'watsonxai' or 'openai' sources")


# choose llm IBM model for predict question model
p_model_id=             settings.answer_relevancy.llm_predict.name
p_decoding_method=      settings.answer_relevancy.llm_predict.decoding_method
p_min_new_tokens=       settings.answer_relevancy.llm_predict.min_new_tokens
p_max_new_tokens=       settings.answer_relevancy.llm_predict.max_new_tokens
p_repetition_penalty=   settings.answer_relevancy.llm_predict.repetition_penalty
# p_mode=                 settings.answer_relevancy.llm_predict.prompt_language
p_model_source=         settings.answer_relevancy.llm_predict.source

if p_model_source == 'watsonxai':
    predict_model  = connect_watsonx_llm(p_model_id, 
                                 p_decoding_method, p_min_new_tokens, p_max_new_tokens, p_repetition_penalty)
else:   raise ValueError(f"Invalid input: '{p_model_source}' model is not supported. Please choose model from 'watsonxai' sources")


file_location = settings.answer_relevancy.content_csv_location
file_name = settings.answer_relevancy.content_csv_name
print(f'evaluating {file_location}{file_name} with {p_model_id}')

content_df = pd.read_csv(f'{file_location}{file_name}')
new_df = content_df.loc[:,["question","answer","contexts"]]

for i in content_df.index:
    answer = content_df.loc[i,'answer']
    predicted_question = predict_question_from_answer_llm3_TH(answer, predict_model, 'EN')
    new_df.loc[i,'predicted_question'] = predicted_question

original_q  = new_df.question.values.tolist()
predicted_q = new_df.predicted_question.values.tolist()

if model_embedder_source == 'watsonxai':
    vector_orig = np.array(model_embedder.embed_documents(original_q))
    vector_pred = np.array(model_embedder.embed_documents(predicted_q))
if model_embedder_source == 'huggingface':
    vector_orig = np.array(model_embedder.encode(original_q))
    vector_pred = np.array(model_embedder.encode(predicted_q))

norm = np.linalg.norm(vector_orig, axis=1) * np.linalg.norm(vector_pred, axis=1)
dot_product = npsumdot(vector_orig, vector_pred)


answer_relevancy_array = dot_product / norm
new_df['answer_relevancy']=answer_relevancy_array
content_df['answer_relevancy']=answer_relevancy_array

print('answer_relevancy = ', round(np.average(answer_relevancy_array),5))

now = datetime.datetime.now()
formatted_datetime = now.strftime("%d-%m-%Y_%H%M")

new_df.to_csv(f'{file_location}eval_ansrelevancy_{formatted_datetime}.csv')
new_df.to_excel(f'{file_location}excel/eval_ansrelevancy_{formatted_datetime}.xlsx')
content_df.to_csv(f'{file_location}eval_ansrelevancy_detial_{formatted_datetime}.csv')
content_df.to_excel(f'{file_location}excel/eval_ansrelevancy_detial_{formatted_datetime}.xlsx')