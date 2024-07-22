from connection import read_json_file, read_text_file, recording_results_to_csv
from utils.config import settings
from connection import connect_watsonx_llm
from __customized_matrics.prompt import prompt_generation
import time
import requests
import pandas as pd


# llm evaluation model
e_model_id=             settings.customized_matrics.llm_eval.name
e_decoding_method=      settings.customized_matrics.llm_eval.decoding_method
e_min_new_tokens=       settings.customized_matrics.llm_eval.min_new_tokens
e_max_new_tokens=       settings.customized_matrics.llm_eval.max_new_tokens
e_repetition_penalty=   settings.customized_matrics.llm_eval.repetition_penalty
e_mode=                 settings.customized_matrics.llm_eval.prompt_language
e_model_source=         settings.customized_matrics.llm_eval.source
if e_model_source == 'watsonxai':
    eval_model  = connect_watsonx_llm(e_model_id, 
                                 e_decoding_method, e_min_new_tokens, e_max_new_tokens, e_repetition_penalty)
elif e_model_source == 'openai':    pass
else:   raise ValueError(f"Invalid input: '{e_model_source}' model is not supported. Pleasechoose model from 'watsonx' or 'openai' sources")

file_location = settings.customized_matrics.content_csv_location
file_name = settings.customized_matrics.content_csv_name
print(f'evaluating {file_location}{file_name} with {e_model_id}')

data_df = pd.read_csv(f'{file_location}{file_name}')
content_df = data_df.loc[:,["question","answer","contexts"]]

for i in content_df.index:
    answer = content_df.loc[i,'answer']
    predicted_question = prompt_generation(answer, predict_model, 'EN')
    new_df.loc[i,'predicted_question'] = predicted_question


start = time.time()
# for i in range(0,5 + 0):
#     row = df.iloc[i]
#     print(i/5 * 100, " %")
#     curr_responses = {
#             'qa_set': []
#         }
#     current_question = qa_data[i]['qa_set'][0]['question']
#     written_answer = qa_data[i]['qa_set'][0]['answer']
#     current_reference = text_thai[i]
#     answer_1 = row["answer1"]

#     res_1 = prompt_generation(current_reference, current_question, written_answer, answer_1)
#     res_response_quality_1 = eval_model.generate_text(res_1)
#     # To add model answer, written answer in.
#     current_question = qa_data[i]['qa_set'][1]['question']
#     written_answer = qa_data[i]['qa_set'][1]['answer']
#     current_reference = text_thai[i]
#     answer_2 = row["answer2"]

#     res_2 = prompt_generation(current_reference, current_question, written_answer, answer_2)
#     res_response_quality_2 = eval_model.generate_text(res_2)

#     curr_responses['qa_set'].append({"answer": res_response_quality_1})
#     curr_responses['qa_set'].append({"answer": res_response_quality_2})
#     responses.append(curr_responses)
#     print(res_response_quality_1)
#     print(res_response_quality_2)

# print(responses[0]['qa_set'][0]['answer'])
end = time.time()
total_time_taken = end - start
print(total_time_taken)

# recording_results_to_csv(qa_data, responses, 'mpt-30b', 'used mpt-30b, and no trans', "EN", total_time_taken, 'results/qa_answer_mpt-30b.csv')
recording_results_to_csv(qa_data, responses, name, '', "", total_time_taken, f'scoring-llama3/{name}.csv')