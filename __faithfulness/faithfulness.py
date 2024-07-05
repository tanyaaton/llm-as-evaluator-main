import pandas as pd
from utils.config import settings
from function_faithfulness import store_divided_answer_found
from function import connect_watsonx_llm
import datetime

# llm divide model
d_model_id=             settings.faithfulness.llm_divide.name
d_decoding_method=      settings.faithfulness.llm_divide.decoding_method
d_min_new_tokens=       settings.faithfulness.llm_divide.min_new_tokens
d_max_new_tokens=       settings.faithfulness.llm_divide.max_new_tokens
d_repetition_penalty=   settings.faithfulness.llm_divide.repetition_penalty
d_mode=                 settings.faithfulness.llm_divide.prompt_language
d_model_source=         settings.faithfulness.llm_divide.source
if d_model_source == 'watsonxai':
    divide_model  = connect_watsonx_llm(d_model_id, 
                                 d_decoding_method, d_min_new_tokens, d_max_new_tokens, d_repetition_penalty)
elif d_model_source == 'openai':    pass
else:   raise ValueError(f"Invalid input: '{d_model_source}' model is not supported. Pleasechoose model from 'watsonxai' or 'openai' sources")

# llm evaluation model
e_model_id=             settings.faithfulness.llm_eval.name
e_decoding_method=      settings.faithfulness.llm_eval.decoding_method
e_min_new_tokens=       settings.faithfulness.llm_eval.min_new_tokens
e_max_new_tokens=       settings.faithfulness.llm_eval.max_new_tokens
e_repetition_penalty=   settings.faithfulness.llm_eval.repetition_penalty
e_mode=                 settings.faithfulness.llm_eval.prompt_language
e_model_source=         settings.faithfulness.llm_eval.source
if e_model_source == 'watsonxai':
    eval_model  = connect_watsonx_llm(e_model_id, 
                                 e_decoding_method, e_min_new_tokens, e_max_new_tokens, e_repetition_penalty)
elif d_model_source == 'openai':    pass
else:   raise ValueError(f"Invalid input: '{e_model_source}' model is not supported. Pleasechoose model from 'watsonx' or 'openai' sources")


data_df = pd.read_csv(f'csv_files/content.csv')
content_df = data_df.loc[:,["question","answer","contexts"]]

now = datetime.datetime.now()
formatted_datetime = now.strftime("%d-%m-%Y_%H%M")

new_df = store_divided_answer_found(content_df, divide_model=divide_model, d_mode=d_mode, eval_model=eval_model, e_mode=e_mode)
new_ff_df = new_df.loc[:,["question","answer","contexts","faithfuln"]]

new_df.to_csv(        f'csv_files/eval_faithfulness_detail_{formatted_datetime}.csv')
new_df.to_excel(f'csv_files/excel/eval_faithfulness_detail_{formatted_datetime}.xlsx')
new_ff_df.to_csv(        f'csv_files/eval_faithfulness_{formatted_datetime}.csv')
new_ff_df.to_excel(f'csv_files/excel/eval_faithfulness_{formatted_datetime}.xlsx')
