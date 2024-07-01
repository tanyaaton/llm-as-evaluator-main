import pandas as pd
from utils.config import settings
from function_faithfulness import store_divided_answer_found
from function import connect_watsonx_llm
import datetime

model_id_llm_llama3_8b='meta-llama/llama-3-8b-instruct'
model_id_llm_llama3_70b='meta-llama/llama-3-70b-instruct'

model_llm_llama3_8b  = connect_watsonx_llm(model_id_llm_llama3_8b)
model_llm_llama3_70b = connect_watsonx_llm(model_id_llm_llama3_70b)

# choose model and mode
ans = '70b'         #'8b' '70b' 'gpt'
d_model = '8b'      #'8b' '70b' 'gpt'
d_mode = 'TH'       #'TH 'EN'
e_model = '70b'     #'8b' '70b' 'gpt'
e_mode = 'EN'       #'TH 'EN'

if d_model == '8b':     divide_model = model_llm_llama3_8b
elif d_model == '70b':  divide_model = model_llm_llama3_70b
elif d_mode == 'gpt':   divide_model = 'gpt'
else:                   raise ValueError(f"Invalid input: '{d_mode}' is not allowed.")

if e_model == '8b':     eval_model = model_llm_llama3_8b
elif e_model == '70b':  eval_model = model_llm_llama3_70b
elif e_model == 'gpt':  eval_model = 'gpt'
else:                   raise ValueError(f"Invalid input: '{e_mode}' is not allowed.")

data_df = pd.read_csv(f'csv_faithfulness_TH_{ans}/lm3_{ans}_answer.csv')
content_df = data_df.loc[:,["question","answer","contexts"]]

now = datetime.datetime.now()
formatted_datetime = now.strftime("%d-%m-%Y_%H%M")

new_df = store_divided_answer_found(content_df, divide_model=divide_model, d_mode=d_mode, eval_model=eval_model, e_mode=e_mode)

new_df.to_csv(        f'csv_faithfulness_TH_{ans}/test_D8b{d_mode}_E70b{e_mode}_{formatted_datetime}.csv')
new_df.to_excel(f'csv_faithfulness_TH_{ans}/excel/test_D8b{d_mode}_E70b{e_mode}_{formatted_datetime}.xlsx')
