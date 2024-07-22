
import pandas as pd
from __customized_matrics.prompt import prompt_generation

# def store_divided_answer_found(df, eval_model, e_mode):
#     for i in df.index:
#         print('****************')
#         print('question no:', i)
#         found_in_doc_combine = get_faithfulness_scores_llm3_TH(context=df.loc[i,'contexts'], answer=divided_answer, model_llm_llama3=eval_model, mode=e_mode)
#         new_df.loc[ii,'llama3_label']=found_in_doc_combine
#         ii+=1
#         if ('1' in found_in_doc_combine[0]):
#             yes_count += 1
#         elif ('0' in found_in_doc_combine[0]):
#             no_count  += 1
#         new_df.loc[ii_freeze,"faithfuln"]=yes_count/(yes_count+no_count)
#         new_df.loc[ii_freeze,"yes_count"]=yes_count
#         new_df.loc[ii_freeze,"no_count"] =no_count
#         print(i, ';',yes_count, no_count )
#     return new_df