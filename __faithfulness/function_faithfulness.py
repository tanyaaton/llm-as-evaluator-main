from openai import OpenAI
from __faithfulness.prompt_old import ( divide_answer_instruction_TH, divided_answer_example_llama3_TH, faithfulness_instruction_TH,
                     divide_answer_instruction_EN, divided_answer_example_llama3_EN, faithfulness_instruction_EN )
import pandas as pd
import logging
import datetime
from __faithfulness.prompt import ( faithfulness_divide_answer_prompt_TH, faithfulness_divide_answer_prompt_EN, 
                                   faithfulness_evaluation_prompt_EN )
from prompt_template import get_prompt_template


now = datetime.datetime.now()
formatted_datetime = now.strftime("%d-%m-%Y_%H%M")
logging.basicConfig(filename=f'log/faithfulness_{formatted_datetime}.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# divide answer
# def divide_answer_llm3_TH(answer, model_llm_llama3, mode):
#     if mode == 'TH':
#         print('divide - TH mode')
#         template: str = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> {divide_answer_instruction_TH} <|eot_id|>

# {divided_answer_example_llama3_TH}

# <|start_header_id|>user<|end_header_id|>
# คำตอบ: {answer}
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# คำตอบย่อย: """
#         template = f"""คุณจะได้รับประโยคคำตอบ โปรดทำความเข้าใจคำตอบและแบ่งประโยคคำตอบออกเป็นหลายคำตอบย่อยที่เข้าใจได้ง่ายที่สุด โปรดอย่าใช้คำสรพพนามในคำตอบย่อย

# Input:คำตอบ: อัลเบิร์ต ไอน์สไตน์เป็นนักฟิสิกส์ทฤษฎีชาวเยอรมันมาแต่โดยกำเนิด ซึ่งเป็นที่ยอมรับกันอย่างกว้างขวางว่าเป็นหนึ่งในนักฟิสิกส์ที่ยิ่งใหญ่ที่สุดตลอดกาล เขาได้เป็นที่รู้จักกันในการพัฒนาทฤษฎีสัมพัทธภาพ แต่เขายังมีส่วนสำคัญในการพัฒนาทฤษฎีกลศาสตร์ควอนตัม
# Output:0:อัลเบิร์ต ไอน์สไตน์เป็นนักฟิสิกส์ทฤษฎีชาวเยอรมันมาแต่โดยกำเนิด
# 1:อัลเบิร์ต ไอน์สไตน์เป็นที่ยอมรับกันอย่างกว้างขวางว่าเป็นหนึ่งในนักฟิสิกส์ที่ยิ่งใหญ่ที่สุดตลอดกาล
# 2:อัลเบิร์ต ไอน์สไตน์เป็นที่รู้จักกันในการพัฒนาทฤษฎีสัมพัทธภาพ
# 3:อัลเบิร์ต ไอน์สไตน์มีส่วนสำคัญในการพัฒนาทฤษฎีกลศาสตร์ควอนตัม

# Input:{answer}
# Output:"""
        
#     elif mode == 'EN':
#         print('divide - EN mode')
#         template: str = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> {divide_answer_instruction_EN} <|eot_id|>

# {divided_answer_example_llama3_EN}

# <|start_header_id|>user<|end_header_id|>
# Statement: {answer}
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# Answer: """
#     else:
#         raise ValueError(f"Invalid input: '{mode}' is not allowed.")

#     evaluate_response = model_llm_llama3.generate_text(template)
    
#     return evaluate_response


# evaluate faithfulness
# def get_faithfulness_scores_llm3_TH(context, answer, model_llm_llama3, mode):
#     if mode == 'TH':    
#         print('eval - TH mode')        
#         template: str = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> {faithfulness_instruction_TH} <|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# ข้อความ:{answer}
# ข้อมูลสนับสนุน: {context}
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# คำตอบ: """
        
#     elif mode == 'EN':  
#         print('eval - EN mode')
#         template: str = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> {faithfulness_instruction_EN} <|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# Statement:{answer}
# Contexts: {context}
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# Answer: """
#     else:               
#         raise ValueError(f"Invalid input: '{mode}' is not allowed.")

#     logging.info(template)
#     evaluate_response = model_llm_llama3.generate_text(template)
#     logging.info(evaluate_response)
#     return evaluate_response


def get_faithfulness_scores_openai_TH(context, answer, mode):
    if mode == 'TH':    
        print('gpt eval - TH mode')        
        faithfulness_instruction = faithfulness_instruction_TH
    elif mode == 'EN':  
        print('gpt eval - EN mode')
        faithfulness_instruction = faithfulness_instruction_EN
    else:               
        raise ValueError(f"Invalid input: '{mode}' is not allowed.")

    client = OpenAI()
    response = client.chat.completions.create(
        model= 'gpt-4o',
        messages=[
        {"role": "system", "content": f"{faithfulness_instruction}"},
        {'role': 'user', 
        'content': f"""
        Statement: {answer}
        Contexts: {context}
        """}
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content


def store_divided_answer_found(df, divide_model, d_mode, eval_model, e_mode):
    ii=0
    new_df = pd.DataFrame()
    for i in df.index:
        yes_count = 0
        no_count = 0
        print('****************')
        print('question no:', i)
        message_d = faithfulness_divide_answer_prompt_TH(df.loc[i,'answer'])
        prompt_d = get_prompt_template(divide_model, message_d)
        logging.info(prompt_d)
        print('prompt-D', prompt_d)
        response_d = divide_model.generate_text(prompt_d)
        print('--response--',response_d)
        logging.info(response_d)
        divided_answers_list = response_d.strip().split('\n')
        new_df.loc[ii,'question_no']=i
        new_df.loc[ii,'answer']=df.loc[i,'answer']
        new_df.loc[ii,'question']=df.loc[i,'question']
        new_df.loc[ii,'contexts']=df.loc[i,'contexts']
        ii_freeze = ii
        for divided_answer in divided_answers_list:
            new_df.loc[ii,'divided_answer']=divided_answer
            message_e = faithfulness_evaluation_prompt_EN(context=df.loc[i,'contexts'], answer=divided_answer)
            prompt_e = get_prompt_template(eval_model, message_e)
            logging.info(prompt_e)
            response_e = eval_model.generate_text(prompt_e)
            logging.info(response_e)
            new_df.loc[ii,'llama3_label']=response_e
            ii+=1
            print(response_e[0])
            if ('0' in response_e[0]):
                no_count += 1
            elif ('1' in response_e[0]):
                yes_count  += 1
        new_df.loc[ii_freeze,"faithfuln"]=yes_count/(yes_count+no_count)
        new_df.loc[ii_freeze,"yes_count"]=yes_count
        new_df.loc[ii_freeze,"no_count"] =no_count
        print(i, ';',yes_count, no_count )
    return new_df