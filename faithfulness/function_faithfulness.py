from openai import OpenAI
from prompt import ( divide_answer_instruction_TH, divided_answer_example_llama3_TH, faithfulness_instruction_TH,
                     divide_answer_instruction_EN, divided_answer_example_llama3_EN, faithfulness_instruction_EN )
import pandas as pd
import logging
import datetime

now = datetime.datetime.now()
formatted_datetime = now.strftime("%d-%m-%Y_%H%M")
logging.basicConfig(filename=f'log/faithfulness_{formatted_datetime}.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# divide answer
def divide_answer_llm3_TH(answer, model_llm_llama3, mode):
    if mode == 'TH':
        print('divide - TH mode')
        divide_answer_instruction = divide_answer_instruction_TH
        divided_answer_example_llama3 = divided_answer_example_llama3_TH
    elif mode == 'EN':
        print('divide - EN mode')
        divide_answer_instruction = divide_answer_instruction_EN
        divided_answer_example_llama3 = divided_answer_example_llama3_TH
    else:
        raise ValueError(f"Invalid input: '{mode}' is not allowed.")
    
    template: str = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> {divide_answer_instruction} <|eot_id|>

{divided_answer_example_llama3}

<|start_header_id|>user<|end_header_id|>
คำตอบ: {answer}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
คำตอบย่อย: """

    evaluate_response = model_llm_llama3.generate_text(template)
    
    return evaluate_response

def divide_answer_openai_TH(rag_response):
    divide_answer_instruction_TH = """คุณจะได้รับประโยคคำตอบ โปรดทำความเข้าใจคำตอบและแบ่งประโยคคำตอบออกเป็นหลายคำตอบย่อยที่เข้าใจได้ง่ายที่สุด โปรดอย่าใช้คำสรพพนามในคำตอบย่อย"""
    
    client = OpenAI()
    response = client.chat.completions.create(
        model= 'gpt-4o',
        messages=[
        {"role": "system", "content": f"{divide_answer_instruction_TH}"},
        
        {'role': 'user', 
        'content': """
        คำตอบ: อัลเบิร์ต ไอน์สไตน์เป็นนักฟิสิกส์ทฤษฎีชาวเยอรมันมาแต่โดยกำเนิด ซึ่งเป็นที่ยอมรับกันอย่างกว้างขวางว่าเป็นหนึ่งในนักฟิสิกส์ที่ยิ่งใหญ่ที่สุดตลอดกาล เขาได้เป็นที่รู้จักกันในการพัฒนาทฤษฎีสัมพัทธภาพ แต่เขายังมีส่วนสำคัญในการพัฒนาทฤษฎีกลศาสตร์ควอนตัม
        """},
        {'role': 'assistant', 
        'content': """
        คำตอบย่อย: 
        0:อัลเบิร์ต ไอน์สไตน์เป็นนักฟิสิกส์ทฤษฎีชาวเยอรมันมาแต่โดยกำเนิด
        1:อัลเบิร์ต ไอน์สไตน์เป็นที่ยอมรับกันอย่างกว้างขวางว่าเป็นหนึ่งในนักฟิสิกส์ที่ยิ่งใหญ่ที่สุดตลอดกาล
        2:อัลเบิร์ต ไอน์สไตน์เป็นที่รู้จักกันในการพัฒนาทฤษฎีสัมพัทธภาพ
        3:อัลเบิร์ต ไอน์สไตน์มีส่วนสำคัญในการพัฒนาทฤษฎีกลศาสตร์ควอนตัม
            """},

        {'role': 'user', 
        'content': f"""\
        คำตอบย่อย: {rag_response}
        """}
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content


# evaluate faithfulness
def get_faithfulness_scores_llm3_TH(context, answer, model_llm_llama3, mode):
    if mode == 'TH':    
        print('eval - TH mode')        
        faithfulness_instruction = faithfulness_instruction_TH
    elif mode == 'EN':  
        print('eval - EN mode')
        faithfulness_instruction = faithfulness_instruction_EN
    else:               
        raise ValueError(f"Invalid input: '{mode}' is not allowed.")

    template: str = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> {faithfulness_instruction} <|eot_id|>
<|start_header_id|>user<|end_header_id|>
Statement:{answer}
Contexts: {context}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Answer: """
    logging.info(template)
    evaluate_response = model_llm_llama3.generate_text(template)
    logging.info(evaluate_response)
    return evaluate_response


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
        if divide_model == 'gpt':
            divided_answers = divide_answer_openai_TH(df.loc[i,'answer'])
        else:
            divided_answers = divide_answer_llm3_TH(answer=df.loc[i,'answer'], model_llm_llama3=divide_model,mode=d_mode)        
        print(divided_answers)
        logging.info(divided_answers)
        divided_answers_list = divided_answers.strip().split('\n')
        new_df.loc[ii,'question_no']=i
        new_df.loc[ii,'answer']=df.loc[i,'answer']
        new_df.loc[ii,'question']=df.loc[i,'question']
        new_df.loc[ii,'contexts']=df.loc[i,'contexts']
        ii_freeze = ii
        for divided_answer in divided_answers_list:
            new_df.loc[ii,'divided_answer']=divided_answer
            if eval_model == 'gpt':
                found_in_doc_combine = get_faithfulness_scores_openai_TH(df.loc[i,'contexts'], divided_answer, mode=e_mode)
            else:
                found_in_doc_combine = get_faithfulness_scores_llm3_TH(context=df.loc[i,'contexts'], answer=divided_answer, model_llm_llama3=eval_model, mode=e_mode)
            new_df.loc[ii,'llama3_label']=found_in_doc_combine
            ii+=1
            if ('1' in found_in_doc_combine[0]):
                yes_count += 1
            elif ('0' in found_in_doc_combine[0]):
                no_count  += 1
        new_df.loc[ii_freeze,"faithfuln"]=yes_count/(yes_count+no_count)
        new_df.loc[ii_freeze,"yes_count"]=yes_count
        new_df.loc[ii_freeze,"no_count"] =no_count
        print(i, ';',yes_count, no_count )
    return new_df