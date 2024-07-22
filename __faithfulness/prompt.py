def faithfulness_divide_answer_prompt_TH(answer):
    messages = {
        "messages": [
            {   "content": "คุณจะได้รับประโยคคำตอบ โปรดทำความเข้าใจคำตอบและแบ่งประโยคคำตอบออกเป็นหลายคำตอบย่อยที่เข้าใจได้ง่ายที่สุด โปรดอย่าใช้คำสรพพนามในคำตอบย่อย",
                "role": "system"
            },
            {
            "content": """คำตอบ: อัลเบิร์ต ไอน์สไตน์เป็นนักฟิสิกส์ทฤษฎีชาวเยอรมันมาแต่โดยกำเนิด ซึ่งเป็นที่ยอมรับกันอย่างกว้างขวางว่าเป็นหนึ่งในนักฟิสิกส์ที่ยิ่งใหญ่ที่สุดตลอดกาล เขาได้เป็นที่รู้จักกันในการพัฒนาทฤษฎีสัมพัทธภาพ แต่เขายังมีส่วนสำคัญในการพัฒนาทฤษฎีกลศาสตร์ควอนตัม""",
            "role": "user"
            },
            {
            "content": """คำตอบย่อย: 
0:อัลเบิร์ต ไอน์สไตน์เป็นนักฟิสิกส์ทฤษฎีชาวเยอรมันมาแต่โดยกำเนิด
1:อัลเบิร์ต ไอน์สไตน์เป็นที่ยอมรับกันอย่างกว้างขวางว่าเป็นหนึ่งในนักฟิสิกส์ที่ยิ่งใหญ่ที่สุดตลอดกาล
2:อัลเบิร์ต ไอน์สไตน์เป็นที่รู้จักกันในการพัฒนาทฤษฎีสัมพัทธภาพ
3:อัลเบิร์ต ไอน์สไตน์มีส่วนสำคัญในการพัฒนาทฤษฎีกลศาสตร์ควอนตัม""",
            "role": "assistant"
            },
            {
            "content": f"คำตอบ: {answer}",
            "role": "user"
            },
            {
            "content": "คำตอบย่อย: ",
            "role": "assistant"
            },
        ]
    }
    return messages["messages"]

def faithfulness_divide_answer_prompt_EN(answer):
    messages = {
        "messages": [
            {
            "content": "Given a Thai statement, analyze the complexity of each statement and break down each statement into one to nine fully understandable sub-statements in Thai language, while also ensuring no pronouns are used in each sub-statement. Format output as a list of sentences in Thai language. Do not provide any further information",
            "role": "system"
            },
            {
            "content": f"Statement: {answer}",
            "role": "user"
            },
            {
            "content": "Sub-statements: ",
            "role": "assistant"
            },
        ]
    }
    return messages["messages"]

def faithfulness_evaluation_prompt_EN(context, answer):
    messages = {
        "messages": [
            {
            "content": """Your task is to judge the faithfulness of a statements in Thai based on a given context in Thai. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context, or 0 if the statement can not be directly inferred based on the context.""",
            "role": "system"
            },
            {
            "content": f"""Statement: {answer}
Context: {context}""",
            "role": "user"
            },
            {
            "content": "Answer: ",
            "role": "assistant"
            },
        ]
    }
    return messages["messages"]