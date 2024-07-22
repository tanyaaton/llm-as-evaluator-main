from __answer_relevancy.prompt import predict_question_instruction_TH, predict_question_instruction_EN
import numpy as np

def predict_question_from_answer_llm3_TH(answer, model_llm_llama3, mode):
    if mode == 'TH':
        predict_question_instruction = predict_question_instruction_TH
        # divided_answer_example_llama3 = predict_question_example_llama3_TH
    elif mode == 'EN':
        predict_question_instruction = predict_question_instruction_EN
        # print(predict_question_instruction)
        # divided_answer_example_llama3 = predict_question_example_llama3_EN
    else:
        raise ValueError(f"Invalid input: '{mode}' is not allowed.")
    
    template: str = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> {predict_question_instruction} <|eot_id|>

<|start_header_id|>user<|end_header_id|>
คำตอบ: {answer}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
คำถาม: """

    evaluate_response = model_llm_llama3.generate_text(template)
    
    return evaluate_response

def npsumdot(x, y):
    return np.sum(x*y, axis=1)
