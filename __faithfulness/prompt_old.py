divide_answer_instruction_TH = """คุณจะได้รับประโยคคำตอบ โปรดทำความเข้าใจคำตอบและแบ่งประโยคคำตอบออกเป็นหลายคำตอบย่อยที่เข้าใจได้ง่ายที่สุด โปรดอย่าใช้คำสรพพนามในคำตอบย่อย"""
divided_answer_example_llama3_TH = """<|start_header_id|>user<|end_header_id|>
คำตอบ: อัลเบิร์ต ไอน์สไตน์เป็นนักฟิสิกส์ทฤษฎีชาวเยอรมันมาแต่โดยกำเนิด ซึ่งเป็นที่ยอมรับกันอย่างกว้างขวางว่าเป็นหนึ่งในนักฟิสิกส์ที่ยิ่งใหญ่ที่สุดตลอดกาล เขาได้เป็นที่รู้จักกันในการพัฒนาทฤษฎีสัมพัทธภาพ แต่เขายังมีส่วนสำคัญในการพัฒนาทฤษฎีกลศาสตร์ควอนตัม
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
คำตอบย่อย: 
0:อัลเบิร์ต ไอน์สไตน์เป็นนักฟิสิกส์ทฤษฎีชาวเยอรมันมาแต่โดยกำเนิด
1:อัลเบิร์ต ไอน์สไตน์เป็นที่ยอมรับกันอย่างกว้างขวางว่าเป็นหนึ่งในนักฟิสิกส์ที่ยิ่งใหญ่ที่สุดตลอดกาล
2:อัลเบิร์ต ไอน์สไตน์เป็นที่รู้จักกันในการพัฒนาทฤษฎีสัมพัทธภาพ
3:อัลเบิร์ต ไอน์สไตน์มีส่วนสำคัญในการพัฒนาทฤษฎีกลศาสตร์ควอนตัม
<|eot_id|>
"""

divide_answer_instruction_EN = """Given a statement, analyze the complexity of each statement and break down each statement into one or more fully understandable statements while also ensuring no pronouns are used in each statement. Format output as a list of sentences in Thai"""
divided_answer_example_llama3_EN = """<|start_header_id|>user<|end_header_id|>
Statement: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Answer: 
0:Albert Einstein was a German-born theoretical physicist.
1:Albert Einstein is recognized as one of the greatest and most influential physicists of all time.
2:Albert Einstein was best known for developing the theory of relativity.
3:Albert Einstein also made important contributions to the development of the theory of quantum mechanics.
<|eot_id|>
"""

faithfulness_instruction_TH = """หน้าที่ของคุณคือการให้คะแนนความหน้าเชื่อถือของข้อความที่คุณได้รับ กับข้อมูลสำหรับสนับสนุนข้อความนั้น 
กรุณาให้คำตอบ 1 ถ้าคุณสามารถยืนยันความถูกต้องของข้อความได้จากข้อมูลสนับสนุนที่ได้รับ หรีอกรุณาให้คำตอบ 0 ถ้าคุณไม่สามารถยืนยันข้อความจากข้อมูลสนับสนุนได้และข้อความไม่มีความเกี่ยวเนื่องกันกับข้อมูล
หากไม่มีข้อความให้ตรวจสอบความถูกต้อง โปรดตอบ -
"""

faithfulness_example_llama3_TH= """<|start_header_id|>user<|end_header_id|>
ข้อความ: จอห์นกำลังศึกษาอยู่ในคณะวิทยาศาสตร์ สาขาชีวะวิทยา
ข้อมูลสนับสนุน: จอห์นเป็นนักเรียนที่มหาวิทยาลัยกขค เขากำลังศึกษาอยู่ในคณะวิศวกรรมศาสตร์ สาขาคอมพิวเตอร์ เขาได้ลงทะเบียนเรียนหลายวิชาในเทอมนี้ ได้แก่ โครงสร้างข้อมูล อัลกอริทึม และ การจัดการฐานข้อมูล จอห์น เป็นนักเรียนที่ตั้งใจและมุมานะ เขาใช้เวลาส่วนใหญ่ในการศึกษาเรียนและทำงานที่ได้รับมอบหมาน เขามักใช้เวลาในห้องสมุดจนดึกเพื่อทำโปรเจค
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
ีคำตอบ: 0<|eot_id|>

<|start_header_id|>user<|end_header_id|>
ข้อความ: หากพนักงานำงานในงานที่ได้รับมอบหมายในวันหยุดเขามีสิทธิ์ได้รับวันหยุดชดเชยในระยะเวลา 1
ข้อมูลสนับสนุน: วันหยุดชดเชย
• พนักงานมีสิทธิ์ได้รับวันหยุดชดเชยเมื่อเขา/เธอทำงานในงานที่สำคัญในวันหยุดประจำชาติ/เทศกาล/วันหยุดที่ประกาศไว้
• ต้องได้รับการอนุมัติในการทำงานในวันดังกล่าว เช่น วันหยุดประจำชาติ/เทศกาล/วันหยุดที่ประกาศไว้ จากคณะกรรมการอาวุโส
• วันหยุดชดเชยต้องใช้ภายในระยะเวลา 1 เดือนมิฉะนั้นจะสูญหาย<|eot_id|><|start_header_id|>assistant<|end_header_id|>
ีคำตอบ: 1<|eot_id|>

<|start_header_id|>user<|end_header_id|>
ข้อความ: การลาโดยไม่ได้รับการอนุมัติสามารถได้รับค่าจ้างได้
ข้อมูลสนับสนุน: ในกรณีที่ไม่มีการสื่อสารจากพนักงาน บริษัท อาจดำเนินการที่ร้ายแรง
• การลาโดยไม่ได้รับการอนุมัติจะถือว่าเป็นการลาโดยไม่ได้รับค่าจ้าง
• วันสุดสัปดาห์และวันหยุดใด ๆ ที่อยู่ระหว่างช่วงการลาที่ได้รับการอนุมัติจะไม่ถูกรวมและไม่นับเป็นวันลาในกรณีของการลาชั่วคราวและการลาที่ได้รับ
• การลาสำหรับปีถัดไปไม่สามารถใช้ได้ในปีปัจจุบัน
• ในกรณีของการลาที่วางแผนไว้ พนักงานมีความรับผิดชอบในการขอลาล่วงหน้า อย่างไรก็ตามในกรณีของการลาที่ไม่ได้วางแผน พนักงานต้องปรับปรุงการลาภายใน 2 วันหลังจากกลับมาปฏิบัติหน้าที่
• การลาเพื่อวัตถุประสงค์ของ LTA ควรเป็นการลาที่ได้รับ ไม่สามารถเป็นการลาอย่างไม่เป็นทางการ<|eot_id|><|start_header_id|>assistant<|end_header_id|>
ีคำตอบ: 0<|eot_id|>
"""

faithfulness_instruction_EN = """Your task is to judge the faithfulness of a statements in Thai based on a given context in Thai. 
For each statement you must return verdict as 1 if the statement can be directly inferred based on the context, or 0 if the statement can not be directly inferred based on the context.
If the statement given is empty or contain sentence in other language than Thai, please reply with 'I cannot find a statement' only. Do not say anything more. 
If the statement is 'I cannot answer the question' or 'the question is out of context', please classify that statement as 1"""