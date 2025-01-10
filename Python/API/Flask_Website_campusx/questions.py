import json


class QuestionAnswer:

    def cat_name_var (self,category):
        cat_name_var = {
                        'machine_learning':'ml_interview_questions_and_answers',
                        'deep_learning':'dl_interview_questions_and_answers',
                        'python':'python_question',
                        'mlops': 'mlops'
                        }
        return cat_name_var[category]
    

    def get_question_answer(self,category):
        with open ('questions.json','r') as rf:
            response = json.load(rf)
            category_key = self.cat_name_var(category)
            return response[category_key]
            





# from constant import question
# import os 
# import json 

# category_var = {}
# for key, value in question.items():
#     category_var[key] = {}
#     for k , v in value.items():
#         for kk, vv in v.items():
#             key_name = str(k)+'_'+kk
#             category_var[key][key_name] = vv
# category_var

# file = 'questions.json'
# if os.path.exists(file):
#     os.remove(file)

# with open(file,'w') as wf:
#     json.dump(category_var, wf, indent=4)

