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
            
            

