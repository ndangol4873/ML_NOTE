import json
import os
import hashlib

class Database:

    def insert(self,name, email, password):
        """Insert Register Credential"""
        if os.path.exists('users.json'):
            with open('users.json', 'r') as rf:
                users = json.load(rf)

            if email in users:
                return False
            else:
                hashed_password = hashlib.sha256(password.encode()).hexdigest()
                users[email] = [name, hashed_password]
            
            with open('users.json','w') as wf:
                json.dump(users, wf, indent=4)
                return True
        else:
            users = {}
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            users[email] = [name, hashed_password] 
            with open('users.json', 'w') as wf: 
                json.dump(users, wf, indent=4)
                return True
            

            
    def authenticate(self,email, password): 
        """Authenticate Login Credential"""
        if os.path.exists('users.json'): 
            with open('users.json', 'r') as rf: 
                users = json.load(rf) 
                if email in users: 
                    stored_name, stored_hashed_password = users[email] 
                    hashed_password = hashlib.sha256(password.encode()).hexdigest() 
                    if hashed_password == stored_hashed_password: 
                        return True 
                    return False 
                return False

        



    # def search(self,email,password):
    #     with open('users.json','r') as rf:
    #         users = json.load(rf)

    #         if email in users:
    #             if users[email][1] == password:
    #                 return 1
    #             else:
    #                 return 0
    #         else:
    #             return 0