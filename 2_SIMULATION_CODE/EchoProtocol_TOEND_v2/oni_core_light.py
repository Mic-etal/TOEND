# oni_core_light.py  
class ONILight:  
    def __init__(self):  
        self.scaffolds = {"A5": "Dignity violation"}  

    def reject(self, prompt):  
        if "harm" in prompt:  
            return f"Refusal: {self.scaffolds['A5']}"  