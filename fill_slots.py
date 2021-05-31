from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import numpy as np
from enum import Enum, auto

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
softmax = torch.nn.Softmax(dim=1)

class Slots(Enum):
    CLIENT = auto()
    CONTACT_METHOD = auto()
    CONTACT_TIME = auto()
    PROJECT_NAME = auto()
    DUMMY = auto()

slot_questions = {
    Slots.CLIENT: "Who?",
    Slots.CONTACT_METHOD: "How was the client contacted?",
    Slots.CONTACT_TIME: "When?",
    Slots.PROJECT_NAME: "What was the project name?",
    Slots.DUMMY: "Is this the way To Amarillo?"  # Dummy question for understanding failure cases
}

print(slot_questions)

def fill_slots(contexts):
    res = []
    for context in contexts:
        res.append({
            slot: fill_slot(slot, context)
            for slot in slot_questions
        })
    return res
        
def fill_slot(slot, context):
    """Fill in a missing slot based on a question

    Keyword arguments:
    question -- Question used to define the slot
    context -- Text containing information to query

    Returns a tuple with the text answer and softmax of the start pointer
    """
    question = slot_questions[slot]
    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
    
    # The following is unnecessary for running in production, but useful for investigating the output
    # Including this line is a tradeoff between clean code and being able to debug the output
    start_softmax = np.squeeze(softmax(answer_start_scores).detach().numpy()).tolist()

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    
    return answer, start_softmax