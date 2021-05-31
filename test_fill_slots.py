import matplotlib.pyplot as plt
from fill_slots import fill_slots, Slots
import argparse

# Test harness for slot filling function

parser = argparse.ArgumentParser(description='Test slot filling function')
parser.add_argument('--plot', action='store_true',
                    help='Show plots of start token softmax')
args = parser.parse_args()

contexts = [
    "21 Jan: Phoned John S about signing them up for phrase 2 of Project Alpha.",
    "21 Jan: Emailed John S about signing them up for phrase 2 of Project Alpha.",
    "Phoned John S yesterday about signing them up for phrase 2 of Project Alpha.",
    "21 Jan: Talked with John S about the next phase of Project Alpha over the phone."
]
 
for context, filled in zip(contexts, fill_slots(contexts)):
    print(f"Context = {context}")
    for slot_name, slot in Slots.__members__.items():
        answer, softmax = filled[slot]
        if args.plot:
            plt.bar(range(len(softmax)), softmax)
            plt.show()
        print(f"{slot_name}: {answer}")
