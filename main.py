import re
import json
from sklearn.model_selection import train_test_split
from transformers import TextDataset, DataCollatorForLanguageModeling  # Preprocessing
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer  # Training
from transformers import pipeline  # Testing

INPUT_FILENAME = "./dataset/recipes_raw_nosource_ar.json"
# INPUT_FILENAME = "./dataset/recipes_raw_nosource_epi.json"
# INPUT_FILENAME = "./dataset/recipes_raw_nosource_fn.json"

TRAIN_PATH = "train_dataset.txt"
TEST_PATH = "test_dataset.txt"
OUTPUT_PATH = "./model"
LOG_PATH = "./logs"
TOKENIZER_NAME = "gpt2"
MODEL_NAME = "gpt2"


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


with open(INPUT_FILENAME) as f:
    data = json.load(f)


def build_file(data, filepath):
    file = open(filepath, 'w')
    string = ''

    for texts in data:
        if 'instructions' in texts:
            instructions = texts['instructions']

            instructions = re.sub(r'[^\x00-\x7f]', r' ', instructions)  # Remove non-Unicode characters
            instructions = re.sub(r"\s+", " ", instructions)  # Remove trailing tabs and spaces

            string += instructions + tokenizer.eos_token

    file.write(string)
    file.close()


data_list = []

for key, value in data.items():  # Remove key hash, not relevant
    data_list.append(value)

train, test = train_test_split(data_list, test_size=0.15)

build_file(train, TRAIN_PATH)
build_file(test, TEST_PATH)

print("Train dataset length: " + str(len(train)))
print("Test dataset length: " + str(len(test)))


train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=TRAIN_PATH,
    block_size=128)

test_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=TEST_PATH,
    block_size=128)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    logging_dir=LOG_PATH,
    overwrite_output_dir=True,  # Overwrite the output directory
    num_train_epochs=5,
    warmup_steps=500,  # Number of steps used for a linear warmup from 0
    eval_steps=500,  # Number of update steps between two evaluations.
    per_device_train_batch_size=64,  # Batch size for training
    per_device_eval_batch_size=64,  # Batch size for evaluation
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

model = pipeline('text-generation', model=OUTPUT_PATH, tokenizer=TOKENIZER_NAME)
model("Chicken soup")
