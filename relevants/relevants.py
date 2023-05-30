import pandas as pd
tweets=pd.read_pickle('relevants.pickle')


from sklearn.model_selection import train_test_split
train, valid = train_test_split(tweets, test_size=.2)
train.reset_index(drop=True,inplace=True)
valid.reset_index(drop=True,inplace=True)


valid, test = train_test_split(valid, test_size=.5)
test.reset_index(drop=True,inplace=True)
valid.reset_index(drop=True,inplace=True)


from transformers import AutoTokenizer
modelname='classla/bcms-bertic'  ## the hugging face name of Ljub and co model
tokenizer = AutoTokenizer.from_pretrained(modelname)  ## loading the tokenizer of that model

# tokenization by the Ljub and co model. train and val texts are converted to train and val encodings:
train_encodings = tokenizer(train['text'].to_list(), is_split_into_words=False, padding=True, truncation=True)
val_encodings = tokenizer(valid['text'].to_list(), is_split_into_words=False, padding=True, truncation=True)
test_encodings = tokenizer(test['text'].to_list(), is_split_into_words=False, padding=True, truncation=True)

train_labels=train['Vesna'].to_list()
val_labels=valid['Vesna'].to_list()
test_labels=test['Vesna'].to_list()

"""
Ovo sto sledi je prosto torch klasa za konverziju inputa u torch tipove podataka
"""
from dataset import TweetsDataset
import pickle

train_dataset = TweetsDataset(train_encodings, train_labels)
val_dataset = TweetsDataset(val_encodings, val_labels)
test_dataset = TweetsDataset(test_encodings, test_labels)


filename = './val_relevants.pickle'
outfile = open(filename,'wb')
bb=[val_dataset, valid['id']]
pickle.dump(bb,outfile)
outfile.close()


filename = './train_relevants.pickle'
outfile = open(filename,'wb')
bb=[train_dataset, train['id']]
pickle.dump(bb,outfile)
outfile.close()

filename = './test_relevants.pickle'
outfile = open(filename,'wb')
bb=[test_dataset, test['id']]
pickle.dump(bb,outfile)
outfile.close()

from transformers import ElectraForSequenceClassification #AutoModelForPreTraining  
model = ElectraForSequenceClassification.from_pretrained(modelname, num_labels=2)


from transformers import Trainer, TrainingArguments
# these are training settings
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=6,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    save_steps=200,
    evaluation_strategy='steps',
    eval_steps = 200, # Evaluation and Save happens every 200 steps
    save_total_limit = 10, # Only last 5 models are saved. Older ones are deleted.
    load_best_model_at_end=True
    )

trainer = Trainer(
    model=model,                         # the instantiated �~_�~W Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset          # evaluation dataset
    )

trainer.train()

model_path = "./BERTICrelevants"  
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
