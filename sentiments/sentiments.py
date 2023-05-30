import pickle
from dataset import TweetsDataset

filename = './train_sentiments.pickle'
infile = open(filename,'rb')
bb = pickle.load(infile)
train_dataset=bb[0]

train_ids=bb[1]
infile.close()

filename = './val_sentiments.pickle'
infile = open(filename,'rb')
bb = pickle.load(infile)
val_dataset=bb[0]
val_ids=bb[1]
infile.close()


modelname='classla/bcms-bertic'
from transformers import ElectraForSequenceClassification #AutoModelForPreTraining  
model = ElectraForSequenceClassification.from_pretrained(modelname, num_labels=3)


from transformers import Trainer, TrainingArguments
# these are training settings
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    save_steps=200,
    evaluation_strategy='steps',
    eval_steps = 200, # Evaluation and Save happens every 10 steps
    save_total_limit = 10, # Only last 10 models are saved. Older ones are deleted.
    load_best_model_at_end=True
    )

trainer = Trainer(
    model=model,                         # the instantiated �~_�~W Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset            # evaluation dataset
    )

trainer.train()

model_path = "./BERTICsentiments"  
model.save_pretrained(model_path)
