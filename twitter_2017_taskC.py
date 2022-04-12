# Import libraries

import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer


# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Load Test and Validation data
train_data_df = pd.read_csv("data/twitter-2016train-CE.txt", delimiter='\t', header=None, names=['id','topic','label', 'tweet'])
val_data_df = pd.read_csv("data/twitter-2016devtest-CE.txt", delimiter='\t', header=None, names=['id','topic','label', 'tweet'])

train_tweet = train_data_df.tweet.values
train_topic = train_data_df.topic.values
y_train = train_data_df.label.values

val_tweet = val_data_df.tweet.values
val_topic = val_data_df.topic.values
y_val = val_data_df.label.values

# Convert string classes into numeric classee
train_labels=[]
val_labels=[]
label_dict = {-2:0,-1:1,0:2,1:3,2:4}

for label in y_train:
  train_labels.append(label_dict[label])

for label in y_val:
  val_labels.append(label_dict[label])

# Train and validation data size
print(len(train_tweet))
print(len(val_tweet))

# Data Processing
def processdata(topics,tweets,labels):
  input_ids = []
  attention_masks = []
  token_type_ids = []
  for topic, tweet in zip(topics,tweets):
    encoded_dict = tokenizer.encode_plus(topic,
                        tweet,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        return_token_type_ids = True,
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    token_type_ids.append(encoded_dict['token_type_ids'])

  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  token_type_ids = torch.cat(token_type_ids,dim=0)
  labels = torch.tensor(labels)
  return input_ids,attention_masks,token_type_ids,labels

train_input_ids,train_attention_masks,train_token_type_ids,train_labels = processdata(train_topic,train_tweet,train_labels)
val_input_ids,val_attention_masks,val_token_type_ids,val_labels = processdata(val_topic,val_tweet,val_labels)


# Convert to TensorData
train_dataset = TensorDataset(train_input_ids, train_attention_masks,train_token_type_ids,train_labels)
val_dataset = TensorDataset(val_input_ids, val_attention_masks,val_token_type_ids,val_labels)

# Create DataLoader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
batch_size = 32

# Create the DataLoaders for our training and validation sets.
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


# Load BERT Model
from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels = 5, # The number of output labels = 5
                    
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()



# Define Optimizer
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

# Epochs and Scheduler
from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
EPOCHS = 4

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * EPOCHS

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


# Accuracy function
def accuracy(y_pred, y_test):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
  return acc


import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Train Model
training_stats=[]
def train(model, train_loader, val_loader, optimizer,scheduler):  
  total_step = len(train_loader)

  for epoch in range(EPOCHS):
    # Measure how long the training epoch takes.
    train_start = time.time()
    model.train()

    # Reset the total loss and accuracy for this epoch.
    total_train_loss = 0
    total_train_acc  = 0
    for batch_idx, (pair_token_ids, mask_ids,seg_ids, y) in enumerate(train_loader):

      # Unpack this training batch from our dataloader. 
      pair_token_ids = pair_token_ids.to(device)
      mask_ids = mask_ids.to(device)
      seg_ids = seg_ids.to(device)
      labels = y.to(device)

      #clear any previously calculated gradients before performing a backward pass
      optimizer.zero_grad()

      #Get the loss and prediction
      loss, prediction = model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()

      acc = accuracy(prediction, labels)
      
      # Accumulate the training loss and accuracy over all of the batches so that we can
      # calculate the average loss at the end
      total_train_loss += loss.item()
      total_train_acc  += acc.item()

      # Perform a backward pass to calculate the gradients.
      loss.backward()

      # Clip the norm of the gradients to 1.0.
      # This is to help prevent the "exploding gradients" problem.
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      # Update parameters and take a step using the computed gradient.
      optimizer.step()

      # Update the learning rate.
      scheduler.step()

    # Calculate the average accuracy and loss over all of the batches.
    train_acc  = total_train_acc/len(train_loader)
    train_loss = total_train_loss/len(train_loader)
    train_end = time.time()

    # Put the model in evaluation mode
    model.eval()

    total_val_acc  = 0
    total_val_loss = 0
    val_start = time.time()
    with torch.no_grad():
      for batch_idx, (pair_token_ids, mask_ids,seg_ids,y) in enumerate(val_loader):

        #clear any previously calculated gradients before performing a backward pass
        optimizer.zero_grad()

        # Unpack this validation batch from our dataloader. 
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = y.to(device)
        
        #Get the loss and prediction
        loss, prediction = model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()

        # Calculate the accuracy for this batch
        acc = accuracy(prediction, labels)

        # Accumulate the validation loss and Accuracy
        total_val_loss += loss.item()
        total_val_acc  += acc.item()

    # Calculate the average accuracy and loss over all of the batches.
    val_acc  = total_val_acc/len(val_loader)
    val_loss = total_val_loss/len(val_loader)

    #end = time.time()
    val_end = time.time()
    hours, rem = divmod(val_end-train_start, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': train_loss,
            'Valid. Loss': val_loss,
            'Valid. Accur.': val_acc,
            'Training Time': train_end-train_start,
            'Validation Time': val_end-val_start
        }
    )

train(model, train_dataloader, validation_dataloader, optimizer,scheduler)

import pandas as pd

# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# A hack to force the column headers to wrap.
#df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

# Display the table.
print(df_stats)



# Visualize the training and validation loss
import matplotlib.pyplot as plt
import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])

plt.show()


# Load Test data
# Load the dataset into a pandas dataframe.
import pandas as pd
df = pd.read_csv("data/twitter-2016test-CE.txt", delimiter='\t', header=None, names=['id','topic','label','tweet'])

# Report the number of sentences.
print('Number of test sentences: {:,}\n'.format(df.shape[0]))

# Create sentence and label lists

y_test = df.label.values
test_topic = df.topic.values
test_tweet = df.tweet.values


labels=[]
for label in y_test:
  labels.append(label_dict[label])

input_ids,attention_masks,token_type_ids,labels = processdata(test_topic,test_tweet,labels)

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks,token_type_ids,labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#Test data preposessing
def test(model,prediction_dataloader): 

  total_test_acc  = 0
  total_F1_Score = 0
  total_precision = 0
  total_recall = 0

  for batch_idx, (pair_token_ids, mask_ids,token_type_ids,y) in enumerate(prediction_dataloader):
    pair_token_ids = pair_token_ids.to(device)
    mask_ids = mask_ids.to(device)
    labels = y.to(device)
    token_type_ids=token_type_ids.to(device)

    loss, prediction = model(pair_token_ids, 
                             token_type_ids=token_type_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()


    acc = accuracy(prediction, labels)

    f1 = metrics.f1_score(labels.cpu(), torch.argmax(prediction, -1).cpu(), labels=[0, 1, 2, 3, 4], average='weighted')
    precision = precision_score(labels.cpu(), torch.argmax(prediction, -1).cpu(),labels=[0, 1, 2, 3, 4], average='weighted')
    recall = recall_score(labels.cpu(), torch.argmax(prediction, -1).cpu(),labels=[0, 1, 2, 3, 4], average='weighted')

    total_test_acc  += acc.item()
    total_F1_Score += f1
    total_precision += precision
    total_recall += recall
  
  test_acc  = total_test_acc/len(prediction_dataloader)
  test_f1 = total_F1_Score/len(prediction_dataloader)
  test_precision = total_precision/len(prediction_dataloader)
  test_recall = total_recall/len(prediction_dataloader)


  print(f'test_acc: {test_acc:.4f}')
  print(f'f1 Score: {test_f1:.4f}')
  print(f'precision: {test_precision:.4f}')
  print(f'recall: {test_recall:.4f}')
  
test(model,prediction_dataloader)
