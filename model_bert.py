# import libraries
import numpy as np
import pandas as pd
import re
import torch.nn as nn
import json 
import torch
from torchinfo import summary
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


# specify GPU
USE_CUDA = torch.cuda.is_available()
device = torch.device("cpu")

# Initialize The encoder
le = LabelEncoder()
#load intents
data_file = open('intents_negotiation.json').read()
intents = json.loads(data_file)
# Load the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Import the DistilBert pretrained model
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
for param in bert.parameters():
      param.requires_grad = False
torch.cuda.is_available()
# freeze all the parameters. This will prevent updating of model weights during fine-tuning.
for param in bert.parameters():
      param.requires_grad = False

# Hyperparameters 
epochs = 150
batch_size = 16
max_seq_len = 8

def split_data(intents):
    '''1- we get at first a dataframe with (label and text)
    '''
    classes = []
    documents = []

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            #add documents in the corpus
            documents.append((pattern, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])            
    df = pd.DataFrame(documents, columns = ["text", "label"])
    return df

def encoding_train_labels(data):

    '''
    encoding train set ( labels)
    '''
    # encoding labesl
    data['label'] = le.fit_transform(data['label'])
    '''
        check class distribution
        df['label'].value_counts(normalize = True)
    '''
    return data['label']

def encoding_train_texts(data):
    '''
    '''
    train_labels = encoding_train_labels(data)
    train_text = data['text']
    # tokenize and encode sequences in the training set(texts)
    tokens_train = tokenizer(
        train_text.tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )
    # for train set
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())
    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)
    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)
    # DataLoader for train set
    train_set_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_set_dataloader
'''
# get length of all the messages in the train set
seq_len = [len(i.split()) for i in train_text]
pd.Series(seq_len).hist(bins = 10)
# Based on the histogram we are selecting the max len as 8 and that's why max_lenght is equal to 8
'''

class BERT_Arch(nn.Module):
   def __init__(self, bert):      
       super(BERT_Arch, self).__init__()
       self.bert = bert 
      
       # dropout layer
       self.dropout = nn.Dropout(0.2)
      
       # relu activation function
       self.relu =  nn.ReLU()
       # dense layer
       self.fc1 = nn.Linear(768,512)
       self.fc2 = nn.Linear(512,256)
       self.fc3 = nn.Linear(256,8)
       #softmax activation function
       self.softmax = nn.LogSoftmax(dim=1)
       #define the forward pass
   def forward(self, sent_id, mask):
      #pass the inputs to the model  
      cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]
      
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      
      x = self.fc2(x)
      x = self.relu(x)
      x = self.dropout(x)
      # output layer
      x = self.fc3(x)
   
      # apply softmax activation
      x = self.softmax(x)
      return x


def classes_to_tensors(intents):
    '''
    '''
    train_labels = encoding_train_labels(split_data(intents))
    class_wts = compute_class_weight( class_weight = "balanced",
                                            classes = np.unique(train_labels),
                                            y = train_labels)

    return torch.tensor(class_wts,dtype=torch.float).to(device)


def train():
  '''
  '''
  model = BERT_Arch(bert)
# push the model to GPU /cpu
  model = model.to(device)
# from torchinfo import summary
# summary(model)
# define the optimizer
  optimizer = AdamW(model.parameters(), lr = 1e-3) 
  #   
  model.train()
  total_loss = 0
  # We can also use learning rate scheduler to achieve better results
#   lr_sch = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
  # 
  weights = classes_to_tensors(intents)

  # loss function
  cross_entropy = nn.NLLLoss(weight=weights)

#   define dataloader
  train_dataloader = encoding_train_texts(split_data(intents)) 
  # empty list to save model predictions
  total_preds=[]
  
  # iterate over batches
  for step,batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step,    len(train_dataloader)))
    # push the batch to gpu
    batch = [r.to(device) for r in batch] 
    sent_id, mask, labels = batch
    # get model predictions for the current batch
    preds = model(sent_id, mask)
    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)
    # add on to the total loss
    total_loss = total_loss + loss.item()
    # backward pass to calculate the gradients
    loss.backward()
    # clip the the gradients to 1.0. It helps in preventing the    exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # update parameters
    optimizer.step()
    # clear calculated gradients
    optimizer.zero_grad()
  
    # We are not using learning rate scheduler as of now
    # lr_sch.step()
    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()
    # append the model predictions
    total_preds.append(preds)
# compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    #returns the loss and predictions
    return avg_loss, total_preds
  

def train_run():
    # push the model to GPU/CPU
    model = BERT_Arch(bert).to(device)
    # summary(model)
    # empty lists to store training and validation loss of each epoch
    train_losses=[]
    # number of training epochs 
    for epoch in range(epochs):
        train_loss, _ = train() 
        print('\n Epoch {:} / {:} / loss {:}'.format(epoch + 1, epochs, train_loss))
        # append training and validation loss
        train_losses.append(train_loss)
        # it can make your experiment reproducible, similar to set  random seed to all options where there needs a random seed.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f'\nTraining Loss: {train_loss:.3f}')

    torch.save(model.state_dict(), "model.h5")


def get_prediction(str, max_seq_len):
    #load the model
    model = BERT_Arch(bert)
    model.load_state_dict(torch.load("modelll.h5", device))
    model = model.to(device)
    encoding_train_labels(split_data(intents))
    str = re.sub(r'[^a-zA-Z ]+', '', str)
    test_text = [str]
    model.eval()
    tokens_test_data = tokenizer(
    test_text,
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
    )
    test_seq = torch.tensor(tokens_test_data['input_ids'])
    test_mask = torch.tensor(tokens_test_data['attention_mask'])
    
    preds = None
    with torch.no_grad():
      preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis = 1)
    return le.inverse_transform(preds)[0]