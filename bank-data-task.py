#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv

train_texts = []
train_labels = []
with open('data/banking_data/train.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    headers = next(reader, None)
    for row in reader:
        train_texts.append(row[0])
        train_labels.append(row[1])

test_texts = []
test_labels = []
with open('data/banking_data/test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    headers = next(reader, None)
    for row in reader:
        test_texts.append(row[0])
        test_labels.append(row[1])

# In[2]:


BATCH_SIZE = 32
MAX_LEN = 128

# In[3]:


from transformers import BertTokenizer, BertForSequenceClassification
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# In[4]:


import nltk
# Uncomment to download "stopwords"
# nltk.download("stopwords")
from nltk.corpus import stopwords
import re


def text_preprocessing(s):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    return s


# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


# In[5]:


train_texts = [text_preprocessing(text) for text in train_texts]
test_texts = [text_preprocessing(text) for text in test_texts]
train_inputs, train_masks = preprocessing_for_bert(train_texts)
test_inputs, test_masks = preprocessing_for_bert(test_texts)

# In[6]:


list_categories = [
    "card_arrival",
    "card_linking",
    "exchange_rate",
    "card_payment_wrong_exchange_rate",
    "extra_charge_on_statement",
    "pending_cash_withdrawal",
    "fiat_currency_support",
    "card_delivery_estimate",
    "automatic_top_up",
    "card_not_working",
    "exchange_via_app",
    "lost_or_stolen_card",
    "age_limit",
    "pin_blocked",
    "contactless_not_working",
    "top_up_by_bank_transfer_charge",
    "pending_top_up",
    "cancel_transfer",
    "top_up_limits",
    "wrong_amount_of_cash_received",
    "card_payment_fee_charged",
    "transfer_not_received_by_recipient",
    "supported_cards_and_currencies",
    "getting_virtual_card",
    "card_acceptance",
    "top_up_reverted",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "card_payment_not_recognised",
    "edit_personal_details",
    "why_verify_identity",
    "unable_to_verify_identity",
    "get_physical_card",
    "visa_or_mastercard",
    "topping_up_by_card",
    "disposable_card_limits",
    "compromised_card",
    "atm_support",
    "direct_debit_payment_not_recognised",
    "passcode_forgotten",
    "declined_cash_withdrawal",
    "pending_card_payment",
    "lost_or_stolen_phone",
    "request_refund",
    "declined_transfer",
    "Refund_not_showing_up",
    "declined_card_payment",
    "pending_transfer",
    "terminate_account",
    "card_swallowed",
    "transaction_charged_twice",
    "verify_source_of_funds",
    "transfer_timing",
    "reverted_card_payment?",
    "change_pin",
    "beneficiary_not_allowed",
    "transfer_fee_charged",
    "receiving_money",
    "failed_transfer",
    "transfer_into_account",
    "verify_top_up",
    "getting_spare_card",
    "top_up_by_cash_or_cheque",
    "order_physical_card",
    "virtual_card_not_working",
    "wrong_exchange_rate_for_cash_withdrawal",
    "get_disposable_virtual_card",
    "top_up_failed",
    "balance_not_updated_after_bank_transfer",
    "cash_withdrawal_not_recognised",
    "exchange_charge",
    "top_up_by_card_charge",
    "activate_my_card",
    "cash_withdrawal_charge",
    "card_about_to_expire",
    "apple_pay_or_google_pay",
    "verify_my_identity",
    "country_support"
]


def label2id(label):
    return list_categories.index(label)


def id2label(id_):
    return list_categories[id_]


# In[7]:


train_labels = [label2id(label) for label in train_labels]
test_labels = [label2id(label) for label in test_labels]
# Convert other data types to torch.Tensor
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

# In[8]:


import torch

print(train_inputs.shape)
print(test_inputs.shape)
print(train_labels.shape)
print(test_labels.shape)

# In[9]:


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# Create the DataLoader for our validation set
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

# In[10]:


# model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
#                                                       num_labels=len(list_categories),
#                                                       # The number of output labels--2 for binary classification.
#                                                       # You can increase this for multi-class tasks.
#                                                       output_attentions=False,
#                                                       # Whether the model returns attentions weights.
#                                                       output_hidden_states=False,
#                                                       )  # Whether the model returns all hidden-states.)
#
# model.cuda()


from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, num_class, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 512, num_class

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters()[0:5]:
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)
        return logits


model = BertClassifier(len(list_categories), freeze_bert=False)
model.to(device)
# In[11]:


# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())
print('The BERT model has {:} different named parameters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== First Transformer ====\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# In[15]:


# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
from transformers import AdamW

optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  )


from transformers import get_linear_schedule_with_warmup

epochs = 50

total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)
loss_fn = nn.CrossEntropyLoss()
# In[20]:


import random
import numpy as np
from tqdm import tqdm

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = preds.flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Store the average loss after each epoch so we can plot them.
loss_values = []
# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    model.train()
    total_loss = 0
    # For each batch of training data...
    for step, batch in enumerate(tqdm(train_dataloader)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # model.zero_grad()
        optimizer.zero_grad()

        logits = model(b_input_ids,
                        attention_mask=b_input_mask)

        loss = loss_fn(logits, b_labels)

        total_loss += loss.item()

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)

    loss_values.append(avg_train_loss)

    print('Loss values: ')
    print(np.mean(np.array(loss_values)))

    print("")
    print("Running Validation...")

    model.eval()
    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            )

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs
        # Move labels to CPU
        label_ids = b_labels.to('cpu').numpy()
        preds = torch.argmax(logits, dim=1)
        preds = preds.detach().cpu().numpy()
        # Calculate the accuracy for this batch of test sentences.

        tmp_eval_accuracy = flat_accuracy(preds, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy
        # Track the number of batches
        nb_eval_steps += 1
    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.4f}".format(eval_accuracy / nb_eval_steps))
