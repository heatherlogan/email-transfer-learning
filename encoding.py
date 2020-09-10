from imports import *

def encode_dataset(train, tokenizer, MAX_LEN = 128):

  """
  train - dataframe of training examples; title of string to be processed should be 'text'
  tokenizer - the pretrained tokenizer used for the task
  MAX_LEN - max length of tokens to be passed in. BERT can only handle up to 512 tokens, XLNET has no max. 
  default = 128 as this does not cause usage limits; but this is altered when used
  """

  # if no tokenizer is passed in, declare XLNet. 
  if tokenizer==None:

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

  # store in list
  input_ids = []
  lengths = []

  # iterate through every 'text' row in the dataframe. 
  for sentence in train.text:

    # progress report
    if ((len(input_ids) % 10000) == 0):
      print('Read {:,} posts'.format(len(input_ids)))

    # encode function to tokenize, prepend [CLS] and [SEP] tokens, and map tokens to their IDS
    encoded_sentence = tokenizer.encode(sentence, add_special_tokens=True)

    # store encodings
    input_ids.append(encoded_sentence)
    lengths.append(len(encoded_sentence))


  # for use in BERT when max length is exceeded. reports number of truncated inputs. 
  lengths = [min(l,512) for l in lengths]
  num_truncated= lengths.count(512)
  num_sentences = len(lengths)
  percent = float(num_truncated)/float(num_sentences)

  print("{:,} of {:,} posts ({:.1%}) are longer than 512 tokens".format(num_truncated, num_sentences, percent))
  print()

  # pad and truncate posts
  input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', value=0, truncating='post', padding='post')


  # return attention masks
  attention_masks = []
  for sentence in input_ids:
    att_mask = [int(token_id != 0) for token_id in sentence]
    attention_masks.append(att_mask)

  return input_ids, lengths, attention_masks




def get_training_validation_dataloaders(input_ids, attention_masks, labels, batch_size=32):
  
  """
  get the encoding, lengths and attention_masks for each dataset then split into training & validation inputs, labels and masks 
  params - 
  input_ids: id of tokens inside the encoded language embeddings produced by encode_dataset
  attention_masks: np.array of attention, same dimensions as input_ids
  labels: array of labels correpsonding to train samples
  batch_size: the number of samples included in each batch for forming the datalaoders.  
  """

  # split for the encodings into train validation sets
  train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels ,
                                                                                    random_state=200, test_size=0.2)  
  # split for the masks
  train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels , random_state=200, test_size=0.2)  

  # convert to torch tensors
  train_inputs = torch.tensor(train_inputs)
  validation_inputs = torch.tensor(validation_inputs)

  train_labels = torch.tensor(train_labels)
  validation_labels = torch.tensor(validation_labels)

  train_masks = torch.tensor(train_masks)
  validation_masks = torch.tensor(validation_masks)


  # get tensors/sampler/dataloader to be used in training loop
  train_data = TensorDataset(train_inputs, train_masks, train_labels)
  train_sampler= RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

  validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
  validation_sampler = SequentialSampler(validation_data)
  validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

  return train_data, train_sampler, train_dataloader, validation_data, validation_sampler, validation_dataloader




def get_test_dataloader(input_ids, attention_masks, labels, batch_size=32):
  """
  formatting the test set to be used with pytorch as above, but with no labels. 
  """
  input_ids = torch.tensor(input_ids)
  attention_masks = torch.tensor(attention_masks)
  labels = torch.tensor(labels)

  test_data = TensorDataset(input_ids, attention_masks, labels)
  test_sampler = RandomSampler(test_data)
  test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

  return test_dataloader



