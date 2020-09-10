from imports import *


def tokenize_inputs(text_list, tokenizer, MAX_LEN=512):

    """
    params:
    - text_list: list of text to be processed
    - tokenizer: XLNET or BERT tokenizer
    - MAX_LEN: length after which text should be truncated
    converts raw text into lang embeddings with tokens mapped to an id and padded to be of uniform length
    special tokens are needed for the classification task, [CLS] is a special symbol added in front of every input example, and [SEP] is a special separator token
    they are needed as the models were trained on data with these tokens. 
    """
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:MAX_LEN-2], text_list))
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
   
    return input_ids


def create_attn_masks(input_ids):
    """
    params:
    - input_ids: embedding vectors
    creates corresponding attention vectors telling the model where it should be paying atttention to 
    i.e. zero for padding, 1 for any token present
    """
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks


def predictions_to_class(pred_array, classifying_task):

  """
  params:
   - pred_array: array of binary nums output my model
   - classifying_task: energy, informatioin, organisatioon or decision
  Converts the predicted labels to string labels
  """

  class_key = { 'energy':{0:'Introversion', 1:'Extroversion'}, 
                'information':{0:'Intuition', 1:'Sensing'},
                'organisation':{0:'Perceiving', 1:"Judging"}, 
                'decision':{0:'Feeling', 1:"Thinking"}
  }

  label_key = class_key[classifying_task.lower()]

  return [label_key.get(num) for num in pred_array]




def predict(model, tokenizer, email_df, num_labels=2, batch_size=32):

  """
  params
  - model; pretrained classification model 
  - df: email dataframe
  - num_labels: number of classes model can predict 
  - batch_size: defaultt is 32 as over 64 was causing usage limits
  outputs
  - pred_probs: model output class probailities probabilities 
  """

  print('Preparing emails...')

  # get token ids and attention masks for emails 
  email_input_ids = tokenize_inputs(email_df['text'].tolist(), tokenizer)
  email_attention_masks = create_attn_masks(email_input_ids)

  # append as columns on original dataframe
  email_df["features"] = email_input_ids.tolist()
  email_df["masks"] = email_attention_masks

  # finds how many iterations we need depending on batch size
  num_iter = math.ceil(email_df.shape[0]/batch_size)
  
  # keep track of probability predictions
  pred_probs = np.array([]).reshape(0, num_labels)
  
  # set model to evaluation mode
  model.to(device)
  model.eval()
  
  print('Classifying emails')
  
    # iterate through number of batches
  for i in tqdm(range(num_iter)):
    
    # locate batch contents
    df_subset = email_df.iloc[i*batch_size:(i+1)*batch_size,:]

    X = df_subset["features"].values.tolist()
    masks = df_subset["masks"].values.tolist()
    X = torch.tensor(X)
    
    masks = torch.tensor(masks, dtype=torch.long)
    X = X.to(device)
    masks = masks.to(device)

    with torch.no_grad():
      # feed to model
      logits = model(input_ids=X, attention_mask=masks)
      logits = logits[0].sigmoid().detach().cpu().numpy()

      # add probabilities
      pred_probs = np.vstack([pred_probs, logits])
  
  return pred_probs



  