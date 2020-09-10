from imports import *


class_names = {0:'negative', 1:'neutral', 2:'positive'}


def predict_sentiment(text, model):


    """
    predicts sentiment for any textual input. not for use on large datasets. 
    param:
    - text: string sentence
    - model: embedding model (XLNET, BERT..)
    """

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=False)  

    # max_len is safe to be set at 512, here as there is one item entered at a time so isn't demandinig on
    # resources - in training cases it may need to be lowered. 
    MAX_LEN = 512


    # encode_plus returns dictionary item containing encodings and attention masks
    # less efficient than .encode() only but fine here as small input. 
    # parameters state to add [cls] and [sep] tokens which the model recognises, not to return token_type_ids
    # which are not needed here (arrays of [1s and 0s] and not to )
    encoded_review = tokenizer.encode_plus(
    text, 
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=False,
    return_attention_mask=True,
    return_tensors='pt',
    )

    # transform inputs with padding and convert to tensors 

    input_ids = pad_sequences(encoded_review['input_ids'], maxlen=MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
    input_ids = input_ids.astype(dtype = 'int64')
    input_ids = torch.tensor(input_ids) 

    attention_mask = pad_sequences(encoded_review['attention_mask'], maxlen=MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
    attention_mask = attention_mask.astype(dtype = 'int64')
    attention_mask = torch.tensor(attention_mask) 

    # put tensors on the gpu 
    input_ids = input_ids.reshape(1,512).to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # take outputs, generate prediction using softmax function, then turn off gradient and take off gpu 
    outputs = outputs[0][0].cpu().detach()
    probs = F.softmax(outputs, dim=-1).cpu().detach().numpy().tolist()
    _, prediction = torch.max(outputs, dim =-1)

    print("Positive score:", probs[1])
    print("Negative score:", probs[0])
    print(f'Review text: {text}')
    print(f'Sentiment  : {class_names.get(prediction.item())}')

