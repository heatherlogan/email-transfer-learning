from imports import *


# helper function for tracking time
def format_time(elapsed):
  elapsed_rounded = int(round(elapsed))
  return str(datetime.timedelta(seconds=elapsed_rounded))


def evaluate_test_set(model, test_dataloader):



  # seed value for reducability / consistent predictions
  seed_val = 50
  random.seed(seed_val)
  np.random.seed(seed_val)    
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

  # model into evaluation model
  model.eval()

  predictions, true_labels = [], []

  t0 = time.time()

  # loop through each batch of examples
  for (step, batch) in enumerate(test_dataloader):

    # put batch to GPU/cuda
    batch = tuple(t.to(device) for t in batch)

    if step % 100 == 0 and not step==0:

      elapsed = format_time(time.time() - t0 )
      print('Batch {:>5,} of {:>5,}. Elapsed: {:}. '.format(step, len(test_dataloader), elapsed))

    # unpack inputs from dataloader
    b_input_ids, b_input_mask, b_labels = batch

    # turn off learning 
    with torch.no_grad():

      # pass inputs and attention masks to model. token_type_ids is a [1,0] array 
      # which is useful for 
      outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    # log probabilities 
    logits = outputs[0]

    # take the sigmoid of the logits to squash into (1,0) range, 
    # detach turns off gradient calculations 
    # cpu removes item from the gpu as the processing power is no longer needed
    logits = logits.sigmoid().detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # keep track of predictions vs. true labels 
    predictions.append(logits)
    true_labels.append(label_ids)

  print('Done. ')
  return predictions, true_labels

  