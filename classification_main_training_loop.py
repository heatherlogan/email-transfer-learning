from imports import *

print("GPU Available: {}".format(torch.cuda.is_available()))
n_gpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Number of GPU Available: {}".format(n_gpu))
print("GPU: {}".format(torch.cuda.get_device_name(0)))


"""

helper functions for calculating performance metrics
parameters 

"""


def binary_validation_loss(preds, labels):

  pred_tensor = torch.FloatTensor(preds)

  # converts y to shape of predictions 
  y = torch.zeros(len(labels), 2)
  y[range(y.shape[0]), labels]=1
  
  # use cross entropy loss
  lossfunc =  BCEWithLogitsLoss()
  return lossfunc(pred_tensor, y)


def multiclass_validation_loss(preds, labels):

  # convert to tensors 
  pred_tensor = torch.FloatTensor(preds)
  label_tensor = torch.LongTensor(labels)

  # loss function is cross entropy for multi-class classification
  lossfunc = CrossEntropyLoss()

  return lossfunc(pred_tensor, label_tensor)


def flat_accuracy(preds, labels):
  
  # takes the highest probability accross class log probs
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  acc = np.sum(pred_flat == labels_flat)/len(labels_flat)

  return acc


# helper function for tracking time
def format_time(elapsed):
  elapsed_rounded = int(round(elapsed))
  return str(datetime.timedelta(seconds=elapsed_rounded))



def train_loop(model, optimizer, train_dataloader, validation_dataloader, num_epochs=5):

  # seed value for reproducability 

  """
  main training loop for classification_tasks. 
  params:
  - model: Pretrained langauge embedding model that will be trained. (BERT, XLNET etc. )
  - optimizer: previously declared optimiser to be used with model. (AdamW etc. )
  - training_dataloader: data structure created at pre-processing stage containing training data, labels in batches. 
  - validation_dataloader: as above, but with the validation dataloader
  - num_epochs: the number of epochs/cycles the model will be trained over, default is 5. 

  returns training loss, validation loss and validation accuracies as arrays. 
  """


  # setting seed values keeps rresults more consistent when re-run.
  seed_val = 50
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)


  """
  From doc: Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after a
  warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
  """
  epochs = num_epochs 
  total_steps = len(train_dataloader)*epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                              num_training_steps = total_steps)

  # keep track of training loss and validation accuracies
  loss_values = []
  validation_loss_values = []
  accuracies = [] 

  for e in range(0, epochs):

    print("============= Epoch {:}/{:} ==============".format(e+1, epochs))

    print('\nTraining...')

    t0 = time.time()

    total_loss= 0

    # set model to train mode
    model.train()

    for step, batch in enumerate(train_dataloader):

      if step%100==0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print("Batch {:>5,} of {:>5,}. Elapsed {:}".format(step, len(train_dataloader), elapsed))


      ################## Training ##################

      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_labels = batch[2].to(device)

      # clear previously calculated gradients before backward pass
      model.zero_grad()


      # pass the batch contents to the model for learning
      outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

      # training loss
      loss = outputs[0]

      # accumulate loss over batches 
      total_loss += loss.item()

      # calculate gradients with backward pass
      loss.backward()

      # clip norm of gradient - prevents exploding gradients (accumulating gradients cause large update to NN weigihts )
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      # update optimizer
      optimizer.step()

      # update learning rate
      scheduler.step()
    
    # average training loss between batches
    avg_train_loss = total_loss / len(train_dataloader) 
    loss_values.append(avg_train_loss)
    
    print("\nAverage training loss: {0:.5f}".format(avg_train_loss))
    print("Time taken for epoch: {:}".format(format_time(time.time() - t0)))


    ################## Validation ##################

    t0 = time.time()

    model.eval()

    eval_loss, eval_accuracy = 0,0 
    nb_eval_steps, nb_eval_examples= 0, 0 

    # iterate through validatoin batches
    for batch in validation_dataloader:


      # unpack items from batch and put onto gpu 
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch

      # turn off model learning
      with torch.no_grad():

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)


      # logits are log probabilities of the output belonging to each class; e.g. [0.1, 0.1. 0.8] for a 3 class problem. 
      logits = outputs[0]
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()

      # helper functions to calculate loss and accuracy 
      tmp_eval_accuracy = flat_accuracy(logits, label_ids)

      # calculate loss with appropriate method (binary cross entropy for binary classification, regualr crossn entropy for multiclass)
      if model.num_labels > 2:
        # multiclass loss 
         tmp_eval_loss = multiclass_validation_loss(logits, label_ids)
      else:
         tmp_eval_loss = binary_validation_loss(logits, label_ids)

         
      eval_loss += tmp_eval_loss
      eval_accuracy += tmp_eval_accuracy
      nb_eval_steps +=1 

    # calculate average over batches
    val_accuracy = eval_accuracy/nb_eval_steps
    val_loss = eval_loss/len(validation_dataloader)
    
    accuracies.append(val_accuracy)
    validation_loss_values.append(val_loss)

    print("Accuracy on Validation set: {0:.5f}".format(val_accuracy))
    print("Validation Loss: {0:.5f}".format(val_loss))

    # convert from tensors to list 
    validation_list = [t.item() for t in validation_loss_values]


  return loss_values, validation_list, accuracies

