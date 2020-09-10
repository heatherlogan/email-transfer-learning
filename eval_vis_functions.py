import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score


def plot_training_loss(loss_values, num_epochs, dataset_name):

  """
  plots training loss on its own
  params;
  - loss_values: array of training loss
  - num_epochs
  - dataset_name: e.g. introvert/extrovert/intuition/sensing for tthe graph tittle

  """

  sns.set(style='darkgrid')
  sns.set(font_scale=1.5)
  plt.rcParams['figure.figsize'] = (12,6)

  x_axis = [x for x in range(1,num_epochs+1)]
  plt.plot(x_axis, loss_values, 'b-o')

  plt.title('Training Loss ({} Classifier)'.format(dataset_name))
  plt.xlabel('Loss')
  plt.ylabel('Accuracy')

  plt.show()



def plot_training_validation_loss(training_loss_values,validation_loss_values, num_epochs, dataset_name):

  """
  plots training loss against validation loss
  """

  sns.set(style='darkgrid')
  sns.set(font_scale=1.5)
  plt.rcParams['figure.figsize'] = (12,6)

  x_axis = [x for x in range(1,num_epochs+1)]
  plt.plot(x_axis, training_loss_values, 'b-o', label='Training loss')
  plt.plot(x_axis, validation_loss_values, 'r-o', label='Validation loss')

  plt.title('Training Loss vs Validation Loss ({} Classifier)'.format(dataset_name))
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.show()



def plot_val_accuracy(accuracies, num_epochs,  dataset_name):

  sns.set(style='darkgrid')
  sns.set(font_scale=1.5)
  plt.rcParams['figure.figsize'] = (12,6)

  x_axis = [x for x in range(1,num_epochs+1)]
  plt.plot(x_axis, accuracies, 'g-o')

  plt.title('Validation Accuracy ({} Classifier)'.format(dataset_name))
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')

  plt.show()



def plot_confusion_matrix(true_labels, pred, class_labels):

  """
  confusion matrix to illustrate classification metrics TP/FN etc. of classification on test_set
  params
  - true labels: array of the correct predictions for a number of samples
  - pred: the labels predicted by the model
  - class_labels: the corresponding classes  
  """
  C = confusion_matrix(true_labels, pred)
  conf = C / C.astype(np.float).sum(axis=1)    
  df_cm = pd.DataFrame(conf, index = class_labels, columns=class_labels)
  sns.heatmap(df_cm, annot=True, cmap='Blues')



def mbti_classification_performance(predictions, true_labels, name=""):

  """
  displays metrics/report for classification problem. 
  params
  - predictions: array of the models predicted labels
  - true_labels: array of correct labels
  - name: the myers briggs dimension which classification is being assessed (Energy)

  peformance metrics for the sentiment analysis classification 
  accuracy: number of correctly predicted 
  f1-score: combination measure of total precision and recall 
  AUC: area under receiver operating characteristic curve whiich measures true positive rate vs false positive rate 
  classification report: individual class mettrics, acc, precision, recall
  confusion matrix; visualise performance true classes vs predicted classes for every class

  """

  if name.lower()=="energy":
    class_labels = ['Introversion' , 'Extroversion']
  elif name.lower()=='information':
    class_labels = ["Sensing", "Intuition"]
  elif name.lower() == "organisation":
    class_labels = ['Percieving', "Judging"]
  elif name.lower()=="decision":
    class_labels = ["Feeling", "Thinking"]
  else:
    class_labels = ['0', '1']

  # use model output for label 1 as out predictions

  p1 = predictions[:,1]
  pred_flat = np.argmax(predictions, axis=1).flatten()

  print("\n\n **** Evaluation *****")

  acc = accuracy_score(true_labels, pred_flat)
  print("Accuracy: {:.3f}".format(acc))

  f1score = f1_score(true_labels, pred_flat)
  print('F1-Score: {:.3f}'.format(f1score))

  auc = roc_auc_score(true_labels, p1)
  print('AUC:  {:.3f}'.format(auc))
  print()
  print()
  
  report = classification_report(true_labels, pred_flat, target_names=class_labels)
  print(report)
  plot_confusion_matrix(true_labels, pred_flat, class_labels=class_labels)



def sentiment_analysis_performance(predictions, true_labels):

  """
  peformance metrics for the sentiment analysis classification 
  accuracy: number of correctly predicted 
  f1-score: combination measure of total precision and recall 
  classification report: individual class mettrics, acc, precision, recall
  confusion matrix; visualise performance true classes vs predicted classes for every class

  """

  # 0 = negative, 1 = neutral, 2 = positive
  class_labels = ['negative', 'neutral', 'positive']

  p1 = predictions[:,1]
  pred_flat = np.argmax(predictions, axis=1).flatten()

  print("\n\n **** Evaluation *****")

  acc = accuracy_score(true_labels, pred_flat)
  print("Accuracy: {:.3f}".format(acc))

  f1score = f1_score(true_labels, pred_flat, average="weighted")
  print('F1-Score: {:.3f}'.format(f1score))

  # auc = roc_auc_score(true_labels, p1, average='weighted', multi_class='ovr')
  # print('AUC:  {:.3f}'.format(auc))

  print()
  print()
  report = classification_report(true_labels, pred_flat, target_names=class_labels)

  print(report)

  plot_confusion_matrix(true_labels, pred_flat, class_labels=class_labels)




