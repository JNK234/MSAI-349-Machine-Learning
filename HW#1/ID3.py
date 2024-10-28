from node import Node
import math
import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings("ignore")

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node)
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  # Get the feature and Target
  target = "Class"
  features = [k for k in examples[0].keys() if k != target]

  # Data Cleaning
  df = pd.DataFrame(examples)
  df = df.replace('?', np.nan)

  for col in features:
    mode_value = df[col].mode()[0]  # Get the most frequent value
    df[col].fillna(mode_value, inplace=True)

  # Build Tree
  root_node = build_tree(df, features, target)

  return root_node

def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''

  if not examples:
    return node

  if not node.children:
      return node

  # Recursively prune children
  for attribute_value, child in node.children.items():
      child_examples = [e for e in examples if e[node.label_name] == attribute_value]
      node.children[attribute_value] = prune(child, child_examples)

  # Calculate accuracy before pruning
  true_accuracy = test(node, examples)

  # Create a leaf node with majority class
  majority_class = max(set(e['Class'] for e in examples), key=lambda x: sum(e['Class'] == x for e in examples))
  pruned_node = Node()
  pruned_node.label_name = majority_class
  pruned_node.leaf_node = True

  # Calculate accuracy after pruning
  pruned_accuracy = test(pruned_node, examples)

  # If pruning improves accuracy, return the pruned node
  if pruned_accuracy >= true_accuracy:
      return pruned_node
  else:
      return node

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  test_inputs, test_targets = zip(*[(dict((k, v) for k, v in d.items() if k != 'Class'), d['Class']) for d in examples])
  test_preds = [evaluate(node, item) for item in test_inputs]

  return sum(np.equal(test_preds, test_targets))/len(test_targets)

def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''

  if node.leaf_node:
    return node.label_name
    
  feature_value = example.get(node.label_name)
  if feature_value in node.children:
    return evaluate(node.children[feature_value], example)



# ======================= CUSTOM FUNCTIONS =======================


def entropy(column):
  # Identify the counts of labels
  elements, counts  = np.unique(column, return_counts=True)
  # Calculate the total entropy
  total_entropy = sum([-(counts[i]/sum(counts) * np.log2(counts[i]/sum(counts))) for i in range(len(elements))])
  return total_entropy

def get_information_gain(Data, Attribute, Target):
  feature_values = Data[Attribute].unique()

  target_entropy = entropy(Target)

  # For every color, we have to find probability and individual entrpy for that value
  cond_entropy = 0
  for value in feature_values:
    # Get the data subset
    subset = Data[Data[Attribute] == value]
    target_entropy =  entropy(subset[Target])
    ratio = len(subset) / len(Data)
    cond_entropy += ratio * target_entropy

  #  Information Gain Calculation
  information_gain = target_entropy - cond_entropy
  return information_gain


def build_tree(df, features, target):
  # Create a Node
  node = Node()

  # Assign a most Common Class label
  node.label_name = df[target].mode().iloc[0]

  if len(np.unique(df[target])) == 1:
    node.label_name = df[target].iloc[0]
    node.leaf_node = True
    return node

  if len(features) == 0:
    node.leaf_node = True
    # node.label = Data[target].mode().iloc[0]
    return node
  
  # Choose the best attribute with max information gain
  best_attribute = max(features, key=lambda f: get_information_gain(df, f, target))

  node.label_name = best_attribute

  # Recursively build the tree
  for value in df[best_attribute].unique():
      subset = df[df[best_attribute] == value]
      if len(subset):
          sub_attributes = [a for a in features if a != best_attribute]
          child_node = build_tree(subset, sub_attributes, target)
          node.children[value] = child_node
      else:
          child_node = Node()
          child_node.leaf_node = True
          child_node.label = df[target].mode().iloc[0]
          node.children[child_node.label] = child_node

  return node