import torch
import math
from collections import Counter
from typing import List, Dict, Any, Union
# labels=[]
# for i in examples:
#   labels.append(i["PlayTennis"])

def calculate_entropy(labels: List[Any]) -> float:
    """
    Compute the Shannon entropy of the list of labels.
    labels: list of any hashable items.
    Returns a Python float.
    """
    val=Counter(labels)
    total_samples=len(labels)
    probs=[]
    for k,v in val.items():
      probs.append(v/total_samples)

    shanon_entropy=0
    for p in probs:
      shanon_entropy-=p*(math.log2(p))
    return shanon_entropy

def calculate_information_gain(
    examples: List[Dict[str, Any]],
    attr: str,
    target_attr: str
) -> float:
    """
    Compute information gain for splitting `examples` on `attr` w.r.t. `target_attr`.
    Returns a Python float.
    """
    split_groups={}
    for example in examples:
      attr_value=example[attr]

      if attr_value not in split_groups:
        split_groups[attr_value]=[]

      split_groups[attr_value].append(example)
    example_labels=[]
    for example in examples:
      example_labels.append(example[target_attr])

    parent_entropy=calculate_entropy(example_labels)

    weights=[]
    for k,v in split_groups.items():
      weights.append(len(v)/len(example_labels))







    entropies=[]

    for k,v in split_groups.items():
      labels=[]
      for r in v:
        # print(r["PlayTennis"])
        labels.append(r[target_attr])
      s=calculate_entropy(labels)
      entropies.append(s)
    child_entropy=0
    for i in range(len(entropies)):
      child_entropy+= (entropies[i]*weights[i])


    return parent_entropy - child_entropy


def majority_class(
    examples: List[Dict[str, Any]],
    target_attr: str
) -> Any:
    """
    Return the most common value of `target_attr` in `examples`.
    In case of a tie, return the class that comes first alphabetically.
    """
    # Your implementation here
    labels=[]
    for example in examples:
      labels.append(example[target_attr])
    counts=Counter(labels)
    max_count=max(counts.values())
    candidates=[]
    for label,count in counts.items():
      if max_count==count:
        candidates.append(label)

    return sorted(candidates)[0]
    

def learn_decision_tree(
    examples,
    attributes,
    target_attr
):

    labels = [
        e[target_attr]
        for e in examples
    ]

    # Case 1:
    # all labels same
    if len(set(labels)) == 1:
        return labels[0]

    # Case 2:
    # no attributes left
    if not attributes:
        return majority_class(
            examples,
            target_attr
        )

    # Choose best attribute
    best_attr = max(
        attributes,
        key=lambda attr:
            calculate_information_gain(
                examples,
                attr,
                target_attr
            )
    )

    tree = {
        best_attr: {}
    }

    # Partition data
    partitions = {}

    for example in examples:
        value = example[best_attr]

        if value not in partitions:
            partitions[value] = []

        partitions[value].append(example)

    remaining_attributes = [
        attr
        for attr in attributes
        if attr != best_attr
    ]

    # Recurse
    for value, subset in partitions.items():

        subtree = learn_decision_tree(
            subset,
            remaining_attributes,
            target_attr
        )

        tree[best_attr][value] = subtree

    return tree

