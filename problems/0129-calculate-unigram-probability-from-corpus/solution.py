import torch
from collections import Counter
def unigram_probability(corpus: str, word: str) -> torch.Tensor:
    """Calculate the unigram probability of a word in a corpus.
    
    Args:
        corpus: A string containing the text corpus with tokens separated by spaces
        word: The word to calculate probability for
        
    Returns:
        A torch.Tensor containing the probability rounded to 4 decimal places
    """
    tokens=corpus.split(" ")
    count=Counter(tokens)
    word_dict=dict(count)
    vocab=0
    for k,v in word_dict.items():
      vocab+=v
    return torch.tensor(round(word_dict[word]/vocab,4))
    
      
    
    