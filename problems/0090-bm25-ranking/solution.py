import torch
from collections import Counter
import math


def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75) -> torch.Tensor:
    """
    Implement the BM25 ranking function using PyTorch tensor operations.

    Args:
        corpus: List of tokenized documents (list of list of strings)
        query: List of query tokens (list of strings)
        k1: Term frequency saturation parameter (default 1.5)
        b: Document length normalization parameter (default 0.75)

    Returns:
        torch.Tensor: BM25 scores for each document, rounded to 3 decimal places
    """
    # Your code here

    BM_25=0
    avg=0
    for c in corpus:
      avg+=len(c)
    avg=avg/len(corpus)
    N=len(corpus)
    IDF=[]
    for q in query:
      df=0
      for c in corpus:
        if q in c:
          df+=1

      val=math.log((N+1)/(df+1))
      IDF.append(val)

    IDF=torch.as_tensor(IDF,dtype=torch.float32)
    
    BM_25_final=[]
    for doc in corpus:
      count=Counter(doc)
      BM_25_list=[]
      for q in query:
        f1=count[q]
        
        denominator=f1+k1*(1-b+b*(len(doc)/avg))+0.00000001
        numerator=f1*(k1+1)
        BM_25=numerator/denominator
        BM_25_list.append(BM_25)
      BM_25_list=torch.as_tensor(BM_25_list,dtype=torch.float32)
      BM_25_list=IDF*BM_25_list
      BM_25_final.append(torch.sum(BM_25_list,dim=0))
    return torch.stack(BM_25_final)







