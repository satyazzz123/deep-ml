
import torch
from typing import List
import math
from collections import Counter
def compute_tf_idf(corpus: List[List[str]], query: List[str]) -> torch.Tensor:

  
  DF_list=[]
  N=len(corpus)
  for q in query:
    DF=0
    for c in corpus:
      if q in c:
        DF+=1
    DF_list.append(DF)

  IDF=[]
  for DF in DF_list:
    
      IDF.append(math.log((N + 1) / (DF + 1)) + 1)


  TF=[]

  for q in query:
    TF_i=[]
    for c in corpus:
      if q in c:
        Tf_counter=Counter(c)
        q_counter=Tf_counter[q]
        TF_i.append(q_counter/len(c))
      else:
        TF_i.append(0)

    TF.append(TF_i)
  TF=torch.as_tensor(TF,dtype=torch.float32)
  IDF=torch.as_tensor(IDF,dtype=torch.float32)
  TF_IDF=[]

  for i in TF.T:
        TF_IDF.append(IDF * i)

  TF_IDF = torch.stack(TF_IDF)

  return torch.round(TF_IDF, decimals=5)

  


        






    


      
