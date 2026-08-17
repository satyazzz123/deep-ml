
def mask_instruction_loss(batches, pad_token_id, ignore_index=-100):
    final_input=[]
    final_label=[]
    max_len_batch=0
    requires_pad=False
    for b in batches:
      tokens=b["tokens"]
      if len(tokens)>max_len_batch:
        max_len_batch=len(tokens)
    for batch in batches:
      unpadded_tokens=batch["tokens"]
      num_pad=max_len_batch-len(unpadded_tokens)
      tokens=unpadded_tokens+(num_pad)*[pad_token_id]

     


      input_tokens=[]
      labels=[]
      input_tokens=tokens[:len(tokens)-1]
      labels=tokens[1:len(tokens)]
      instruction_length=batch["instruction_length"]-1
      
      for i in range(len(input_tokens)):
        if i in list(range(instruction_length)) or input_tokens[i]==pad_token_id:
          labels[i]=ignore_index
      
      final_input.append(input_tokens)
      final_label.append(labels)
      
    return{'inputs':final_input,'targets':final_label}
    