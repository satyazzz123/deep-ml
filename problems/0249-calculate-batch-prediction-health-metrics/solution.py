import torch

def calculate_batch_health(statuses: torch.Tensor, confidences: torch.Tensor, confidence_threshold: float = 0.5) -> dict:
    """
    Calculate health metrics for a batch prediction job.
    
    Args:
        statuses: 1D tensor with 1 for 'success' and 0 for 'error'
        confidences: 1D tensor of confidence values (meaningful only for successful predictions)
        confidence_threshold: threshold below which a prediction is considered low confidence
    
    Returns:
        dict with keys: 'success_rate', 'avg_confidence', 'low_confidence_rate'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    statuses=torch.as_tensor(statuses,dtype=torch.float32)
    confidences=torch.as_tensor(confidences,dtype=torch.float32)
    if len(statuses)==0 or len(confidences)==0:
      return {} 
    status_mask=(statuses==1)
    success_rate=torch.sum(status_mask)/(len(statuses)+1e-7)

    # confidence_mask_that_are_successful=torch.where(status_mask,confidences,torch.tensor(0))
    confidence_mask_that_are_successful=confidences[status_mask]

    avg_confidence=torch.sum(confidence_mask_that_are_successful)/(len(confidence_mask_that_are_successful)+1e-7)
    low_confidence_rate=torch.sum(confidence_mask_that_are_successful<confidence_threshold)/(len(confidence_mask_that_are_successful)+1e-7)
    return {
        "success_rate":round(success_rate.item()*100,2),
        "avg_confidence":round(avg_confidence.item()*100,2),
        "low_confidence_rate":round(low_confidence_rate.item()*100,2)
        }
