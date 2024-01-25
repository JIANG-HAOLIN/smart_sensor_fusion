import torch


def top_k_accuracy(logits, label, k=3):
    # Sort the logits in descending order along the last dimension
    _, sorted_indices = torch.sort(logits, dim=-1, descending=True)

    # Retrieve the indices of the top k logits
    top_k_indices = sorted_indices[:, :k]

    # Check if the true label is among the top k predicted indices
    correct_predictions = torch.any(top_k_indices == label.unsqueeze(1), dim=-1)

    # Compute the top-k accuracy
    top_k_acc = torch.mean(correct_predictions.float())

    return top_k_acc.item()  # Convert to Python float

