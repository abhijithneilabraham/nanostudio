import torch
from torch.cuda.amp import autocast

class Evaluator:
    def __init__(self, model, data_loader, device, dtype='float32'):
        """
        Initialize the evaluator with a model and data loader.
        
        Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): The DataLoader providing the validation dataset.
        device (str): Device to run the evaluation on ('cuda' or 'cpu').
        dtype (str): Data type for mixed precision ('float32' or 'float16').
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.dtype = dtype

    def evaluate(self):
        """
        Evaluate the model on the entire validation set and return the average loss.
        
        Returns:
        float: The average loss over the validation dataset.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in self.data_loader.get_validation_batches():
                x, y = x.to(self.device), y.to(self.device)
                with autocast(enabled=self.dtype == 'float16'):
                    logits, loss = self._forward_pass(x, y)
                    total_loss += loss.item()
        
        avg_loss = total_loss / len(self.data_loader.validation_data)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def _forward_pass(self, x, y):
        """
        Perform a forward pass on the model, calculating logits and loss.
        
        Args:
        x (Tensor): Inputs to the model.
        y (Tensor): True labels for input data.
        
        Returns:
        tuple: A tuple containing logits from the model and the calculated loss.
        """
        logits = self.model(x)
        loss = self._calculate_loss(logits, y)
        return logits, loss

    def _calculate_loss(self, logits, labels):
        """
        Calculate the loss using the logits and labels.
        
        Args:
        logits (Tensor): Logits returned by the model.
        labels (Tensor): True labels for the inputs.
        
        Returns:
        Tensor: The loss value.
        """
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
