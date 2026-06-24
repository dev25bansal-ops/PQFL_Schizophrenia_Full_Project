"""PQFL Flower client for federated quantum learning.

Implements the PQFLClient that:
1. Receives shared parameters from the Flower server
2. Retains personal (local) parameters
3. Trains locally using AdamW with cosine annealing
4. Returns only shared parameter updates

FedPer implementation follows Arivazhagan et al. (2019):
- Shared layers: classical encoder + VQC base (federated)
- Personal layers: VQC personalization + classification head (local)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class PQFLClient:
    """Personalized Quantum Federated Learning client.
    
    Each client represents a site in the federation and:
    - Maintains a local HybridVQC model
    - Splits parameters into shared (federated) and personal (local)
    - Performs local training with the site's data
    - Reports only shared parameter updates to the server
    
    Args:
        model: HybridVQC model instance.
        train_loader: Training DataLoader for this site.
        val_loader: Optional validation DataLoader.
        site_id: Numeric site identifier.
        site_name: Human-readable site name.
        local_epochs: Number of local training epochs per round.
        learning_rate: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for regularization.
        fedprox_mu: FedProx proximal term coefficient. 0 = no proximal term.
        gradient_clip: Maximum gradient norm for clipping.
        device: PyTorch device ("cpu" or "cuda").
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        site_id: int = 0,
        site_name: str = "Unknown",
        local_epochs: int = 5,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        fedprox_mu: float = 0.0,
        gradient_clip: float = 1.0,
        label_smoothing: float = 0.0,
        device: str = "cpu",
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.site_id = site_id
        self.site_name = site_name
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.fedprox_mu = fedprox_mu
        self.gradient_clip = gradient_clip
        self.label_smoothing = label_smoothing
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Loss function — supports both binary (BCE) and multi-class (CE) via class_weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device),
                label_smoothing=label_smoothing,
            )
            self._class_weights_set = True
            logger.info(f"Site {site_name}: using provided class weights: {class_weights.tolist()}")
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self._class_weights_set = False
        
        # Tracking
        self._history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        
        # Store global parameters for FedProx
        self._global_params = None
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get shared parameters to send to server.
        
        Only shared (federated) parameters are returned.
        Personal parameters stay local.
        
        Returns:
            List of numpy arrays (shared parameters).
        """
        from .parameter_utils import get_shared_parameters
        return get_shared_parameters(self.model)
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set shared parameters received from server.
        
        Personal parameters are preserved (not overwritten).
        
        Args:
            parameters: Shared parameters from the Flower server.
        """
        from .parameter_utils import set_shared_parameters
        set_shared_parameters(self.model, parameters)
        
        # Store for FedProx
        if self.fedprox_mu > 0:
            self._global_params = [p.copy() for p in parameters]
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Optional[Dict] = None,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Perform local training.
        
        1. Receive shared parameters from server
        2. Train locally for local_epochs
        3. Return updated shared parameters
        
        Args:
            parameters: Shared parameters from server.
            config: Optional configuration from server.
        
        Returns:
            Tuple of (updated_shared_parameters, num_samples, metrics).
        """
        # Set received parameters
        self.set_parameters(parameters)
        
        # Configure training
        lr = config.get("learning_rate", self.learning_rate) if config else self.learning_rate
        epochs = config.get("local_epochs", self.local_epochs) if config else self.local_epochs
        
        # Setup optimizer with cosine annealing
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        # Set class weights on first fit call (need data to compute)
        if not self._class_weights_set and hasattr(self.train_loader.dataset, 'get_class_weights'):
            try:
                weights = self.train_loader.dataset.get_class_weights()
                self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device), label_smoothing=self.label_smoothing)
                self._class_weights_set = True
                logger.info(f"Site {self.site_name}: class weights set: HC={weights[0]:.3f}, SZ={weights[1]:.3f}")
            except Exception:
                pass
        
        # Training loop
        self.model.train()
        epoch_losses = []
        epoch_accs = []
        
        for epoch in range(epochs):
            batch_losses = []
            correct = 0
            total = 0
            
            for batch in self.train_loader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    x = batch["tangent_features"].to(self.device)
                    y = batch["label"].to(self.device)
                    fdt = batch.get("fdt_features")
                    if fdt is not None:
                        fdt = fdt.to(self.device)
                else:
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    fdt = None
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(x, fdt_features=fdt)
                loss = self.criterion(logits, y)
                
                # FedProx regularization
                if self.fedprox_mu > 0 and self._global_params is not None:
                    prox_loss = self._compute_proximal_term()
                    loss = loss + (self.fedprox_mu / 2) * prox_loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                optimizer.step()
                
                batch_losses.append(loss.item())
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            
            scheduler.step()
            
            epoch_loss = np.mean(batch_losses)
            epoch_acc = correct / total if total > 0 else 0
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc)
        
        # Track history
        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accs)
        self._history["train_loss"].append(avg_loss)
        self._history["train_acc"].append(avg_acc)
        
        # Validation
        val_metrics = {}
        if self.val_loader is not None:
            val_metrics = self.evaluate()
        
        # Return updated shared parameters
        updated_params = self.get_parameters()
        num_samples = len(self.train_loader.dataset)
        
        metrics = {
            "site_id": self.site_id,
            "site_name": self.site_name,
            "train_loss": avg_loss,
            "train_accuracy": avg_acc,
            **val_metrics,
        }
        
        # Log both train and val metrics
        val_ba = val_metrics.get("balanced_accuracy", 0)
        val_auc = val_metrics.get("auc_roc", "N/A")
        val_str = f", val_BA={val_ba:.4f}"
        if isinstance(val_auc, float):
            val_str += f", val_AUC={val_auc:.4f}"
        logger.info(
            f"Site {self.site_name} (id={self.site_id}): "
            f"train_loss={avg_loss:.4f}, train_acc={avg_acc:.4f}{val_str}"
        )
        
        return updated_params, num_samples, metrics
    
    def evaluate(
        self,
        parameters: Optional[List[np.ndarray]] = None,
        config: Optional[Dict] = None,
    ) -> Dict:
        """Evaluate the model on validation data.
        
        Args:
            parameters: Optional shared parameters to set before evaluation.
            config: Optional configuration.
        
        Returns:
            Dictionary with evaluation metrics.
        """
        if parameters is not None:
            self.set_parameters(parameters)
        
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0
        n_batches = 0
        n_classes = 2  # default; updated from first batch logits
        
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, dict):
                    x = batch["tangent_features"].to(self.device)
                    y = batch["label"].to(self.device)
                    fdt = batch.get("fdt_features")
                    if fdt is not None:
                        fdt = fdt.to(self.device)
                else:
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    fdt = None
                
                logits = self.model(x, fdt_features=fdt)
                loss = self.criterion(logits, y)
                
                total_loss += loss.item()
                n_batches += 1
                
                # Track n_classes from logits shape
                if logits.dim() >= 2 and logits.shape[1] > 1:
                    n_classes = logits.shape[1]
                
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                # Save full probability matrix (supports both binary and multi-class)
                all_probs.append(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        # Concatenate per-batch prob matrices → (n_samples, n_classes)
        if all_probs:
            all_probs = np.concatenate(all_probs, axis=0)
            # For backward compatibility with binary case: if shape is (N, 2),
            # also expose just the positive-class column via y_prob[:, 1]
        else:
            all_probs = None
        
        # Compute metrics with probabilities for AUC
        from ..evaluation.metrics import compute_classification_metrics
        metrics = compute_classification_metrics(
            all_labels, all_preds, y_prob=all_probs, n_classes=n_classes,
        )
        metrics["val_loss"] = total_loss / max(n_batches, 1)
        
        self._history["val_loss"].append(metrics["val_loss"])
        self._history["val_acc"].append(metrics.get("accuracy", 0))
        
        return metrics
    
    def _compute_proximal_term(self) -> torch.Tensor:
        """Compute FedProx proximal term: ||θ - θ_global||^2.
        
        Penalizes divergence from the global model.
        
        Returns:
            Proximal loss scalar.
        """
        if self._global_params is None:
            return torch.tensor(0.0)
        
        prox_loss = torch.tensor(0.0, device=self.device)
        current_params = self.get_parameters()
        
        for current, global_p in zip(current_params, self._global_params):
            current_t = torch.tensor(current, device=self.device)
            global_t = torch.tensor(global_p, device=self.device)
            prox_loss += torch.sum((current_t - global_t) ** 2)
        
        return prox_loss
    
    def get_history(self) -> Dict:
        """Return training history."""
        return self._history.copy()
