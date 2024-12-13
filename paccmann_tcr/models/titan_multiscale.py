from typing import Dict, Tuple
import torch
import torch.nn as nn
from paccmann_predictor.models import MODEL_FACTORY

class MultiScaleModule(nn.Module):
    def __init__(self, input_dim: int, scales: list = [1, 3, 5]):
        """
        Multi-scale feature extraction module
        
        Args:
            input_dim: Input feature dimension
            scales: Kernel sizes for different scales
        """
        super().__init__()
        self.scales = scales
        
        # Create parallel convolution paths for each scale
        self.conv_paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=scale, padding=scale//2),
                nn.BatchNorm1d(input_dim),
                nn.ReLU()
            ) for scale in scales
        ])
        
        # Scale attention
        self.scale_attention = nn.Sequential(
            nn.Linear(input_dim * len(scales), input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, len(scales)),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_dim, seq_length)
        batch_size = x.size(0)
        
        # Extract features at different scales
        multi_scale_features = []
        for conv in self.conv_paths:
            features = conv(x)  # (batch_size, input_dim, seq_length)
            multi_scale_features.append(features)
            
        # Concatenate features for attention
        concat_features = torch.cat([f.mean(dim=2) for f in multi_scale_features], dim=1)
        
        # Calculate attention weights
        attention = self.scale_attention(concat_features)  # (batch_size, num_scales)
        
        # Apply attention weights
        weighted_features = []
        for i, features in enumerate(multi_scale_features):
            weighted = features * attention[:, i].view(batch_size, 1, 1)
            weighted_features.append(weighted)
            
        # Combine features
        output = sum(weighted_features)
        return output

class TITANMultiScale(nn.Module):
    """TITAN model with multi-scale feature fusion"""
    
    def __init__(self, params: Dict):
        super().__init__()
        
        # Initialize base TITAN model components
        self.base_model = MODEL_FACTORY[params.get('model_fn', 'bimodal_mca')](params)
        
        # Add multi-scale modules
        hidden_dim = params.get('hidden_size', 512)
        self.tcr_multiscale = MultiScaleModule(hidden_dim)
        self.epitope_multiscale = MultiScaleModule(hidden_dim)
        
        # Adjust final layers for multi-scale features
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, tcr_input: torch.Tensor, epitope_input: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Get base embeddings and attention 
        base_output, pred_dict = self.base_model(tcr_input, epitope_input)
        
        # Extract intermediate representations
        tcr_features = pred_dict['tcr_features']  # (batch_size, hidden_dim, seq_length)
        epitope_features = pred_dict['epitope_features']
        
        # Apply multi-scale processing
        tcr_ms = self.tcr_multiscale(tcr_features)
        epitope_ms = self.epitope_multiscale(epitope_features)
        
        # Combine features
        combined_features = torch.cat([
            tcr_ms.mean(dim=2),  # Global average pooling
            epitope_ms.mean(dim=2)
        ], dim=1)
        
        # Final prediction
        output = self.fusion_layer(combined_features)
        
        # Update prediction dictionary
        pred_dict.update({
            'tcr_multiscale': tcr_ms,
            'epitope_multiscale': epitope_ms
        })
        
        return output, pred_dict

# Register the multi-scale model
MODEL_FACTORY['bimodal_mca_multiscale'] = TITANMultiScale
