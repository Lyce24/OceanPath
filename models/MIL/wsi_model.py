import torch
import torch.nn as nn

from models.MIL.ABMIL import ABMIL
from models.MIL.TransMIL import TransMIL
# from models.MIL.DSMIL import DSMIL
# from models.MIL.CLAM import CLAMSB
# from models.MIL.WIKGMIL import WIKGMIL
from models.MIL.StaticMIL import StaticMIL
# from models.MIL.CHAMP import CHAMP

class WSIModel(nn.Module):
    def __init__(self, 
                 input_feature_dim,
                 encoder_type='ABMIL',
                 n_classes=1,
                 encoder_attrs = {},
                 encoder_weights_path=None,
                 encoder_weights = None,
                 freeze_encoder=False):
        super().__init__()

        self.encoder_type = encoder_type
        if encoder_type == 'ABMIL':
            self.feature_encoder = ABMIL(
                in_dim=input_feature_dim,
                embed_dim=encoder_attrs.get("embed_dim", 512),
                num_fc_layers=encoder_attrs.get("num_fc_layers", 1),
                dropout=encoder_attrs.get("dropout", 0.25),
                attn_dim=encoder_attrs.get("attn_dim", 384),
                gate=encoder_attrs.get("gate", True),
                num_classes=0
            )
        elif encoder_type == 'TransMIL':
            self.feature_encoder = TransMIL(
                in_dim=input_feature_dim,
                embed_dim=encoder_attrs.get("embed_dim", 512),
                num_fc_layers=encoder_attrs.get("num_fc_layers", 1),
                dropout=encoder_attrs.get("dropout", 0.25),
                num_attention_layers=encoder_attrs.get("num_attention_layers", 2),
                num_heads=encoder_attrs.get("num_heads", 4),
                num_classes=0,
            )
        elif encoder_type == 'Mean' or encoder_type == 'Max':
            self.feature_encoder = StaticMIL(
                in_dim=input_feature_dim,
                embed_dim=encoder_attrs.get("embed_dim", 512),
                num_fc_layers=encoder_attrs.get("num_fc_layers", 1),
                dropout=encoder_attrs.get("dropout", 0.25),
                method=encoder_type,
                num_classes=0,
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        
        if encoder_weights_path is not None:
            print(f"Loading encoder weights from: {encoder_weights_path}")
            
            state_dict = torch.load(encoder_weights_path, map_location='cpu')['state_dict']

            # ---- 1. Filter CXR encoder keys ----
            encoder_state = {k: v for k, v in state_dict.items() if k.startswith("encoder.aggregator.")}

            # ---- 2. Strip the "cxr_encoder." prefix ----
            encoder_state_stripped = {}
            for k, v in encoder_state.items():
                new_key = k.replace("encoder.aggregator.", "")
                encoder_state_stripped[new_key] = v  
            
            missing, unexpected = self.feature_encoder.load_state_dict(encoder_state_stripped, strict=False)
            if len(missing) > 0:
                print(f"Missing keys when loading encoder weights: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected keys when loading encoder weights: {unexpected}")
                
            if len(missing) == 0 and len(unexpected) == 0:
                print("Successfully loaded encoder weights.")
            
        if encoder_weights is not None:
            print(f"Loading encoder weights from provided state_dict")
            missing, unexpected = self.feature_encoder.load_state_dict(encoder_weights, strict=False)
            if len(missing) > 0:
                print(f"Missing keys when loading encoder weights: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected keys when loading encoder weights: {unexpected}")
        
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.feature_encoder.parameters():
                param.requires_grad = False
        print(f"Encoder '{encoder_type}' initialized. Freeze encoder: {self.freeze_encoder}")
        
        # one head per task
        self.n_classes = n_classes
        head_dim = encoder_attrs.get("embed_dim", 512)
        self.heads = nn.Linear(head_dim, n_classes)
        
        if encoder_weights_path is None and encoder_weights is None:
            print("Initializing model weights...")
            self.initialize_weights() # otherwise, assume loaded weights are pre-trained
        else:
            print("Using loaded encoder weights; initializing head weights...")
            for layer in self.heads.modules():
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
                elif isinstance(layer, nn.Conv1d):
                    layer.reset_parameters()

    def forward_features(self, x, return_raw_attention=False):
        features, log_dict = self.feature_encoder.forward_features(x, return_attention=return_raw_attention) # [B, D] or [B, N, D]
        return features, log_dict

    def forward(self, x, return_raw_attention=False, labels=None):
        features, log_dict = self.feature_encoder.forward_features(x, return_attention=return_raw_attention)

        logits = self.heads(features)  # [B, n_classes] or [B, 1]
        
        if self.n_classes == 1:
            logits = logits.squeeze(dim=-1)

        return logits, log_dict

    def initialize_weights(self):
        if not self.freeze_encoder:
            self.feature_encoder.initialize_weights()
            
        for layer in self.heads.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
            elif isinstance(layer, nn.Conv1d):
                layer.reset_parameters()
                    
if __name__ == "__main__":
        # import dataset and dataloader for training testing
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch.amp import autocast
    from torch.amp import GradScaler
    
    # Example usage
    for encoder_type in ['Mean', 'Max', 'ABMIL','TransMIL']:
        print(f"Testing encoder type: {encoder_type}")
        model = WSIModel(input_feature_dim=1024, encoder_type=encoder_type, n_classes=2)

        # Create dummy data
        x = torch.randn(100, 100, 1024)
        
        # convert to float16
        x = x.half()  # Convert to float16 for training

        y = torch.randint(0, 2, (100,))  # Binary classification
        
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        use_amp = True  # Use AMP if available
        scaler = GradScaler(enabled=use_amp)
        
        # Training loop
        model.train()
        for epoch in range(5):  # 5 epochs
            for i, batch in enumerate(dataloader):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                opt.zero_grad()
                with autocast(device_type=device.type, enabled=use_amp):

                    outputs, log_dict = model(inputs, return_raw_attention=True)
                        
                    if epoch == 0 and i == 0:
                        print(f"Outputs shape: {outputs.shape}, Attention shape: {log_dict['attention'].shape if log_dict['attention'] is not None else 'N/A'}")

            loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()

            scaler.step(opt)
            scaler.update()
            
        model.initialize_weights()  # Reinitialize weights after training