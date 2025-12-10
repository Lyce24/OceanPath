import torch
import torch.nn as nn

from models.base_modules import create_mlp, GlobalAttention, GlobalGatedAttention, MIL
from models.ABMIL import ABMIL
from models.TransMIL import TransMIL
from models.DSMIL import DSMIL
from models.CLAM import CLAMSB
from models.WIKGMIL import WIKGMIL
from models.StaticMIL import StaticMIL
from models.CHAMP import CHAMP

class WSIModel(nn.Module):
    def __init__(self, 
                 input_feature_dim,
                 encoder_type='ABMIL',
                 num_fc_layers=1, 
                 head_dim=512,
                 head_dropout=0.25, 
                 ds_dropout=0.2,
                 hidden_dim=128, 
                 simple_mlp=False, 
                 n_classes=1,
                 encoder_attrs = {},
                 freeze_encoder=False):
        super().__init__()

        self.encoder_type = encoder_type
        if encoder_type == 'ABMIL':
            self.feature_encoder = ABMIL(
                in_dim=input_feature_dim,
                embed_dim=head_dim,
                num_fc_layers=num_fc_layers,
                dropout=head_dropout,
                attn_dim=encoder_attrs.get("attn_dim", 384),
                gate=encoder_attrs.get("gate", True),
                num_classes=0
            )
        elif encoder_type == 'TransMIL':
            self.feature_encoder = TransMIL(
                in_dim=input_feature_dim,
                embed_dim=head_dim,
                num_fc_layers=num_fc_layers,
                dropout=head_dropout,
                num_attention_layers=encoder_attrs.get("num_attention_layers", 2),
                num_classes=0,
                num_heads=encoder_attrs.get("num_heads", 4)
            )
        # elif encoder_type == 'ILRA':
        #     self.feature_encoder = ILRA(
        #         in_dim=input_feature_dim,
        #         embed_dim=encoder_attrs.get("embed_dim", head_dim),
        #         num_heads=encoder_attrs.get("num_heads", 8),
        #         topk=encoder_attrs.get("topk", 64),
        #         num_attention_layers=encoder_attrs.get("num_attention_layers", 2),
        #         num_classes=0,
        #         ln=encoder_attrs.get("ln", True),
        #         mode=encoder_attrs.get("mode", 'classification')
        #     )
        elif encoder_type == 'CLAM':
            self.feature_encoder = CLAMSB(
                in_dim=input_feature_dim,
                embed_dim=head_dim,
                n_fc_layers=num_fc_layers,
                dropout=head_dropout,
                gate=encoder_attrs.get("gate", True),
                attention_dim=encoder_attrs.get("attn_dim", 384),
                num_classes=n_classes,
                k_sample=encoder_attrs.get("k_sample", 8),
                subtyping=encoder_attrs.get("subtyping", False),
                instance_loss_fn=encoder_attrs.get("instance_loss_fn", 'svm'),
                bag_weight=encoder_attrs.get("bag_weight", 0.7)
            )
            # Instance Classifiers: One classifier per class for instance-level prediction
            instance_classifiers = [nn.Linear(head_dim, 2) for _ in range(n_classes)]  # Binary classifier per class
            self.instance_classifiers = nn.ModuleList(instance_classifiers)

        elif encoder_type == 'DSMIL':
            self.feature_encoder = DSMIL(
                in_dim=input_feature_dim,
                embed_dim=head_dim,
                num_fc_layers=num_fc_layers,
                dropout=head_dropout,
                attn_dim=encoder_attrs.get("attn_dim", 384),
                dropout_v=encoder_attrs.get("dropout_v", 0.0),
                num_classes=n_classes,
                layernorm=encoder_attrs.get("layernorm", True)
            )
        elif encoder_type == 'WIKGMIL':
            self.feature_encoder = WIKGMIL(
                in_dim=input_feature_dim,
                embed_dim=head_dim,
                num_classes=n_classes,
                agg_type=encoder_attrs.get("agg_type", 'bi-interaction'),
                pool=encoder_attrs.get("pool", 'attn'),
                dropout=head_dropout,
                topk=encoder_attrs.get("topk", 4)
            )
        elif encoder_type == 'CHAMP':
            self.feature_encoder = CHAMP(
                in_dim=input_feature_dim,
                embed_dim=head_dim,
                num_classes=n_classes,
                attn_dim=encoder_attrs.get("attn_dim", 384),
                dropout=head_dropout,
                gate=encoder_attrs.get("gate", True),
                target_prevalence=encoder_attrs.get("target_prevalence", 0.1),
                base_tau=encoder_attrs.get("base_tau", 1.0),
                gamma=encoder_attrs.get("gamma", 1.0),
                top_frac=encoder_attrs.get("top_frac", 0.1),
                bottom_frac=encoder_attrs.get("bottom_frac", 0.1),
            )
        # elif encoder_type == 'RRTMIL':
        #     self.feature_encoder = RRTMIL(
        #         in_dim=input_feature_dim,
        #         embed_dim=head_dim,
        #         mlp_dim=encoder_attrs.get("mlp_dim", 512),
        #         act=encoder_attrs.get("act", 'relu'),
        #         dropout=head_dropout,
        #         pos_pos=encoder_attrs.get("pos_pos", 0),
        #         pos=encoder_attrs.get("pos", 'ppeg'),
        #         peg_k=encoder_attrs.get("peg_k", 7),
        #         attn=encoder_attrs.get("attn", 'ntrans'),
        #         pool=encoder_attrs.get("pool", 'attn'),
        #         region_num=encoder_attrs.get("region_num", 8),
        #         n_layers=encoder_attrs.get("n_layers", 2),
        #         n_heads=encoder_attrs.get("n_heads", 8),
        #         multi_scale=encoder_attrs.get("multi_scale", False),
        #         drop_path=encoder_attrs.get("drop_path", 0.0),
        #         da_act=encoder_attrs.get("da_act", 'relu'),
        #         trans_dropout=encoder_attrs.get("trans_dropout", 0.1),
        #         ffn=encoder_attrs.get("ffn", False),
        #         ffn_act=encoder_attrs.get("ffn_act", 'gelu'),
        #         mlp_ratio=encoder_attrs.get("mlp_ratio", 4.0),
        #         da_gated=encoder_attrs.get("da_gated", False),
        #         da_bias=encoder_attrs.get("da_bias", False),
        #         da_dropout=encoder_attrs.get("da_dropout", False),
        #         trans_dim=encoder_attrs.get("trans_dim", 64),
        #         n_cycle=encoder_attrs.get("n_cycle", 1),
        #         epeg=encoder_attrs.get("epeg", False),
        #         min_region_num=encoder_attrs.get("min_region_num", 0),
        #         qkv_bias=encoder_attrs.get("qkv_bias", True),
        #         shift_size=encoder_attrs.get("shift_size", False),
        #         no_norm=encoder_attrs.get("no_norm", False)
        #     )
        elif encoder_type == 'Mean' or encoder_type == 'Max':
            self.feature_encoder = StaticMIL(
                in_dim=input_feature_dim,
                embed_dim=head_dim,
                num_fc_layers=num_fc_layers,
                dropout=head_dropout,
                num_classes=0,
                method=encoder_type
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.feature_encoder.parameters():
                param.requires_grad = False

        # one head per task
        self.simple_mlp = simple_mlp
        self.n_classes = n_classes

        if self.simple_mlp:
            self.heads = create_mlp(
                in_dim=head_dim if encoder_type not in ['Mean', 'Max'] else input_feature_dim,
                hid_dims=[hidden_dim],
                dropout=ds_dropout,
                out_dim=n_classes,
                end_with_fc=True,
                end_with_dropout=False,
                bias=True
            )
        else:
            self.heads = nn.Linear(head_dim, n_classes)
        
        if self.encoder_type == "DSMIL":
            self.heads = nn.Conv1d(n_classes, n_classes, kernel_size=head_dim)
        
        self.initialize_weights()

    def forward_features(self, x, return_raw_attention=False):
        features, log_dict = self.feature_encoder.forward_features(x, return_attention=return_raw_attention) # [B, D] or [B, N, D]
        return features, log_dict

    def forward(self, x, return_raw_attention=False, labels=None):
        features, log_dict = self.feature_encoder.forward_features(x, return_attention=return_raw_attention)

        logits = self.heads(features)  # [B, n_classes] or [B, 1]
        
        if self.encoder_type == "DSMIL":
            logits = logits.squeeze(dim=-1)  # [B, n_classes] if Conv1d head
            if 'instance_classes' in log_dict and labels is not None:
                max_prediction, _ = torch.max(log_dict['instance_classes'], 1)
                max_loss = torch.nn.CrossEntropyLoss()(max_prediction, labels)
                log_dict['instance_loss'] = max_loss.item() if max_loss is not None else -1
            
        if self.encoder_type == "CLAM" and "instance_feats" in log_dict and labels is not None and "attention" in log_dict:
            instance_loss = self.feature_encoder.forward_instance_heads(log_dict["instance_feats"], log_dict['attention'], labels)
            log_dict['instance_loss'] = instance_loss if instance_loss is not None else -1

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
    for encoder_type in ['Mean', 'Max', 'ABMIL', 'DSMIL', 'TransMIL', 'CHAMP', 'CLAM']:
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
                    if model.encoder_type == "CLAM" or model.encoder_type == "DSMIL":
                        outputs, log_dict = model(inputs, return_raw_attention=True, labels=labels)
                    else:
                        outputs, log_dict = model(inputs, return_raw_attention=True)
                        
                    if epoch == 0 and i == 0:
                        print(f"Outputs shape: {outputs.shape}, Attention shape: {log_dict['attention'].shape if log_dict['attention'] is not None else 'N/A'}")

            loss = loss_fn(outputs, labels)
            if model.encoder_type == "CLAM" and 'instance_loss' in log_dict and log_dict['instance_loss'] != -1:
                loss = 0.7 * loss + 0.3 * log_dict['instance_loss']
            elif model.encoder_type == "DSMIL" and 'instance_loss' in log_dict and log_dict['instance_loss'] != -1:
                loss = 0.5 * loss + 0.5 * log_dict['instance_loss']
            
            scaler.scale(loss).backward()

            scaler.step(opt)
            scaler.update()
            
        model.initialize_weights()  # Reinitialize weights after training