from data.data_module import WSIMILDataModule
from modules.sl_lit import WSIClassificationModule
import lightning.pytorch as pl

csv_path = "./src/blca_k_fold.csv"      # or .xlsx if you use read_excel
feature_dir = "../BLCA_AI/combined_features/wsi_processed_no_artifacts/features_uni_v1" 
id_col = "De ID"
target_col = "Binary WHO 2022"
current_fold = 0

dm = WSIMILDataModule(
    csv_path=csv_path,
    feature_dir=feature_dir,
    id_col=id_col,
    target_col=target_col,
    batch_size=1,
    num_workers=4,
    precision=16,
    current_fold=current_fold,
    return_key=False,
)
dm.setup("fit")

n_classes = dm.num_classes
D = dm.dim_features
class_weights = dm.class_weight_tensor

model = WSIClassificationModule(
    mil="ABMIL",                # or "TransMIL", "CLAM", ...
    n_classes=n_classes,
    feature_dim=D,
    lr=3e-4,
    l2_reg=1e-3,
    max_epochs=60,
    warmup_epochs=5,
    scheduler="cosine",
    step_on_epochs=True,
    class_weights=class_weights,
)

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(
        monitor="val/roc_auc" if n_classes <= 2 else "val/balanced_accuracy",
        mode="max",
        patience=10,
    ),
    ModelCheckpoint(
        monitor="val/roc_auc" if n_classes <= 2 else "val/balanced_accuracy",
        mode="max",
        save_top_k=1,
        filename=f"fold{current_fold}" + "-{epoch}-{step}",
    ),
]

trainer = pl.Trainer(
    max_epochs=60,
    precision="16-mixed",      # Lightning handles AMP for you
        callbacks=callbacks,
        log_every_n_steps=10,
    )

trainer.fit(model, datamodule=dm)
