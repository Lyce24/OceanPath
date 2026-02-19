from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from oceanpath.eval.metrics import classification_metrics
from oceanpath.modules.wsi_module import WSIClassificationModule


def _load_feature_tensor(path: str) -> torch.Tensor:
	feature_path = Path(path)
	if not feature_path.exists():
		raise FileNotFoundError(f"Feature file not found: {feature_path}")

	if feature_path.suffix in {".pt", ".pth"}:
		x = torch.load(feature_path, map_location="cpu")
	elif feature_path.suffix == ".npy":
		x = torch.from_numpy(np.load(feature_path))
	else:
		raise ValueError(f"Unsupported feature extension: {feature_path.suffix}")

	if not torch.is_tensor(x):
		x = torch.as_tensor(x)
	return x.float()


class FeatureBagDataset(Dataset):
	def __init__(self, df: pd.DataFrame):
		self.df = df.reset_index(drop=True)

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		row = self.df.iloc[idx]
		x = _load_feature_tensor(str(row["feature_path"]))
		y = torch.tensor(int(row["label"]), dtype=torch.long)
		return x, y


def _collate_single_bag(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
	xs, ys = zip(*batch)
	if len(xs) != 1:
		raise ValueError("Variable-length bag collation currently expects batch_size=1")
	return xs[0].unsqueeze(0), torch.stack(ys)


@dataclass
class CVConfig:
	csv_path: str
	output_dir: str
	feature_dim: int
	n_classes: int
	mil: str = "ABMIL"
	mil_attrs: Dict[str, Any] = field(default_factory=dict)
	mode: str = "ft"
	encoder_weights_path: Optional[str] = None
	batch_size: int = 1
	num_workers: int = 4
	max_epochs: int = 40
	patience: int = 6
	seed: int = 42
	precision: str = "16-mixed"
	encoder_lr: float = 3e-5
	head_lr: float = 3e-4
	encoder_wd: float = 1e-3
	head_wd: float = 1e-3


class RunCV:
	def __init__(self, config: CVConfig):
		self.config = config
		self.output_dir = Path(config.output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)

		self.df = pd.read_csv(config.csv_path)
		required_cols = {"feature_path", "label", "k_fold"}
		missing = required_cols - set(self.df.columns)
		if missing:
			raise ValueError(f"CSV missing required columns: {sorted(missing)}")

	def _dataloader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
		return DataLoader(
			FeatureBagDataset(df),
			batch_size=self.config.batch_size,
			shuffle=shuffle,
			num_workers=self.config.num_workers,
			collate_fn=_collate_single_bag,
			pin_memory=torch.cuda.is_available(),
		)

	def _class_weights(self, labels: np.ndarray) -> torch.Tensor:
		classes = np.unique(labels)
		weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
		out = np.ones(self.config.n_classes if self.config.n_classes > 1 else 2, dtype=np.float32)
		out[classes.astype(int)] = weights.astype(np.float32)
		return torch.from_numpy(out)

	@torch.no_grad()
	def _predict(self, module: WSIClassificationModule, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
		module.eval()
		module.freeze()
		device = module.device

		y_true_list: List[np.ndarray] = []
		y_prob_list: List[np.ndarray] = []
		for x, y in loader:
			x = x.to(device)
			logits, _ = module(x)

			if module.n_classes == 1:
				probs = torch.sigmoid(logits).detach().cpu().numpy()
			elif module.n_classes == 2:
				probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
			else:
				probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

			y_true_list.append(y.numpy())
			y_prob_list.append(probs)

		y_true = np.concatenate(y_true_list)
		if module.n_classes > 2:
			raise NotImplementedError("Current evaluation helper supports binary tasks only")
		y_prob = np.concatenate(y_prob_list)
		return y_true, y_prob

	def run(self) -> Dict[str, Any]:
		pl.seed_everything(self.config.seed, workers=True)

		df_train = self.df[self.df["k_fold"] >= 0].copy()
		df_test = self.df[self.df["k_fold"] == -1].copy()
		folds = sorted(df_train["k_fold"].unique().tolist())

		fold_summaries: List[Dict[str, Any]] = []
		test_predictions: List[pd.DataFrame] = []

		for fold in folds:
			train_df = df_train[df_train["k_fold"] != fold].copy()
			val_df = df_train[df_train["k_fold"] == fold].copy()

			train_loader = self._dataloader(train_df, shuffle=True)
			val_loader = self._dataloader(val_df, shuffle=False)
			test_loader = self._dataloader(df_test, shuffle=False) if len(df_test) > 0 else None

			class_weights = self._class_weights(train_df["label"].to_numpy())

			module = WSIClassificationModule(
				mil=self.config.mil,
				n_classes=self.config.n_classes,
				feature_dim=self.config.feature_dim,
				mil_attrs=self.config.mil_attrs,
				mode=self.config.mode,
				encoder_weights_path=self.config.encoder_weights_path,
				encoder_lr=self.config.encoder_lr,
				head_lr=self.config.head_lr,
				encoder_wd=self.config.encoder_wd,
				head_wd=self.config.head_wd,
				max_epochs=self.config.max_epochs,
				class_weights=class_weights,
			)

			fold_dir = self.output_dir / f"fold_{fold}"
			fold_dir.mkdir(parents=True, exist_ok=True)

			ckpt = ModelCheckpoint(
				dirpath=str(fold_dir),
				filename="best",
				monitor="val/roc_auc",
				mode="max",
				save_top_k=1,
			)
			early = EarlyStopping(monitor="val/roc_auc", mode="max", patience=self.config.patience)

			trainer = pl.Trainer(
				max_epochs=self.config.max_epochs,
				accelerator="auto",
				devices=1,
				precision=self.config.precision,
				callbacks=[ckpt, early],
				logger=False,
				enable_progress_bar=True,
			)

			trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

			best_ckpt = ckpt.best_model_path
			best_module = WSIClassificationModule.load_from_checkpoint(best_ckpt, map_location="cpu")

			y_val_true, y_val_prob = self._predict(best_module, val_loader)
			val_metrics = classification_metrics(y_val_true, y_val_prob)

			fold_summary: Dict[str, Any] = {
				"fold": int(fold),
				"checkpoint": best_ckpt,
				"val_metrics": val_metrics,
			}

			if test_loader is not None:
				y_test_true, y_test_prob = self._predict(best_module, test_loader)
				test_metrics = classification_metrics(y_test_true, y_test_prob)
				fold_summary["test_metrics"] = test_metrics

				test_predictions.append(
					pd.DataFrame(
						{
							"fold": int(fold),
							"y_true": y_test_true,
							"y_prob": y_test_prob,
						}
					)
				)

			fold_summaries.append(fold_summary)

		result: Dict[str, Any] = {
			"config": asdict(self.config),
			"folds": fold_summaries,
		}

		if test_predictions:
			merged = pd.concat(test_predictions, ignore_index=True)
			ensemble = (
				merged.groupby(merged.index % len(df_test))
				.agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
				.reset_index(drop=True)
			)

			ensemble_metrics = classification_metrics(
				ensemble["y_true"].to_numpy(),
				ensemble["y_prob"].to_numpy(),
			)
			result["ensemble_test_metrics"] = ensemble_metrics

			ensemble_path = self.output_dir / "ensemble_test_predictions.csv"
			ensemble.to_csv(ensemble_path, index=False)
			result["ensemble_predictions_csv"] = str(ensemble_path)

		with open(self.output_dir / "cv_results.json", "w", encoding="utf-8") as f:
			json.dump(result, f, indent=2)

		return result

