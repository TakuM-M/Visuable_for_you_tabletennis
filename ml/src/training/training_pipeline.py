import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from src.models.play_classifier_lstm import PlayClassifierLSTM, PlayClassifierCNNLSTM
from src.datasets import MultiCSVPoseDataset, collate_fn
from src.training.config import TrainingPipelineConfig
from src.training.exceptions import DataInputError, ExportError


class TrainingPipeline:
    """プレー検知モデルの学習パイプライン"""

    def __init__(self, config: TrainingPipelineConfig):
        """
        初期化

        Args:
            config: 学習パイプライン設定
        """
        self.config = config
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.writer = None
        self.train_loader = None
        self.val_loader = None
        self.output_dir = None

        # 学習履歴
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }

        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.early_stopping_counter = 0
        
    def run(self) -> Dict[str, Any]:
        """
        学習パイプラインを実行

        Returns:
            学習結果の統計情報

        Raises:
            DataInputError: データの読み込みに失敗
            ExportError: モデル保存に失敗
        """
        print(f"\n{'='*70}")
        print("学習パイプライン開始")
        print(f"{'='*70}\n")

        # 初期化
        self._setup_device()
        self._setup_output_dir()
        self._save_config()
        self._setup_dataloaders()
        self._setup_model()
        self._setup_optimizer()
        self._setup_criterion()
        self._setup_tensorboard()

        # 学習情報を表示
        self._print_training_info()

        # 学習実行
        self._train_loop()

        # 最終モデル保存
        self._save_checkpoint(self.config.training.epochs, "final_model.pth")
        self._save_history()

        # 結果
        results = {
            'best_val_f1': self.best_val_f1,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.config.training.epochs,
            'output_dir': str(self.output_dir),
            'best_model_path': str(self.output_dir / 'best_model.pth'),
            'final_model_path': str(self.output_dir / 'final_model.pth'),
            'history_path': str(self.output_dir / 'training_history.json')
        }

        print(f"\n{'='*70}")
        print("学習完了")
        print(f"  Best Val F1: {self.best_val_f1:.4f}")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")
        print(f"  Best Model: {results['best_model_path']}")
        print(f"  Final Model: {results['final_model_path']}")
        print(f"{'='*70}\n")

        return results

    def _setup_device(self):
        """デバイスの設定"""
        device_name = self.config.training.device

        if device_name == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA not available, using CPU")
            device_name = 'cpu'
        elif device_name == 'mps' and not torch.backends.mps.is_available():
            print("警告: MPS not available, using CPU")
            device_name = 'cpu'

        self.device = torch.device(device_name)
        print(f"デバイス: {self.device}\n")

    def _setup_output_dir(self):
        """出力ディレクトリの作成"""
        base_dir = Path(self.config.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = base_dir / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"出力ディレクトリ: {self.output_dir}\n")

    def _save_config(self):
        """設定をJSONで保存"""
        config_dict = {
            'model': {
                'model_type': self.config.model.model_type,
                'hidden_size': self.config.model.hidden_size,
                'num_layers': self.config.model.num_layers,
                'dropout': self.config.model.dropout,
                'use_attention': self.config.model.use_attention,
                'cnn_channels': self.config.model.cnn_channels
            },
            'dataset': {
                'train_data_dirs': self.config.dataset.train_data_dirs,
                'val_data_dirs': self.config.dataset.val_data_dirs,
                'csv_filename': self.config.dataset.csv_filename,
                'label_filename': self.config.dataset.label_filename,
                'sequence_length': self.config.dataset.sequence_length,
                'stride': self.config.dataset.stride,
                'batch_size': self.config.dataset.batch_size,
                'num_workers': self.config.dataset.num_workers,
                'use_motion_features': self.config.dataset.use_motion_features
            },
            'optimizer': {
                'learning_rate': self.config.optimizer.learning_rate,
                'weight_decay': self.config.optimizer.weight_decay,
                'scheduler_patience': self.config.optimizer.scheduler_patience,
                'scheduler_factor': self.config.optimizer.scheduler_factor,
                'scheduler_min_lr': self.config.optimizer.scheduler_min_lr
            },
            'training': {
                'epochs': self.config.training.epochs,
                'save_every': self.config.training.save_every,
                'device': self.config.training.device,
                'early_stopping_patience': self.config.training.early_stopping_patience
            }
        }

        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"設定保存: {config_path}\n")

    def _setup_dataloaders(self):
        """データローダーの作成（複数CSV対応）"""
        print("データセット読み込み中...")

        # 訓練データ（複数CSV）
        try:
            train_dataset = MultiCSVPoseDataset.from_directories(
                data_dirs=self.config.dataset.train_data_dirs,
                csv_filename=self.config.dataset.csv_filename,
                label_filename=self.config.dataset.label_filename,
                sequence_length=self.config.dataset.sequence_length,
                stride=self.config.dataset.stride,
                use_motion_features=self.config.dataset.use_motion_features
            )
            print(f"  訓練データ: {len(train_dataset)} シーケンス")
        except Exception as e:
            raise DataInputError(
                str(self.config.dataset.train_data_dirs),
                f"訓練データの読み込みに失敗: {str(e)}"
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=True,
            num_workers=self.config.dataset.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if self.config.training.device == 'cuda' else False
        )

        # 検証データ（複数CSV）
        if self.config.dataset.val_data_dirs and len(self.config.dataset.val_data_dirs) > 0:
            try:
                val_dataset = MultiCSVPoseDataset.from_directories(
                    data_dirs=self.config.dataset.val_data_dirs,
                    csv_filename=self.config.dataset.csv_filename,
                    label_filename=self.config.dataset.label_filename,
                    sequence_length=self.config.dataset.sequence_length,
                    stride=self.config.dataset.stride,
                    use_motion_features=self.config.dataset.use_motion_features
                )
                print(f"  検証データ: {len(val_dataset)} シーケンス")

                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.dataset.batch_size,
                    shuffle=False,
                    num_workers=self.config.dataset.num_workers,
                    collate_fn=collate_fn,
                    pin_memory=True if self.config.training.device == 'cuda' else False
                )
            except Exception as e:
                raise DataInputError(
                    str(self.config.dataset.val_data_dirs),
                    f"検証データの読み込みに失敗: {str(e)}"
                )
        else:
            self.val_loader = None
            print("  検証データ: なし")

        print()

    def _setup_model(self):
        """モデルの作成"""
        print("モデル作成���...")

        # 特徴量次元: 座標のみ=34, 速度・加速度追加=102
        input_size = 102 if self.config.dataset.use_motion_features else 34

        if self.config.model.model_type == 'lstm':
            self.model = PlayClassifierLSTM(
                input_size=input_size,
                hidden_size=self.config.model.hidden_size,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout,
                use_attention=self.config.model.use_attention
            )
        elif self.config.model.model_type == 'cnn_lstm':
            self.model = PlayClassifierCNNLSTM(
                input_size=input_size,
                cnn_channels=self.config.model.cnn_channels,
                hidden_size=self.config.model.hidden_size,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model.model_type}")

        self.model = self.model.to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"  モデルタイプ: {self.config.model.model_type}")
        print(f"  パラメータ数: {num_params:,}\n")

    def _setup_optimizer(self):
        """最適化器とスケジューラの設定"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.optimizer.scheduler_factor,
            patience=self.config.optimizer.scheduler_patience,
            min_lr=self.config.optimizer.scheduler_min_lr
        )

    def _setup_criterion(self):
        """損失関数の設定（クラス不均衡対応）"""
        # 訓練データからpos_weightを自動計算
        pos_weight = self._compute_pos_weight()
        if pos_weight is not None:
            print(f"  pos_weight: {pos_weight:.2f}")
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=self.device)
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def _compute_pos_weight(self) -> float:
        """訓練データのクラス不均衡からpos_weightを計算"""
        total_positive = 0
        total_negative = 0
        for _, labels, _ in self.train_loader:
            total_positive += (labels == 1).sum().item()
            total_negative += (labels == 0).sum().item()

        if total_positive == 0:
            print("  警告: 正例が見つかりません。pos_weight=1.0を使用")
            return None

        pos_weight = total_negative / total_positive
        print(f"  クラス分布: positive={total_positive}, negative={total_negative}")
        return pos_weight

    def _setup_tensorboard(self):
        """TensorBoardの設定"""
        if self.config.training.use_tensorboard:
            log_dir = self.output_dir / 'logs'
            self.writer = SummaryWriter(log_dir=str(log_dir))
            print(f"TensorBoard: {log_dir}\n")

    def _print_training_info(self):
        """学習情報を表示"""
        print(f"{'='*70}")
        print("学習設定")
        print(f"{'='*70}")
        print(f"  エポック数: {self.config.training.epochs}")
        print(f"  バッチサイズ: {self.config.dataset.batch_size}")
        print(f"  学習率: {self.config.optimizer.learning_rate}")
        print(f"  シーケンス長: {self.config.dataset.sequence_length}")
        print(f"  ストライド: {self.config.dataset.stride}")
        if self.config.training.early_stopping_patience:
            print(f"  Early Stopping: {self.config.training.early_stopping_patience} epochs")
        print(f"{'='*70}\n")

    def _train_loop(self):
        """学習メインループ"""
        for epoch in range(1, self.config.training.epochs + 1):
            # 訓練
            train_metrics = self._train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])

            # 検証
            val_metrics = {}
            if self.val_loader:
                val_metrics = self._validate_epoch(epoch)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['val_f1'].append(val_metrics['f1'])

                # スケジューラ更新
                self.scheduler.step(val_metrics['loss'])

            # ログ出力
            self._log_epoch_results(epoch, train_metrics, val_metrics)

            # TensorBoard
            if self.writer:
                self._log_to_tensorboard(epoch, train_metrics, val_metrics)

            # モデル保存
            if epoch % self.config.training.save_every == 0:
                self._save_checkpoint(epoch, f"checkpoint_epoch_{epoch}.pth")

            # ベストモデル保存
            if val_metrics and self._is_best_model(val_metrics):
                self.best_val_f1 = val_metrics['f1']
                self.best_val_loss = val_metrics['loss']
                self._save_checkpoint(epoch, "best_model.pth")
                print(f"  ✓ Best model saved (F1: {self.best_val_f1:.4f})")
                self.early_stopping_counter = 0
            elif val_metrics:
                self.early_stopping_counter += 1

            # Early Stopping
            if self._should_early_stop():
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for features, labels, metadata in pbar:
            # デバイスに転送
            features = features.to(self.device)
            labels = labels.to(self.device)

            # 順伝播
            self.optimizer.zero_grad()
            outputs = self.model(features)
            outputs = outputs.squeeze(-1)

            # 損失計算
            loss = self.criterion(outputs, labels)

            # 逆伝播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 統計（logitsにsigmoidを適用して確率化）
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().detach().numpy()
            labels_np = labels.cpu().detach().numpy()
            all_preds.append(preds.flatten())
            all_labels.append(labels_np.flatten())

            pbar.set_postfix({'loss': loss.item()})

        # エポック全体の統計
        avg_loss = total_loss / len(self.train_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        metrics = self._compute_metrics(all_labels, all_preds)
        metrics['loss'] = avg_loss

        return metrics

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """1エポックの検証"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for features, labels, metadata in pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)

                # 順伝播
                outputs = self.model(features)
                outputs = outputs.squeeze(-1)

                # 損失
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # 統計（logitsにsigmoidを適用して確率化）
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float().cpu().numpy()
                labels_np = labels.cpu().numpy()
                all_preds.append(preds.flatten())
                all_labels.append(labels_np.flatten())

                pbar.set_postfix({'loss': loss.item()})

        # 検証全体の統計
        avg_loss = total_loss / len(self.val_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        metrics = self._compute_metrics(all_labels, all_preds)
        metrics['loss'] = avg_loss

        return metrics

    def _compute_metrics(self, labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
        """評価指標の計算"""
        accuracy = np.mean(labels == preds)

        tp = np.sum((labels == 1) & (preds == 1))
        fp = np.sum((labels == 0) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _log_epoch_results(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """エポック結果をログ出力"""
        print(f"\nEpoch {epoch}/{self.config.training.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        if val_metrics:
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")

    def _log_to_tensorboard(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """TensorBoardにログを記録"""
        self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        self.writer.add_scalar('F1/train', train_metrics['f1'], epoch)

        if val_metrics:
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/val', val_metrics['f1'], epoch)

    def _is_best_model(self, val_metrics: Dict[str, float]) -> bool:
        """ベストモデルかどうか判定"""
        return val_metrics['f1'] > self.best_val_f1

    def _should_early_stop(self) -> bool:
        """Early Stoppingすべきか判定"""
        if self.config.training.early_stopping_patience is None:
            return False
        return self.early_stopping_counter >= self.config.training.early_stopping_patience

    def _save_checkpoint(self, epoch: int, filename: str):
        """チェックポイント保存"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_f1': self.best_val_f1,
            'best_val_loss': self.best_val_loss
        }
        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)

    def _save_history(self):
        """学習履歴を保存"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
