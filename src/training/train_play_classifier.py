"""
プレー検知モデルの学習スクリプト

LSTMモデルを使って、プレー中/プレー外を判別するモデルを学習する
"""
import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.play_classifier import PlayClassifierLSTM, PlayClassifierCNNLSTM
from src.dataset.dataset import PoseSequenceDataset, collate_fn


class Trainer:
    """モデル学習用クラス"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        output_dir: str = 'output/training',
        class_weights: list = None
    ):
        """
        初期化

        Args:
            model: 学習するモデル
            train_loader: 訓練データローダー
            val_loader: 検証データローダー
            device: 使用デバイス
            learning_rate: 学習率
            output_dir: 出力ディレクトリ
            class_weights: クラスの重み [非プレー, プレー]
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 損失関数（クラス不均衡に対応）
        if class_weights:
            weights = torch.FloatTensor(class_weights).to(device)
            self.criterion = nn.BCELoss(weight=weights)
        else:
            self.criterion = nn.BCELoss()

        # オプティマイザ
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 学習率スケジューラ
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))

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

    def train_epoch(self, epoch: int) -> dict:
        """1エポックの学習"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (features, labels, metadata) in enumerate(pbar):
            # デバイスに転送
            features = features.to(self.device)
            labels = labels.to(self.device)

            # 順伝播
            self.optimizer.zero_grad()
            outputs = self.model(features)  # (batch, seq, 1)
            outputs = outputs.squeeze(-1)   # (batch, seq)

            # 損失計算
            loss = self.criterion(outputs, labels)

            # 逆伝播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 統計
            total_loss += loss.item()
            preds = (outputs > 0.5).float().cpu().detach().numpy()
            labels_np = labels.cpu().detach().numpy()
            all_preds.append(preds.flatten())
            all_labels.append(labels_np.flatten())

            # プログレスバー更新
            pbar.set_postfix({'loss': loss.item()})

        # エポック全体の統計
        avg_loss = total_loss / len(self.train_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        metrics = self._compute_metrics(all_labels, all_preds)
        metrics['loss'] = avg_loss

        return metrics

    def validate(self, epoch: int) -> dict:
        """検証"""
        if self.val_loader is None:
            return {}

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

                # 統計
                preds = (outputs > 0.5).float().cpu().numpy()
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

    def _compute_metrics(self, labels: np.ndarray, preds: np.ndarray) -> dict:
        """評価指標の計算"""
        # 精度
        accuracy = np.mean(labels == preds)

        # Precision, Recall, F1（プレー中クラスに対して）
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

    def train(self, num_epochs: int, save_every: int = 10):
        """学習メインループ"""
        print(f"\n{'='*60}")
        print(f"学習開始")
        print(f"  エポック数: {num_epochs}")
        print(f"  デバイス: {self.device}")
        print(f"  出力ディレクトリ: {self.output_dir}")
        print(f"{'='*60}\n")

        for epoch in range(1, num_epochs + 1):
            # 訓練
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])

            # 検証
            val_metrics = self.validate(epoch)
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['val_f1'].append(val_metrics['f1'])

                # 学習率スケジューラ
                self.scheduler.step(val_metrics['loss'])

            # ログ出力
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")
            if val_metrics:
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}, "
                      f"F1: {val_metrics['f1']:.4f}")

            # TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/train', train_metrics['f1'], epoch)
            if val_metrics:
                self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('F1/val', val_metrics['f1'], epoch)

            # モデル保存
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch}.pth")

            # ベストモデル保存
            if val_metrics:
                if val_metrics['f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1']
                    self.save_checkpoint(epoch, "best_model.pth")
                    print(f"  ✓ Best model saved (F1: {self.best_val_f1:.4f})")

        # 最終モデル保存
        self.save_checkpoint(num_epochs, "final_model.pth")
        self.save_history()

        print(f"\n{'='*60}")
        print(f"学習完了")
        print(f"  Best Val F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}\n")

    def save_checkpoint(self, epoch: int, filename: str):
        """チェックポイント保存"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_f1': self.best_val_f1
        }
        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)

    def save_history(self):
        """学習履歴を保存"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='プレー検知モデルの学習')

    # データ
    parser.add_argument('--train-csv', type=str, required=True,
                        help='訓練データCSVパス')
    parser.add_argument('--train-labels', type=str, required=True,
                        help='訓練ラベルCSVパス')
    parser.add_argument('--val-csv', type=str, default=None,
                        help='検証データCSVパス')
    parser.add_argument('--val-labels', type=str, default=None,
                        help='検証ラベルCSVパス')

    # モデル
    parser.add_argument('--model-type', type=str, default='lstm',
                        choices=['lstm', 'cnn_lstm'],
                        help='モデルタイプ')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='LSTM隠れ層のサイズ')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='LSTMの層数')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='ドロップアウト率')
    parser.add_argument('--no-attention', action='store_true',
                        help='Attention機構を使用しない')

    # 学習
    parser.add_argument('--epochs', type=int, default=50,
                        help='エポック数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学習率')
    parser.add_argument('--sequence-length', type=int, default=30,
                        help='シーケンス長（フレーム数）')
    parser.add_argument('--stride', type=int, default=5,
                        help='シーケンスのストライド')

    # その他
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='使用デバイス')
    parser.add_argument('--output-dir', type=str, default='output/training',
                        help='出力ディレクトリ')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoaderのワーカー数')
    parser.add_argument('--save-every', type=int, default=10,
                        help='チェックポイント保存間隔')

    args = parser.parse_args()

    # デバイス設定
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = 'cpu'

    device = torch.device(args.device)

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 設定保存
    config = vars(args)
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"設定:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # データセット作成
    print(f"\nデータセット読み込み中...")
    train_dataset = PoseSequenceDataset(
        csv_path=args.train_csv,
        label_path=args.train_labels,
        sequence_length=args.sequence_length,
        stride=args.stride
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == 'cuda' else False
    )

    val_loader = None
    if args.val_csv and args.val_labels:
        val_dataset = PoseSequenceDataset(
            csv_path=args.val_csv,
            label_path=args.val_labels,
            sequence_length=args.sequence_length,
            stride=args.stride
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if args.device == 'cuda' else False
        )

    # モデル作成
    print(f"\nモデル作成中...")
    if args.model_type == 'lstm':
        model = PlayClassifierLSTM(
            input_size=34,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_attention=not args.no_attention
        )
    else:  # cnn_lstm
        model = PlayClassifierCNNLSTM(
            input_size=34,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    print(f"  パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # 学習器作成
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=str(device),
        learning_rate=args.lr,
        output_dir=str(output_dir)
    )

    # 学習開始
    trainer.train(num_epochs=args.epochs, save_every=args.save_every)

    print(f"\n学習済みモデル: {output_dir / 'best_model.pth'}")
    print(f"学習履歴: {output_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()
