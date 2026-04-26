"""
共通カスタム例外クラス

パイプライン処理・学習処理で共通利用する例外
"""


class PipelineError(Exception):
    """パイプライン処理の基底例外クラス"""
    pass


class TableDetectionError(PipelineError):
    """卓球台検出に失敗した場合の例外"""

    def __init__(self, message: str, attempts: int = 0, min_confidence: float = 0.0):
        """
        Args:
            message: エラーメッセージ
            attempts: 試行回数
            min_confidence: 要求された最小信頼度
        """
        self.attempts = attempts
        self.min_confidence = min_confidence
        super().__init__(message)


class VideoProcessingError(PipelineError):
    """動画処理に失敗した場合の例外"""
    pass


class VideoInputError(PipelineError):
    """動画ファイルの読み込みに失敗した場合の例外"""

    def __init__(self, video_path: str, reason: str = ""):
        """
        Args:
            video_path: 動画ファイルのパス
            reason: 失敗理由
        """
        self.video_path = video_path
        self.reason = reason
        message = f"動画ファイルを開けませんでした: {video_path}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


class ExportError(PipelineError):
    """データのエクスポートに失敗した場合の例外"""

    def __init__(self, output_path: str, reason: str = ""):
        """
        Args:
            output_path: 出力先パス
            reason: 失敗理由
        """
        self.output_path = output_path
        self.reason = reason
        message = f"データのエクスポートに失敗しました: {output_path}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


class DataInputError(PipelineError):
    """データファイルの読み込みに失敗した場合の例外"""

    def __init__(self, input_path: str, reason: str = ""):
        """
        Args:
            input_path: 入力ファイルのパス
            reason: 失敗理由
        """
        self.input_path = input_path
        self.reason = reason
        message = f"データファイルを読み込めませんでした: {input_path}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


class AugmentationError(PipelineError):
    """データ拡張処理に失敗した場合の例外"""

    def __init__(self, message: str, sample_index: int = -1):
        """
        Args:
            message: エラーメッセージ
            sample_index: エラーが発生したサンプルのインデックス
        """
        self.sample_index = sample_index
        if sample_index >= 0:
            message = f"[Sample {sample_index}] {message}"
        super().__init__(message)


class TrainingError(PipelineError):
    """モデル学習に失敗した場合の例外"""

    def __init__(self, message: str, epoch: int = -1):
        """
        Args:
            message: エラーメッセージ
            epoch: エラーが発生したエポック
        """
        self.epoch = epoch
        if epoch >= 0:
            message = f"[Epoch {epoch}] {message}"
        super().__init__(message)
