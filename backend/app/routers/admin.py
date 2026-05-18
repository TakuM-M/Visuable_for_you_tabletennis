from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.deps import require_internal_api_key
from app.db.session import get_db
from app.schemas.admin import StorageMetricsResponse
from app.services import metrics_service

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get(
    "/metrics",
    response_model=StorageMetricsResponse,
    dependencies=[Depends(require_internal_api_key)],
)
def get_storage_metrics(db: Session = Depends(get_db)) -> StorageMetricsResponse:
    """R2 使用量・DB レコード数のストレージ統計を返す（管理者用）"""
    return metrics_service.collect_storage_metrics(db)
