"""metrics_service の集計テスト。

collect_storage_metrics は DB集計（生の集計SQL）と R2集計（外部サービス）を
1つにまとめて返す。
  - DB集計の正しさ＝SQLの正しさなので「実DB」(db fixture)で検証する
  - R2 はネットワーク越しの外部依存なので必ず Mock に差し替える
"""
from unittest.mock import Mock, patch

from app.repositories import user as user_repo
from app.services import metrics_service
from app.repositories import video as video_repo


def _client_with_pages(pages):
    """list_objects_v2 のページ列を返す偽 R2 client を組み立てるヘルパ"""
    client = Mock()
    client.get_paginator.return_value.paginate.return_value = pages
    return client


def test_collect_sums_r2_objects(db):
    """R2 の各ページの Size を合計し、オブジェクト数を数える"""
    fake = _client_with_pages([
        {"Contents": [{"Size": 100}, {"Size": 200}]},
        {"Contents": [{"Size": 50}]},
    ])
    with patch("app.services.storage_service._get_client", return_value=fake):
        result = metrics_service.collect_storage_metrics(db)
    assert result.r2_total_bytes == 350
    assert result.r2_object_count == 3
    
def test_collect_counts_videos_per_user(db, user):
    """DB の動画件数とユーザー別件数（user→2本 / other→1本）が正しいか"""
    # 別ユーザーを作る（email はユニークなので fixture の owner@ と別アドレスに）
    other = user_repo.create(
        db=db,
        email="other@example.com",
        password_hash="hashed",
        display_name="Other",
    )
    # user に2本、other に1本（引数は storage_path）
    video_repo.create(db, user_id=user.id, title="v1", storage_path="videos/v1.mp4")
    video_repo.create(db, user_id=user.id, title="v2", storage_path="videos/v2.mp4")
    video_repo.create(db, user_id=other.id, title="v3", storage_path="videos/v3.mp4")

    # R2 は空にして DB集計だけを検証
    fake = _client_with_pages([{"Contents": []}])
    with patch("app.services.storage_service._get_client", return_value=fake):
        result = metrics_service.collect_storage_metrics(db)

    assert result.db_video_count == 3
    assert result.videos_per_user == {str(user.id): 2, str(other.id): 1}

def test_collect_survives_r2_failure(db, video):
    with patch("app.services.storage_service._get_client", side_effect=Exception("boom")):
        result = metrics_service.collect_storage_metrics(db)
    # R2失敗でも例外が外に漏れず、DB側の集計は返る
    assert result.r2_total_bytes == 0
    assert result.r2_object_count == 0
    assert result.db_video_count == 1


# --- 以下、自分で書く分 ---

# A) test_collect_counts_videos_per_user(db, user):
#    狙い: DB集計（件数・ユーザー別）が正しいか → 実DBで検証
#    準備:
#      - user fixture の他にもう1人 user_repo.create(...) で作る
#      - video_repo.create(db, user_id=..., ...) で
#        user に2本、もう1人に1本など投入
#      - R2 は邪魔なので空に: _client_with_pages([{"Contents": []}]) を patch
#    検証:
#      - result.db_video_count == 3
#      - result.videos_per_user == {str(user.id): 2, str(other.id): 1}
#        ※ キーは str(uid)。サービスが str() で文字列化しているので int で比較しない

# C) test_collect_survives_r2_failure(db, video):
#    狙い: R2 が落ちても例外を投げず、DB側だけは返す（resilience）
#    準備:
#      - patch の side_effect=Exception("boom") で _get_client が例外を投げるように
#        （email_service の失敗テストと同じ手口）
#      - video fixture で DB に1本入れておく
#    検証:
#      - 例外が外に漏れない（呼んでも落ちない＝ assert まで到達する）
#      - result.r2_total_bytes == 0 / result.r2_object_count == 0
#      - result.db_video_count == 1  ← R2失敗でもDB側は返る、が肝