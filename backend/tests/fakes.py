"""リポジトリ Protocol を満たす Fake の土台。

`patch("app.services.xxx.yyy_repo.zzz")` のように import パスを文字列で差し替える
代わりに、サービスへ引数で注入するための偽リポジトリを提供する。

各メソッドは `NotImplementedError` を送出するだけ。テストは必要なメソッドだけを
サブクラスで上書きする。想定外のメソッドがサービスから呼ばれた場合はその場で
落ちるので、MagicMock のように黙って何かを返してしまうことがない。
"""

import uuid

from sqlalchemy.orm import Session

from app.models.user import User


class FakeUserRepository:
    """UserRepository Protocol の形だけを満たす土台"""

    def create(
        self, db: Session, email: str, password_hash: str, display_name: str
    ) -> User:
        raise NotImplementedError

    def get_by_id(self, db: Session, user_id: uuid.UUID) -> User | None:
        raise NotImplementedError

    def get_by_email(self, db: Session, email: str) -> User | None:
        raise NotImplementedError

    def verify_email(self, db: Session, user_id: uuid.UUID) -> None:
        raise NotImplementedError

    def update(
        self, db: Session, user_id: uuid.UUID, display_name: str, password_hash: str
    ) -> User | None:
        raise NotImplementedError
