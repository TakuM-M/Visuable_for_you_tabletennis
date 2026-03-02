# データモデル

## ER図

```mermaid
erDiagram
    users {
        UUID id PK
        VARCHAR email
        VARCHAR password_hash
        VARCHAR display_name
        BOOLEAN email_verified
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }

    videos {
        UUID id PK
        UUID user_id FK
        VARCHAR title
        VARCHAR storage_path
        FLOAT duration
        ENUM status
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }

    jobs {
        UUID id PK
        UUID video_id FK
        ENUM status
        TIMESTAMPTZ started_at
        TIMESTAMPTZ completed_at
        TEXT error_message
        TIMESTAMPTZ created_at
    }

    clips {
        UUID id PK
        UUID video_id FK
        UUID job_id FK
        FLOAT start_time
        FLOAT end_time
        VARCHAR storage_path
        TIMESTAMPTZ created_at
    }

    notification_logs {
        BIGSERIAL id PK
        UUID user_id FK
        UUID job_id FK
        VARCHAR email
        ENUM status
        TIMESTAMPTZ sent_at
        TIMESTAMPTZ created_at
    }

    users ||--o{ videos : "投稿"
    videos ||--o{ jobs : "処理"
    jobs ||--o{ clips : "生成"
    videos ||--o{ clips : "元動画"
    users ||--o{ notification_logs : "通知先"
    jobs ||--o{ notification_logs : "対象"
```

## テーブル定義

### users — ユーザーアカウント

| カラム | 型 | NULL | 説明 |
|---|---|---|---|
| id | UUID | NOT NULL | PK |
| email | VARCHAR | NOT NULL UNIQUE | ログイン・通知先メールアドレス |
| password_hash | VARCHAR | NOT NULL | ハッシュ化済みパスワード |
| display_name | VARCHAR | NOT NULL | 表示名 |
| email_verified | BOOLEAN | NOT NULL DEFAULT FALSE | メール認証済みフラグ |
| created_at | TIMESTAMPTZ | NOT NULL | |
| updated_at | TIMESTAMPTZ | NOT NULL | |

### videos — アップロードされた動画

| カラム | 型 | NULL | 説明 |
|---|---|---|---|
| id | UUID | NOT NULL | PK |
| user_id | UUID | NOT NULL | FK → users.id |
| title | VARCHAR | NOT NULL | 動画タイトル |
| storage_path | VARCHAR | NOT NULL | ストレージ上のファイルパス |
| duration | FLOAT | NULLABLE | 動画長（秒）。アップロード後に設定 |
| status | ENUM | NOT NULL | `uploaded` / `queued` / `processing` / `completed` / `failed` |
| created_at | TIMESTAMPTZ | NOT NULL | |
| updated_at | TIMESTAMPTZ | NOT NULL | |

### jobs — 切り抜き処理ジョブ

| カラム | 型 | NULL | 説明 |
|---|---|---|---|
| id | UUID | NOT NULL | PK |
| video_id | UUID | NOT NULL | FK → videos.id |
| status | ENUM | NOT NULL | `queued` / `processing` / `completed` / `failed` |
| started_at | TIMESTAMPTZ | NULLABLE | 処理開始時刻 |
| completed_at | TIMESTAMPTZ | NULLABLE | 処理完了時刻 |
| error_message | TEXT | NULLABLE | 失敗時のエラー内容 |
| created_at | TIMESTAMPTZ | NOT NULL | |

> 動画とジョブのライフサイクルを分離することで、再処理・リトライが可能になる。

### clips — 生成された切り抜き動画

| カラム | 型 | NULL | 説明 |
|---|---|---|---|
| id | UUID | NOT NULL | PK |
| video_id | UUID | NOT NULL | FK → videos.id（元動画） |
| job_id | UUID | NOT NULL | FK → jobs.id（生成ジョブ） |
| start_time | FLOAT | NOT NULL | 開始位置（秒） |
| end_time | FLOAT | NOT NULL | 終了位置（秒） |
| storage_path | VARCHAR | NOT NULL | ストレージ上のファイルパス |
| created_at | TIMESTAMPTZ | NOT NULL | |

### notification_logs — メール通知履歴

| カラム | 型 | NULL | 説明 |
|---|---|---|---|
| id | BIGSERIAL | NOT NULL | PK |
| user_id | UUID | NOT NULL | FK → users.id |
| job_id | UUID | NOT NULL | FK → jobs.id |
| email | VARCHAR | NOT NULL | 送信先アドレス（送信時点の値を記録） |
| status | ENUM | NOT NULL | `pending` / `sent` / `failed` |
| sent_at | TIMESTAMPTZ | NULLABLE | 実際の送信時刻 |
| created_at | TIMESTAMPTZ | NOT NULL | |

> ユーザーがメールアドレスを変更した後でも送信履歴を追跡できるよう、送信時点のアドレスをスナップショットとして保持する。


## 作成メモ
1. PostgreSQL — データベース本体
役割: データを実際に保存・管理するサーバー

SQL（Structured Query Language）を使って、テーブルの作成・データの読み書きを行う。Pythonからは直接触らず、後述のSQLAlchemyを通じて操作する。

2. psycopg2 — PostgreSQL ドライバ
役割: PythonとPostgreSQLを接続する「橋渡し」

PythonはそのままではPostgreSQLと通信できない。psycopg2がTCP接続の確立・SQLの送受信・型変換を担う。直接使うことはほぼなく、SQLAlchemyの内部で自動的に使われる。

3. SQLAlchemy — ORM（Object-Relational Mapper）
役割: テーブルをPythonクラスとして扱えるようにする

生のSQLを書かずに、Pythonオブジェクトを操作するだけでデータの読み書きができる。

4. Alembic — マイグレーション管理
役割: テーブル定義の変更履歴を管理し、DBに安全に適用する

SQLAlchemyのモデルクラスを変更しただけでは、PostgreSQL側のテーブルは変わらない。Alembicがモデルの差分を検出して「マイグレーションファイル（変更スクリプト）」を自動生成し、それをDBに適用することでテーブル構造を更新する。

5. Pydantic Settings — 設定管理
役割: 接続先DBのURLやパスワードを .env から安全に読み込む

DBのホスト名・ユーザー名・パスワードをコードにハードコードするのは危険。Pydantic Settingsが .env ファイルを読み込み、型安全な設定オブジェクトとして提供する。PostgreSQL	データの永続化
