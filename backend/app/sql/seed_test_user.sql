-- テストユーザー挿入用SQL
-- 使い方:
--   docker exec -i tabletennis-postgres psql -U postgres -d tabletennis < backend/app/sql/seed_test_user.sql
--
-- ログイン情報:
--   Email:    test@example.com
--   Password: password123

INSERT INTO users (id, email, password_hash, display_name, email_verified, created_at, updated_at)
VALUES (
    gen_random_uuid(),
    'test@example.com',
    '$2b$12$zivZN77qVmf8DrSDmzqKy.nmiRjm69y2Ow0MP9hbzJIlNza3hqTcO',
    'テストユーザー',
    true,
    now(),
    now()
)
ON CONFLICT (email) DO NOTHING;
