import type { CSSProperties, ReactNode } from 'react'
import Logo from '../components/ui/Logo'
import { getToken } from '../lib/auth'

type LpCtaProps = {
  kind?: 'primary' | 'secondary' | 'accent' | 'ghost' | 'inverted'
  size?: 'lg' | 'sm'
  href?: string
  style?: CSSProperties
  children: ReactNode
  onClick?: () => void
}

export function LpCta({ children, kind = 'primary', size = 'lg', href, style, onClick, ...rest }: LpCtaProps) {
  const pad = size === 'lg' ? '13px 20px' : '10px 16px'
  const fs = size === 'lg' ? 14.5 : 13
  const base: CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 8,
    padding: pad,
    borderRadius: 8,
    fontSize: fs,
    fontWeight: 500,
    fontFamily: 'var(--font-sans)',
    cursor: 'pointer',
    border: '1px solid transparent',
    lineHeight: 1,
    letterSpacing: '-0.005em',
    transition: 'background 120ms, border-color 120ms, transform 120ms',
  }

  const kinds = {
    primary: { background: 'var(--fg)', color: 'var(--bg)', borderColor: 'var(--fg)' },
    secondary: { background: 'transparent', color: 'var(--fg)', borderColor: 'var(--border-strong)' },
    accent: { background: 'var(--accent)', color: 'white', borderColor: 'var(--accent)' },
    ghost: { background: 'transparent', color: 'var(--fg-2)', borderColor: 'transparent' },
    inverted: { background: 'var(--bg)', color: 'var(--fg)', borderColor: 'var(--bg)' },
  }

  const element = href ? 'a' : 'button'
  const Element = element as any

  return (
    <Element
      href={href}
      {...rest}
      onClick={onClick}
      style={{ ...base, ...kinds[kind], ...style }}
    >
      {children}
    </Element>
  )
}

export function Eyebrow({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div
      style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
        color: 'var(--fg-4)',
        textTransform: 'uppercase',
        letterSpacing: '0.14em',
        ...style,
      }}
    >
      {children}
    </div>
  )
}

export function LpTopbar() {
  const loggedIn = !!getToken()
  const loginHref = loggedIn ? '/videos' : '/login'
  const registerHref = loggedIn ? '/videos' : '/register'
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '18px 48px',
        borderBottom: '1px solid var(--border)',
        background: 'color-mix(in oklab, var(--surface) 85%, transparent)',
        backdropFilter: 'saturate(140%) blur(8px)',
        position: 'sticky',
        top: 0,
        zIndex: 5,
      }}
    >
      <Logo />
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <LpCta size="sm" kind="secondary" href={loginHref}>
          ログイン
        </LpCta>
        <LpCta size="sm" kind="primary" href={registerHref}>
          無料でβに登録
        </LpCta>
      </div>
    </div>
  )
}

export function LpTopbarMobile() {
  const loggedIn = !!getToken()
  const loginHref = loggedIn ? '/videos' : '/login'
  const registerHref = loggedIn ? '/videos' : '/register'
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '14px 20px',
        borderBottom: '1px solid var(--border)',
        background: 'color-mix(in oklab, var(--surface) 85%, transparent)',
        backdropFilter: 'saturate(140%) blur(8px)',
        position: 'sticky',
        top: 0,
        zIndex: 5,
      }}
    >
      <Logo />
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <LpCta size="sm" kind="secondary" href={loginHref}>
          ログイン
        </LpCta>
        <LpCta size="sm" kind="primary" href={registerHref}>
          登録
        </LpCta>
      </div>
    </div>
  )
}

const MOCK_ROWS = [
  { title: '練習試合 vs 田中さん', duration: 1523, status: 'completed', created_at: '2025-05-18' },
  { title: '大会予選 第2ラウンド', duration: 2847, status: 'completed', created_at: '2025-05-15' },
  { title: 'フォーム確認 バックハンド', duration: null, status: 'processing', created_at: '2025-05-12' },
  { title: 'ダブルス練習', duration: 3156, status: 'completed', created_at: '2025-05-08' },
]

function fmtDuration(seconds: number | null | undefined) {
  if (seconds == null) return '—'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

const STATUS_SPEC: Record<string, { dot: string; bg: string; fg: string; border: string; label: string; pulse?: boolean }> = {
  completed:  { dot: 'var(--ok)',   bg: 'var(--ok-soft)',   fg: 'var(--ok-ink)',   border: 'color-mix(in oklab, var(--ok-ink) 15%, transparent)',   label: '完了' },
  processing: { dot: 'var(--warn)', bg: 'var(--warn-soft)', fg: 'var(--warn-ink)', border: 'color-mix(in oklab, var(--warn-ink) 15%, transparent)', label: '処理中', pulse: true },
  queued:     { dot: 'var(--warn)', bg: 'var(--warn-soft)', fg: 'var(--warn-ink)', border: 'color-mix(in oklab, var(--warn-ink) 15%, transparent)', label: '処理待ち' },
  failed:     { dot: 'var(--err)',  bg: 'var(--err-soft)',  fg: 'var(--err-ink)',  border: 'color-mix(in oklab, var(--err-ink) 15%, transparent)',  label: '失敗' },
}

function StatusBadgeMock({ status }: { status: string }) {
  const s = STATUS_SPEC[status] ?? STATUS_SPEC.completed
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '2px 8px',
        borderRadius: 999,
        border: `1px solid ${s.border}`,
        background: s.bg,
        color: s.fg,
        fontSize: 11,
        fontWeight: 500,
        whiteSpace: 'nowrap',
      }}
    >
      <span
        className={s.pulse ? 'animate-pulseDot' : undefined}
        style={{ width: 6, height: 6, borderRadius: 999, background: s.dot, color: s.dot }}
      />
      {s.label}
    </span>
  )
}

export function AppPreview({ height = 520 }: { height?: number }) {
  return (
    <div
      style={{
        borderRadius: 12,
        overflow: 'hidden',
        border: '1px solid var(--border)',
        background: 'var(--surface)',
        boxShadow:
          '0 24px 60px -24px color-mix(in oklab, var(--fg) 22%, transparent), 0 2px 6px color-mix(in oklab, var(--fg) 8%, transparent)',
        height,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* fake titlebar */}
      <div
        style={{
          height: 30,
          background: 'var(--subtle)',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '0 12px',
          flexShrink: 0,
        }}
      >
        <span style={{ width: 9, height: 9, borderRadius: 99, background: 'oklch(0.85 0.04 30)' }} />
        <span style={{ width: 9, height: 9, borderRadius: 99, background: 'oklch(0.88 0.04 90)' }} />
        <span style={{ width: 9, height: 9, borderRadius: 99, background: 'oklch(0.85 0.05 150)' }} />
        <span
          style={{
            marginLeft: 14,
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--fg-4)',
          }}
        >
          visuable.app/videos
        </span>
      </div>
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {/* AppShell — row 1 (brand / search / avatar) */}
        <div
          style={{
            height: 42,
            flexShrink: 0,
            background: 'var(--surface)',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            gap: 16,
            padding: '0 16px',
          }}
        >
          <Logo />
          <span style={{ width: 1, height: 16, background: 'var(--border)' }} />
          <div style={{ flex: 1 }} />
          <div
            style={{
              width: 24,
              height: 24,
              borderRadius: 999,
              background: 'var(--accent-soft)',
              color: 'var(--accent-ink)',
              display: 'grid',
              placeItems: 'center',
              fontSize: 10.5,
              fontWeight: 600,
              letterSpacing: '0.02em',
            }}
          >
            TM
          </div>
        </div>

        {/* AppShell — row 2 (tabs) */}
        <div
          style={{
            height: 38,
            flexShrink: 0,
            background: 'var(--surface)',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'stretch',
            gap: 22,
            padding: '0 16px',
          }}
        >
          {[
            { label: '動画一覧', active: true },
            { label: 'アップロード', active: false },
            { label: 'プロフィール', active: false },
          ].map((t, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                alignItems: 'center',
                fontSize: 12,
                fontWeight: t.active ? 500 : 400,
                color: t.active ? 'var(--fg)' : 'var(--fg-2)',
                borderBottom: t.active ? '2px solid var(--fg)' : '2px solid transparent',
              }}
            >
              {t.label}
            </div>
          ))}
        </div>

        {/* Page header */}
        <div
          style={{
            padding: '14px 16px',
            borderBottom: '1px solid var(--border)',
            flexShrink: 0,
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'space-between',
            gap: 12,
          }}
        >
          <div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--fg-4)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 6 }}>
              Library
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, margin: 0 }}>動画一覧</div>
            <div style={{ fontSize: 12, color: 'var(--fg-3)', marginTop: 4 }}>{MOCK_ROWS.length}件</div>
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                padding: '5px 10px',
                borderRadius: 6,
                border: '1px solid var(--border-strong)',
                background: 'transparent',
                color: 'var(--fg)',
                fontSize: 11.5,
                fontWeight: 500,
              }}
            >
              絞り込み
            </span>
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                padding: '5px 10px',
                borderRadius: 6,
                background: 'var(--fg)',
                color: 'var(--bg)',
                fontSize: 11.5,
                fontWeight: 500,
              }}
            >
              + 動画を追加
            </span>
          </div>
        </div>

        {/* Table header */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 100px 100px 100px',
            gap: 0,
            padding: '0 16px',
            borderBottom: '1px solid var(--border)',
            height: 28,
            alignItems: 'center',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--fg-4)',
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            flexShrink: 0,
          }}
        >
          <span>タイトル</span>
          <span style={{ textAlign: 'right' }}>再生時間</span>
          <span style={{ textAlign: 'right' }}>状態</span>
          <span style={{ textAlign: 'right' }}>アップロード</span>
        </div>

        {/* Rows */}
        <div style={{ flex: 1, overflow: 'auto' }}>
          {MOCK_ROWS.map((row, i) => (
            <div
              key={i}
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 100px 100px 100px',
                gap: 0,
                padding: '8px 16px',
                borderBottom: '1px solid var(--border)',
                alignItems: 'center',
                fontSize: 12,
                color: 'var(--fg)',
              }}
            >
              <div style={{ minWidth: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
                <div className="stripes" style={{ width: 32, height: 20, flexShrink: 0, borderRadius: 3 }} />
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div style={{ fontSize: 12, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {row.title}
                  </div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--fg-4)', marginTop: 2 }}>
                    MP4
                  </div>
                </div>
              </div>
              <div style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--fg-2)' }}>
                {fmtDuration(row.duration)}
              </div>
              <div style={{ textAlign: 'right' }}>
                <StatusBadgeMock status={row.status} />
              </div>
              <div style={{ textAlign: 'right', fontSize: 11, color: 'var(--fg-3)' }}>
                {row.created_at}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export function StatRow({ items, style }: { items: Array<{ v: string | number; k: string }>; style?: CSSProperties }) {
  return (
    <div
      style={{
        display: 'flex',
        gap: 32,
        flexWrap: 'wrap',
        borderTop: '1px solid var(--border)',
        paddingTop: 18,
        ...style,
      }}
    >
      {items.map((it, i) => (
        <div key={i}>
          <div style={{ fontSize: 28, fontWeight: 600, letterSpacing: '-0.02em' }}>{it.v}</div>
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--fg-4)',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
              marginTop: 4,
            }}
          >
            {it.k}
          </div>
        </div>
      ))}
    </div>
  )
}

export function LpFooter() {
  const BALL = 'oklch(0.68 0.18 50)'
  return (
    <footer
      style={{
        borderTop: '1px solid var(--border)',
        background: 'var(--surface)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div
        aria-hidden="true"
        style={{
          position: 'absolute',
          right: -100,
          bottom: -100,
          width: 240,
          height: 240,
          borderRadius: 999,
          background: `radial-gradient(circle at 30% 30%, color-mix(in oklab, ${BALL} 14%, transparent), transparent 65%)`,
          pointerEvents: 'none',
        }}
      />
      <div
        style={{
          padding: '24px 48px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 14,
          position: 'relative',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <Logo />
        </div>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--fg-4)',
            letterSpacing: '0.06em',
          }}
        >
          <span style={{ width: 6, height: 6, borderRadius: 999, background: BALL }} />
          <span>© 2025 VISUABLE, INC.</span>
          <span style={{ color: 'var(--border-strong)' }}>/</span>
          <span>TOKYO, JAPAN</span>
          <span style={{ color: 'var(--border-strong)' }}>/</span>
          <span>v0.4.2-BETA</span>
        </div>
      </div>
    </footer>
  )
}

export function LpFooterMobile() {
  const BALL = 'oklch(0.68 0.18 50)'
  return (
    <footer
      style={{
        borderTop: '1px solid var(--border)',
        background: 'var(--surface)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div
        aria-hidden="true"
        style={{
          position: 'absolute',
          right: -80,
          bottom: -80,
          width: 180,
          height: 180,
          borderRadius: 999,
          background: `radial-gradient(circle at 30% 30%, color-mix(in oklab, ${BALL} 16%, transparent), transparent 65%)`,
        }}
      />
      <div
        style={{
          padding: '20px 22px',
          display: 'flex',
          flexDirection: 'column',
          gap: 14,
          position: 'relative',
        }}
      >
        <Logo />
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            flexWrap: 'wrap',
            fontFamily: 'var(--font-mono)',
            fontSize: 10.5,
            color: 'var(--fg-4)',
            letterSpacing: '0.08em',
          }}
        >
          <span style={{ width: 6, height: 6, borderRadius: 999, background: BALL }} />
          <span>© 2025 VISUABLE, INC.</span>
          <span style={{ color: 'var(--border-strong)' }}>/</span>
          <span>TOKYO</span>
          <span style={{ color: 'var(--border-strong)' }}>/</span>
          <span>v0.4.2-BETA</span>
        </div>
      </div>
    </footer>
  )
}
