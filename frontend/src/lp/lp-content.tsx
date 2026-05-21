import type { CSSProperties, ReactNode } from 'react'
import { LP_NAV, LP_FAQS } from './lp-shared'
import Logo from '../components/ui/Logo'
import { IconPlus } from '../components/ui/Icons'
import { useState } from 'react'

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
      <nav style={{ display: 'flex', gap: 28 }}>
        {LP_NAV.map((n) => (
          <a
            key={n.label}
            href={n.anchor}
            style={{
              fontSize: 13.5,
              color: 'var(--fg-2)',
              textDecoration: 'none',
              fontWeight: 450,
            }}
          >
            {n.label}
          </a>
        ))}
      </nav>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <a href="/login" style={{ fontSize: 13.5, color: 'var(--fg-2)', textDecoration: 'none' }}>
          ログイン
        </a>
        <LpCta size="sm" kind="primary" href="/register">
          無料でβに登録
        </LpCta>
      </div>
    </div>
  )
}

export function LpTopbarMobile() {
  const [menuOpen, setMenuOpen] = useState(false)
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
      <button
        onClick={() => setMenuOpen(!menuOpen)}
        style={{
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
          padding: 0,
          color: 'var(--fg-2)',
        }}
        aria-label="Menu"
      >
        <svg
          width={20}
          height={20}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M4 6h16M4 12h16M4 18h10" />
        </svg>
      </button>
    </div>
  )
}

const MOCK_ROWS = [
  { title: '練習試合 vs 田中さん', clips: 23, date: '05/18', done: true },
  { title: '大会予選 第2ラウンド', clips: 17, date: '05/15', done: true },
  { title: 'フォーム確認 バックハンド', clips: null, date: '05/12', done: false },
  { title: 'ダブルス練習', clips: 31, date: '05/08', done: true },
]

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
      <div style={{ height: height - 30, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {/* Mock video list */}
        <div style={{ flex: 1, overflow: 'auto', padding: '8px' }}>
          {MOCK_ROWS.map((row, i) => (
            <div
              key={i}
              style={{
                padding: '12px 14px',
                borderBottom: '1px solid var(--border)',
                fontSize: 13,
                color: 'var(--fg)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 500, marginBottom: 2 }}>{row.title}</div>
                <div style={{ fontSize: 11, color: 'var(--fg-4)' }}>{row.date}</div>
              </div>
              {row.done ? (
                <div style={{ fontSize: 12, color: 'var(--accent)', fontWeight: 500 }}>{row.clips} シーン</div>
              ) : (
                <div style={{ fontSize: 11, color: 'var(--fg-4)' }}>処理中…</div>
              )}
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

export function FaqList({ alwaysOpen = false }: { alwaysOpen?: boolean }) {
  const [open, setOpen] = useState<Set<number>>(alwaysOpen ? new Set([0, 1, 2, 3, 4]) : new Set([0]))

  const toggle = (i: number) => {
    const n = new Set(open)
    n.has(i) ? n.delete(i) : n.add(i)
    setOpen(n)
  }

  return (
    <div style={{ borderTop: '1px solid var(--border)' }}>
      {LP_FAQS.map((f, i) => (
        <div key={i} style={{ borderBottom: '1px solid var(--border)' }}>
          <button
            onClick={() => toggle(i)}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              width: '100%',
              padding: '20px 4px',
              background: 'transparent',
              border: 0,
              cursor: 'pointer',
              textAlign: 'left',
              fontFamily: 'inherit',
            }}
          >
            <span style={{ fontSize: 16, fontWeight: 500, color: 'var(--fg)', letterSpacing: '-0.005em' }}>
              {f.q}
            </span>
            <IconPlus
              size={16}
              style={{
                color: 'var(--fg-3)',
                transition: 'transform 200ms',
                transform: open.has(i) ? 'rotate(45deg)' : 'rotate(0)',
              }}
            />
          </button>
          {open.has(i) && (
            <div
              style={{
                padding: '0 4px 22px',
                fontSize: 14.5,
                color: 'var(--fg-2)',
                lineHeight: 1.7,
                maxWidth: '70ch',
              }}
            >
              {f.a}
            </div>
          )}
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
