import { LP_HERO, LP_PAINS, LP_HOW } from './lp-shared'
import {
  LpCta,
  Eyebrow,
  LpTopbar,
  LpTopbarMobile,
  AppPreview,
  LpFooter,
  LpFooterMobile,
} from './lp-content'
import { IconChevR } from '../components/ui/Icons'
import type { CSSProperties } from 'react'

const BALL = 'oklch(0.68 0.18 50)'

function BallTrail({ width = 240, height = 48, style }: { width?: number; height?: number; style?: CSSProperties }) {
  return (
    <svg
      viewBox="0 0 280 56"
      width={width}
      height={height}
      aria-hidden="true"
      style={{ display: 'block', overflow: 'visible', ...style }}
    >
      <path
        d="M6,50 Q40,-2 76,50 T146,50 T216,50 T270,50"
        stroke="var(--fg-4)"
        strokeWidth="1.2"
        strokeDasharray="2 3"
        strokeLinecap="round"
        fill="none"
        opacity="0.55"
      />
      <circle cx="270" cy="50" r="5.5" fill={BALL} />
      <circle
        cx="270"
        cy="50"
        r="5.5"
        fill="none"
        stroke={`color-mix(in oklab, ${BALL} 30%, transparent)`}
        strokeWidth="3"
      />
    </svg>
  )
}

function NetRule({ style }: { style?: CSSProperties }) {
  return (
    <div
      aria-hidden="true"
      style={{
        height: 1,
        width: '100%',
        backgroundImage:
          'repeating-linear-gradient(90deg, var(--border-strong) 0 12px, transparent 12px 22px)',
        opacity: 0.55,
        ...style,
      }}
    />
  )
}

export function LpVariationA() {
  return (
    <div className="vsbl" style={{ background: 'var(--bg)', color: 'var(--fg)', minHeight: '100%' }}>
      <LpTopbar />

      {/* HERO */}
      <section
        style={{
          padding: '96px 48px 96px',
          textAlign: 'center',
          background: 'linear-gradient(180deg, var(--surface) 0%, var(--bg) 100%)',
          borderBottom: '1px solid var(--border)',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <div
          aria-hidden="true"
          style={{
            position: 'absolute',
            left: -40,
            top: 80,
            width: 160,
            height: 160,
            borderRadius: 999,
            background: `radial-gradient(circle, color-mix(in oklab, ${BALL} 16%, transparent), transparent 60%)`,
            pointerEvents: 'none',
          }}
        />
        <div
          aria-hidden="true"
          style={{
            position: 'absolute',
            right: -60,
            bottom: 200,
            width: 220,
            height: 220,
            borderRadius: 999,
            background: `radial-gradient(circle, color-mix(in oklab, ${BALL} 11%, transparent), transparent 60%)`,
            pointerEvents: 'none',
          }}
        />

        <div style={{ position: 'relative', maxWidth: 880, margin: '0 auto' }}>
          <BallTrail style={{ margin: '0 auto 18px' }} />
          <Eyebrow style={{ marginBottom: 24 }}>{LP_HERO.eyebrow}</Eyebrow>
          <h1
            style={{
              fontSize: 64,
              fontWeight: 600,
              letterSpacing: '-0.03em',
              lineHeight: 1.05,
              margin: 0,
            }}
          >
            {LP_HERO.title_a}
            <br />
            {LP_HERO.title_b}
          </h1>
          <p
            style={{
              fontSize: 18,
              color: 'var(--fg-2)',
              lineHeight: 1.6,
              marginTop: 24,
              maxWidth: 620,
              marginLeft: 'auto',
              marginRight: 'auto',
            }}
          >
            {LP_HERO.sub}
          </p>
          <div style={{ display: 'flex', gap: 10, justifyContent: 'center', marginTop: 36 }}>
            <LpCta kind="primary" href="/register">
              {LP_HERO.cta_primary}
            </LpCta>
            <LpCta kind="secondary" href="#how">
              {LP_HERO.cta_secondary} <IconChevR size={14} />
            </LpCta>
          </div>
          <div style={{ marginTop: 14, fontSize: 12.5, color: 'var(--fg-4)' }}>{LP_HERO.meta}</div>
        </div>

        <div style={{ position: 'relative', maxWidth: 1200, margin: '72px auto 0', padding: '0 24px' }}>
          <AppPreview height={580} />
        </div>
      </section>

      {/* PAINS */}
      <section id="pains" style={{ padding: '96px 48px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: 56 }}>
            <Eyebrow style={{ marginBottom: 12 }}>The problem</Eyebrow>
            <h2 style={{ fontSize: 36, fontWeight: 600, letterSpacing: '-0.02em', margin: 0 }}>
              こんな悩み、ありませんか。
            </h2>
            <NetRule style={{ maxWidth: 120, margin: '20px auto 0' }} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24 }}>
            {LP_PAINS.map((p, i) => (
              <div
                key={i}
                style={{
                  padding: 32,
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 12,
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                    color: 'var(--accent)',
                    letterSpacing: '0.1em',
                    marginBottom: 18,
                  }}
                >
                  0{i + 1}
                </div>
                <h3
                  style={{
                    fontSize: 20,
                    fontWeight: 600,
                    letterSpacing: '-0.015em',
                    margin: 0,
                    lineHeight: 1.35,
                  }}
                >
                  {p.head}
                  <br />
                  {p.head2}
                </h3>
                <p style={{ fontSize: 14, color: 'var(--fg-2)', lineHeight: 1.65, marginTop: 14 }}>
                  {p.body}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section id="how" style={{ padding: '96px 48px', background: 'var(--surface)' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <div style={{ marginBottom: 56 }}>
            <Eyebrow style={{ marginBottom: 12 }}>How it works</Eyebrow>
            <h2 style={{ fontSize: 36, fontWeight: 600, letterSpacing: '-0.02em', margin: 0, maxWidth: 720 }}>
              撮って、上げる。あとは AI が、ラリーを切り出して並べます。
            </h2>
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: 0,
              borderTop: '1px solid var(--border)',
            }}
          >
            {LP_HOW.map((s, i) => (
              <div
                key={i}
                style={{
                  padding: '32px 28px 32px 0',
                  borderRight: i < 2 ? '1px solid var(--border)' : 'none',
                  paddingLeft: i > 0 ? 28 : 0,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                  <span
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: 999,
                      background: i === 1 ? BALL : 'var(--fg-4)',
                      boxShadow:
                        i === 1 ? `0 0 0 3px color-mix(in oklab, ${BALL} 22%, transparent)` : 'none',
                    }}
                  />
                  <span
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: 11,
                      color: 'var(--fg-4)',
                      letterSpacing: '0.12em',
                    }}
                  >
                    STEP {s.num}
                  </span>
                </div>
                <h3 style={{ fontSize: 22, fontWeight: 600, letterSpacing: '-0.015em', margin: 0 }}>
                  {s.title}
                </h3>
                <p style={{ fontSize: 14, color: 'var(--fg-2)', lineHeight: 1.7, marginTop: 12 }}>
                  {s.body}
                </p>
                <div
                  style={{
                    marginTop: 18,
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                    color: 'var(--fg-4)',
                    letterSpacing: '0.04em',
                  }}
                >
                  {s.caption}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <LpFooter />
    </div>
  )
}

export function LpVariationAMobile() {
  return (
    <div className="vsbl" style={{ background: 'var(--bg)', color: 'var(--fg)', minHeight: '100%' }}>
      <LpTopbarMobile />

      <section
        style={{
          padding: '44px 22px 56px',
          textAlign: 'left',
          borderBottom: '1px solid var(--border)',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <div
          aria-hidden="true"
          style={{
            position: 'absolute',
            right: -40,
            top: 40,
            width: 140,
            height: 140,
            borderRadius: 999,
            background: `radial-gradient(circle, color-mix(in oklab, ${BALL} 18%, transparent), transparent 60%)`,
            pointerEvents: 'none',
          }}
        />
        <div style={{ position: 'relative' }}>
          <BallTrail width={170} height={32} style={{ marginBottom: 12 }} />
          <Eyebrow style={{ marginBottom: 18 }}>{LP_HERO.eyebrow}</Eyebrow>
          <h1
            style={{
              fontSize: 36,
              fontWeight: 600,
              letterSpacing: '-0.025em',
              lineHeight: 1.1,
              margin: 0,
            }}
          >
            {LP_HERO.title_a}
            <br />
            {LP_HERO.title_b}
          </h1>
          <p style={{ fontSize: 15, color: 'var(--fg-2)', lineHeight: 1.6, marginTop: 18 }}>
            {LP_HERO.sub}
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 28 }}>
            <LpCta kind="primary" href="/register" style={{ justifyContent: 'center' }}>
              {LP_HERO.cta_primary}
            </LpCta>
            <LpCta kind="secondary" href="#how" style={{ justifyContent: 'center' }}>
              {LP_HERO.cta_secondary}
            </LpCta>
          </div>
          <div style={{ marginTop: 14, fontSize: 12, color: 'var(--fg-4)', textAlign: 'center' }}>
            {LP_HERO.meta}
          </div>
          <div
            style={{
              marginTop: 32,
              borderRadius: 10,
              overflow: 'hidden',
              border: '1px solid var(--border)',
              height: 280,
              position: 'relative',
            }}
          >
            <AppPreview height={280} />
          </div>
        </div>
      </section>

      <section style={{ padding: '56px 22px', borderBottom: '1px solid var(--border)' }}>
        <Eyebrow style={{ marginBottom: 10 }}>The problem</Eyebrow>
        <h2 style={{ fontSize: 26, fontWeight: 600, letterSpacing: '-0.02em', margin: 0, marginBottom: 28 }}>
          こんな悩み、ありませんか。
        </h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {LP_PAINS.map((p, i) => (
            <div
              key={i}
              style={{
                padding: 20,
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: 12,
              }}
            >
              <div
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 10.5,
                  color: 'var(--accent)',
                  letterSpacing: '0.12em',
                  marginBottom: 10,
                }}
              >
                0{i + 1}
              </div>
              <h3 style={{ fontSize: 17, fontWeight: 600, margin: 0, lineHeight: 1.35 }}>
                {p.head}
                {p.head2}
              </h3>
              <p style={{ fontSize: 13.5, color: 'var(--fg-2)', lineHeight: 1.65, marginTop: 10 }}>
                {p.body}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section style={{ padding: '56px 22px', background: 'var(--surface)' }}>
        <Eyebrow style={{ marginBottom: 10 }}>How it works</Eyebrow>
        <h2 style={{ fontSize: 26, fontWeight: 600, letterSpacing: '-0.02em', margin: 0, marginBottom: 28 }}>
          3ステップ。
        </h2>
        {LP_HOW.map((s, i) => (
          <div
            key={i}
            style={{
              paddingTop: 22,
              paddingBottom: 22,
              borderTop: i === 0 ? '1px solid var(--border)' : 'none',
              borderBottom: '1px solid var(--border)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: 999,
                  background: i === 1 ? BALL : 'var(--fg-4)',
                  boxShadow:
                    i === 1 ? `0 0 0 3px color-mix(in oklab, ${BALL} 22%, transparent)` : 'none',
                }}
              />
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 11,
                  color: 'var(--fg-4)',
                  letterSpacing: '0.12em',
                }}
              >
                STEP {s.num}
              </span>
            </div>
            <h3 style={{ fontSize: 18, fontWeight: 600, margin: 0 }}>{s.title}</h3>
            <p style={{ fontSize: 13.5, color: 'var(--fg-2)', lineHeight: 1.65, marginTop: 8 }}>
              {s.body}
            </p>
            <div
              style={{
                marginTop: 12,
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                color: 'var(--fg-4)',
                letterSpacing: '0.04em',
              }}
            >
              {s.caption}
            </div>
          </div>
        ))}
      </section>

      <LpFooterMobile />
    </div>
  )
}
