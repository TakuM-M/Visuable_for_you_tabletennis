type Props = {
  size?: number;
  withLabel?: boolean;
  className?: string;
};

/**
 * Visuable mark. Dark rounded glyph with a small orange ball dot —
 * the ball is the only piece of color, used sparingly elsewhere.
 */
export default function Logo({ size = 22, withLabel = true, className = "" }: Props) {
  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <div
        className="relative grid place-items-center flex-none rounded-md text-bg bg-fg font-mono font-semibold leading-none"
        style={{
          width: size,
          height: size,
          fontSize: size * 0.55,
          letterSpacing: "-0.02em",
        }}
      >
        V
        <span
          className="absolute right-[-2px] bottom-[-2px] rounded-full bg-ball"
          style={{
            width: 7,
            height: 7,
            border: "1.5px solid var(--color-bg)",
          }}
        />
      </div>
      {withLabel && (
        <div className="leading-[1.05]">
          <div className="text-[14px] font-semibold tracking-[-0.01em]">VisuableForYou</div>
          <div className="font-mono text-[10px] uppercase tracking-[0.08em] text-fg-3">
            tabletennis
          </div>
        </div>
      )}
    </div>
  );
}
