type Props = {
  label?: string;
  className?: string;
};

/**
 * Striped placeholder block — used wherever a real image/thumbnail/video frame
 * would go. Renders a subtle diagonal stripe pattern with an optional
 * monospace label.
 */
export default function Stripes({ label, className = "" }: Props) {
  return (
    <div
      className={`stripes flex h-full w-full items-center justify-center font-mono text-[11px] tracking-[0.04em] lowercase text-fg-3 ${className}`}
    >
      {label}
    </div>
  );
}
