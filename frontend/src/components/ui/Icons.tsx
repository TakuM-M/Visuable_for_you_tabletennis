import type { SVGProps, ReactNode } from "react";

/* Minimal hairline icon set. Stroke-based unless otherwise noted. */

type Props = Omit<SVGProps<SVGSVGElement>, 'stroke' | 'fill'> & {
  size?: number;
  strokeWidth?: number;
  stroke?: string;
  fill?: string;
};

function Svg({
  size = 16,
  strokeWidth = 1.5,
  children,
  fill = "none",
  stroke = "currentColor",
  ...rest
}: Props & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={fill}
      stroke={stroke}
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const IconList     = (p: Props) => <Svg {...p}><path d="M4 6h16M4 12h16M4 18h10" /></Svg>;
export const IconUpload   = (p: Props) => <Svg {...p}><path d="M12 16V4M6 10l6-6 6 6M4 20h16" /></Svg>;
export const IconUser     = (p: Props) => <Svg {...p}><circle cx="12" cy="8" r="3.5" /><path d="M4 20c1.5-3.5 4.5-5 8-5s6.5 1.5 8 5" /></Svg>;
export const IconLogout   = (p: Props) => <Svg {...p}><path d="M15 4h4v16h-4M10 8l-4 4 4 4M6 12h11" /></Svg>;
export const IconSearch   = (p: Props) => <Svg {...p}><circle cx="11" cy="11" r="6.5" /><path d="M20 20l-4.5-4.5" /></Svg>;
export const IconPlus     = (p: Props) => <Svg {...p}><path d="M12 5v14M5 12h14" /></Svg>;
export const IconChevR    = (p: Props) => <Svg {...p}><path d="M9 6l6 6-6 6" /></Svg>;
export const IconChevL    = (p: Props) => <Svg {...p}><path d="M15 6l-6 6 6 6" /></Svg>;
export const IconChevD    = (p: Props) => <Svg {...p}><path d="M6 9l6 6 6-6" /></Svg>;
export const IconPlay     = (p: Props) => <Svg {...p} fill="currentColor" stroke="none"><path d="M7 5v14l12-7z" /></Svg>;
export const IconVol      = (p: Props) => <Svg {...p}><path d="M4 9v6h4l5 4V5L8 9H4z M16 8a5 5 0 010 8 M18.5 5.5a9 9 0 010 13" /></Svg>;
export const IconFull     = (p: Props) => <Svg {...p}><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" /></Svg>;
export const IconFilm     = (p: Props) => <Svg {...p}><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 9h18M3 15h18M8 5v14M16 5v14"/></Svg>;
export const IconDownload = (p: Props) => <Svg {...p}><path d="M12 4v12M6 10l6 6 6-6M4 20h16" /></Svg>;
export const IconMore     = (p: Props) => <Svg {...p}><circle cx="5" cy="12" r="1.2" fill="currentColor"/><circle cx="12" cy="12" r="1.2" fill="currentColor"/><circle cx="19" cy="12" r="1.2" fill="currentColor"/></Svg>;
export const IconClock    = (p: Props) => <Svg {...p}><circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3.5 2"/></Svg>;
export const IconClose    = (p: Props) => <Svg {...p}><path d="M6 6l12 12M18 6L6 18" /></Svg>;
export const IconCheck    = (p: Props) => <Svg {...p}><path d="M5 12l4.5 4.5L19 7" /></Svg>;
export const IconRefresh  = (p: Props) => <Svg {...p}><path d="M3 12a9 9 0 0115-6.7L21 8M21 3v5h-5M21 12a9 9 0 01-15 6.7L3 16M3 21v-5h5" /></Svg>;
export const IconTrash    = (p: Props) => <Svg {...p}><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v6M14 11v6" /></Svg>;
export const IconPencil   = (p: Props) => <Svg {...p}><path d="M4 20l1-4L16.5 4.5a2.1 2.1 0 013 3L8 19l-4 1zM13.5 7.5l3 3" /></Svg>;
