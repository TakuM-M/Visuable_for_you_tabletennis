import type { ButtonHTMLAttributes, ReactNode } from "react";

type Kind = "primary" | "secondary" | "ghost" | "accent" | "danger";
type Size = "sm" | "md";

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  kind?: Kind;
  size?: Size;
  children: ReactNode;
};

const KIND: Record<Kind, string> = {
  primary:   "bg-fg text-bg border-fg hover:bg-fg-2 hover:border-fg-2",
  secondary: "bg-surface text-fg border-border hover:bg-subtle",
  ghost:     "bg-transparent text-fg-2 border-transparent hover:bg-subtle-2 hover:text-fg",
  accent:    "bg-accent text-white border-accent hover:brightness-105",
  danger:    "bg-transparent text-err border-transparent hover:bg-err-soft",
};

const SIZE: Record<Size, string> = {
  sm: "px-2.5 py-1.5 text-[12.5px]",
  md: "px-3.5 py-2 text-[13.5px]",
};

export default function Button({
  kind = "ghost",
  size = "md",
  className = "",
  children,
  ...rest
}: Props) {
  return (
    <button
      {...rest}
      className={`
        inline-flex items-center gap-1.5 rounded-lg border font-medium leading-none
        transition-colors disabled:opacity-50 disabled:pointer-events-none
        ${SIZE[size]} ${KIND[kind]} ${className}
      `}
    >
      {children}
    </button>
  );
}
