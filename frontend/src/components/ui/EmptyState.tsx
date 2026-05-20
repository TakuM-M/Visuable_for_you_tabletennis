import type { ReactNode } from "react";

type Props = {
  icon?: ReactNode;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
};

export default function EmptyState({ icon, title, description, actions, className = "" }: Props) {
  return (
    <div
      className={`rounded-[10px] border border-dashed border-border bg-surface px-6 py-12 text-center ${className}`}
    >
      {icon && (
        <div className="mx-auto grid h-9 w-9 place-items-center rounded-lg bg-subtle-2 text-fg-3">
          {icon}
        </div>
      )}
      <div className="mt-3 text-[14px] font-medium">{title}</div>
      {description && (
        <div className="mt-1.5 text-[12.5px] leading-[1.6] text-fg-3">{description}</div>
      )}
      {actions && <div className="mt-4 flex justify-center gap-2">{actions}</div>}
    </div>
  );
}
