import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

type MenuItem = {
  label: string;
  icon?: ReactNode;
  onClick: () => void;
  variant?: "default" | "danger";
  disabled?: boolean;
};

type Props = {
  items: MenuItem[];
  children: ReactNode;
  align?: "left" | "right";
};

export default function DropdownMenu({ items, children, align = "right" }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      document.addEventListener("keydown", handleEscape);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [isOpen]);

  const handleMenuClick = (callback: () => void) => {
    callback();
    setIsOpen(false);
  };

  return (
    <div ref={menuRef} className="relative inline-block">
      <div onClick={() => setIsOpen(!isOpen)}>{children}</div>

      {isOpen && (
        <div
          className={`
            absolute top-full mt-1 bg-surface border border-border rounded-lg
            shadow-lg py-1 min-w-36 z-50
            ${align === "left" ? "left-0" : "right-0"}
          `}
        >
          {items.map((item, idx) => (
            <button
              key={idx}
              onClick={() => handleMenuClick(item.onClick)}
              disabled={item.disabled}
              className={`
                w-full flex items-center gap-2 px-3 py-2 text-[12.5px]
                font-medium transition-colors text-left
                ${
                  item.variant === "danger"
                    ? "text-err hover:bg-err-soft disabled:opacity-50 disabled:pointer-events-none"
                    : "text-fg hover:bg-subtle disabled:opacity-50 disabled:pointer-events-none"
                }
              `}
            >
              {item.icon && <span className="flex-shrink-0">{item.icon}</span>}
              <span>{item.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
