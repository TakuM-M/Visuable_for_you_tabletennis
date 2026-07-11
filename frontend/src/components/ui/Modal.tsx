import { useEffect, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { IconClose } from "./Icons";

type Props = {
  open: boolean;
  onClose: () => void;
  title?: ReactNode;
  children: ReactNode;
  className?: string;
};

/**
 * 最小限のモーダル。背景クリック / Esc で閉じる。body 直下にポータルで描画し、
 * 開いている間は背景スクロールをロックする。
 */
export default function Modal({ open, onClose, title, children, className = "" }: Props) {
  // 背景クリックで閉じるのは「押下も背景で始まった」場合のみ。
  // ダイアログ内でドラッグを始めて背景上で離すと click が背景で発火するため、
  // pointerdown の開始位置を記録して誤クローズを防ぐ。
  const downOnBackdrop = useRef(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [open, onClose]);

  if (!open) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[100] grid place-items-center bg-black/60 p-4"
      onPointerDown={(e) => {
        downOnBackdrop.current = e.target === e.currentTarget;
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget && downOnBackdrop.current) onClose();
        downOnBackdrop.current = false;
      }}
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        onClick={(e) => e.stopPropagation()}
        className={`w-full max-w-2xl overflow-hidden rounded-[12px] border border-border bg-surface shadow-xl ${className}`}
      >
        {title != null && (
          <div className="flex items-center justify-between border-b border-border px-4 py-3">
            <h2 className="m-0 text-[14px] font-semibold tracking-[-0.01em]">{title}</h2>
            <button
              onClick={onClose}
              aria-label="閉じる"
              className="cursor-pointer rounded-md border-none bg-transparent p-1 text-fg-3 hover:bg-subtle-2 hover:text-fg"
            >
              <IconClose size={16} />
            </button>
          </div>
        )}
        <div className="p-4">{children}</div>
      </div>
    </div>,
    document.body,
  );
}
