import { useState } from "react";
import Stripes from "./Stripes";

type Props = {
  /** backend が発行した presigned URL。未生成の動画では null */
  src?: string | null;
  alt?: string;
  label?: string;
  className?: string;
};

/**
 * 動画のサムネイル。src が無い（生成に失敗した動画・機能追加以前の動画）場合と、
 * presigned URL の期限切れなどで読み込みに失敗した場合は Stripes プレースホルダに
 * フォールバックする。
 */
export default function Thumbnail({ src, alt = "", label, className = "" }: Props) {
  const [failed, setFailed] = useState(false);

  if (!src || failed) return <Stripes label={label} className={className} />;

  return (
    <img
      src={src}
      alt={alt}
      loading="lazy"
      onError={() => setFailed(true)}
      className={`h-full w-full object-cover ${className}`}
    />
  );
}
