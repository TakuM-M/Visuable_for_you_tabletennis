import { useQuery } from "@tanstack/react-query";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import type { ReactNode } from "react";
import { getMeUsersMeGet } from "../../api/generated";
import { authHeaders } from "../../lib/auth";
import Logo from "../ui/Logo";
import { IconSearch } from "../ui/Icons";

type Props = { children: ReactNode };

const TABS = [
  { to: "/",            label: "動画一覧",       match: (p: string) => p === "/" || p.startsWith("/videos") },
  { to: "/videos/new",  label: "アップロード",   match: (p: string) => p === "/videos/new" },
  { to: "/profile",     label: "プロフィール",   match: (p: string) => p === "/profile" },
];

/**
 * App-wide shell — horizontal topbar with search + avatar + tabs.
 * Used on every authenticated page (list / detail / upload / profile).
 */
export default function AppShell({ children }: Props) {
  const navigate = useNavigate();
  const location = useLocation();

  const { data: meRes } = useQuery({
    queryKey: ["me"],
    queryFn: () => getMeUsersMeGet({ headers: authHeaders() }),
  });
  const me = meRes?.status === 200 ? meRes.data : null;
  const initials = me?.display_name?.slice(0, 2).toUpperCase() ?? "—";

  return (
    <div className="grid h-full grid-rows-[auto_1fr] bg-bg">
      <header className="bg-surface border-b border-border">
        {/* Row 1 — brand, search, primary actions */}
        <div className="flex h-[52px] items-center gap-6 px-6">
          <Link to="/" className="no-underline text-fg">
            <Logo />
          </Link>

          <div className="hidden h-[18px] w-px bg-border md:block" />

          <div className="relative hidden w-[280px] md:block">
            <IconSearch
              size={14}
              className="pointer-events-none absolute left-2.5 top-[9px] text-fg-4"
            />
            <input
              readOnly
              placeholder="動画を検索…"
              className="w-full rounded-md border border-border bg-subtle py-[7px] pl-8 pr-3 text-[12.5px] text-fg-3 outline-none placeholder:text-fg-4 focus:border-accent focus:bg-surface focus:text-fg"
            />
          </div>

          <div className="flex-1" />

          <button
            onClick={() => navigate("/profile")}
            className="grid h-7 w-7 place-items-center rounded-full bg-accent-soft text-[11.5px] font-semibold text-accent-ink"
            aria-label="プロフィール"
          >
            {initials}
          </button>
        </div>

        {/* Row 2 — section tabs */}
        <nav className="flex items-stretch gap-6 border-t border-border px-6">
          {TABS.map((t) => {
            const active = t.match(location.pathname);
            return (
              <NavLink
                key={t.to}
                to={t.to}
                end={t.to === "/"}
                className={`flex h-[52px] items-center text-[13px] ${
                  active
                    ? "border-b-2 border-fg font-medium text-fg"
                    : "border-b-2 border-transparent text-fg-2 hover:text-fg"
                }`}
              >
                {t.label}
              </NavLink>
            );
          })}
        </nav>
      </header>

      <main className="overflow-hidden bg-bg">{children}</main>
    </div>
  );
}
