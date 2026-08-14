import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

/**
 * Three type roles, used strictly (CLAUDE.md §11): Fraunces for headings only,
 * Public Sans for prose, JetBrains Mono for anything telemetric — the trace,
 * metrics, model names. Telemetry should read as telemetry.
 *
 * Loaded by the browser rather than through `next/font/google`, which fetches
 * the files during `next build`. That makes the build depend on reaching Google,
 * and it fails on a restricted network. `display=swap` plus the fallback stacks
 * in tailwind.config.ts mean text is readable before the fonts land.
 */
const FONTS_HREF =
  "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Public+Sans:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap";

export const metadata: Metadata = {
  title: "Cascade",
  description:
    "Routes your question to the cheapest model that can answer it correctly, checks the answer, and escalates only when the check fails.",
};

const NAV = [
  { href: "/", label: "Ask" },
  { href: "/history", label: "History" },
  { href: "/metrics", label: "Metrics" },
];

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link rel="stylesheet" href={FONTS_HREF} />
      </head>
      <body className="min-h-screen">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-ink focus:px-3 focus:py-2 focus:text-paper"
        >
          Skip to content
        </a>

        <header className="border-b border-ink/10">
          <div className="mx-auto flex max-w-3xl items-baseline justify-between px-5 py-5">
            <Link href="/" className="font-display text-lg font-semibold tracking-tight">
              Cascade
            </Link>
            <nav aria-label="Main">
              <ul className="flex gap-5 font-mono text-xs uppercase tracking-widest text-muted">
                {NAV.map((item) => (
                  <li key={item.href}>
                    <Link href={item.href} className="hover:text-ink">
                      {item.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </nav>
          </div>
        </header>

        <main id="main" className="mx-auto max-w-3xl px-5 py-10">
          {children}
        </main>

        <footer className="mx-auto max-w-3xl px-5 pb-12 pt-6 text-xs text-muted">
          <p>
            Costs shown are estimates from published per-token prices. Every model
            call runs on a free tier, so no money is spent.
          </p>
        </footer>
      </body>
    </html>
  );
}
