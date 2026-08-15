import type { Config } from "tailwindcss";

/**
 * Tier colours are semantic, not decorative: the same three carry tier identity
 * everywhere — trace, badges, charts, history rows — so a reader learns the
 * mapping once (CLAUDE.md §11).
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#F2F4F1",
        ink: "#14171A",
        muted: "#6B7280",
        t1: "#4B7B6E",
        t2: "#C08A2E",
        t3: "#8B4A6B",
      },
      fontFamily: {
        display: ["Fraunces", "Georgia", "serif"],
        body: ["'Public Sans'", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "SFMono-Regular", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
