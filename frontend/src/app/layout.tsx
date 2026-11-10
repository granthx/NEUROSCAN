import type { Metadata } from "next";
import { Barlow_Condensed, IBM_Plex_Mono, VT323 } from "next/font/google";
import "./globals.css";
import { DISCLAIMER, PROJECT_NAME, PROJECT_TAGLINE } from "@/lib/config";

const vt323 = VT323({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-vt323",
  display: "swap",
});

const plex = IBM_Plex_Mono({
  weight: ["400", "500", "600", "700"],
  subsets: ["latin"],
  variable: "--font-plex",
  display: "swap",
});

const barlow = Barlow_Condensed({
  weight: ["600", "700"],
  subsets: ["latin"],
  variable: "--font-barlow",
  display: "swap",
});

export const metadata: Metadata = {
  title: `${PROJECT_NAME} — ${PROJECT_TAGLINE}`,
  description: `${PROJECT_TAGLINE} (T1, T1ce, T2, FLAIR). ${DISCLAIMER}`,
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "any" },
      { url: "/icon.svg", type: "image/svg+xml" },
      { url: "/icon.png", type: "image/png" },
    ],
    apple: [
      { url: "/apple-icon.png" },
    ],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      data-crt="full"
      className={`${vt323.variable} ${plex.variable} ${barlow.variable}`}
    >
      <body className="min-h-screen bg-crt text-phos antialiased">{children}</body>
    </html>
  );
}

