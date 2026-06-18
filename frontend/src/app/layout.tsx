import type { Metadata } from "next";
import { Inter, Space_Mono } from "next/font/google";
import "./globals.css";
import { TopNav } from "@/components/layout/topnav";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const spaceMono = Space_Mono({ weight: ["400", "700"], subsets: ["latin"], variable: "--font-space-mono" });

export const metadata: Metadata = {
  title: "Crypto Filter Pro",
  description: "Advanced quantitative analysis platform for cryptocurrency markets",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${spaceMono.variable} font-sans antialiased bg-background text-foreground min-h-screen flex flex-col selection:bg-primary/30`}>
        <TopNav />
        <main className="flex-1 pt-24 px-8 pb-8 relative min-h-screen overflow-x-hidden">
          {children}
        </main>
      </body>
    </html>
  );
}
