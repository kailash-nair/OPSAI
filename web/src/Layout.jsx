import React from "react";
import { Link, useLocation } from "react-router-dom";
import { createPageUrl } from "@/utils";
import { Mic, Upload, Zap, Archive } from "lucide-react";

import {
  SidebarProvider,
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarTrigger,
  SidebarInset,
} from "@/components/ui/sidebar";

export default function Layout({ children }) {
  const location = useLocation();

  const nav = [
    { title: "Upload Recording", url: createPageUrl("Upload"), icon: Upload },
    { title: "Processing Dashboard", url: createPageUrl("Dashboard"), icon: Zap },
    { title: "Meeting Archive", url: createPageUrl("Archive"), icon: Archive },
  ];

  return (
    <SidebarProvider>
      <style>{`
        :root {
          /* Base design tokens */
          --glass-bg: rgba(30, 27, 46, 0.3);
          --glass-border: rgba(255, 255, 255, 0.15);
          --glass-shadow: rgba(0, 0, 0, 0.25);
          --text-primary: rgba(255, 255, 255, 0.98);
          --text-secondary: rgba(230, 230, 250, 0.8);

          /* Sidebar sizing */
          --sidebar-width: 16rem;            /* expanded width (matches your original) */
          --sidebar-width-icon: 4rem;        /* collapsed icon rail width */
        }

        body {
          background: linear-gradient(135deg, #1a1d3d 0%, #2e2240 50%, #4a2f52 100%);
          min-height: 100vh;
          color: var(--text-primary);
          text-shadow: 0 1px 3px rgba(0,0,0,.4);
        }
        .glass-panel { background: var(--glass-bg); backdrop-filter: blur(25px); -webkit-backdrop-filter: blur(25px); border: 1px solid var(--glass-border); box-shadow: 0 8px 32px var(--glass-shadow); }
        .glass-button { background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.2); transition: all .25s ease; }
        .glass-button:hover { background: rgba(255,255,255,.15); transform: translateY(-1px); }
        .glass-active { background: rgba(255,255,255,.2); box-shadow: inset 0 2px 4px rgba(0,0,0,.2); }

        /* Ambient glows (unchanged design) */
        .ambient {
          position: fixed; inset: 0; overflow: hidden; pointer-events: none; z-index: -1;
        }

        /* COLLAPSE FIX:
           1) When the <Sidebar> is collapsed, use :has() to set the shared
              --sidebar-width variable on the root container to the icon width.
           2) The content wrapper (SidebarInset) uses that variable for left padding.
        */
        .sidebar-root:has(> aside[data-collapsible="icon"][data-state="collapsed"]) {
          --sidebar-width: var(--sidebar-width-icon);
        }
        .sidebar-inset {
          padding-left: var(--sidebar-width);
          transition: padding-left .2s ease;
        }
        @media (max-width: 768px) {
          /* On mobile, sidebar becomes off-canvas; don't reserve space */
          .sidebar-inset { padding-left: 0 !important; }
        }

        /* Hide label text when the sidebar itself is collapsed (icon rail) */
        aside[data-collapsible="icon"][data-state="collapsed"] .sidebar-label {
          display: none;
        }
        /* Center icons when collapsed */
        aside[data-collapsible="icon"][data-state="collapsed"] .sidebar-item {
          justify-content: center;
          padding-left: .5rem;
          padding-right: .5rem;
        }
      `}</style>

      {/* Ambient gradient blobs (keep original look) */}
      <div className="ambient">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-r from-purple-400/20 to-pink-400/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-gradient-to-r from-blue-400/20 to-cyan-400/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 right-1/3 w-64 h-64 bg-gradient-to-r from-orange-400/20 to-red-400/20 rounded-full blur-3xl animate-pulse delay-2000"></div>
      </div>

      {/* Root container so :has() can detect the collapsed sidebar and adjust content */}
      <div className="sidebar-root min-h-screen w-full relative flex">
        {/* Collapsible sidebar (expanded → icon rail) */}
        <Sidebar
          variant="sidebar"
          side="left"
          collapsible="icon"
          className="glass-panel border-r border-white/10 w-[var(--sidebar-width)] transition-[width] duration-200"
        >
          <SidebarHeader className="px-4 py-3 border-b border-white/10">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl bg-white/10 flex items-center justify-center">
                  <Mic className="w-5 h-5" />
                </div>
                <div className="leading-tight">
                  <h2 className="font-bold text-white">OpsAI Agent</h2>
                  <p className="text-xs text-white/70">Malayalam → English</p>
                </div>
              </div>
              {/* Desktop collapse/expand trigger */}
              <SidebarTrigger className="glass-button p-2 rounded-lg hidden md:inline-flex" />
            </div>
          </SidebarHeader>

          <SidebarContent className="py-2">
            <SidebarGroup>
              <SidebarGroupLabel className="px-3 text-xs uppercase tracking-wider text-white/60">
                Navigation
              </SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {nav.map((item) => {
                    const active = location.pathname === item.url;
                    return (
                      <SidebarMenuItem key={item.title}>
                        <SidebarMenuButton
                          asChild
                          isActive={active}
                          className={`sidebar-item glass-button text-white hover:text-white rounded-lg mb-2 ${active ? "glass-active" : ""}`}
                        >
                          <Link to={item.url} className="flex items-center gap-3 w-full px-3 py-2">
                            <item.icon className="w-5 h-5 shrink-0" />
                            <span className="sidebar-label font-medium truncate">{item.title}</span>
                          </Link>
                        </SidebarMenuButton>
                      </SidebarMenuItem>
                    );
                  })}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>
        </Sidebar>

        {/* Content area; padding-left auto-syncs with sidebar width */}
        <SidebarInset className="sidebar-inset flex min-h-screen flex-1 flex-col">
          {/* Mobile top bar with trigger to open off-canvas */}
          <header className="md:hidden sticky top-0 z-20 px-4 py-3 bg-black/20 backdrop-blur border-b border-white/10 flex items-center gap-3">
            <SidebarTrigger className="glass-button p-2 rounded-lg" />
            <h1 className="text-base font-semibold">OpsAI Agent</h1>
          </header>

          {/* Page content */}
          <div className="flex-1">{children}</div>
        </SidebarInset>
      </div>
    </SidebarProvider>
  );
}
