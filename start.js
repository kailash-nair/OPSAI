/**
 * Cross‑platform one-click starter:
 * - Runs `npm run dev` in /server and /web
 * - Waits for a dev URL and opens the browser
 */

const { spawn } = require("child_process");
const path = require("path");
const http = require("http");
const https = require("https");

// dynamic import for ESM-only 'open' package
const openBrowser = async (url) => {
  const mod = await import("open");
  return mod.default(url);
};

// Adjust if needed
const SERVER_DIR = path.join(__dirname, "server");
const WEB_DIR = path.join(__dirname, "web");

// Common dev URLs to probe
const CANDIDATE_URLS = [
  "http://localhost:3000",
  "http://localhost:5173",
  "http://localhost:8080",
  "http://localhost:8000"
];

function run(command, cwd) {
  const child = spawn(command, { cwd, stdio: "inherit", shell: true });
  child.on("close", (code) => {
    if (code !== 0) console.error("Process exited in ${cwd} with code ${code}");
  });
  return child;
}

function sleep(ms) {
  return new Promise((res) => setTimeout(res, ms));
}

function checkUrlOnce(url, timeoutMs = 1500) {
  return new Promise((resolve) => {
    const lib = url.startsWith("https") ? https : http;
    const req = lib.get(url, (res) => {
      // Treat <400 as reachable
      const ok = res.statusCode && res.statusCode < 400;
      res.resume();
      resolve(ok);
    });
    req.on("error", () => resolve(false));
    req.setTimeout(timeoutMs, () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForAnyUrl(urls, attempts = 40, delayMs = 500) {
  for (let i = 0; i < attempts; i++) {
    for (const url of urls) {
      if (await checkUrlOnce(url)) return url;
    }
    await sleep(delayMs);
  }
  return null;
}

(async () => {
  console.log("Starting AI Agent…");

  console.log("Launching server: npm run dev");
  run("npm run dev", SERVER_DIR);

  console.log("Launching web: npm run dev");
  run("npm run dev", WEB_DIR);

  console.log("⏳ Waiting for frontend to become reachable…");
  const url = await waitForAnyUrl(CANDIDATE_URLS, 40, 500); // ~20s max

  if (url) {
    console.log("Opening browser at ${url}");
    await openBrowser(url);
  } else {
    const fallback = CANDIDATE_URLS[1] || CANDIDATE_URLS[0]; // likely Vite: 5173
    console.log("Couldn’t confirm server. Opening fallback: ${fallback}");
    await openBrowser(fallback);
  }
})();
