import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FileText, Eye, Download, RefreshCw, Calendar, User, Copy, Check } from "lucide-react";
// import { format } from "date-fns"; // REMOVED
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Meeting } from "@/entities/Meeting"; // NEW: use new API helpers

const STATUS_CONFIG = {
  uploaded:   { color: "text-blue-300",   bg: "bg-blue-500/20",   border: "border-blue-400/30" },
  processing: { color: "text-yellow-300", bg: "bg-yellow-500/20", border: "border-yellow-400/30" },
  completed:  { color: "text-green-300",  bg: "bg-green-500/20",  border: "border-green-400/30" },
  failed:     { color: "text-red-300",    bg: "bg-red-500/20",    border: "border-red-400/30" }
};

const METHOD_LABELS = {
  openai_whisper: "OpenAI Whisper",
  faster_whisper: "Faster-Whisper",
  hf_whisper_malayalam: "HF Whisper ML",
  hf_wav2vec2: "HF Wav2Vec2"
};

// ---------- NEW: simple date formatter (no dependencies) ----------
function formatDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

export default function RecentMeetings({ meetings, onRefresh }) {
  const [showDetails, setShowDetails] = useState(false);
  const [active, setActive] = useState(null);
  const [copied, setCopied] = useState(false);

  // ---------- NEW: dialog mode + loaded content ----------
  const [mode, setMode] = useState("summary"); // "summary" | "transcript"
  const [loaded, setLoaded] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text || "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      console.error("Clipboard error:", e);
    }
  };

  const downloadText = (text, name, mime = "text/plain") => {
    if (!text) return;
    const blob = new Blob([text], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadSummary = (meeting) => {
    const content = meeting?.markdown_output || loaded; // prefer loaded when opened
    if (!content) return;
    const safe = (meeting.title || "meeting").replace(/[^a-z0-9]/gi, "_").toLowerCase();
    downloadText(content, `${safe}_summary.md`, "text/markdown");
  };

  // ---------- NEW ----------
  const downloadTranscript = (meeting) => {
    const content = loaded; // transcript loads into `loaded` when mode==="transcript"
    if (!content) return;
    const safe = (meeting.title || "meeting").replace(/[^a-z0-9]/gi, "_").toLowerCase();
    downloadText(content, `${safe}_transcript.txt`, "text/plain");
  };

  const formatFileSize = (bytes) => {
    if (!bytes && bytes !== 0) return "Unknown size";
    const sizes = ["Bytes", "KB", "MB", "GB"];
    if (bytes === 0) return "0 Byte";
    const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
    return Math.round((bytes / Math.pow(1024, i)) * 100) / 100 + " " + sizes[i];
  };

  // open dialog in a specific mode and fetch content
  const openSummary = async (m) => {
    setActive(m);
    setMode("summary");
    setShowDetails(true);
  };

  const openTranscript = async (m) => {
    setActive(m);
    setMode("transcript");
    setShowDetails(true);
  };

  // ---------- NEW: load content when dialog opens/mode changes ----------
  useEffect(() => {
    let ignore = false;
    async function load() {
      if (!showDetails || !active) return;
      setLoading(true); setErr(""); setLoaded("");
      try {
        if (mode === "summary") {
          // If the meeting already has markdown_output, use it (no round-trip).
          // Otherwise call the endpoint (useful after a refresh or if you want always-fresh).
          if (active.markdown_output) {
            setLoaded(active.markdown_output);
          } else {
            const { markdown } = await Meeting.getSummary(active.id);
            if (!ignore) setLoaded(markdown || "");
          }
        } else {
          const { text } = await Meeting.getTranscript(active.id);
          if (!ignore) setLoaded(text || "");
        }
      } catch (e) {
        if (!ignore) setErr(String(e.message || e));
      } finally {
        if (!ignore) setLoading(false);
      }
    }
    load();
    return () => { ignore = true; };
  }, [showDetails, active, mode]);

  return (
    <>
      <Card className="glass-panel border-white/20 bg-white/10">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Recent Meetings ({meetings.length})
            </CardTitle>
            <Button
              onClick={onRefresh}
              variant="ghost"
              size="icon"
              className="text-white/70 hover:text-white hover:bg-white/10"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {meetings.length > 0 ? (
            <div className="space-y-4">
              {meetings.map((meeting) => (
                <Card key={meeting.id} className="glass-panel border-white/10 bg-white/5">
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <h4 className="font-semibold text-white mb-1">{meeting.title}</h4>
                        <p className="text-white/60 text-sm mb-2">{meeting.file_name}</p>

                        <div className="flex flex-wrap items-center gap-3 text-sm text-white/70">
                          <div className="flex items-center gap-1">
                            <Calendar className="w-4 h-4" />
                            {formatDate(meeting.created_date)}
                          </div>
                          <div className="flex items-center gap-1">
                            <User className="w-4 h-4" />
                            {meeting.created_by}
                          </div>
                          <span>{formatFileSize(meeting.file_size)}</span>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <Badge
                          className={`${STATUS_CONFIG[meeting.status]?.bg} ${STATUS_CONFIG[meeting.status]?.border} ${STATUS_CONFIG[meeting.status]?.color} border`}
                        >
                          {meeting.status}
                        </Badge>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <Badge variant="outline" className="border-white/30 text-white/70 text-xs">
                        {METHOD_LABELS[meeting.processing_method]}
                      </Badge>

                      <div className="flex gap-2">
                        {/* Open original file if present */}
                        {meeting.original_file_url && (
                          <Button
                            size="sm"
                            variant="ghost"
                            className="text-white/70 hover:text-white hover:bg-white/10"
                            onClick={() => window.open(meeting.original_file_url, "_blank")}
                            title="Open original file"
                          >
                            <Eye className="w-4 h-4" />
                          </Button>
                        )}

                        {/* ---------- NEW: Separate buttons ---------- */}
                        <Button
                          size="sm"
                          className="glass-button text-white hover:text-white"
                          onClick={() => openSummary(meeting)}
                          title="View Summary"
                        >
                          Summary
                        </Button>

                        <Button
                          size="sm"
                          variant="outline"
                          className="border-white/30 text-white hover:bg-white/10"
                          onClick={() => openTranscript(meeting)}
                          title="View Full English Transcript"
                        >
                          Transcript
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="w-16 h-16 mx-auto mb-4 glass-panel rounded-full flex items-center justify-center">
                <FileText className="w-8 h-8 text-white/50" />
              </div>
              <p className="text-white/60">No meetings found</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Details Dialog */}
      <Dialog open={showDetails} onOpenChange={setShowDetails}>
        <DialogContent className="glass-panel border-white/30 bg-gray-900/95 backdrop-blur-xl max-w-4xl max-h-[80vh] overflow-auto text-white">
          <DialogHeader>
            <DialogTitle className="text-white text-xl">
              {mode === "summary" ? "Meeting Summary" : "Full English Transcript"} — {active?.title || "Meeting"}
            </DialogTitle>
          </DialogHeader>

          {active ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-white/60">Status</p>
                  <Badge className={`${STATUS_CONFIG[active.status]?.bg} ${STATUS_CONFIG[active.status]?.color} mt-1`}>
                    {active.status}
                  </Badge>
                </div>
                <div>
                  <p className="text-white/60">Created</p>
                  <p className="text-white font-medium">{formatDate(active.created_date)}</p>
                </div>
                <div>
                  <p className="text-white/60">File Size</p>
                  <p className="text-white font-medium">{formatFileSize(active.file_size)}</p>
                </div>
                <div>
                  <p className="text-white/60">Method</p>
                  <p className="text-white font-medium">{METHOD_LABELS[active.processing_method]}</p>
                </div>
              </div>

              {/* Content area */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="text-lg font-semibold text-white">
                    {mode === "summary" ? "Summary" : "Transcript"}
                  </div>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      className="border-white/30 text-white hover:bg-white/10"
                      onClick={() => copyToClipboard(loaded)}
                      disabled={!loaded}
                      title="Copy to clipboard"
                    >
                      {copied ? <Check className="w-4 h-4 mr-2" /> : <Copy className="w-4 h-4 mr-2" />}
                      {copied ? "Copied!" : "Copy"}
                    </Button>

                    {mode === "summary" ? (
                      <Button
                        size="sm"
                        className="glass-button text-white hover:text-white"
                        onClick={() => downloadSummary(active)}
                        disabled={!loaded}
                        title="Download summary (.md)"
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Download
                      </Button>
                    ) : (
                      <Button
                        size="sm"
                        className="glass-button text-white hover:text-white"
                        onClick={() => downloadTranscript(active)}
                        disabled={!loaded}
                        title="Download transcript (.txt)"
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Download
                      </Button>
                    )}
                  </div>
                </div>

                <div className="glass-panel border-white/20 bg-white/5 rounded-lg p-4 max-h-96 overflow-auto">
                  {loading && <div className="text-white/70">Loading…</div>}
                  {err && <div className="text-red-300">{err}</div>}
                  {!loading && !err && (
                    mode === "summary"
                      ? <pre className="whitespace-pre-wrap text-white/90 text-sm font-mono">{loaded || "—"}</pre>
                      : <pre className="whitespace-pre-wrap text-white/90 text-sm font-mono">{loaded || "—"}</pre>
                  )}
                </div>
              </div>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </>
  );
}
