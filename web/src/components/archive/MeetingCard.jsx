import React, { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FileText, Download, Eye, Calendar, User, Clock, Copy, Check } from "lucide-react";
import { format } from "date-fns";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const STATUS_CONFIG = {
  "uploaded": { color: "text-blue-300", bg: "bg-blue-500/20", border: "border-blue-400/30" },
  "processing": { color: "text-yellow-300", bg: "bg-yellow-500/20", border: "border-yellow-400/30" },
  "completed": { color: "text-green-300", bg: "bg-green-500/20", border: "border-green-400/30" },
  "failed": { color: "text-red-300", bg: "bg-red-500/20", border: "border-red-400/30" }
};

const METHOD_LABELS = {
  "openai_whisper": "OpenAI Whisper",
  "faster_whisper": "Faster-Whisper",
  "hf_whisper_malayalam": "HF Whisper ML",
  "hf_wav2vec2": "HF Wav2Vec2"
};

export default function MeetingCard({ meeting }) {
  const [showDetails, setShowDetails] = useState(false);
  const [copied, setCopied] = useState(false);

  const formatFileSize = (bytes) => {
    if (!bytes) return "Unknown size";
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Byte';
    const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error("Failed to copy to clipboard:", error);
    }
  };

  const downloadSummary = () => {
    if (!meeting.markdown_output) return;
    
    const blob = new Blob([meeting.markdown_output], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${meeting.title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_summary.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <Card className="glass-panel border-white/20 bg-white/10 hover:bg-white/15 transition-all duration-300 group">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 glass-panel rounded-lg flex items-center justify-center">
                <FileText className="w-5 h-5 text-blue-400" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-white truncate">{meeting.title}</h3>
                <p className="text-white/60 text-sm truncate">{meeting.file_name}</p>
              </div>
            </div>
            <Badge 
              className={`${STATUS_CONFIG[meeting.status]?.bg} ${STATUS_CONFIG[meeting.status]?.border} ${STATUS_CONFIG[meeting.status]?.color} border text-xs`}
            >
              {meeting.status}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Meeting Info */}
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2 text-white/70">
              <Calendar className="w-4 h-4" />
              <span>{format(new Date(meeting.created_date), "MMM d, yyyy")}</span>
            </div>
            <div className="flex items-center gap-2 text-white/70">
              <User className="w-4 h-4" />
              <span className="truncate">{meeting.created_by}</span>
            </div>
            <div className="flex items-center gap-2 text-white/70">
              <Clock className="w-4 h-4" />
              <span>{formatFileSize(meeting.file_size)}</span>
            </div>
          </div>

          {/* Processing Method */}
          <Badge variant="outline" className="border-white/30 text-white/80 text-xs">
            {METHOD_LABELS[meeting.processing_method]}
          </Badge>

          {/* Action Buttons */}
          <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            {meeting.status === "completed" && meeting.markdown_output && (
              <>
                <Button
                  size="sm"
                  variant="ghost"
                  className="flex-1 text-white/70 hover:text-white hover:bg-white/10"
                  onClick={() => copyToClipboard(meeting.markdown_output)}
                >
                  {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="flex-1 text-white/70 hover:text-white hover:bg-white/10"
                  onClick={downloadSummary}
                >
                  <Download className="w-4 h-4" />
                </Button>
              </>
            )}
            <Button
              size="sm"
              className="flex-1 glass-button text-white hover:text-white"
              onClick={() => setShowDetails(true)}
            >
              <Eye className="w-4 h-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Details Modal */}
      <Dialog open={showDetails} onOpenChange={setShowDetails}>
        <DialogContent className="glass-panel border-white/30 bg-gray-900/95 backdrop-blur-xl max-w-4xl max-h-[80vh] overflow-auto text-white">
          <DialogHeader>
            <DialogTitle className="text-white text-xl">{meeting.title}</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-6">
            {/* Meeting Details */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-white/60">Status</p>
                <Badge 
                  className={`${STATUS_CONFIG[meeting.status]?.bg} ${STATUS_CONFIG[meeting.status]?.color} mt-1`}
                >
                  {meeting.status}
                </Badge>
              </div>
              <div>
                <p className="text-white/60">Created</p>
                <p className="text-white font-medium">{format(new Date(meeting.created_date), "MMM d, yyyy")}</p>
              </div>
              <div>
                <p className="text-white/60">File Size</p>
                <p className="text-white font-medium">{formatFileSize(meeting.file_size)}</p>
              </div>
              <div>
                <p className="text-white/60">Method</p>
                <p className="text-white font-medium">{METHOD_LABELS[meeting.processing_method]}</p>
              </div>
            </div>

            {/* Summary Content */}
            {meeting.status === "completed" && meeting.markdown_output && (
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">Meeting Summary</h3>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      className="border-white/30 text-white hover:bg-white/10"
                      onClick={() => copyToClipboard(meeting.markdown_output)}
                    >
                      {copied ? <Check className="w-4 h-4 mr-2" /> : <Copy className="w-4 h-4 mr-2" />}
                      {copied ? "Copied!" : "Copy"}
                    </Button>
                    <Button
                      size="sm"
                      className="glass-button text-white hover:text-white"
                      onClick={downloadSummary}
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </Button>
                  </div>
                </div>
                <div className="glass-panel border-white/20 bg-white/5 rounded-lg p-4 max-h-96 overflow-auto">
                  <pre className="whitespace-pre-wrap text-white/90 text-sm font-mono">
                    {meeting.markdown_output}
                  </pre>
                </div>
              </div>
            )}

            {/* Processing Status */}
            {meeting.status === "processing" && (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-2 border-white border-t-transparent mx-auto mb-4" />
                <p className="text-white/70">This meeting is still being processed...</p>
              </div>
            )}

            {meeting.status === "failed" && (
              <div className="text-center py-8">
                <div className="w-16 h-16 mx-auto mb-4 glass-panel rounded-full flex items-center justify-center">
                  <FileText className="w-8 h-8 text-red-400" />
                </div>
                <p className="text-red-300">Processing failed for this meeting</p>
                <p className="text-white/60 text-sm mt-2">Please try uploading again</p>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}