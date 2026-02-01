import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Clock, FileVideo, Shield, ShieldOff } from "lucide-react";
import { format } from "date-fns";

const METHOD_LABELS = {
  "openai_whisper": "OpenAI Whisper",
  "faster_whisper": "Faster-Whisper",
  "hf_whisper_malayalam": "HF Whisper ML",
  "hf_wav2vec2": "HF Wav2Vec2"
};

export default function ProcessingCard({ meeting }) {
  return (
    <Card className="glass-panel border-white/30 bg-white/5">
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 glass-panel rounded-lg flex items-center justify-center">
              <FileVideo className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <h4 className="font-medium text-white">{meeting.title}</h4>
              <p className="text-white/60 text-sm">{meeting.file_name}</p>
            </div>
          </div>
          <Badge className="glass-panel border-yellow-400/30 bg-yellow-500/20 text-yellow-300">
            <div className="animate-pulse w-2 h-2 bg-yellow-400 rounded-full mr-2" />
            Processing
          </Badge>
        </div>

        <div className="space-y-3">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-white/70">Progress</span>
              <span className="text-white">{meeting.progress || 0}%</span>
            </div>
            <Progress 
              value={meeting.progress || 0} 
              className="h-2 bg-white/10"
            />
          </div>

          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2 text-white/70">
              <Clock className="w-4 h-4" />
              {format(new Date(meeting.created_date), "MMM d, h:mm a")}
            </div>
            <div className="flex items-center gap-2">
              {meeting.validate_summary ? (
                <Shield className="w-4 h-4 text-green-400" title="Validation enabled" />
              ) : (
                <ShieldOff className="w-4 h-4 text-white/40" title="Validation disabled" />
              )}
              <Badge variant="outline" className="border-white/30 text-white/80 text-xs">
                {METHOD_LABELS[meeting.processing_method]}
              </Badge>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}