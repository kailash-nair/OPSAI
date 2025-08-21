
import React, { useState, useRef } from "react";
import { Meeting } from "@/entities/Meeting";
import { UploadFile } from "@/integrations/Core";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Upload, FileVideo, FileAudio, Zap, AlertCircle, CheckCircle } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { createPageUrl } from "@/utils";

const PROCESSING_METHODS = [
  {
    value: "openai_whisper",
    label: "OpenAI Whisper API (Cloud)",
    description: "High accuracy, requires internet connection",
    icon: "🌐"
  },
  {
    value: "faster_whisper",
    label: "Faster-Whisper (Local)",
    description: "Fast processing, works offline",
    icon: "⚡"
  },
  {
    value: "hf_whisper_malayalam",
    label: "HF Whisper-Medium Malayalam (Local)",
    description: "Fine-tuned for Malayalam, then LLM translation",
    icon: "🎯"
  },
  {
    value: "hf_wav2vec2",
    label: "HF Wav2Vec2 Malayalam (Local)",
    description: "Local CTC model with LLM translation",
    icon: "🔧"
  }
];

export default function UploadPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [meetingTitle, setMeetingTitle] = useState("");
  const [processingMethod, setProcessingMethod] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleFileSelect = (file) => {
    const validTypes = ['video/mp4', 'video/mov', 'video/avi', 'audio/mp3', 'audio/wav', 'audio/m4a'];
    if (!validTypes.includes(file.type)) {
      setUploadError("Please upload a video (MP4, MOV, AVI) or audio file (MP3, WAV, M4A)");
      return;
    }
    
    setSelectedFile(file);
    setUploadError(null);
    if (!meetingTitle) {
      setMeetingTitle(file.name.split('.')[0]);
    }
  };

  const handleUpload = async () => {
  if (!selectedFile || !meetingTitle || !processingMethod) {
    setUploadError("Please fill in all required fields");
    return;
  }

  setIsUploading(true);
  setUploadError(null);

  try {
    // 1) Upload the file
    const resp = await UploadFile({ file: selectedFile });
    // Be tolerant of different key names just in case
    const fileUrl  = resp.file_url  ?? resp.fileUrl  ?? resp.url;
    const filePath = resp.file_path ?? resp.filePath ?? resp.path ?? "";

    if (!fileUrl) {
      throw new Error("Upload response missing file_url");
    }

    // (Optional) Inspect what came back
    // console.log("Upload response:", resp);

    // 2) Create the meeting record
    await Meeting.create({
      title: meetingTitle,
      original_file_url: fileUrl,
      original_file_path: filePath,        // <— SAFE even if empty (mock mode)
      file_name: selectedFile.name,
      file_size: selectedFile.size,
      processing_method: processingMethod,
      status: "uploaded",
    });

    setUploadSuccess(true);
    setTimeout(() => {
      navigate(createPageUrl("Dashboard"));
    }, 1200);
  } catch (error) {
    console.error("Upload error:", error);
    setUploadError(error?.message || "Failed to upload meeting. Please try again.");
  } finally {
    setIsUploading(false);
  }
};

  const formatFileSize = (bytes) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Byte';
    const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white gradient-text mb-2">
            Upload Meeting Recording
          </h1>
          <p className="text-white/70 text-lg">
            Transform your Malayalam operations meetings into structured English summaries
          </p>
        </div>

        {/* TEMP sanity check button — you can remove later */}
        <Button className="glass-button text-white hover:text-white mb-6">
          Shadcn Button Renders
        </Button>

        {uploadError && (
          <Alert className="mb-6 glass-panel border-red-300/30 bg-red-500/10">
            <AlertCircle className="h-4 w-4 text-red-400" />
            <AlertDescription className="text-red-200">{uploadError}</AlertDescription>
          </Alert>
        )}

        {uploadSuccess && (
          <Alert className="mb-6 glass-panel border-green-300/30 bg-green-500/10">
            <CheckCircle className="h-4 w-4 text-green-400" />
            <AlertDescription className="text-green-200">
              Meeting uploaded successfully! Redirecting to dashboard...
            </AlertDescription>
          </Alert>
        )}

        <div className="space-y-6">
          {/* File Upload Card */}
          <Card className="glass-panel border-white/20 bg-transparent">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Select Recording File
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div
                className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
                  dragActive 
                    ? "border-blue-400/50 bg-blue-500/10" 
                    : "border-white/30 hover:border-white/50"
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*,audio/*"
                  onChange={handleFileInput}
                  className="hidden"
                />
                
                {!selectedFile ? (
                  <div>
                    <div className="w-20 h-20 mx-auto mb-4 glass-panel rounded-full flex items-center justify-center">
                      <FileVideo className="w-10 h-10 text-white/70" />
                    </div>
                    <h3 className="text-xl font-semibold text-white mb-2">
                      Drop your meeting recording here
                    </h3>
                    <p className="text-white/60 mb-4">
                      Supports MP4, MOV, AVI, MP3, WAV, M4A files
                    </p>
                    <Button
                      onClick={() => fileInputRef.current?.click()}
                      className="glass-button text-white hover:text-white"
                    >
                      Browse Files
                    </Button>
                  </div>
                ) : (
                  <div className="glass-panel rounded-lg p-4">
                    <div className="flex items-center gap-4">
                      {selectedFile.type.startsWith('video/') ? (
                        <FileVideo className="w-8 h-8 text-blue-400" />
                      ) : (
                        <FileAudio className="w-8 h-8 text-purple-400" />
                      )}
                      <div className="flex-1 text-left">
                        <p className="font-medium text-white">{selectedFile.name}</p>
                        <p className="text-white/60">{formatFileSize(selectedFile.size)}</p>
                      </div>
                      <Button
                        onClick={() => setSelectedFile(null)}
                        variant="ghost"
                        className="text-white/70 hover:text-white hover:bg-white/10"
                      >
                        Remove
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Meeting Details Card */}
          <Card className="glass-panel border-white/20 bg-transparent">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <FileVideo className="w-5 h-5" />
                Meeting Details
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="title" className="text-white/90">Meeting Title</Label>
                <Input
                  id="title"
                  value={meetingTitle}
                  onChange={(e) => setMeetingTitle(e.target.value)}
                  placeholder="Enter meeting title..."
                  className="mt-2 glass-panel border-white/30 bg-transparent text-white"
                />
              </div>

              <div>
                <Label className="text-white/90">Processing Method</Label>
                <Select value={processingMethod} onValueChange={setProcessingMethod}>
                  <SelectTrigger className="mt-2 glass-panel border-white/30 bg-transparent text-white">
                    <SelectValue placeholder="Choose processing method..." />
                  </SelectTrigger>
                  <SelectContent className="glass-panel border-white/30 bg-gray-900/80 backdrop-blur-xl">
                    {PROCESSING_METHODS.map((method) => (
                      <SelectItem 
                        key={method.value} 
                        value={method.value}
                        className="text-white hover:bg-white/10"
                      >
                        <div className="flex items-center gap-3">
                          <span className="text-lg">{method.icon}</span>
                          <div>
                            <div className="font-medium">{method.label}</div>
                            <div className="text-sm text-white/60">{method.description}</div>
                          </div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Action Button */}
          <div className="flex justify-center">
            <Button
              onClick={handleUpload}
              disabled={!selectedFile || !meetingTitle || !processingMethod || isUploading}
              className="glass-button text-white hover:text-white px-8 py-3 text-lg font-medium"
            >
              {isUploading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent mr-2" />
                  Uploading...
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5 mr-2" />
                  Start Processing
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
