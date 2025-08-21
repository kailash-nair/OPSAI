
import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, Filter } from "lucide-react";

const PROCESSING_METHODS = [
  { value: "openai_whisper", label: "OpenAI Whisper" },
  { value: "faster_whisper", label: "Faster-Whisper" },
  { value: "hf_whisper_malayalam", label: "HF Whisper ML" },
  { value: "hf_wav2vec2", label: "HF Wav2Vec2" }
];

export default function SearchFilters({
  searchTerm,
  setSearchTerm,
  statusFilter,
  setStatusFilter,
  methodFilter,
  setMethodFilter,
  totalCount
}) {
  return (
    <Card className="glass-panel border-white/20 bg-transparent mb-8">
      <CardContent className="p-6">
        <div className="flex flex-col md:flex-row gap-4 items-end">
          {/* Search Input */}
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-white/50" />
              <Input
                placeholder="Search meetings by title, filename, or creator..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 glass-panel border-white/30 bg-transparent text-white"
              />
            </div>
          </div>

          {/* Status Filter */}
          <div className="w-full md:w-48">
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="glass-panel border-white/30 bg-transparent text-white">
                <SelectValue placeholder="All Statuses" />
              </SelectTrigger>
              <SelectContent className="glass-panel border-white/30 bg-gray-900/90 backdrop-blur-xl">
                <SelectItem value="all" className="text-white hover:bg-white/10">All Statuses</SelectItem>
                <SelectItem value="uploaded" className="text-white hover:bg-white/10">Uploaded</SelectItem>
                <SelectItem value="processing" className="text-white hover:bg-white/10">Processing</SelectItem>
                <SelectItem value="completed" className="text-white hover:bg-white/10">Completed</SelectItem>
                <SelectItem value="failed" className="text-white hover:bg-white/10">Failed</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Method Filter */}
          <div className="w-full md:w-48">
            <Select value={methodFilter} onValueChange={setMethodFilter}>
              <SelectTrigger className="glass-panel border-white/30 bg-transparent text-white">
                <SelectValue placeholder="All Methods" />
              </SelectTrigger>
              <SelectContent className="glass-panel border-white/30 bg-gray-900/90 backdrop-blur-xl">
                <SelectItem value="all" className="text-white hover:bg-white/10">All Methods</SelectItem>
                {PROCESSING_METHODS.map((method) => (
                  <SelectItem 
                    key={method.value} 
                    value={method.value}
                    className="text-white hover:bg-white/10"
                  >
                    {method.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Results Count */}
        <div className="flex items-center gap-2 mt-4 text-white/70">
          <Filter className="w-4 h-4" />
          <span className="text-sm">
            Showing {totalCount} meeting{totalCount !== 1 ? 's' : ''}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
