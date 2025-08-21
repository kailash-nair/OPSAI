import React, { useState, useEffect } from "react";
import { Meeting } from "@/entities/Meeting";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input.jsx";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge.jsx";
import { Search, Archive, Download, Eye, Calendar, Filter } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select.jsx";
import { format } from "date-fns";

import MeetingCard from "../components/archive/MeetingCard";
import SearchFilters from "../components/archive/SearchFilters";

export default function ArchivePage() {
  const [meetings, setMeetings] = useState([]);
  const [filteredMeetings, setFilteredMeetings] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [methodFilter, setMethodFilter] = useState("all");

  useEffect(() => {
    loadMeetings();
  }, []);

  useEffect(() => {
    filterMeetings();
  }, [meetings, searchTerm, statusFilter, methodFilter]);

  const loadMeetings = async () => {
    setIsLoading(true);
    try {
      const data = await Meeting.list("-created_date");
      setMeetings(data);
    } catch (error) {
      console.error("Error loading meetings:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const filterMeetings = () => {
    let filtered = meetings;

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(meeting =>
        meeting.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        meeting.file_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        meeting.created_by.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Status filter
    if (statusFilter !== "all") {
      filtered = filtered.filter(meeting => meeting.status === statusFilter);
    }

    // Method filter
    if (methodFilter !== "all") {
      filtered = filtered.filter(meeting => meeting.processing_method === methodFilter);
    }

    setFilteredMeetings(filtered);
  };

  const exportAllSummaries = () => {
    const completedMeetings = filteredMeetings.filter(m => m.status === "completed" && m.markdown_output);
    
    if (completedMeetings.length === 0) {
      alert("No completed meetings with summaries found");
      return;
    }

    const combinedContent = completedMeetings.map(meeting => {
      return `# ${meeting.title}\n\n**Date:** ${format(new Date(meeting.created_date), "MMMM d, yyyy")}\n**Processing Method:** ${meeting.processing_method}\n\n${meeting.markdown_output}\n\n---\n\n`;
    }).join("");

    const blob = new Blob([combinedContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `meeting-summaries-${format(new Date(), "yyyy-MM-dd")}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen p-4 md:p-8">
        <div className="max-w-7xl mx-auto">
          <div className="animate-pulse space-y-6">
            <div className="glass-panel rounded-xl h-20"></div>
            <div className="glass-panel rounded-xl h-16"></div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Array(6).fill(0).map((_, i) => (
                <div key={i} className="glass-panel rounded-xl h-48"></div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
              <h1 className="text-4xl font-bold text-white gradient-text mb-2">
                Meeting Archive
              </h1>
              <p className="text-white/70 text-lg">
                Browse and search through all your processed meetings
              </p>
            </div>
            <div className="flex gap-3">
              <Button
                onClick={exportAllSummaries}
                className="glass-button text-white hover:text-white"
                disabled={filteredMeetings.filter(m => m.status === "completed").length === 0}
              >
                <Download className="w-4 h-4 mr-2" />
                Export All Summaries
              </Button>
            </div>
          </div>
        </div>

        {/* Search and Filters */}
        <SearchFilters
          searchTerm={searchTerm}
          setSearchTerm={setSearchTerm}
          statusFilter={statusFilter}
          setStatusFilter={setStatusFilter}
          methodFilter={methodFilter}
          setMethodFilter={setMethodFilter}
          totalCount={filteredMeetings.length}
        />

        {/* Meetings Grid */}
        {filteredMeetings.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredMeetings.map((meeting) => (
              <MeetingCard key={meeting.id} meeting={meeting} />
            ))}
          </div>
        ) : (
          <Card className="glass-panel border-white/20 bg-white/10">
            <CardContent className="text-center py-12">
              <div className="w-20 h-20 mx-auto mb-4 glass-panel rounded-full flex items-center justify-center">
                <Archive className="w-10 h-10 text-white/50" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">
                {searchTerm || statusFilter !== "all" || methodFilter !== "all" 
                  ? "No meetings found" 
                  : "No meetings archived yet"
                }
              </h3>
              <p className="text-white/60">
                {searchTerm || statusFilter !== "all" || methodFilter !== "all"
                  ? "Try adjusting your search criteria"
                  : "Upload and process your first meeting to get started"
                }
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}