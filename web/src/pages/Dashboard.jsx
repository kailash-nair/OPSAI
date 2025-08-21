import React, { useState, useEffect } from "react";
import { Meeting } from "@/entities/Meeting";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FileText, Zap } from "lucide-react";
import { Link } from "react-router-dom";
import { createPageUrl } from "@/utils";

import ProcessingCard from "../components/dashboard/ProcessingCard";
import StatsOverview from "../components/dashboard/StatsOverview";
import RecentMeetings from "../components/dashboard/RecentMeetings";

export default function DashboardPage() {
  const [meetings, setMeetings] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadMeetings();
  }, []);

  const loadMeetings = async () => {
    setIsLoading(true);
    try {
      const data = await Meeting.list("-created_date", 50);
      setMeetings(data);
    } catch (error) {
      console.error("Error loading meetings:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const processingMeetings = meetings.filter((m) => m.status === "processing");
  const completedMeetings = meetings.filter((m) => m.status === "completed");
  const failedMeetings = meetings.filter((m) => m.status === "failed");

  if (isLoading) {
    return (
      <div className="min-h-screen p-4 md:p-8">
        <div className="max-w-7xl mx-auto">
          <div className="animate-pulse space-y-6">
            <div className="glass-panel rounded-xl h-32"></div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="glass-panel rounded-xl h-24"></div>
              <div className="glass-panel rounded-xl h-24"></div>
              <div className="glass-panel rounded-xl h-24"></div>
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
          {/* Smaller title so it doesn't clash with the sidebar */}
          <h1 className="text-2xl md:text-3xl font-bold text-white gradient-text mb-2">
            Processing Dashboard
          </h1>
          <p className="text-white/70 text-lg">
            Monitor your meeting transcriptions and view summaries
          </p>
        </div>

        {/* Stats Overview */}
        <StatsOverview
          totalMeetings={meetings.length}
          completedMeetings={completedMeetings.length}
          processingMeetings={processingMeetings.length}
          failedMeetings={failedMeetings.length}
        />

        {/* Processing Queue */}
        {processingMeetings.length > 0 && (
          <Card className="glass-panel border-white/20 bg-white/10 mb-8">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                Currently Processing ({processingMeetings.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {processingMeetings.map((meeting) => (
                  <ProcessingCard key={meeting.id} meeting={meeting} />
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Recent Meetings */}
        <RecentMeetings meetings={meetings} onRefresh={loadMeetings} />

        {/* Empty State */}
        {meetings.length === 0 && (
          <Card className="glass-panel border-white/20 bg-white/10">
            <CardContent className="text-center py-12">
              <div className="w-20 h-20 mx-auto mb-4 glass-panel rounded-full flex items-center justify-center">
                <FileText className="w-10 h-10 text-white/50" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">
                No meetings uploaded yet
              </h3>
              <p className="text-white/60 mb-6">
                Upload your first Malayalam meeting recording to get started
              </p>
              <Link to={createPageUrl("Upload")}>
                <button className="glass-button text-white hover:text-white px-4 py-2 rounded-lg">
                  Upload Recording
                </button>
              </Link>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
