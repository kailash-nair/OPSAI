import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { FileText, CheckCircle, Zap, AlertCircle } from "lucide-react";

export default function StatsOverview({ 
  totalMeetings, 
  completedMeetings, 
  processingMeetings, 
  failedMeetings 
}) {
  const stats = [
    {
      title: "Total Meetings",
      value: totalMeetings,
      icon: FileText,
      color: "text-blue-400",
      bgColor: "bg-blue-500/20"
    },
    {
      title: "Completed",
      value: completedMeetings,
      icon: CheckCircle,
      color: "text-green-400",
      bgColor: "bg-green-500/20"
    },
    {
      title: "Processing",
      value: processingMeetings,
      icon: Zap,
      color: "text-yellow-400",
      bgColor: "bg-yellow-500/20"
    },
    {
      title: "Failed",
      value: failedMeetings,
      icon: AlertCircle,
      color: "text-red-400",
      bgColor: "bg-red-500/20"
    }
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {stats.map((stat) => (
        <Card key={stat.title} className="glass-panel border-white/20 bg-white/10">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-white/70 text-sm font-medium">{stat.title}</p>
                <p className="text-2xl font-bold text-white mt-1">{stat.value}</p>
              </div>
              <div className={`p-3 rounded-xl ${stat.bgColor}`}>
                <stat.icon className={`w-6 h-6 ${stat.color}`} />
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}