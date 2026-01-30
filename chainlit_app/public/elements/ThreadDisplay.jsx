import React from 'react';
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";

export default function ThreadDisplay(props) {
  // Comprehensive debugging - check all possible prop locations
  console.log('=== ThreadDisplay FULL Debug ===');
  console.log('1. Full props:', JSON.stringify(props, null, 2));
  console.log('2. Props keys:', Object.keys(props));
  console.log('3. Props values:', Object.values(props));

  // Check different possible locations
  console.log('4. props.element:', props.element);
  console.log('5. props.data:', props.data);
  console.log('6. props.props:', props.props);
  console.log('7. Direct access - contact_name:', props.contact_name);
  console.log('8. Direct access - content:', props.content);

  // Try to find ANY key that contains our data
  for (const key in props) {
    console.log(`9. Found key "${key}":`, props[key]);
  }
  console.log('================================');

  // Parse the thread content into individual messages
  const parseMessages = (content) => {
    if (!content || typeof content !== 'string') {
      return [];
    }
    const lines = content.split('\n').filter(line => line.trim());
    const messages = [];

    for (const line of lines) {
      // Match pattern: [timestamp] sender: text
      const match = line.match(/\[(.*?)\]\s*([^:]*?):\s*(.+)/);
      if (match) {
        const [, timestamp, sender, text] = match;
        messages.push({
          timestamp: timestamp.trim(),
          sender: sender.trim() || 'Other',
          text: text.trim()
        });
      }
    }

    return messages;
  };

  const messages = parseMessages(props.content);

  // If no valid content, show error state
  if (!props.content || messages.length === 0) {
    return (
      <Card className="w-full max-w-2xl">
        <CardContent className="pt-6">
          <div className="text-center text-muted-foreground">
            No messages to display
          </div>
        </CardContent>
      </Card>
    );
  }

  // Format date for header (iOS-style)
  const formatDateHeader = (dateStr) => {
    try {
      const date = new Date(dateStr);
      const options = { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' };
      return date.toLocaleDateString('en-US', options);
    } catch {
      return dateStr;
    }
  };

  // Format time for individual messages
  const formatTime = (timestamp) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    } catch {
      return timestamp;
    }
  };

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader className="pb-3">
        {/* Date header */}
        <div className="text-center text-sm text-muted-foreground font-medium">
          {formatDateHeader(props.start_time)}
        </div>
      </CardHeader>

      <CardContent>
        <ScrollArea className="max-h-96 pr-4">
          <div className="space-y-3">
            {messages.map((msg, index) => {
              const isMe = msg.sender === 'me';

              return (
                <div
                  key={index}
                  className={`flex ${isMe ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[70%] rounded-2xl px-4 py-2 shadow-sm ${
                      isMe
                        ? 'bg-[#007AFF] text-white rounded-br-sm'
                        : 'bg-[#E5E5EA] text-black rounded-bl-sm'
                    }`}
                  >
                    <div className="text-[15px] leading-relaxed mb-1">
                      {msg.text}
                    </div>
                    <div
                      className={`text-[11px] opacity-70 ${
                        isMe ? 'text-right text-white' : 'text-left text-black'
                      }`}
                    >
                      {formatTime(msg.timestamp)}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </ScrollArea>

        <Separator className="my-4" />

        {/* Contact name footer */}
        <div className="text-center text-sm text-muted-foreground">
          Conversation with {props.contact_name || 'Unknown'}
        </div>
      </CardContent>
    </Card>
  );
}
