import React from 'react';

export default function AnalyticsButton() {
    const [isExpanded, setIsExpanded] = React.useState(false);
    const [selectedOption, setSelectedOption] = React.useState(null);

    const options = [
        { id: 'time', label: 'Time Analytics', icon: '📊' },
    ];

    const handleOptionClick = (option) => {
        setSelectedOption(option);
        // Will add functionality later
        console.log('Selected:', option.label);
    };

    return (
        <div className="w-full">
            {/* Main Analytics Section Header */}
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="w-full flex items-center justify-between px-4 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
            >
                <div className="flex items-center gap-2">
                    <span className="text-lg">📊</span>
                    <span className="font-medium">Analytics</span>
                </div>
                <span className={`text-xs transition-transform ${isExpanded ? 'rotate-180' : ''}`}>
                    ▼
                </span>
            </button>

            {/* Expanded Options - Now below the button instead of absolute positioned */}
            {isExpanded && (
                <div className="mt-2 w-full bg-card border border-border rounded-md">
                    <div className="p-2">
                        <div className="text-sm font-semibold text-muted-foreground px-3 py-2">
                            Select Analytics Type
                        </div>
                        <div className="space-y-1">
                            {options.map((option) => (
                                <button
                                    key={option.id}
                                    onClick={() => handleOptionClick(option)}
                                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-left transition-colors ${
                                        selectedOption?.id === option.id
                                            ? 'bg-primary text-primary-foreground'
                                            : 'hover:bg-accent hover:text-accent-foreground'
                                    }`}
                                >
                                    <span className="text-xl">{option.icon}</span>
                                    <span className="font-medium">{option.label}</span>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
