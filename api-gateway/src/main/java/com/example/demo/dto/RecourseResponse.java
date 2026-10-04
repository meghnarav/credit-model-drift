package com.example.demo.dto;

import java.util.Map;

public record RecourseResponse(
    String applicantId,
    String decision,
    double approvalProbability,
    Map<String, Double> shapAttribution,
    RecourseDetails recourse
) {
    public record RecourseDetails(
        boolean achievable,
        Map<String, Object> recommendedChanges,
        ReliabilityMetrics reliability
    ) {}

    public record ReliabilityMetrics(
        String subgroup,
        double empiricalInvalidationRate,
        String reliabilityTier,
        String warningNotice
    ) {}
}
