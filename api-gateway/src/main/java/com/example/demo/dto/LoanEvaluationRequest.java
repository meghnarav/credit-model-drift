package com.example.demo.dto;

import java.util.Map;

public record LoanEvaluationRequest(
    String applicantId,
    Map<String, Object> features,
    String subgroup
) {}
