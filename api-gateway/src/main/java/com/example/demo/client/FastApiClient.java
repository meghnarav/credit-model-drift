package com.example.demo.client;

import com.example.demo.dto.LoanEvaluationRequest;
import com.example.demo.dto.RecourseResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

@Component
public class FastApiClient {

    private final RestClient restClient;

    public FastApiClient(@Value("${ml.engine.base-url:http://localhost:8000}") String mlEngineUrl) {
        this.restClient = RestClient.builder()
            .baseUrl(mlEngineUrl)
            .defaultHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
            .build();
    }

    public RecourseResponse evaluateAndGenerateRecourse(LoanEvaluationRequest request) {
        return restClient.post()
            .uri("/predict")
            .body(request)
            .retrieve()
            .body(RecourseResponse.class);
    }

    public String getCohortSummary() {
        return restClient.get()
            .uri("/api/v1/cohort/summary")
            .retrieve()
            .body(String.class);
    }

    public String getCohortApplicants(int page, int limit, String subgroup, String riskTier) {
        return restClient.get()
            .uri(uriBuilder -> uriBuilder
                .path("/api/v1/cohort/applicants")
                .queryParam("page", page)
                .queryParam("limit", limit)
                .queryParam("subgroup", subgroup != null ? subgroup : "ALL")
                .queryParam("risk_tier", riskTier != null ? riskTier : "ALL")
                .build())
            .retrieve()
            .body(String.class);
    }

    public String getAlePdpCurves() {
        return restClient.get()
            .uri("/api/v1/analytics/ale-pdp")
            .retrieve()
            .body(String.class);
    }

    public String getCumulativeProfiles(String applicantId) {
        return restClient.get()
            .uri(uriBuilder -> uriBuilder
                .path("/api/v1/analytics/cumulative-profiles")
                .queryParam("applicant_id", applicantId)
                .build())
            .retrieve()
            .body(String.class);
    }
}
