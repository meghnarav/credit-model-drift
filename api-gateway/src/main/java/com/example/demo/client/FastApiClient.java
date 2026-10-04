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
            .uri("/predict") // Adjusted to match the FastAPI endpoint we created earlier, though a dedicated one could be made
            .body(request)
            .retrieve()
            .body(RecourseResponse.class);
    }
}
