package com.example.demo.controller;

import com.example.demo.client.FastApiClient;
import com.example.demo.dto.LoanEvaluationRequest;
import com.example.demo.dto.RecourseResponse;
import com.example.demo.entity.RecourseAuditEntity;
import com.example.demo.repository.RecourseAuditRepository;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.time.Instant;

@RestController
@RequestMapping("/api/v1/applications")
public class LoanApplicationController {

    private final FastApiClient fastApiClient;
    private final RecourseAuditRepository auditRepo;

    public LoanApplicationController(FastApiClient fastApiClient, RecourseAuditRepository auditRepo) {
        this.fastApiClient = fastApiClient;
        this.auditRepo = auditRepo;
    }

    @PostMapping("/evaluate")
    public ResponseEntity<RecourseResponse> evaluateApplicant(@RequestBody LoanEvaluationRequest request) {
        RecourseResponse response = fastApiClient.evaluateAndGenerateRecourse(request);
        
        // Persist for compliance auditing
        double invalidationRate = 0.0;
        if (response.recourse() != null && response.recourse().reliability() != null) {
            invalidationRate = response.recourse().reliability().empiricalInvalidationRate();
        }
        
        auditRepo.save(new RecourseAuditEntity(
            request.applicantId(),
            response.decision(),
            invalidationRate,
            Instant.now()
        ));
        
        return ResponseEntity.ok(response);
    }
}
