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
            Instant.now(),
            "{}", // t0AdviceSnapshot
            "PENDING", // t1RecheckStatus
            0.0 // costDistance
        ));
        
        return ResponseEntity.ok(response);
    }

    @GetMapping("/cohort-summary")
    public ResponseEntity<String> getCohortSummary() {
        return ResponseEntity.ok(fastApiClient.getCohortSummary());
    }

    @GetMapping("/list")
    public ResponseEntity<String> getApplicantsList(
        @RequestParam(defaultValue = "1") int page,
        @RequestParam(defaultValue = "50") int limit,
        @RequestParam(required = false) String subgroup,
        @RequestParam(required = false) String riskTier
    ) {
        return ResponseEntity.ok(fastApiClient.getCohortApplicants(page, limit, subgroup, riskTier));
    }

    @GetMapping("/analytics/curves")
    public ResponseEntity<String> getCurves() {
        return ResponseEntity.ok(fastApiClient.getAlePdpCurves());
    }

    @GetMapping("/analytics/cumulative-profiles")
    public ResponseEntity<String> getCumulativeProfiles(@RequestParam(defaultValue = "APP-9824") String applicantId) {
        return ResponseEntity.ok(fastApiClient.getCumulativeProfiles(applicantId));
    }

    @GetMapping("/ml/benchmarks")
    public ResponseEntity<String> getMlBenchmarks() {
        return ResponseEntity.ok(fastApiClient.getMlBenchmarks());
    }

    @GetMapping("/ml/diagnostic-curves")
    public ResponseEntity<String> getMlDiagnosticCurves() {
        return ResponseEntity.ok(fastApiClient.getMlDiagnosticCurves());
    }

    @PostMapping("/ml/sandbox/score")
    public ResponseEntity<String> scoreSandbox(@RequestBody Object payload) {
        return ResponseEntity.ok(fastApiClient.scoreSandboxApplicant(payload));
    }
}

