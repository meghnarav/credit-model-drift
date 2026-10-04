package com.example.demo.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import java.time.Instant;

@Entity
public class RecourseAuditEntity {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    private String applicantId;
    private String decision;
    private double empiricalInvalidationRate;
    private Instant auditTimestamp;
    
    // New Audit Fields
    private String t0AdviceSnapshot; // JSON representation of the advice
    private String t1RecheckStatus; // e.g. "VALID", "INVALIDATED"
    private double costDistance; // L1 distance of the required changes

    protected RecourseAuditEntity() {}

    public RecourseAuditEntity(String applicantId, String decision, double empiricalInvalidationRate, Instant auditTimestamp, String t0AdviceSnapshot, String t1RecheckStatus, double costDistance) {
        this.applicantId = applicantId;
        this.decision = decision;
        this.empiricalInvalidationRate = empiricalInvalidationRate;
        this.auditTimestamp = auditTimestamp;
        this.t0AdviceSnapshot = t0AdviceSnapshot;
        this.t1RecheckStatus = t1RecheckStatus;
        this.costDistance = costDistance;
    }

    public Long getId() { return id; }
    public String getApplicantId() { return applicantId; }
    public String getDecision() { return decision; }
    public double getEmpiricalInvalidationRate() { return empiricalInvalidationRate; }
    public Instant getAuditTimestamp() { return auditTimestamp; }
    public String getT0AdviceSnapshot() { return t0AdviceSnapshot; }
    public String getT1RecheckStatus() { return t1RecheckStatus; }
    public double getCostDistance() { return costDistance; }
}
