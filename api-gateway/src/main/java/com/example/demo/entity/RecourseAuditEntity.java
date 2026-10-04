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

    protected RecourseAuditEntity() {}

    public RecourseAuditEntity(String applicantId, String decision, double empiricalInvalidationRate, Instant auditTimestamp) {
        this.applicantId = applicantId;
        this.decision = decision;
        this.empiricalInvalidationRate = empiricalInvalidationRate;
        this.auditTimestamp = auditTimestamp;
    }

    public Long getId() { return id; }
    public String getApplicantId() { return applicantId; }
    public String getDecision() { return decision; }
    public double getEmpiricalInvalidationRate() { return empiricalInvalidationRate; }
    public Instant getAuditTimestamp() { return auditTimestamp; }
}
