package com.example.demo.entity;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
public class LoanApplication {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String applicantExternalId;
    
    @Column(columnDefinition = "TEXT")
    private String features; // JSON representation of raw features
    
    private String decision; // APPROVED / DENIED
    private String subgroup; // THIN_FILE / THICK_FILE
    private Instant createdAt;

    protected LoanApplication() {}

    public LoanApplication(String applicantExternalId, String features, String decision, String subgroup) {
        this.applicantExternalId = applicantExternalId;
        this.features = features;
        this.decision = decision;
        this.subgroup = subgroup;
        this.createdAt = Instant.now();
    }

    // Getters and Setters omitted for brevity
    public Long getId() { return id; }
}
