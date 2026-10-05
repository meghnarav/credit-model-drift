package com.example.demo.entity;

import jakarta.persistence.*;

@Entity
public class RecoursePlan {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String applicationId;
    
    @Column(columnDefinition = "TEXT")
    private String recommendedChanges; // JSON
    
    private String baselineModel; // XGBoost vs LR
    private double empiricalInvalidationRate;
    private String reliabilityTier; // LOW_RISK, MEDIUM_RISK, HIGH_RISK

    protected RecoursePlan() {}

    public RecoursePlan(String applicationId, String recommendedChanges, String baselineModel, double empiricalInvalidationRate, String reliabilityTier) {
        this.applicationId = applicationId;
        this.recommendedChanges = recommendedChanges;
        this.baselineModel = baselineModel;
        this.empiricalInvalidationRate = empiricalInvalidationRate;
        this.reliabilityTier = reliabilityTier;
    }

    public Long getId() { return id; }
}
