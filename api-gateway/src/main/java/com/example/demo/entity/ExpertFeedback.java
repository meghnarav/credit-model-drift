package com.example.demo.entity;

import jakarta.persistence.*;

@Entity
public class ExpertFeedback {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String applicationId;
    private String reviewerId;
    private int actionabilityScore; // 1-5
    private String feasibilityVerdict; // ACCEPT/REJECT/MODIFY
    
    @Column(columnDefinition = "TEXT")
    private String reviewerNotes;

    protected ExpertFeedback() {}

    public ExpertFeedback(String applicationId, String reviewerId, int actionabilityScore, String feasibilityVerdict, String reviewerNotes) {
        this.applicationId = applicationId;
        this.reviewerId = reviewerId;
        this.actionabilityScore = actionabilityScore;
        this.feasibilityVerdict = feasibilityVerdict;
        this.reviewerNotes = reviewerNotes;
    }

    public Long getId() { return id; }
}
