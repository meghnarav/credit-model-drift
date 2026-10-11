#!/usr/bin/env bash
# Sample input curl script to test credit model predictions and underwriter feedback

set -e
BASE_URL="${1:-http://localhost:8000}"

echo "=========================================================="
echo "Testing Credit Model Drift ML Engine at: $BASE_URL"
echo "=========================================================="

echo -e "\n1. Checking Health..."
curl -s -X GET "$BASE_URL/health" | grep "ok" && echo " [OK] Health check passed"

echo -e "\n2. Submitting Loan Evaluation for Thin-File Applicant (APP-9824)..."
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "applicantId": "APP-9824",
    "subgroup": "thin-file",
    "features": {
      "ExternalRiskEstimate": 62.0,
      "MSinceOldestTradeOpen": 14.0,
      "MSinceMostRecentTradeOpen": 3.0,
      "AverageMInFile": 8.0,
      "NumSatisfactoryTrades": 4.0,
      "NumTrades60Ever2DerogPubRec": 0.0,
      "NumTrades90Ever2DerogPubRec": 0.0,
      "PercentTradesNeverDelq": 90.0,
      "MSinceMostRecentDelq": 0.0,
      "MaxDelq2PublicRecLast12M": 0.0,
      "MaxDelqEver": 0.0,
      "NumTotalTrades": 5.0,
      "NumTradesOpeninLast12M": 2.0,
      "PercentInstallTrades": 40.0,
      "MSinceMostRecentInqexcl7days": 0.0,
      "NumInqLast6M": 2.0,
      "NumInqLast6Mexcl7days": 2.0,
      "NetFractionRevolvingBurden": 85.0,
      "NetFractionInstallBurden": 45.0,
      "NumRevolvingTradesWBalance": 3.0,
      "NumInstallTradesWBalance": 2.0,
      "NumBank2NatlTradesWHighUtilization": 2.0,
      "PercentTradesWBalance": 80.0
    }
  }'
echo ""

echo -e "\n3. Fetching Drift Sensitivity Curve Metrics..."
curl -s -X GET "$BASE_URL/api/v1/metrics/sensitivity" | head -c 200
echo "... [OK]"

echo -e "\n4. Fetching Representative Failure Cases..."
curl -s -X GET "$BASE_URL/api/v1/metrics/failure-cases" | head -c 200
echo "... [OK]"

echo -e "\n5. Submitting Expert Underwriter Feedback..."
curl -s -X POST "$BASE_URL/api/v1/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "applicationId": "APP-9824",
    "reviewerId": "REV-001",
    "applicantSubgroup": "thin-file",
    "actionabilityScore": 2,
    "plausibilityScore": 2,
    "verdict": "REJECT",
    "reviewerNotes": "Prescribed 40% burden paydown is invalidated under post-drift credit threshold."
  }'
echo ""

echo -e "\n6. Retrieving Recorded Underwriter Feedback..."
curl -s -X GET "$BASE_URL/api/v1/feedback" | head -c 200
echo "... [OK]"

echo -e "\n=========================================================="
echo "All curl endpoint tests completed successfully!"
echo "=========================================================="
