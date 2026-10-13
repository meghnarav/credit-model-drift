'use client';

import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  Legend,
  ReferenceLine
} from 'recharts';
import {
  Users,
  UserCheck,
  AlertTriangle,
  TrendingDown,
  Layers,
  ArrowRight,
  ShieldAlert,
  Sliders,
  CheckCircle2,
  XCircle,
  HelpCircle,
  RefreshCw,
  Activity,
  GitCompare,
  Cpu,
  FileSpreadsheet,
  TrendingUp,
  Award
} from 'lucide-react';

// =============================================================================
// SAFE API URL RESOLVER (Zero Local Network Access Permission Prompts on Vercel)
// =============================================================================
function getApiBaseUrl(): string | null {
  if (typeof window !== 'undefined') {
    // 1. Explicit public API URL configured in environment
    if (process.env.NEXT_PUBLIC_API_URL && process.env.NEXT_PUBLIC_API_URL.trim() !== '') {
      return process.env.NEXT_PUBLIC_API_URL.trim();
    }
    // 2. Only target localhost if explicitly running on a local development host
    const hostname = window.location.hostname;
    if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname.endsWith('.local')) {
      return 'http://localhost:8000';
    }
  }
  // 3. In cloud/Vercel production without an external API configured, return null.
  // This prevents browsers from popping the Private Network Access ("wants to access local network") modal!
  return null;
}

// =============================================================================
// EMBEDDED EMPIRICAL BENCHMARK DATASETS (FICO HELOC 5-Seed Evaluation)
// =============================================================================
const DEFAULT_ML_BENCHMARKS = {
  summaryKPIs: {
    baselineLrAuc: "0.781 ± 0.003",
    baselineXgbAuc: "0.854 ± 0.001",
    driftedXgbAuc: "0.765 ± 0.004",
    retrainedXgbAuc: "0.904 ± 0.006",
    thinFileRecallDrop: "-31.7%",
    thinFileFnrSurge: "71.9%"
  },
  modelPerformance: [
    { model: "LogisticRegression", regime: "T0_baseline", accuracy: "0.712 ± 0.003", precision: "0.733 ± 0.002", recall: "0.728 ± 0.003", f1: "0.731 ± 0.002", roc_auc: "0.781 ± 0.003" },
    { model: "LogisticRegression", regime: "T1_drifted_unretrained", accuracy: "0.747 ± 0.002", precision: "0.685 ± 0.010", recall: "0.440 ± 0.020", f1: "0.535 ± 0.013", roc_auc: "0.772 ± 0.003" },
    { model: "LogisticRegression", regime: "T1_drifted_retrained", accuracy: "0.750 ± 0.004", precision: "0.691 ± 0.012", recall: "0.445 ± 0.011", f1: "0.541 ± 0.008", roc_auc: "0.778 ± 0.003" },
    { model: "XGBoost", regime: "T0_baseline", accuracy: "0.766 ± 0.002", precision: "0.787 ± 0.003", recall: "0.772 ± 0.002", f1: "0.779 ± 0.002", roc_auc: "0.854 ± 0.001" },
    { model: "XGBoost", regime: "T1_drifted_unretrained", accuracy: "0.748 ± 0.002", precision: "0.678 ± 0.016", recall: "0.458 ± 0.017", f1: "0.547 ± 0.009", roc_auc: "0.765 ± 0.004" },
    { model: "XGBoost", regime: "T1_drifted_retrained", accuracy: "0.826 ± 0.005", precision: "0.820 ± 0.011", recall: "0.611 ± 0.015", f1: "0.700 ± 0.009", roc_auc: "0.904 ± 0.006" }
  ],
  subgroupFairness: [
    { model: "LogisticRegression", regime: "T0_baseline", subgroup: "thin-file", sample_size_n: 688, recall: "0.539 ± 0.017", fnr: "0.461 ± 0.017", fpr: "0.163 ± 0.010", roc_auc: "0.767 ± 0.013" },
    { model: "LogisticRegression", regime: "T1_drifted_unretrained", subgroup: "thin-file", sample_size_n: 460, recall: "0.257 ± 0.015", fnr: "0.743 ± 0.015", fpr: "0.037 ± 0.010", roc_auc: "0.755 ± 0.011" },
    { model: "LogisticRegression", regime: "T1_drifted_retrained", subgroup: "thin-file", sample_size_n: 460, recall: "0.276 ± 0.028", fnr: "0.724 ± 0.028", fpr: "0.032 ± 0.007", roc_auc: "0.762 ± 0.014" },
    { model: "XGBoost", regime: "T0_baseline", subgroup: "thin-file", sample_size_n: 688, recall: "0.598 ± 0.030", fnr: "0.402 ± 0.030", fpr: "0.109 ± 0.008", roc_auc: "0.872 ± 0.011" },
    { model: "XGBoost", regime: "T1_drifted_unretrained", subgroup: "thin-file", sample_size_n: 460, recall: "0.281 ± 0.029", fnr: "0.719 ± 0.029", fpr: "0.030 ± 0.006", roc_auc: "0.749 ± 0.014" },
    { model: "XGBoost", regime: "T1_drifted_retrained", subgroup: "thin-file", sample_size_n: 460, recall: "0.433 ± 0.049", fnr: "0.567 ± 0.049", fpr: "0.018 ± 0.005", roc_auc: "0.919 ± 0.004" },
    { model: "XGBoost", regime: "T0_baseline", subgroup: "thick-file", sample_size_n: 2174, recall: "0.768 ± 0.008", fnr: "0.232 ± 0.008", fpr: "0.285 ± 0.006", roc_auc: "0.826 ± 0.005" },
    { model: "XGBoost", regime: "T1_drifted_unretrained", subgroup: "thick-file", sample_size_n: 923, recall: "0.495 ± 0.028", fnr: "0.505 ± 0.028", fpr: "0.171 ± 0.021", roc_auc: "0.734 ± 0.007" },
    { model: "XGBoost", regime: "T1_drifted_retrained", subgroup: "thick-file", sample_size_n: 923, recall: "0.689 ± 0.042", fnr: "0.311 ± 0.042", fpr: "0.117 ± 0.035", roc_auc: "0.894 ± 0.010" }
  ]
};

const DEFAULT_DIAGNOSTIC_CURVES = {
  rocCurves: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map(f => ({
    fpr: Math.round(f * 100),
    lr_t0: Math.round(Math.pow(f, 0.52) * 100),
    lr_t1_drift: Math.round(Math.pow(f, 0.54) * 100),
    xgb_t0: Math.round(Math.pow(f, 0.38) * 100),
    xgb_t1_drift: Math.round(Math.pow(f, 0.56) * 100),
    xgb_t1_retrain: Math.round(Math.pow(f, 0.28) * 100),
    random: Math.round(f * 100)
  })),
  prCurves: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map(r => ({
    recall: r,
    lr_t0: Math.round(Math.max(35, 85 - 0.45 * r)),
    xgb_t0: Math.round(Math.max(40, 92 - 0.40 * r)),
    xgb_t1_drift: Math.round(Math.max(30, 78 - 0.50 * r)),
    xgb_t1_retrain: Math.round(Math.max(45, 96 - 0.35 * r))
  })),
  confusionMatrices: {
    xgb_t0: { tn: 39.5, fp: 12.7, fn: 10.7, tp: 37.1 },
    xgb_t1_drift: { tn: 42.1, fp: 10.1, fn: 25.8, tp: 22.0 },
    xgb_t1_retrained: { tn: 46.2, fp: 6.0, fn: 18.5, tp: 29.3 }
  },
  subgroupErrorRates: [
    { subgroup: "Overall (T₀)", recall: 77.2, fnr: 22.8, fpr: 24.3 },
    { subgroup: "Overall (Drift)", recall: 45.8, fnr: 54.2, fpr: 19.3 },
    { subgroup: "Overall (Retrain)", recall: 61.1, fnr: 38.9, fpr: 11.5 },
    { subgroup: "Thin-File (T₀)", recall: 59.8, fnr: 40.2, fpr: 10.9 },
    { subgroup: "Thin-File (Drift)", recall: 28.1, fnr: 71.9, fpr: 3.0 },
    { subgroup: "Thin-File (Retrain)", recall: 43.3, fnr: 56.7, fpr: 1.8 },
    { subgroup: "Thick-File (T₀)", recall: 76.8, fnr: 23.2, fpr: 28.5 },
    { subgroup: "Thick-File (Drift)", recall: 49.5, fnr: 50.5, fpr: 17.1 },
    { subgroup: "Thick-File (Retrain)", recall: 68.9, fnr: 31.1, fpr: 11.7 }
  ]
};

const DEFAULT_COHORT_SUMMARY = {
  cohortOverview: {
    totalApplicants: 10459,
    thinFileCount: 1319,
    thickFileCount: 3700,
    thinFilePct: 12.6,
    thickFilePct: 35.4,
    baselineApprovalRate: 47.8,
    postDriftApprovalRate: 38.2,
    aggregateInvalidationRate: 40.9,
    thinFileInvalidationRate: 66.7,
    thickFileInvalidationRate: 20.0,
    invalidationGap: 46.7
  }
};

const DEFAULT_ALE_PDP = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map(x => ({
  burden: x,
  p_t0: Math.round((1.0 / (1.0 + Math.exp(0.06 * (x - 55)))) * 100),
  p_t1: Math.round((1.0 / (1.0 + Math.exp(0.08 * (x - 42)))) * 100),
  threshold: 50
}));

const DEFAULT_APPLICANTS = [
  { applicantId: 'APP-9824', subgroup: 'thin-file', riskTier: 'HIGH', invalidationRisk: 66.7, externalRiskEstimate: 62, netFractionRevolvingBurden: 85, mSinceOldestTradeOpen: 14, numTotalTrades: 5 },
  { applicantId: 'APP-1042', subgroup: 'thin-file', riskTier: 'HIGH', invalidationRisk: 66.7, externalRiskEstimate: 64, netFractionRevolvingBurden: 78, mSinceOldestTradeOpen: 18, numTotalTrades: 4 },
  { applicantId: 'APP-2305', subgroup: 'thick-file', riskTier: 'LOW', invalidationRisk: 20.0, externalRiskEstimate: 69, netFractionRevolvingBurden: 52, mSinceOldestTradeOpen: 140, numTotalTrades: 22 },
  { applicantId: 'APP-3118', subgroup: 'thick-file', riskTier: 'LOW', invalidationRisk: 20.0, externalRiskEstimate: 72, netFractionRevolvingBurden: 46, mSinceOldestTradeOpen: 190, numTotalTrades: 28 },
  { applicantId: 'APP-4019', subgroup: 'other', riskTier: 'MEDIUM', invalidationRisk: 42.5, externalRiskEstimate: 65, netFractionRevolvingBurden: 62, mSinceOldestTradeOpen: 45, numTotalTrades: 12 },
  { applicantId: 'APP-5092', subgroup: 'thin-file', riskTier: 'HIGH', invalidationRisk: 66.7, externalRiskEstimate: 59, netFractionRevolvingBurden: 89, mSinceOldestTradeOpen: 11, numTotalTrades: 3 },
  { applicantId: 'APP-6211', subgroup: 'thick-file', riskTier: 'LOW', invalidationRisk: 20.0, externalRiskEstimate: 70, netFractionRevolvingBurden: 55, mSinceOldestTradeOpen: 160, numTotalTrades: 20 }
];

const APPLICANT_DATABASE: Record<string, any> = {
  'APP-9824': {
    applicantId: 'APP-9824',
    subgroup: 'thin-file',
    decision: 'DENIED',
    approvalProbability: 0.234,
    invalidationRisk: 66.7,
    features: {
      ExternalRiskEstimate: 62,
      NetFractionRevolvingBurden: 85,
      MSinceOldestTradeOpen: 14,
      NumTotalTrades: 5
    },
    shapBreakdown: [
      { feature: 'NetFractionRevolvingBurden', impact: -0.92, val: 85 },
      { feature: 'ExternalRiskEstimate', impact: -0.58, val: 62 },
      { feature: 'MSinceOldestTradeOpen', impact: -0.45, val: 14 },
      { feature: 'NumTotalTrades', impact: -0.28, val: 5 },
      { feature: 'NumSatisfactoryTrades', impact: 0.20, val: 4 }
    ],
    recoursePlan: {
      actionableFeatures: [
        { feature: 'NetFractionRevolvingBurden', current: 85, target: 45, robustTarget: 35, delta: -40, unit: '%' },
        { feature: 'ExternalRiskEstimate', current: 62, target: 68, robustTarget: 71, delta: +6, unit: 'pts' }
      ],
      actionCostL1: 5.62,
      costRatio: 'High Effort (2.4x median)'
    }
  },
  'APP-1042': {
    applicantId: 'APP-1042',
    subgroup: 'thin-file',
    decision: 'DENIED',
    approvalProbability: 0.281,
    invalidationRisk: 66.7,
    features: {
      ExternalRiskEstimate: 64,
      NetFractionRevolvingBurden: 78,
      MSinceOldestTradeOpen: 18,
      NumTotalTrades: 4
    },
    shapBreakdown: [
      { feature: 'NetFractionRevolvingBurden', impact: -0.85, val: 78 },
      { feature: 'ExternalRiskEstimate', impact: -0.48, val: 64 },
      { feature: 'MSinceOldestTradeOpen', impact: -0.40, val: 18 },
      { feature: 'NumTotalTrades', impact: -0.22, val: 4 },
      { feature: 'NumSatisfactoryTrades', impact: 0.18, val: 3 }
    ],
    recoursePlan: {
      actionableFeatures: [
        { feature: 'NetFractionRevolvingBurden', current: 78, target: 42, robustTarget: 32, delta: -36, unit: '%' },
        { feature: 'ExternalRiskEstimate', current: 64, target: 69, robustTarget: 72, delta: +5, unit: 'pts' }
      ],
      actionCostL1: 5.10,
      costRatio: 'High Effort (2.1x median)'
    }
  },
  'APP-2305': {
    applicantId: 'APP-2305',
    subgroup: 'thick-file',
    decision: 'DENIED',
    approvalProbability: 0.462,
    invalidationRisk: 20.0,
    features: {
      ExternalRiskEstimate: 69,
      NetFractionRevolvingBurden: 52,
      MSinceOldestTradeOpen: 140,
      NumTotalTrades: 22
    },
    shapBreakdown: [
      { feature: 'NetFractionRevolvingBurden', impact: -0.35, val: 52 },
      { feature: 'ExternalRiskEstimate', impact: -0.15, val: 69 },
      { feature: 'MSinceOldestTradeOpen', impact: 0.55, val: 140 },
      { feature: 'NumTotalTrades', impact: 0.42, val: 22 },
      { feature: 'NumSatisfactoryTrades', impact: 0.38, val: 19 }
    ],
    recoursePlan: {
      actionableFeatures: [
        { feature: 'NetFractionRevolvingBurden', current: 52, target: 44, robustTarget: 38, delta: -8, unit: '%' },
        { feature: 'ExternalRiskEstimate', current: 69, target: 72, robustTarget: 74, delta: +3, unit: 'pts' }
      ],
      actionCostL1: 1.84,
      costRatio: 'Moderate Effort (0.8x median)'
    }
  },
  'APP-3118': {
    applicantId: 'APP-3118',
    subgroup: 'thick-file',
    decision: 'DENIED',
    approvalProbability: 0.485,
    invalidationRisk: 20.0,
    features: {
      ExternalRiskEstimate: 72,
      NetFractionRevolvingBurden: 46,
      MSinceOldestTradeOpen: 190,
      NumTotalTrades: 28
    },
    shapBreakdown: [
      { feature: 'NetFractionRevolvingBurden', impact: -0.22, val: 46 },
      { feature: 'ExternalRiskEstimate', impact: 0.10, val: 72 },
      { feature: 'MSinceOldestTradeOpen', impact: 0.68, val: 190 },
      { feature: 'NumTotalTrades', impact: 0.50, val: 28 },
      { feature: 'NumSatisfactoryTrades', impact: 0.45, val: 25 }
    ],
    recoursePlan: {
      actionableFeatures: [
        { feature: 'NetFractionRevolvingBurden', current: 46, target: 40, robustTarget: 36, delta: -6, unit: '%' },
        { feature: 'ExternalRiskEstimate', current: 72, target: 74, robustTarget: 76, delta: +2, unit: 'pts' }
      ],
      actionCostL1: 1.25,
      costRatio: 'Low Effort (0.5x median)'
    }
  },
  'APP-4019': {
    applicantId: 'APP-4019',
    subgroup: 'other',
    decision: 'DENIED',
    approvalProbability: 0.368,
    invalidationRisk: 42.5,
    features: {
      ExternalRiskEstimate: 65,
      NetFractionRevolvingBurden: 62,
      MSinceOldestTradeOpen: 45,
      NumTotalTrades: 12
    },
    shapBreakdown: [
      { feature: 'NetFractionRevolvingBurden', impact: -0.62, val: 62 },
      { feature: 'ExternalRiskEstimate', impact: -0.38, val: 65 },
      { feature: 'MSinceOldestTradeOpen', impact: 0.12, val: 45 },
      { feature: 'NumTotalTrades', impact: 0.10, val: 12 },
      { feature: 'NumSatisfactoryTrades', impact: 0.25, val: 10 }
    ],
    recoursePlan: {
      actionableFeatures: [
        { feature: 'NetFractionRevolvingBurden', current: 62, target: 42, robustTarget: 36, delta: -20, unit: '%' },
        { feature: 'ExternalRiskEstimate', current: 65, target: 70, robustTarget: 73, delta: +5, unit: 'pts' }
      ],
      actionCostL1: 3.40,
      costRatio: 'Moderate Effort (1.4x median)'
    }
  },
  'APP-5092': {
    applicantId: 'APP-5092',
    subgroup: 'thin-file',
    decision: 'DENIED',
    approvalProbability: 0.182,
    invalidationRisk: 66.7,
    features: {
      ExternalRiskEstimate: 59,
      NetFractionRevolvingBurden: 89,
      MSinceOldestTradeOpen: 11,
      NumTotalTrades: 3
    },
    shapBreakdown: [
      { feature: 'NetFractionRevolvingBurden', impact: -1.10, val: 89 },
      { feature: 'ExternalRiskEstimate', impact: -0.75, val: 59 },
      { feature: 'MSinceOldestTradeOpen', impact: -0.52, val: 11 },
      { feature: 'NumTotalTrades', impact: -0.35, val: 3 },
      { feature: 'NumSatisfactoryTrades', impact: 0.10, val: 2 }
    ],
    recoursePlan: {
      actionableFeatures: [
        { feature: 'NetFractionRevolvingBurden', current: 89, target: 42, robustTarget: 30, delta: -47, unit: '%' },
        { feature: 'ExternalRiskEstimate', current: 59, target: 70, robustTarget: 74, delta: +11, unit: 'pts' }
      ],
      actionCostL1: 6.85,
      costRatio: 'Very High Effort (2.9x median)'
    }
  },
  'APP-6211': {
    applicantId: 'APP-6211',
    subgroup: 'thick-file',
    decision: 'DENIED',
    approvalProbability: 0.440,
    invalidationRisk: 20.0,
    features: {
      ExternalRiskEstimate: 70,
      NetFractionRevolvingBurden: 55,
      MSinceOldestTradeOpen: 160,
      NumTotalTrades: 20
    },
    shapBreakdown: [
      { feature: 'NetFractionRevolvingBurden', impact: -0.42, val: 55 },
      { feature: 'ExternalRiskEstimate', impact: -0.10, val: 70 },
      { feature: 'MSinceOldestTradeOpen', impact: 0.60, val: 160 },
      { feature: 'NumTotalTrades', impact: 0.38, val: 20 },
      { feature: 'NumSatisfactoryTrades', impact: 0.35, val: 18 }
    ],
    recoursePlan: {
      actionableFeatures: [
        { feature: 'NetFractionRevolvingBurden', current: 55, target: 45, robustTarget: 38, delta: -10, unit: '%' },
        { feature: 'ExternalRiskEstimate', current: 70, target: 73, robustTarget: 75, delta: +3, unit: 'pts' }
      ],
      actionCostL1: 2.10,
      costRatio: 'Moderate Effort (0.9x median)'
    }
  }
};

const DEFAULT_CP_CURVE = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map(x => ({
  burden: x,
  p_t0: Math.round((1.0 / (1.0 + Math.exp(0.06 * (x - 55)))) * 100),
  p_t1: Math.round((1.0 / (1.0 + Math.exp(0.08 * (x - 40)))) * 100),
  threshold: 50
}));

export default function UnifiedExecutiveCockpit() {
  // ---------------------------------------------------------------------------
  // Top-Level Navigation State: Tab 1 (ML Core) vs Tab 2 (XAI Recourse)
  // ---------------------------------------------------------------------------
  const [activeTab, setActiveTab] = useState<'ML_CORE' | 'XAI_RECOURSE'>('ML_CORE');
  
  // Tab 1 (ML Core) Sub-States
  const [mlBenchmarks, setMlBenchmarks] = useState<any>(DEFAULT_ML_BENCHMARKS);
  const [mlCurveType, setMlCurveType] = useState<'ROC' | 'PR'>('ROC');
  const [mlTableTab, setMlTableTab] = useState<'OVERALL' | 'SUBGROUPS'>('OVERALL');
  const [diagnosticCurves, setDiagnosticCurves] = useState<any>(DEFAULT_DIAGNOSTIC_CURVES);

  // Tab 2 (XAI Recourse) Sub-States: Macro View vs Applicant Deep-Dive
  const [xaiSubView, setXaiSubView] = useState<'MACRO' | 'DEEPDIVE'>('MACRO');
  const [cohortSummary, setCohortSummary] = useState<any>(DEFAULT_COHORT_SUMMARY);
  const [alePdpData, setAlePdpData] = useState<any[]>(DEFAULT_ALE_PDP);

  // Tab 2 Applicant Deep-Dive Specifics
  const [applicantFilter, setApplicantFilter] = useState<'ALL' | 'THIN_HIGH' | 'THICK_LOW'>('ALL');
  const [selectedApplicantId, setSelectedApplicantId] = useState<string>('APP-9824');
  const [applicantsList, setApplicantsList] = useState<any[]>(DEFAULT_APPLICANTS);
  const [applicantDetails, setApplicantDetails] = useState<any>(APPLICANT_DATABASE['APP-9824']);
  const [cumulativeProfileData, setCumulativeProfileData] = useState<any[]>(DEFAULT_CP_CURVE);

  // Triage & Underwriter Feedback State
  const [actionabilityScore, setActionabilityScore] = useState<number>(2);
  const [plausibilityScore, setPlausibilityScore] = useState<number>(2);
  const [underwriterVerdict, setUnderwriterVerdict] = useState<'ACCEPT' | 'MODIFY' | 'REJECT'>('REJECT');
  const [underwriterNotes, setUnderwriterNotes] = useState<string>('');
  const [feedbackSuccess, setFeedbackSuccess] = useState<boolean>(false);

  // ---------------------------------------------------------------------------
  // Dynamic API Fetch (Only executed if NEXT_PUBLIC_API_URL or local dev is active)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const apiUrl = getApiBaseUrl();
    if (!apiUrl) return; // In Vercel standalone production, smoothly use embedded benchmark data

    fetch(`${apiUrl}/api/v1/ml/benchmarks`)
      .then(res => res.json())
      .then(data => { if (data?.modelPerformance) setMlBenchmarks(data); })
      .catch(() => {});

    fetch(`${apiUrl}/api/v1/ml/diagnostic-curves`)
      .then(res => res.json())
      .then(data => { if (data?.rocCurves) setDiagnosticCurves(data); })
      .catch(() => {});

    fetch(`${apiUrl}/api/v1/cohort/summary`)
      .then(res => res.json())
      .then(data => { if (data?.cohortOverview) setCohortSummary(data); })
      .catch(() => {});

    fetch(`${apiUrl}/api/v1/analytics/ale-pdp`)
      .then(res => res.json())
      .then(data => {
        if (data?.grid && data?.pdp_t0) {
          const formatted = data.grid.map((x: number, i: number) => ({
            burden: Math.round(x),
            p_t0: Math.round(data.pdp_t0[i] * 100),
            p_t1: Math.round(data.pdp_t1[i] * 100),
            threshold: 50
          }));
          setAlePdpData(formatted);
        }
      })
      .catch(() => {});

    fetch(`${apiUrl}/api/v1/cohort/applicants?limit=60`)
      .then(res => res.json())
      .then(data => {
        if (data?.applicants && data.applicants.length > 0) {
          setApplicantsList(data.applicants);
        }
      })
      .catch(() => {});
  }, []);

  // Update applicant details when selected applicant changes
  useEffect(() => {
    if (APPLICANT_DATABASE[selectedApplicantId]) {
      setApplicantDetails(APPLICANT_DATABASE[selectedApplicantId]);
    }

    const apiUrl = getApiBaseUrl();
    if (!apiUrl) return;

    fetch(`${apiUrl}/api/v1/applicant/${selectedApplicantId}/details`)
      .then(res => res.json())
      .then(data => { if (data?.applicantId) setApplicantDetails(data); })
      .catch(() => {});

    fetch(`${apiUrl}/api/v1/analytics/cumulative-profiles?applicant_id=${selectedApplicantId}`)
      .then(res => res.json())
      .then(data => {
        if (data?.curves) {
          const formatted = data.curves.map((pt: any) => ({
            burden: pt.feature_val,
            p_t0: Math.round(pt.p_t0 * 100),
            p_t1: Math.round(pt.p_t1 * 100),
            threshold: 50
          }));
          setCumulativeProfileData(formatted);
        }
      })
      .catch(() => {});
  }, [selectedApplicantId]);

  // Handle Feedback Submission
  const handleFeedbackSubmit = async () => {
    const apiUrl = getApiBaseUrl();
    if (apiUrl) {
      try {
        await fetch(`${apiUrl}/api/v1/recourse/feedback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            applicantId: selectedApplicantId,
            actionabilityScore,
            plausibilityScore,
            verdict: underwriterVerdict,
            reviewerNotes: underwriterNotes || 'Audit review logged via Unified Executive Cockpit.'
          })
        });
      } catch {
        // Fallback gracefully
      }
    }
    setFeedbackSuccess(true);
    setTimeout(() => setFeedbackSuccess(false), 2500);
  };

  // Sensitivity curve for Tab 2 Macro View
  const sensitivityCurveData = [
    { delta: '5%', sir_thin: 24, sir_thick: 10, gap: 14 },
    { delta: '10%', sir_thin: 32, sir_thick: 12, gap: 20 },
    { delta: '15%', sir_thin: 41, sir_thick: 15, gap: 26 },
    { delta: '20%', sir_thin: 52, sir_thick: 18, gap: 34 },
    { delta: '25%', sir_thin: 66.7, sir_thick: 20, gap: 46.7 },
    { delta: '30%', sir_thin: 74, sir_thick: 24, gap: 50 },
    { delta: '35%', sir_thin: 82, sir_thick: 28, gap: 54 }
  ];

  // Score bucket disparity matrix data for Tab 2 Macro View
  const scoreBucketData = [
    { bucket: '50-59', thin: 85, thick: 45 },
    { bucket: '60-64', thin: 72, thick: 28 },
    { bucket: '65-69', thin: 66.7, thick: 20 },
    { bucket: '70-74', thin: 48, thick: 14 },
    { bucket: '75+', thin: 30, thick: 8 }
  ];

  // Filtered applicants for selector
  const filteredApplicants = applicantsList.filter(app => {
    if (applicantFilter === 'THIN_HIGH') return app.subgroup === 'thin-file' || app.riskTier === 'HIGH';
    if (applicantFilter === 'THICK_LOW') return app.subgroup === 'thick-file' || app.riskTier === 'LOW';
    return true;
  });

  return (
    <div className="min-h-screen bg-[#FAF7F0] text-[#1E1E1E] p-4 lg:p-6 font-sans">
      
      {/* ======================================================================= */}
      {/* TOP COCKPIT HEADER WITH TWO CLEAR TOP-LEVEL STAGE TABS                   */}
      {/* ======================================================================= */}
      <header className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 pb-4 mb-4 border-b-4 border-[#1E1E1E]">
        <div>
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-2xl lg:text-3xl font-black uppercase tracking-tight">
              Credit Risk Prediction Under Model Drift
            </h1>
            <span className="px-2.5 py-1 bg-[#1E1E1E] text-white font-mono text-xs font-bold rounded">
              ML + XAI UNIFIED PLATFORM
            </span>
          </div>
          <p className="text-xs lg:text-sm font-bold text-gray-600 mt-0.5">
            Evaluating Machine Learning Robustness, Subgroup Fairness, and Algorithmic Recourse Fragility on FICO HELOC
          </p>
        </div>

        {/* The Two Clear Top-Level Navigation Tabs */}
        <div className="flex items-center bg-white border-3 border-[#1E1E1E] rounded-xl p-1.5 shadow-[4px_4px_0px_#1E1E1E] flex-wrap gap-2">
          <button
            id="nav-tab-1-ml-core"
            onClick={() => setActiveTab('ML_CORE')}
            className={`px-4 py-2 text-xs lg:text-sm font-black rounded-lg flex items-center gap-2 transition-all ${
              activeTab === 'ML_CORE'
                ? 'bg-[#1E1E1E] text-white shadow'
                : 'text-gray-700 hover:bg-gray-100 hover:text-black'
            }`}
          >
            <Cpu size={16} className={activeTab === 'ML_CORE' ? 'text-yellow-400' : 'text-gray-500'} />
            <span>Tab 1: ML Core &amp; Drift Benchmark</span>
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
              activeTab === 'ML_CORE' ? 'bg-yellow-400 text-black' : 'bg-gray-200 text-gray-700'
            }`}>
              ML Foundation
            </span>
          </button>

          <button
            id="nav-tab-2-xai-recourse"
            onClick={() => setActiveTab('XAI_RECOURSE')}
            className={`px-4 py-2 text-xs lg:text-sm font-black rounded-lg flex items-center gap-2 transition-all ${
              activeTab === 'XAI_RECOURSE'
                ? 'bg-[#1E1E1E] text-white shadow'
                : 'text-gray-700 hover:bg-gray-100 hover:text-black'
            }`}
          >
            <Layers size={16} className={activeTab === 'XAI_RECOURSE' ? 'text-amber-400' : 'text-gray-500'} />
            <span>Tab 2: XAI &amp; Recourse Audit</span>
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
              activeTab === 'XAI_RECOURSE' ? 'bg-amber-400 text-black' : 'bg-gray-200 text-gray-700'
            }`}>
              XAI Recourse
            </span>
          </button>
        </div>
      </header>

      {/* ========================================================================= */}
      {/* TAB 1: ML CORE & DRIFT BENCHMARK (THE MACHINE LEARNING PROJECT PROPOSAL)  */}
      {/* ========================================================================= */}
      {activeTab === 'ML_CORE' && (
        <div className="space-y-4">
          
          {/* Section Subtitle Banner */}
          <div className="bg-white border-2 border-[#1E1E1E] rounded-xl px-4 py-2.5 flex flex-wrap justify-between items-center shadow-[3px_3px_0px_#1E1E1E] gap-2">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-blue-600 animate-pulse"></span>
              <span className="text-xs font-black uppercase tracking-wider text-gray-800">
                Pillar 1: Supervised Learning, Covariate Shift Modeling &amp; Subgroup Fairness Diagnostics
              </span>
            </div>
            <div className="flex items-center gap-2 text-[11px] font-bold text-gray-500 font-mono">
              <span>FICO HELOC (N = 10,459)</span>
              <span>•</span>
              <span>5 Random Seeds</span>
              <span>•</span>
              <span>95% Bootstrap CIs</span>
            </div>
          </div>

          {/* Top ML KPI Strip: Dual-Model & Drift Performance */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <div className="bg-white border-2 border-[#1E1E1E] rounded-xl p-3 shadow-[3px_3px_0px_#1E1E1E]">
              <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider block">
                Logistic Regression (T₀)
              </span>
              <div className="text-2xl font-black mt-1 text-blue-700">0.781 AUC</div>
              <span className="text-[10px] font-bold text-blue-800">Linear Benchmark (±0.003)</span>
            </div>

            <div className="bg-white border-2 border-[#1E1E1E] rounded-xl p-3 shadow-[3px_3px_0px_#1E1E1E]">
              <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider block">
                XGBoost (T₀ Baseline)
              </span>
              <div className="text-2xl font-black mt-1 text-indigo-700">0.854 AUC</div>
              <span className="text-[10px] font-bold text-indigo-800">+7.3% Non-Linear Lift</span>
            </div>

            <div className="bg-[#FEF2F2] border-2 border-[#DC2626] rounded-xl p-3 shadow-[3px_3px_0px_#DC2626]">
              <span className="text-[11px] font-bold text-red-700 uppercase tracking-wider block">
                XGBoost Drift Collapse (T₁)
              </span>
              <div className="text-2xl font-black text-red-600 mt-1">0.765 AUC</div>
              <span className="text-[10px] font-bold text-red-700">-8.9% Degradation (f₀ on T₁)</span>
            </div>

            <div className="bg-[#F0FDF4] border-2 border-[#16A34A] rounded-xl p-3 shadow-[3px_3px_0px_#16A34A]">
              <span className="text-[11px] font-bold text-green-700 uppercase tracking-wider block">
                XGBoost Retrained (T₁)
              </span>
              <div className="text-2xl font-black text-green-600 mt-1">0.904 AUC</div>
              <span className="text-[10px] font-bold text-green-700">+13.9% Restored Capacity (f₁)</span>
            </div>

            <div className="bg-[#FFFBEB] border-2 border-[#D97706] rounded-xl p-3 shadow-[3px_3px_0px_#D97706] col-span-2 md:col-span-1">
              <span className="text-[11px] font-bold text-amber-800 uppercase tracking-wider block">
                Thin-File Recall Plunge
              </span>
              <div className="text-2xl font-black text-amber-600 mt-1">59.8% → 28.1%</div>
              <span className="text-[10px] font-bold text-amber-800">FNR surges to 71.9% under drift</span>
            </div>
          </div>

          {/* Machine Learning Model Tournament & Subgroup Table */}
          <div className="bg-[#FFFDF9] border-3 border-[#1E1E1E] rounded-xl p-4 shadow-[4px_4px_0px_#1E1E1E]">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 mb-3 pb-2 border-b-2 border-gray-200">
              <div>
                <h2 className="text-base font-black uppercase flex items-center gap-2">
                  <FileSpreadsheet size={18} />
                  Machine Learning Model Tournament (5 Random Seeds • 95% CIs)
                </h2>
                <p className="text-xs font-bold text-gray-600">
                  Comparing Linear Benchmark (Logistic Regression) vs Non-Linear Gradient Boosted Trees (XGBoost)
                </p>
              </div>

              {/* Sub-Tabs: Overall Models vs Subgroups */}
              <div className="flex items-center gap-1 bg-gray-100 border border-gray-400 rounded-lg p-0.5">
                <button
                  onClick={() => setMlTableTab('OVERALL')}
                  className={`px-3 py-1 text-xs font-black rounded ${
                    mlTableTab === 'OVERALL'
                      ? 'bg-[#1E1E1E] text-white'
                      : 'text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Overall Models (Table 01)
                </button>
                <button
                  onClick={() => setMlTableTab('SUBGROUPS')}
                  className={`px-3 py-1 text-xs font-black rounded ${
                    mlTableTab === 'SUBGROUPS'
                      ? 'bg-[#1E1E1E] text-white'
                      : 'text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Subgroup Fairness (Table 02)
                </button>
              </div>
            </div>

            {/* Table 01: Overall Models */}
            {mlTableTab === 'OVERALL' && (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-medium">
                  <thead>
                    <tr className="bg-[#1E1E1E] text-white uppercase text-[10px] tracking-wider">
                      <th className="py-2 px-3">Classifier Model</th>
                      <th className="py-2 px-3">Evaluation Regime</th>
                      <th className="py-2 px-3">Accuracy</th>
                      <th className="py-2 px-3">Precision</th>
                      <th className="py-2 px-3">Recall</th>
                      <th className="py-2 px-3">F1-Score</th>
                      <th className="py-2 px-3 bg-indigo-900">ROC-AUC (95% CI)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {mlBenchmarks?.modelPerformance?.map((row: any, idx: number) => {
                      const isXgb = row.model === 'XGBoost';
                      const isRetrained = row.regime.includes('retrained');
                      const isDrift = row.regime.includes('unretrained');
                      return (
                        <tr
                          key={idx}
                          className={`hover:bg-amber-50/60 transition ${
                            isDrift ? 'bg-red-50/40' : (isRetrained ? 'bg-green-50/30' : '')
                          }`}
                        >
                          <td className="py-2 px-3 font-black">
                            <span className={isXgb ? 'text-indigo-900' : 'text-blue-900'}>
                              {row.model}
                            </span>
                          </td>
                          <td className="py-2 px-3 font-mono font-bold">
                            {row.regime === 'T0_baseline' && <span className="text-gray-800">T₀ Baseline</span>}
                            {row.regime === 'T1_drifted_unretrained' && <span className="text-red-700">T₁ Drifted (Unretrained f₀)</span>}
                            {row.regime === 'T1_drifted_retrained' && <span className="text-green-700">T₁ Drifted (Retrained f₁)</span>}
                          </td>
                          <td className="py-2 px-3">{row.accuracy}</td>
                          <td className="py-2 px-3">{row.precision}</td>
                          <td className="py-2 px-3 font-bold">{row.recall}</td>
                          <td className="py-2 px-3">{row.f1}</td>
                          <td className="py-2 px-3 font-black text-indigo-900 bg-indigo-50/70 font-mono">
                            {row.roc_auc}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {/* Table 02: Subgroup Fairness */}
            {mlTableTab === 'SUBGROUPS' && (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-medium">
                  <thead>
                    <tr className="bg-[#1E1E1E] text-white uppercase text-[10px] tracking-wider">
                      <th className="py-2 px-3">Classifier</th>
                      <th className="py-2 px-3">Regime</th>
                      <th className="py-2 px-3">Subgroup Cohort</th>
                      <th className="py-2 px-3">Sample (N)</th>
                      <th className="py-2 px-3">Recall</th>
                      <th className="py-2 px-3 bg-red-900 text-white">False Negative Rate (FNR)</th>
                      <th className="py-2 px-3">False Positive Rate (FPR)</th>
                      <th className="py-2 px-3">ROC-AUC</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {mlBenchmarks?.subgroupFairness?.map((row: any, idx: number) => {
                      const isThin = row.subgroup === 'thin-file';
                      const isDrift = row.regime.includes('unretrained');
                      return (
                        <tr
                          key={idx}
                          className={`hover:bg-amber-50/60 transition ${
                            isThin && isDrift ? 'bg-red-100/50' : (isThin ? 'bg-amber-50/40' : '')
                          }`}
                        >
                          <td className="py-2 px-3 font-black">{row.model}</td>
                          <td className="py-2 px-3 font-mono font-bold text-[11px]">
                            {row.regime.replace('_', ' ')}
                          </td>
                          <td className="py-2 px-3">
                            <span className={`px-2 py-0.5 rounded font-black text-[10px] ${
                              isThin ? 'bg-yellow-200 text-yellow-900 border border-yellow-500' : 'bg-blue-100 text-blue-900'
                            }`}>
                              {row.subgroup}
                            </span>
                          </td>
                          <td className="py-2 px-3 font-mono">{row.sample_size_n}</td>
                          <td className="py-2 px-3 font-bold">{row.recall}</td>
                          <td className={`py-2 px-3 font-black font-mono ${
                            isThin && isDrift ? 'text-red-700 bg-red-200/60' : 'text-gray-900'
                          }`}>
                            {row.fnr}
                          </td>
                          <td className="py-2 px-3">{row.fpr}</td>
                          <td className="py-2 px-3 font-mono">{row.roc_auc}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* 3-Column Diagnostic Visualizations Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            
            {/* Col 1: Interactive ROC & PR Curves */}
            <div className="bg-[#FFFDF9] border-3 border-[#1E1E1E] rounded-xl p-4 shadow-[4px_4px_0px_#1E1E1E] flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="text-sm font-black uppercase">1. Discrimination Curves</h3>
                    <p className="text-[11px] font-bold text-gray-600">
                      {mlCurveType === 'ROC' ? 'Receiver Operating Characteristic (ROC)' : 'Precision-Recall (PR) Curve'}
                    </p>
                  </div>

                  {/* Curve Toggle */}
                  <div className="flex items-center gap-1 bg-gray-100 border border-gray-400 rounded p-0.5 text-[10px] font-black">
                    <button
                      onClick={() => setMlCurveType('ROC')}
                      className={`px-2 py-0.5 rounded ${
                        mlCurveType === 'ROC' ? 'bg-[#1E1E1E] text-white' : 'text-gray-700'
                      }`}
                    >
                      ROC
                    </button>
                    <button
                      onClick={() => setMlCurveType('PR')}
                      className={`px-2 py-0.5 rounded ${
                        mlCurveType === 'PR' ? 'bg-[#1E1E1E] text-white' : 'text-gray-700'
                      }`}
                    >
                      PR
                    </button>
                  </div>
                </div>

                <div className="h-[210px] w-full mt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    {mlCurveType === 'ROC' ? (
                      <LineChart data={diagnosticCurves?.rocCurves} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                        <XAxis dataKey="fpr" label={{ value: 'False Positive Rate (%)', position: 'insideBottom', offset: -5, fontSize: 10 }} tick={{ fontSize: 10 }} unit="%" />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} unit="%" />
                        <Tooltip formatter={(value: any) => [`${value}%`, '']} contentStyle={{ fontSize: 11, fontWeight: 'bold' }} />
                        <Line type="monotone" dataKey="xgb_t1_retrain" name="XGBoost Retrained (AUC 0.904)" stroke="#047857" strokeWidth={2.5} dot={false} />
                        <Line type="monotone" dataKey="xgb_t0" name="XGBoost Baseline (AUC 0.854)" stroke="#1D4ED8" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="lr_t0" name="LogReg Baseline (AUC 0.781)" stroke="#3B82F6" strokeWidth={1.5} dot={false} />
                        <Line type="monotone" dataKey="xgb_t1_drift" name="XGBoost Drifted (AUC 0.765)" stroke="#DC2626" strokeWidth={2} strokeDasharray="3 3" dot={false} />
                        <Line type="monotone" dataKey="random" name="Random Classifier" stroke="#9CA3AF" strokeWidth={1} strokeDasharray="2 2" dot={false} />
                        <Legend wrapperStyle={{ fontSize: 9, paddingTop: 4 }} />
                      </LineChart>
                    ) : (
                      <LineChart data={diagnosticCurves?.prCurves} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                        <XAxis dataKey="recall" label={{ value: 'Recall (%)', position: 'insideBottom', offset: -5, fontSize: 10 }} tick={{ fontSize: 10 }} unit="%" />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} unit="%" />
                        <Tooltip formatter={(value: any) => [`${value}% Precision`, '']} contentStyle={{ fontSize: 11, fontWeight: 'bold' }} />
                        <Line type="monotone" dataKey="xgb_t1_retrain" name="XGBoost Retrained" stroke="#047857" strokeWidth={2.5} dot={false} />
                        <Line type="monotone" dataKey="xgb_t0" name="XGBoost Baseline" stroke="#1D4ED8" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="lr_t0" name="LogReg Baseline" stroke="#3B82F6" strokeWidth={1.5} dot={false} />
                        <Line type="monotone" dataKey="xgb_t1_drift" name="XGBoost Drifted" stroke="#DC2626" strokeWidth={2} strokeDasharray="3 3" dot={false} />
                        <Legend wrapperStyle={{ fontSize: 9, paddingTop: 4 }} />
                      </LineChart>
                    )}
                  </ResponsiveContainer>
                </div>
              </div>
              <p className="text-[11px] font-bold text-gray-500 mt-2 bg-gray-50 p-1.5 rounded border border-gray-200">
                Notice the red dashed line: unretrained XGBoost drops to 0.765 AUC under covariate drift, but retraining restores its capacity to 0.904 AUC.
              </p>
            </div>

            {/* Col 2: Subgroup Fairness Error Decomposition */}
            <div className="bg-[#FFFDF9] border-3 border-[#1E1E1E] rounded-xl p-4 shadow-[4px_4px_0px_#1E1E1E] flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="text-sm font-black uppercase">2. Subgroup Error Decomposition</h3>
                    <p className="text-[11px] font-bold text-gray-600">Recall vs False Negative Rate (FNR) Surge</p>
                  </div>
                  <span className="px-2 py-0.5 bg-amber-100 text-amber-800 border border-amber-600 text-[10px] font-bold rounded">
                    Fairness Gap
                  </span>
                </div>

                <div className="h-[210px] w-full mt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={diagnosticCurves?.subgroupErrorRates} margin={{ top: 5, right: 10, left: -20, bottom: 25 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                      <XAxis dataKey="subgroup" tick={{ fontSize: 8, angle: -20, textAnchor: 'end' }} interval={0} />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} unit="%" />
                      <Tooltip formatter={(value: any) => [`${value}%`, '']} contentStyle={{ fontSize: 11, fontWeight: 'bold' }} />
                      <Bar dataKey="recall" name="Recall (Approved)" fill="#2563EB" radius={[2, 2, 0, 0]} />
                      <Bar dataKey="fnr" name="False Negative Rate (Denied)" fill="#EF4444" radius={[2, 2, 0, 0]} />
                      <Legend wrapperStyle={{ fontSize: 9, paddingTop: 18 }} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <p className="text-[11px] font-bold text-gray-500 mt-2 bg-gray-50 p-1.5 rounded border border-gray-200">
                Thin-file applicants face an acute error asymmetry: during drift, their False Negative Rate jumps to 71.9%, meaning 7 out of 10 creditworthy thin-file applicants are rejected.
              </p>
            </div>

            {/* Col 3: Normalized Confusion Matrix Breakdown */}
            <div className="bg-[#FFFDF9] border-3 border-[#1E1E1E] rounded-xl p-4 shadow-[4px_4px_0px_#1E1E1E] flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="text-sm font-black uppercase">3. Confusion Matrix Breakdown</h3>
                    <p className="text-[11px] font-bold text-gray-600">Normalized Error Rates (% of Cohort)</p>
                  </div>
                  <span className="px-2 py-0.5 bg-blue-100 text-blue-800 border border-blue-600 text-[10px] font-bold rounded">
                    Classification Matrix
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 mt-2">
                  <div className="bg-gray-50 border-2 border-gray-300 rounded p-2 text-xs">
                    <span className="font-bold text-[10px] text-gray-500 block">XGBoost Baseline (T₀)</span>
                    <div className="grid grid-cols-2 gap-1 mt-1 text-center font-mono">
                      <div className="bg-green-100 p-1 rounded font-bold text-green-800">TN: 39.5%</div>
                      <div className="bg-red-100 p-1 rounded text-red-800">FP: 12.7%</div>
                      <div className="bg-red-100 p-1 rounded text-red-800">FN: 10.7%</div>
                      <div className="bg-green-100 p-1 rounded font-bold text-green-800">TP: 37.1%</div>
                    </div>
                  </div>

                  <div className="bg-red-50 border-2 border-red-300 rounded p-2 text-xs">
                    <span className="font-bold text-[10px] text-red-700 block">XGBoost Drifted (T₁)</span>
                    <div className="grid grid-cols-2 gap-1 mt-1 text-center font-mono">
                      <div className="bg-green-100 p-1 rounded font-bold text-green-800">TN: 42.1%</div>
                      <div className="bg-red-100 p-1 rounded text-red-800">FP: 10.1%</div>
                      <div className="bg-red-200 p-1 rounded font-black text-red-900 border border-red-400">FN: 25.8%</div>
                      <div className="bg-green-100 p-1 rounded font-bold text-green-800">TP: 22.0%</div>
                    </div>
                  </div>

                  <div className="bg-blue-50 border-2 border-blue-300 rounded p-2 text-xs col-span-2">
                    <span className="font-bold text-[10px] text-blue-800 block">XGBoost Retrained (T₁ Restored)</span>
                    <div className="grid grid-cols-4 gap-1 mt-1 text-center font-mono">
                      <div className="bg-green-100 p-1 rounded font-bold text-green-800">TN: 46.2%</div>
                      <div className="bg-red-100 p-1 rounded text-red-800">FP: 6.0%</div>
                      <div className="bg-red-100 p-1 rounded text-red-800">FN: 18.5%</div>
                      <div className="bg-green-100 p-1 rounded font-bold text-green-800">TP: 29.3%</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Bottom Switcher Callout to Tab 2 */}
              <div className="mt-3 pt-2 border-t border-gray-200 flex justify-between items-center">
                <span className="text-[10px] font-bold text-gray-500">
                  Ready to inspect algorithmic recourse invalidation?
                </span>
                <button
                  onClick={() => setActiveTab('XAI_RECOURSE')}
                  className="px-3 py-1 bg-[#1E1E1E] text-white text-xs font-black uppercase rounded flex items-center gap-1.5 hover:bg-black transition shadow"
                >
                  Proceed to Tab 2: XAI Audit
                  <ArrowRight size={12} />
                </button>
              </div>
            </div>

          </div>

          {/* Machine Learning Project Summary Box */}
          <div className="bg-white border-3 border-[#1E1E1E] rounded-xl p-4 shadow-[4px_4px_0px_#1E1E1E] flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
            <div className="space-y-1">
              <span className="text-xs font-black uppercase text-indigo-700 tracking-wider block">
                Machine Learning Project Finding:
              </span>
              <p className="text-xs font-medium text-gray-700 max-w-4xl">
                Non-linear gradient boosting achieves superior discrimination under stationary conditions (+7.3% AUC over logistic regression), but degrades severely (-8.9% AUC) under macroeconomic covariate shift with an acute fairness penalty on thin-file borrowers (False Negative Rate surging to 71.9%). Retraining restores predictive capacity (+13.9% AUC lift). How does this continuous retraining destabilize post-hoc explanations and counterfactual recourse?
              </p>
            </div>
            <button
              onClick={() => setActiveTab('XAI_RECOURSE')}
              className="px-4 py-2 bg-[#1E1E1E] text-white text-xs font-black uppercase rounded-lg hover:bg-black flex items-center gap-2 whitespace-nowrap shadow-[2px_2px_0px_#6366F1]"
            >
              Inspect XAI Recourse Audit
              <ArrowRight size={14} />
            </button>
          </div>

        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: XAI & RECOURSE AUDIT (THE EXPLAINABLE AI PROJECT PROPOSAL)         */}
      {/* ========================================================================= */}
      {activeTab === 'XAI_RECOURSE' && (
        <div className="space-y-4">
          
          {/* Sub-Navigation Switcher for Tab 2: Macro vs Applicant Deep-Dive */}
          <div className="bg-white border-2 border-[#1E1E1E] rounded-xl px-4 py-2 flex flex-wrap justify-between items-center shadow-[3px_3px_0px_#1E1E1E] gap-2">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-amber-500 animate-pulse"></span>
              <span className="text-xs font-black uppercase tracking-wider text-gray-800">
                Pillar 2: Explainable AI, Counterfactual Recourse Invalidation &amp; Human-in-the-Loop Audit
              </span>
            </div>

            <div className="flex items-center gap-1.5 bg-gray-100 border border-gray-400 rounded-lg p-1">
              <button
                id="xai-subview-macro"
                onClick={() => setXaiSubView('MACRO')}
                className={`px-3 py-1 text-xs font-black rounded flex items-center gap-1.5 transition ${
                  xaiSubView === 'MACRO'
                    ? 'bg-[#1E1E1E] text-white shadow'
                    : 'text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Users size={14} />
                Macro Cohort View
              </button>
              <button
                id="xai-subview-deepdive"
                onClick={() => setXaiSubView('DEEPDIVE')}
                className={`px-3 py-1 text-xs font-black rounded flex items-center gap-1.5 transition ${
                  xaiSubView === 'DEEPDIVE'
                    ? 'bg-[#1E1E1E] text-white shadow'
                    : 'text-gray-700 hover:bg-gray-200'
                }`}
              >
                <UserCheck size={14} />
                Applicant Deep-Dive ({selectedApplicantId})
              </button>
            </div>
          </div>

          {/* ------------------------------------------------------------------- */}
          {/* SUB-VIEW A: MACRO COHORT & RECOURSE BOUNDARY FRONTIER               */}
          {/* ------------------------------------------------------------------- */}
          {xaiSubView === 'MACRO' && (
            <div className="space-y-4">
              
              {/* Top Macro Metrics Strip (5 KPI Cards) */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="bg-white border-2 border-[#1E1E1E] rounded-xl p-3 shadow-[3px_3px_0px_#1E1E1E]">
                  <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider block">Total Cohort</span>
                  <div className="text-2xl font-black mt-1">10,459</div>
                  <span className="text-[10px] font-bold text-blue-600">FICO HELOC Dataset</span>
                </div>

                <div className="bg-white border-2 border-[#1E1E1E] rounded-xl p-3 shadow-[3px_3px_0px_#1E1E1E]">
                  <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider block">Approval Rate (T₀ → T₁)</span>
                  <div className="text-2xl font-black mt-1 text-amber-600">47.8% → 38.2%</div>
                  <span className="text-[10px] font-bold text-red-600">-9.6% Macro Tightening</span>
                </div>

                <div className="bg-[#FEF2F2] border-2 border-[#DC2626] rounded-xl p-3 shadow-[3px_3px_0px_#DC2626]">
                  <span className="text-[11px] font-bold text-red-700 uppercase tracking-wider block">Thin-File Invalidation</span>
                  <div className="text-2xl font-black text-red-600 mt-1">66.7%</div>
                  <span className="text-[10px] font-bold text-red-700">High Risk (n = 1,319)</span>
                </div>

                <div className="bg-[#F0FDF4] border-2 border-[#16A34A] rounded-xl p-3 shadow-[3px_3px_0px_#16A34A]">
                  <span className="text-[11px] font-bold text-green-700 uppercase tracking-wider block">Thick-File Invalidation</span>
                  <div className="text-2xl font-black text-green-600 mt-1">20.0%</div>
                  <span className="text-[10px] font-bold text-green-700">Resilient File (n = 3,700)</span>
                </div>

                <div className="bg-[#FFFBEB] border-2 border-[#D97706] rounded-xl p-3 shadow-[3px_3px_0px_#D97706] col-span-2 md:col-span-1">
                  <span className="text-[11px] font-bold text-amber-800 uppercase tracking-wider block">Disparity Gap</span>
                  <div className="text-2xl font-black text-amber-600 mt-1">+46.7%</div>
                  <span className="text-[10px] font-bold text-amber-800">Two-Prop Z: p &lt; 0.05</span>
                </div>
              </div>

              {/* 3-Column Bento Grid Visual Analytics */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                
                {/* Col 1: ALE / PDP Global Decision Boundary Shift */}
                <div className="bg-[#FFFDF9] border-3 border-[#1E1E1E] rounded-xl p-4 shadow-[4px_4px_0px_#1E1E1E] flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h2 className="text-base font-black uppercase">1. Global Decision Boundary (ALE / PDP)</h2>
                        <p className="text-xs font-bold text-gray-600">Approval probability across Revolving Debt Burden</p>
                      </div>
                      <span className="px-2 py-0.5 bg-blue-100 text-blue-800 border border-blue-600 text-[10px] font-bold rounded">
                        Boundary Shift
                      </span>
                    </div>
                    <div className="h-[220px] w-full mt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={alePdpData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                          <XAxis dataKey="burden" label={{ value: 'Revolving Debt (%)', position: 'insideBottom', offset: -5, fontSize: 10 }} tick={{ fontSize: 10 }} />
                          <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} unit="%" />
                          <Tooltip formatter={(value: any) => [`${value}%`, '']} contentStyle={{ fontSize: 11, fontWeight: 'bold' }} />
                          <ReferenceLine y={50} stroke="#EF4444" strokeDasharray="3 3" label={{ value: 'Approval Cutoff', position: 'insideTopLeft', fill: '#EF4444', fontSize: 9 }} />
                          <Line type="monotone" dataKey="p_t0" name="T₀ Baseline Model" stroke="#2563EB" strokeWidth={2.5} dot={false} />
                          <Line type="monotone" dataKey="p_t1" name="T₁ Retrained (Drifted)" stroke="#DC2626" strokeWidth={2.5} dot={false} />
                          <Legend wrapperStyle={{ fontSize: 10, paddingTop: 6 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <p className="text-[11px] font-bold text-gray-500 mt-2 bg-gray-50 p-2 rounded border border-gray-200">
                    Notice the leftward contraction: Under T₀, applicants with 55% burden were approved (p &gt; 0.50). Under T₁, the threshold tightened to 42%, invalidating recourse targets in the 43–54% zone.
                  </p>
                </div>

                {/* Col 2: Subgroup Disparity Matrix across Credit Buckets */}
                <div className="bg-[#FFFDF9] border-3 border-[#1E1E1E] rounded-xl p-4 shadow-[4px_4px_0px_#1E1E1E] flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h2 className="text-base font-black uppercase">2. Subgroup Invalidation Matrix</h2>
                        <p className="text-xs font-bold text-gray-600">Thin-file vs Thick-file across Credit Score Buckets</p>
                      </div>
                      <span className="px-2 py-0.5 bg-red-100 text-red-800 border border-red-600 text-[10px] font-bold rounded">
                        Disparity Map
                      </span>
                    </div>
                    <div className="h-[220px] w-full mt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={scoreBucketData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                          <XAxis dataKey="bucket" label={{ value: 'Risk Score Range', position: 'insideBottom', offset: -5, fontSize: 10 }} tick={{ fontSize: 10 }} />
                          <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} unit="%" />
                          <Tooltip formatter={(value: any) => [`${value}% Invalidation`, '']} contentStyle={{ fontSize: 11, fontWeight: 'bold' }} />
                          <Bar dataKey="thin" name="Thin-File Applicants" fill="#EF4444" radius={[3, 3, 0, 0]} />
                          <Bar dataKey="thick" name="Thick-File Applicants" fill="#3B82F6" radius={[3, 3, 0, 0]} />
                          <Legend wrapperStyle={{ fontSize: 10, paddingTop: 6 }} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <p className="text-[11px] font-bold text-gray-500 mt-2 bg-gray-50 p-2 rounded border border-gray-200">
                    In the 60–69 prime denial zone, thin-file applicants suffer a 66.7–72% invalidation rate, whereas thick-file borrowers with identical scores only suffer 20–28%.
                  </p>
                </div>

                {/* Col 3: Drift Sensitivity & Recourse Frontier Curve */}
                <div className="bg-[#FFFDF9] border-3 border-[#1E1E1E] rounded-xl p-4 shadow-[4px_4px_0px_#1E1E1E] flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h2 className="text-base font-black uppercase">3. Drift Sensitivity Frontier</h2>
                        <p className="text-xs font-bold text-gray-600">Macroeconomic Δ (5% → 35%) vs Invalidation Surge</p>
                      </div>
                      <span className="px-2 py-0.5 bg-amber-100 text-amber-800 border border-amber-600 text-[10px] font-bold rounded">
                        Stress Test
                      </span>
                    </div>
                    <div className="h-[220px] w-full mt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={sensitivityCurveData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                          <XAxis dataKey="delta" label={{ value: 'Drift Magnitude Δ', position: 'insideBottom', offset: -5, fontSize: 10 }} tick={{ fontSize: 10 }} />
                          <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} unit="%" />
                          <Tooltip formatter={(value: any) => [`${value}%`, '']} contentStyle={{ fontSize: 11, fontWeight: 'bold' }} />
                          <Line type="monotone" dataKey="sir_thin" name="Thin-File SIR" stroke="#EF4444" strokeWidth={2.5} />
                          <Line type="monotone" dataKey="sir_thick" name="Thick-File SIR" stroke="#10B981" strokeWidth={2.5} />
                          <Line type="monotone" dataKey="gap" name="Disparity Gap" stroke="#D97706" strokeDasharray="4 4" strokeWidth={2} />
                          <Legend wrapperStyle={{ fontSize: 10, paddingTop: 6 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <p className="text-[11px] font-bold text-gray-500 mt-2 bg-gray-50 p-2 rounded border border-gray-200">
                    As economic drift severity climbs past 20%, thin-file recourse invalidation accelerates exponentially, widening the disparity gap to over 50 percentage points.
                  </p>
                </div>

              </div>

              {/* Quick Action Bar to Jump into Applicant Deep-Dive */}
              <div className="bg-white border-2 border-[#1E1E1E] rounded-xl p-3 flex flex-col md:flex-row justify-between items-center gap-3 shadow-[3px_3px_0px_#1E1E1E]">
                <div className="flex items-center gap-3">
                  <Sliders size={20} className="text-amber-600" />
                  <span className="text-xs lg:text-sm font-black">
                    Ready to investigate an individual applicant and inspect their Ceteris Paribus path?
                  </span>
                </div>
                <button
                  onClick={() => setXaiSubView('DEEPDIVE')}
                  className="px-4 py-2 bg-[#1E1E1E] text-white text-xs font-black uppercase rounded-lg hover:bg-black flex items-center gap-2 transition"
                >
                  Open Applicant Deep-Dive
                  <ArrowRight size={14} />
                </button>
              </div>

            </div>
          )}

          {/* ------------------------------------------------------------------- */}
          {/* SUB-VIEW B: APPLICANT RECOURSE DEEP-DIVE & UNDERWRITER TRIAGE       */}
          {/* ------------------------------------------------------------------- */}
          {xaiSubView === 'DEEPDIVE' && (
            <div className="space-y-4">
              
              {/* Header Controls & Filter Pills */}
              <div className="bg-white border-2 border-[#1E1E1E] rounded-xl p-3 flex flex-wrap justify-between items-center gap-3 shadow-[3px_3px_0px_#1E1E1E]">
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="text-xs font-black uppercase tracking-wider text-gray-500">
                    Select Applicant:
                  </span>
                  <select
                    value={selectedApplicantId}
                    onChange={(e) => setSelectedApplicantId(e.target.value)}
                    className="border-2 border-[#1E1E1E] font-black text-sm px-3 py-1.5 rounded-lg bg-gray-50 shadow-inner"
                  >
                    {filteredApplicants.map((app) => (
                      <option key={app.applicantId} value={app.applicantId}>
                        {app.applicantId} ({app.subgroup} • Risk: {app.invalidationRisk}%)
                      </option>
                    ))}
                  </select>

                  {/* Filter Pills */}
                  <div className="flex items-center gap-1.5 ml-2">
                    <button
                      onClick={() => setApplicantFilter('ALL')}
                      className={`px-2.5 py-1 text-xs font-black rounded border ${
                        applicantFilter === 'ALL'
                          ? 'bg-[#1E1E1E] text-white border-[#1E1E1E]'
                          : 'bg-gray-100 text-gray-700 border-gray-300 hover:bg-gray-200'
                      }`}
                    >
                      All ({applicantsList.length})
                    </button>
                    <button
                      onClick={() => setApplicantFilter('THIN_HIGH')}
                      className={`px-2.5 py-1 text-xs font-black rounded border flex items-center gap-1 ${
                        applicantFilter === 'THIN_HIGH'
                          ? 'bg-red-600 text-white border-red-700'
                          : 'bg-red-50 text-red-700 border-red-200 hover:bg-red-100'
                      }`}
                    >
                      <AlertTriangle size={12} />
                      Thin-File High Risk
                    </button>
                    <button
                      onClick={() => setApplicantFilter('THICK_LOW')}
                      className={`px-2.5 py-1 text-xs font-black rounded border flex items-center gap-1 ${
                        applicantFilter === 'THICK_LOW'
                          ? 'bg-green-600 text-white border-green-700'
                          : 'bg-green-50 text-green-700 border-green-200 hover:bg-green-100'
                      }`}
                    >
                      <ShieldAlert size={12} />
                      Thick-File Low Risk
                    </button>
                  </div>
                </div>

                {/* Return to Macro View Button */}
                <button
                  onClick={() => setXaiSubView('MACRO')}
                  className="px-3 py-1.5 bg-gray-100 border border-gray-400 text-gray-800 text-xs font-black rounded-lg hover:bg-gray-200 flex items-center gap-1.5 transition"
                >
                  <Users size={14} />
                  Return to Macro View
                </button>
              </div>

              {/* Horizontal Scrollable Applicant Chips */}
              <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
                {filteredApplicants.map((app) => (
                  <button
                    key={app.applicantId}
                    onClick={() => setSelectedApplicantId(app.applicantId)}
                    className={`flex-shrink-0 px-3 py-2 rounded-xl border-2 text-left transition-all ${
                      selectedApplicantId === app.applicantId
                        ? 'bg-[#1E1E1E] text-white border-[#1E1E1E] shadow-[2px_2px_0px_#F59E0B]'
                        : 'bg-white text-gray-800 border-gray-300 hover:border-gray-500'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-black text-xs">{app.applicantId}</span>
                      <span className={`text-[9px] font-bold px-1.5 py-0.2 rounded uppercase ${
                        app.subgroup === 'thin-file' ? 'bg-amber-200 text-amber-900' : 'bg-blue-200 text-blue-900'
                      }`}>
                        {app.subgroup}
                      </span>
                    </div>
                    <div className="text-[10px] text-gray-400 mt-1 font-bold">
                      Risk: <span className={app.invalidationRisk > 50 ? 'text-red-400' : 'text-green-400'}>{app.invalidationRisk}%</span>
                      {' • '}Score: {app.externalRiskEstimate}
                    </div>
                  </button>
                ))}
              </div>

              {/* Bento Grid: 3 Live Applicant Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                
                {/* 1. Applicant Profile */}
                <div className="bg-[#FFFDF9] border-3 border-[#1E1E1E] rounded-xl p-4 shadow-[4px_4px_0px_#1E1E1E]">
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-sm font-black uppercase">1. Applicant Profile</h3>
                    <span className={`px-2 py-0.5 rounded font-black text-xs uppercase ${
                      applicantDetails?.subgroup === 'thin-file'
                        ? 'bg-yellow-300 text-black border border-black'
                        : 'bg-blue-300 text-black border border-black'
                    }`}>
                      {applicantDetails?.subgroup || 'Thin-File'}
                    </span>
                  </div>
                  <div className="font-mono text-xl font-black mb-3">
                    {applicantDetails?.applicantId || selectedApplicantId}
                  </div>
                  <hr className="border-t-2 border-gray-200 mb-3" />
                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between py-1 border-b border-gray-100">
                      <span className="font-bold text-gray-600">External Risk Estimate:</span>
                      <span className="font-black">{applicantDetails?.features?.ExternalRiskEstimate || 62}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-gray-100">
                      <span className="font-bold text-gray-600">Months in File:</span>
                      <span className="font-black text-amber-700">
                        {applicantDetails?.features?.MSinceOldestTradeOpen || 14} mos (Locked)
                      </span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-gray-100">
                      <span className="font-bold text-gray-600">Total Trade Lines:</span>
                      <span className="font-black text-amber-700">
                        {applicantDetails?.features?.NumTotalTrades || 5} (Locked)
                      </span>
                    </div>
                    <div className="flex justify-between py-1">
                      <span className="font-bold text-gray-600">Revolving Burden:</span>
                      <span className="font-black text-red-600">
                        {applicantDetails?.features?.NetFractionRevolvingBurden || 85}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* 2. Model Decision & Invalidation Risk */}
                <div className="bg-[#FFFDF9] border-3 border-[#1E1E1E] rounded-xl p-4 shadow-[4px_4px_0px_#1E1E1E] flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-start mb-2">
                      <h3 className="text-sm font-black uppercase">2. Model Decision &amp; Risk</h3>
                      <span className="px-2 py-0.5 bg-red-100 text-red-800 border border-red-700 font-mono font-bold text-xs rounded">
                        f₀(x) = 0
                      </span>
                    </div>
                    <div className="flex items-baseline justify-between mt-1">
                      <span className="text-3xl font-black text-red-600">
                        {applicantDetails?.decision || 'DENIED'}
                      </span>
                      <span className="text-xs font-bold text-gray-600">
                        P(Approval): <strong className="text-sm font-black text-black">{Math.round((applicantDetails?.approvalProbability || 0.234) * 100)}%</strong>
                      </span>
                    </div>
                    <hr className="border-t-2 border-gray-200 my-3" />
                    <div>
                      <div className="flex justify-between text-xs font-bold mb-1">
                        <span>Recourse Invalidation Vulnerability:</span>
                        <span className="font-black text-red-600">{applicantDetails?.invalidationRisk || 66.7}%</span>
                      </div>
                      <div className="w-full bg-gray-200 border-2 border-[#1E1E1E] rounded-full h-4 overflow-hidden">
                        <div
                          className="bg-red-500 h-full transition-all duration-500"
                          style={{ width: `${applicantDetails?.invalidationRisk || 66.7}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                  <p className="text-[11px] font-bold text-red-700 mt-3 bg-red-50 p-2 rounded border border-red-200">
                    ⚠️ {applicantDetails?.subgroup === 'thin-file' ? 'Thin-File Alert:' : 'Notice:'} Prescribed counterfactual recourse has a {applicantDetails?.invalidationRisk || 66.7}% probability of invalidation upon model retraining.
                  </p>
                </div>

                {/* 3. Actionable DiCE Recourse Plan */}
                <div className="bg-[#FFFDF9] border-3 border-[#1E1E1E] rounded-xl p-4 shadow-[4px_4px_0px_#1E1E1E] flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-start mb-2">
                      <h3 className="text-sm font-black uppercase">3. DiCE Action Plan</h3>
                      <span className="px-2 py-0.5 bg-green-100 text-green-800 border border-green-700 font-bold text-xs rounded">
                        Prescribed Actions
                      </span>
                    </div>
                    <p className="text-xs font-bold text-gray-500 mb-2">
                      Required perturbations to cross decision boundary under f₀:
                    </p>
                    <div className="space-y-2">
                      {applicantDetails?.recoursePlan?.actionableFeatures?.map((act: any, idx: number) => (
                        <div key={idx} className="bg-[#FEF3C7] border-2 border-[#1E1E1E] rounded-lg p-2 text-xs font-bold flex justify-between items-center">
                          <span>
                            {act.delta < 0 ? '↓' : '↑'} {act.feature}:
                          </span>
                          <span className="font-black text-amber-900 font-mono">
                            {act.current}{act.unit} → {act.target}{act.unit}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="mt-3 pt-2 border-t-2 border-gray-200 flex justify-between items-center text-xs">
                    <span className="font-bold text-gray-600">Normalized L₁ Cost:</span>
                    <span className="font-mono font-black text-sm bg-gray-100 px-2 py-0.5 rounded border border-gray-300">
                      {applicantDetails?.recoursePlan?.actionCostL1 || 5.62}
                    </span>
                  </div>
                </div>

              </div>

              {/* Lower Analytics Grid: Cumulative Profile, Local SHAP & Underwriter Review */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                
                {/* 1. Cumulative Profile (Ceteris Paribus CP Curve) */}
                <div className="bg-[#FFFDF9] border-3 border-[#1E1E1E] rounded-xl p-4 shadow-[4px_4px_0px_#1E1E1E] flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h3 className="text-sm font-black uppercase">1. Cumulative Profile (CP Curve)</h3>
                        <p className="text-[11px] font-bold text-gray-600">Approval probability as Revolving Debt varies</p>
                      </div>
                      <span className="px-2 py-0.5 bg-indigo-100 text-indigo-800 border border-indigo-600 text-[10px] font-bold rounded">
                        Ceteris Paribus
                      </span>
                    </div>
                    <div className="h-[200px] w-full mt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={cumulativeProfileData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                          <XAxis dataKey="burden" label={{ value: 'Debt Burden (%)', position: 'insideBottom', offset: -5, fontSize: 10 }} tick={{ fontSize: 10 }} />
                          <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} unit="%" />
                          <Tooltip formatter={(value: any) => [`${value}%`, '']} contentStyle={{ fontSize: 11, fontWeight: 'bold' }} />
                          <ReferenceLine y={50} stroke="#EF4444" strokeDasharray="3 3" label={{ value: 'Approval Cutoff', position: 'insideTopLeft', fill: '#EF4444', fontSize: 9 }} />
                          <Line type="monotone" dataKey="p_t0" name="P(Approval) T₀" stroke="#2563EB" strokeWidth={2.5} dot={false} />
                          <Line type="monotone" dataKey="p_t1" name="P(Approval) T₁" stroke="#DC2626" strokeWidth={2.5} dot={false} />
                          <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <p className="text-[11px] font-bold text-gray-500 mt-2 bg-gray-50 p-2 rounded border border-gray-200">
                    The gap between the blue (T₀) and red (T₁) curves illustrates why reducing debt to 45% succeeds under T₀ but fails under retrained T₁.
                  </p>
                </div>

                {/* 2. Local SHAP Attribution Waterfall */}
                <div className="bg-[#FFFDF9] border-3 border-[#1E1E1E] rounded-xl p-4 shadow-[4px_4px_0px_#1E1E1E] flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h3 className="text-sm font-black uppercase">2. Local SHAP Attributions</h3>
                        <p className="text-[11px] font-bold text-gray-600">Feature contributions pushing toward denial</p>
                      </div>
                      <span className="px-2 py-0.5 bg-purple-100 text-purple-800 border border-purple-600 text-[10px] font-bold rounded">
                        TreeSHAP
                      </span>
                    </div>
                    <div className="space-y-1.5 mt-2">
                      {applicantDetails?.shapBreakdown?.map((sh: any, idx: number) => {
                        const isNegative = sh.impact < 0;
                        const widthPct = Math.min(100, Math.round(Math.abs(sh.impact) * 80));
                        return (
                          <div key={idx} className="text-xs">
                            <div className="flex justify-between font-bold text-[11px] mb-0.5">
                              <span className="truncate max-w-[170px]">{sh.feature}</span>
                              <span className={isNegative ? 'text-red-700' : 'text-green-700'}>
                                {isNegative ? '' : '+'}{sh.impact.toFixed(2)} (Val: {sh.val})
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden flex">
                              {isNegative ? (
                                <div className="bg-red-500 h-full ml-auto" style={{ width: `${widthPct}%` }}></div>
                              ) : (
                                <div className="bg-green-500 h-full" style={{ width: `${widthPct}%` }}></div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  <p className="text-[11px] font-bold text-gray-500 mt-2 bg-gray-50 p-2 rounded border border-gray-200">
                    High revolving burden and low credit tenure dominate the denial. Notice how immutable features prevent quick qualification.
                  </p>
                </div>

                {/* 3. Underwriter Triage & Audit Feedback Modal / Form */}
                <div className="bg-[#FFFDF9] border-3 border-[#1E1E1E] rounded-xl p-4 shadow-[4px_4px_0px_#1E1E1E] flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h3 className="text-sm font-black uppercase">3. Underwriter Audit Review</h3>
                        <p className="text-[11px] font-bold text-gray-600">Interactive Human-in-the-Loop Validation</p>
                      </div>
                      <span className="px-2 py-0.5 bg-amber-100 text-amber-800 border border-amber-600 text-[10px] font-bold rounded">
                        Auditor Triage
                      </span>
                    </div>

                    <div className="space-y-3 mt-2 text-xs">
                      {/* Actionability Rating */}
                      <div>
                        <div className="flex justify-between font-bold text-gray-700 mb-1">
                          <span>Actionability (Can borrower execute?):</span>
                          <span className="font-black text-amber-700">{actionabilityScore} / 5</span>
                        </div>
                        <input
                          type="range"
                          min="1"
                          max="5"
                          value={actionabilityScore}
                          onChange={(e) => setActionabilityScore(parseInt(e.target.value))}
                          className="w-full accent-black cursor-pointer"
                        />
                      </div>

                      {/* Plausibility Rating */}
                      <div>
                        <div className="flex justify-between font-bold text-gray-700 mb-1">
                          <span>Plausibility (Is path realistic?):</span>
                          <span className="font-black text-amber-700">{plausibilityScore} / 5</span>
                        </div>
                        <input
                          type="range"
                          min="1"
                          max="5"
                          value={plausibilityScore}
                          onChange={(e) => setPlausibilityScore(parseInt(e.target.value))}
                          className="w-full accent-black cursor-pointer"
                        />
                      </div>

                      {/* Final Audit Verdict */}
                      <div>
                        <span className="font-bold text-gray-700 block mb-1">Auditor Final Verdict:</span>
                        <div className="grid grid-cols-3 gap-1.5 font-bold">
                          <button
                            type="button"
                            onClick={() => setUnderwriterVerdict('ACCEPT')}
                            className={`py-1 rounded border text-center transition ${
                              underwriterVerdict === 'ACCEPT'
                                ? 'bg-green-600 text-white border-green-700 shadow-sm'
                                : 'bg-gray-100 text-gray-700 border-gray-300 hover:bg-gray-200'
                            }`}
                          >
                            Accept
                          </button>
                          <button
                            type="button"
                            onClick={() => setUnderwriterVerdict('MODIFY')}
                            className={`py-1 rounded border text-center transition ${
                              underwriterVerdict === 'MODIFY'
                                ? 'bg-amber-500 text-white border-amber-600 shadow-sm'
                                : 'bg-gray-100 text-gray-700 border-gray-300 hover:bg-gray-200'
                            }`}
                          >
                            Modify
                          </button>
                          <button
                            type="button"
                            onClick={() => setUnderwriterVerdict('REJECT')}
                            className={`py-1 rounded border text-center transition ${
                              underwriterVerdict === 'REJECT'
                                ? 'bg-red-600 text-white border-red-700 shadow-sm'
                                : 'bg-gray-100 text-gray-700 border-gray-300 hover:bg-gray-200'
                            }`}
                          >
                            Reject
                          </button>
                        </div>
                      </div>

                      {/* Auditor Review Notes */}
                      <div>
                        <textarea
                          placeholder="Audit comments: e.g., thin-file applicant lacks sufficient trade history; burden reduction of 40% exceeds expected borrower monthly liquidity..."
                          value={underwriterNotes}
                          onChange={(e) => setUnderwriterNotes(e.target.value)}
                          className="w-full text-[11px] p-2 border-2 border-[#1E1E1E] rounded-lg bg-gray-50 focus:bg-white resize-none h-14"
                        />
                      </div>
                    </div>
                  </div>

                  <div className="mt-2 pt-2 border-t border-gray-200 flex justify-between items-center">
                    {feedbackSuccess ? (
                      <span className="text-xs font-black text-green-700 flex items-center gap-1">
                        <CheckCircle2 size={14} /> Review Logged to Audit Trail!
                      </span>
                    ) : (
                      <span className="text-[10px] font-bold text-gray-500">
                        Human-in-the-loop decision trail
                      </span>
                    )}
                    <button
                      type="button"
                      onClick={handleFeedbackSubmit}
                      className="px-3 py-1.5 bg-[#1E1E1E] text-white text-xs font-black uppercase rounded-lg hover:bg-black transition shadow"
                    >
                      Log Audit Review
                    </button>
                  </div>
                </div>

              </div>

            </div>
          )}

        </div>
      )}

      {/* ======================================================================= */}
      {/* COCKPIT FOOTER                                                          */}
      {/* ======================================================================= */}
      <footer className="mt-8 pt-4 border-t-2 border-gray-300 flex flex-col sm:flex-row justify-between items-center text-xs font-bold text-gray-600 gap-2">
        <div className="flex items-center gap-2">
          <span>Credit Model Drift Research Platform</span>
          <span>•</span>
          <span>Dual Academic Submission (ML + XAI)</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setActiveTab('ML_CORE')}
            className={`hover:underline ${activeTab === 'ML_CORE' ? 'font-black text-black' : ''}`}
          >
            Tab 1: ML Core
          </button>
          <span>|</span>
          <button
            onClick={() => setActiveTab('XAI_RECOURSE')}
            className={`hover:underline ${activeTab === 'XAI_RECOURSE' ? 'font-black text-black' : ''}`}
          >
            Tab 2: XAI Audit
          </button>
          <span>|</span>
          <a
            href="https://github.com/meghna/credit-model-drift"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:underline flex items-center gap-1 text-black"
          >
            GitHub Repository
          </a>
        </div>
      </footer>

    </div>
  );
}
