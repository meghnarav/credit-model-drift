"use client";
import React, { useState } from 'react';

export default function Dashboard() {
  const [loading, setLoading] = useState(false);
  const [hasEvaluated, setHasEvaluated] = useState(false);

  const handleEvaluate = () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setHasEvaluated(true);
    }, 1500); // Simulate API call
  };

  return (
    <main className="min-h-screen p-8 max-w-6xl mx-auto">
      <header className="mb-10 flex justify-between items-end border-b-2 border-[#1E1E1E] pb-4 border-dashed">
        <div>
          <h1 className="text-5xl font-display text-[#1E1E1E]">Credit Drift Analysis ✦</h1>
          <p className="text-lg text-gray-700 mt-2 font-semibold">Evaluating Recourse Reliability under Covariate Shift</p>
        </div>
        {!hasEvaluated && (
          <button onClick={handleEvaluate} className="brutal-btn brutal-btn-primary text-lg">
            {loading ? 'Evaluating...' : 'Run Evaluation ✨'}
          </button>
        )}
      </header>

      {hasEvaluated ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          
          {/* Top Left: Applicant Profile */}
          <div className="brutal-card relative overflow-hidden">
            <div className="absolute top-2 right-2 text-3xl opacity-20">📝</div>
            <h2 className="text-3xl font-display mb-4">Applicant Profile</h2>
            <div className="flex items-center gap-3 mb-6">
              <span className="brutal-badge bg-[#EBF4FF]">ID: APP-9824</span>
              <span className="brutal-badge brutal-badge-warning flex items-center gap-1">
                <span>⚠️</span> Thin-File
              </span>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between border-b-[1.5px] border-[#1E1E1E] border-dashed pb-1">
                <span className="font-semibold">Oldest Trade Open</span>
                <span>24 Months</span>
              </div>
              <div className="flex justify-between border-b-[1.5px] border-[#1E1E1E] border-dashed pb-1">
                <span className="font-semibold">Total Trades</span>
                <span>5</span>
              </div>
              <div className="flex justify-between border-b-[1.5px] border-[#1E1E1E] border-dashed pb-1">
                <span className="font-semibold">Revolving Burden</span>
                <span>85.0%</span>
              </div>
            </div>
          </div>

          {/* Top Right: Decision & Risk Gauge */}
          <div className="brutal-card bg-[#FFDAB9] relative">
            <div className="absolute top-2 right-2 text-3xl opacity-20">⚖️</div>
            <h2 className="text-3xl font-display mb-4">Base Decision (T₀)</h2>
            <div className="flex flex-col items-center justify-center py-4">
              <div className="text-6xl mb-2 font-display text-[#1E1E1E]">Denied</div>
              <p className="font-bold text-lg">Approval Probability: 41%</p>
            </div>
            <div className="wavy-divider"></div>
            <div className="bg-[#FFFDF9] border-2 border-[#1E1E1E] rounded-xl p-4 mt-4">
              <h3 className="font-display text-xl mb-2 flex items-center gap-2"><span>🚨</span> Drift Vulnerability Risk</h3>
              <div className="flex justify-between items-center">
                <span className="font-bold text-lg">66.67%</span>
                <span className="brutal-badge brutal-badge-danger">High Risk</span>
              </div>
              <p className="text-sm mt-2 font-semibold text-gray-700">
                Recourse invalidation rate for this subgroup under macroeconomic shift (T₁).
              </p>
            </div>
          </div>

          {/* Bottom Left: SHAP Attributions */}
          <div className="brutal-card">
            <h2 className="text-3xl font-display mb-4">SHAP Barriers (T₀)</h2>
            <p className="mb-4 text-sm font-semibold">Key features contributing to the denial.</p>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm font-bold mb-1">
                  <span>NetFractionRevolvingBurden</span>
                  <span className="text-red-500">-1.24</span>
                </div>
                <div className="w-full h-4 border-2 border-[#1E1E1E] rounded-full bg-gray-100 overflow-hidden">
                  <div className="h-full bg-[#FCA5A5] w-3/4 border-r-2 border-[#1E1E1E]"></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm font-bold mb-1">
                  <span>NumSatisfactoryTrades</span>
                  <span className="text-red-500">-0.85</span>
                </div>
                <div className="w-full h-4 border-2 border-[#1E1E1E] rounded-full bg-gray-100 overflow-hidden">
                  <div className="h-full bg-[#FCA5A5] w-1/2 border-r-2 border-[#1E1E1E]"></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm font-bold mb-1">
                  <span>ExternalRiskEstimate</span>
                  <span className="text-red-500">-0.42</span>
                </div>
                <div className="w-full h-4 border-2 border-[#1E1E1E] rounded-full bg-gray-100 overflow-hidden">
                  <div className="h-full bg-[#FCA5A5] w-1/4 border-r-2 border-[#1E1E1E]"></div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Right: DiCE Recourse */}
          <div className="brutal-card bg-[#EBF4FF]">
            <div className="flex justify-between items-start mb-4">
              <h2 className="text-3xl font-display">DiCE Action Plan</h2>
              <span className="text-2xl">🌱</span>
            </div>
            <p className="font-semibold mb-4 border-b-2 border-[#1E1E1E] pb-2 border-dashed">
              Required adjustments to reach <strong>Approved (f₀(x*) = 1)</strong>:
            </p>
            
            <div className="space-y-4">
              <div className="bg-[#FFFDF9] border-2 border-[#1E1E1E] rounded-xl p-3 flex flex-col gap-1 shadow-[2px_2px_0px_#1E1E1E]">
                <div className="flex items-center gap-2">
                  <span className="brutal-badge bg-[#60A5FA]">Actionable</span>
                  <span className="font-bold">Revolving Burden</span>
                </div>
                <div className="flex items-center gap-2 text-lg">
                  <span className="line-through text-gray-500">85.0</span>
                  <span>→</span>
                  <span className="font-black text-[#1E1E1E]">45.2</span>
                </div>
              </div>

              <div className="bg-[#FFFDF9] border-2 border-[#1E1E1E] rounded-xl p-3 flex flex-col gap-1 shadow-[2px_2px_0px_#1E1E1E]">
                <div className="flex items-center gap-2">
                  <span className="brutal-badge bg-[#60A5FA]">Actionable</span>
                  <span className="font-bold">External Risk Estimate</span>
                </div>
                <div className="flex items-center gap-2 text-lg">
                  <span className="line-through text-gray-500">62</span>
                  <span>→</span>
                  <span className="font-black text-[#1E1E1E]">68</span>
                </div>
              </div>

              <div className="bg-gray-200 border-2 border-[#1E1E1E] rounded-xl p-3 opacity-80 border-dashed">
                <span className="text-sm font-bold">🔒 Immutable Context</span>
                <p className="text-sm mt-1">MSinceOldestTradeOpen and NumTotalTrades locked by DiCE constraints.</p>
              </div>
            </div>
          </div>
          
        </div>
      ) : (
        <div className="h-64 flex flex-col items-center justify-center border-4 border-dashed border-[#1E1E1E] rounded-3xl opacity-60">
          <div className="text-6xl mb-4">☕</div>
          <p className="text-xl font-display">Ready to evaluate a loan application...</p>
        </div>
      )}
    </main>
  );
}
