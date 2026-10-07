'use client';

import React, { useState } from 'react';

export default function Dashboard() {
  const [applicantId, setApplicantId] = useState('APP-9824');
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [showSensitivityDrawer, setShowSensitivityDrawer] = useState(false);

  const handleEvaluate = async () => {
    setIsEvaluating(true);
    // Simulating API Call to Spring Boot Gateway -> FastAPI
    setTimeout(() => {
      setIsEvaluating(false);
    }, 2000);
  };

  return (
    <div className="min-h-screen p-8 max-w-7xl mx-auto font-sans">
      
      {/* Header */}
      <header className="mb-8 flex justify-between items-end border-b-4 border-[#1E1E1E] pb-4">
        <div>
          <h1 className="text-4xl font-black tracking-tight text-[#1E1E1E] uppercase font-display">Credit Model Audit</h1>
          <p className="text-lg font-bold text-gray-600">Temporal Fragility & Algorithmic Recourse</p>
        </div>
        <div className="flex gap-4">
          <button onClick={() => setShowSensitivityDrawer(true)} className="brutal-btn bg-white">
            Sensitivity Analysis
          </button>
          <button onClick={handleEvaluate} className="brutal-btn brutal-btn-primary">
            {isEvaluating ? 'Evaluating...' : 'Run Audit'}
          </button>
        </div>
      </header>

      {/* 6-Panel Bento Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Panel 1: Applicant Profile */}
        <div className="brutal-card col-span-1 bg-[#EBF4FF]">
          <h2 className="text-2xl font-black border-b-2 border-[#1E1E1E] pb-2 mb-4">1. Profile</h2>
          <div className="flex justify-between items-center mb-4">
            <span className="font-bold text-lg">{applicantId}</span>
            <span className="brutal-badge brutal-badge-warning">Thin-File</span>
          </div>
          <div className="space-y-3 font-medium">
            <div className="flex justify-between"><span className="text-gray-600">Ext. Risk Estimate:</span> <span>62 (Low)</span></div>
            <div className="flex justify-between"><span className="text-gray-600">Months in File:</span> <span>14 (Locked)</span></div>
            <div className="flex justify-between"><span className="text-gray-600">Revolving Burden:</span> <span>85%</span></div>
          </div>
        </div>

        {/* Panel 2: Decision & Reliability */}
        <div className="brutal-card col-span-1 md:col-span-2 bg-[#FFFDF9]">
          <h2 className="text-2xl font-black border-b-2 border-[#1E1E1E] pb-2 mb-4">2. Decision & Reliability</h2>
          <div className="flex items-center gap-8 h-full pb-4">
            <div className="text-center">
              <div className="text-5xl font-black text-red-500 mb-2">DENIED</div>
              <div className="brutal-badge brutal-badge-danger">f0(x) = 0</div>
            </div>
            <div className="flex-1 bg-red-50 border-2 border-red-200 p-4 rounded-xl">
              <h3 className="font-black text-red-800 mb-2">Recourse Invalidation Risk</h3>
              <div className="w-full bg-gray-200 h-6 rounded-full border-2 border-[#1E1E1E] overflow-hidden">
                <div className="bg-red-500 h-full" style={{ width: '66.7%' }}></div>
              </div>
              <p className="text-sm font-bold text-red-600 mt-2">66.7% Thin-File Alert: High probability of advice failure under drift.</p>
            </div>
          </div>
        </div>

        {/* Panel 3: SHAP Breakdown */}
        <div className="brutal-card col-span-1 md:col-span-2 bg-[#FFFDF9]">
          <h2 className="text-2xl font-black border-b-2 border-[#1E1E1E] pb-2 mb-4">3. SHAP Feature Attribution</h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm font-bold mb-1"><span>NetFractionRevolvingBurden</span><span className="text-red-500">-1.24</span></div>
              <div className="w-full bg-gray-100 h-4 border-2 border-[#1E1E1E]"><div className="bg-red-400 h-full" style={{ width: '75%' }}></div></div>
            </div>
            <div>
              <div className="flex justify-between text-sm font-bold mb-1"><span>ExternalRiskEstimate</span><span className="text-red-500">-0.82</span></div>
              <div className="w-full bg-gray-100 h-4 border-2 border-[#1E1E1E]"><div className="bg-red-400 h-full" style={{ width: '50%' }}></div></div>
            </div>
            <div>
              <div className="flex justify-between text-sm font-bold mb-1"><span>AverageMInFile</span><span className="text-green-600">+0.30</span></div>
              <div className="w-full bg-gray-100 h-4 border-2 border-[#1E1E1E]"><div className="bg-green-400 h-full" style={{ width: '20%' }}></div></div>
            </div>
          </div>
        </div>

        {/* Panel 4: Action Plan */}
        <div className="brutal-card col-span-1 bg-[#FFFDF9]">
          <h2 className="text-2xl font-black border-b-2 border-[#1E1E1E] pb-2 mb-4">4. DiCE Action Plan</h2>
          <div className="space-y-4">
             <div className="p-3 border-2 border-[#1E1E1E] bg-yellow-50 rounded-lg">
                <div className="font-bold text-sm">Revolving Burden</div>
                <div className="flex justify-between mt-1 items-center">
                  <span className="line-through text-gray-500">85%</span>
                  <span className="font-black">→ 45%</span>
                </div>
             </div>
             <div className="p-3 border-2 border-[#1E1E1E] bg-yellow-50 rounded-lg">
                <div className="font-bold text-sm">Risk Estimate</div>
                <div className="flex justify-between mt-1 items-center">
                  <span className="line-through text-gray-500">62</span>
                  <span className="font-black">→ 68</span>
                </div>
             </div>
             <button onClick={() => setShowFeedbackModal(true)} className="w-full brutal-btn bg-[#FFDAB9] mt-4">
               Expert Review
             </button>
          </div>
        </div>
      </div>

      {/* Panel 5: Sensitivity Drawer (Hidden by default) */}
      {showSensitivityDrawer && (
        <div className="fixed inset-0 bg-black/50 flex justify-end z-50">
          <div className="bg-[#FAF7F0] w-full max-w-md h-full border-l-4 border-[#1E1E1E] p-8 shadow-2xl">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-black">5. Sensitivity Curve</h2>
              <button onClick={() => setShowSensitivityDrawer(false)} className="font-black text-xl hover:text-red-500">X</button>
            </div>
            <p className="font-bold mb-4">Drift vs. Invalidation Rate</p>
            <div className="h-64 border-2 border-[#1E1E1E] bg-white flex items-end justify-between p-4 gap-2">
               {/* Mock Bar Chart */}
               {[5, 10, 15, 20, 25, 30, 35].map((val, i) => (
                 <div key={val} className="w-full bg-[#60A5FA] border-2 border-[#1E1E1E] relative group" style={{ height: `${20 + (i * 10)}%` }}>
                    <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs font-bold opacity-0 group-hover:opacity-100">{val}%</span>
                 </div>
               ))}
            </div>
            <div className="flex justify-between text-xs font-bold mt-2">
              <span>5% Drift</span>
              <span>35% Drift</span>
            </div>
          </div>
        </div>
      )}

      {/* Panel 6: Expert Feedback Modal */}
      {showFeedbackModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-[#FFFDF9] border-4 border-[#1E1E1E] p-8 max-w-lg w-full rounded-2xl shadow-[8px_8px_0px_#1E1E1E]">
            <h2 className="text-2xl font-black mb-4">6. Underwriter Feedback</h2>
            <div className="space-y-4 font-bold">
              <div>
                <label>Actionability Score (1-5)</label>
                <input type="range" min="1" max="5" className="w-full mt-2 accent-[#1E1E1E]" />
              </div>
              <div>
                <label>Verdict</label>
                <select className="w-full border-2 border-[#1E1E1E] p-2 mt-2 rounded-lg bg-white">
                  <option>ACCEPT</option>
                  <option>MODIFY</option>
                  <option>REJECT</option>
                </select>
              </div>
              <div>
                <label>Notes</label>
                <textarea className="w-full border-2 border-[#1E1E1E] p-2 mt-2 rounded-lg bg-white" rows={3}></textarea>
              </div>
              <div className="flex gap-4 pt-4">
                <button onClick={() => setShowFeedbackModal(false)} className="brutal-btn flex-1 bg-white">Cancel</button>
                <button onClick={() => setShowFeedbackModal(false)} className="brutal-btn brutal-btn-primary flex-1">Submit</button>
              </div>
            </div>
          </div>
        </div>
      )}
      
    </div>
  );
}
