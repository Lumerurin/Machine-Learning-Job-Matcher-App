import React, { useState, useEffect } from 'react';
import { Briefcase, User, TrendingUp, AlertCircle, CheckCircle, Award, Sparkles, Search, X } from 'lucide-react';

const JobMatcher = () => {
  const [options, setOptions] = useState({
    skills: [],
    qualifications: [],
    experience_levels: [],
    job_roles: []
  });
  
  const [selectedSkills, setSelectedSkills] = useState([]);
  const [selectedQualification, setSelectedQualification] = useState('');
  const [selectedExperience, setSelectedExperience] = useState('');
  const [selectedJobRole, setSelectedJobRole] = useState('');
  
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const [skillSearchQuery, setSkillSearchQuery] = useState('');

  const API_URL = 'http://localhost:5000/api';

  useEffect(() => {
    fetchOptions();
  }, []);

  const fetchOptions = async () => {
    try {
      const response = await fetch(`${API_URL}/options`);
      if (!response.ok) throw new Error('Failed to fetch options');
      const data = await response.json();
      setOptions(data);
      setOptionsLoading(false);
    } catch (err) {
      setError('Failed to connect to API. Make sure Flask server is running on port 5000.');
      setOptionsLoading(false);
    }
  };

  const toggleSkill = (skill) => {
    setSelectedSkills(prev => 
      prev.includes(skill) 
        ? prev.filter(s => s !== skill)
        : [...prev, skill]
    );
  };
  
  const filteredSkills = options.skills.filter(skill =>
    skill.toLowerCase().includes(skillSearchQuery.toLowerCase())
  );

  const predictMatch = async () => {
    if (!selectedSkills.length || !selectedQualification || !selectedExperience || !selectedJobRole) {
      setError('Please fill in all fields');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          skills: selectedSkills,
          qualification: selectedQualification,
          experience_level: selectedExperience,
          job_role: selectedJobRole
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const getScorePercentage = (score, maxScore = 5) => {
    return (score / maxScore) * 100;
  };

  const getScoreColor = (percentage) => {
    if (percentage >= 75) return 'text-green-600';
    if (percentage >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreLabel = (percentage) => {
    if (percentage >= 80) return 'Excellent Match';
    if (percentage >= 60) return 'Good Match';
    if (percentage >= 40) return 'Fair Match';
    return 'Poor Match';
  };

  if (optionsLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading application...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-8">
            <div className="flex items-center gap-3 mb-2">
              <Briefcase size={32} />
              <h1 className="text-3xl font-bold">AI Job Match Predictor</h1>
            </div>
            <p className="text-blue-100">Advanced candidate-job compatibility analysis powered by deep learning</p>
          </div>

          {/* Content */}
          <div className="p-8">
            {error && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
                <AlertCircle className="text-red-500 flex-shrink-0 mt-0.5" size={20} />
                <div className="text-red-700 text-sm">{error}</div>
              </div>
            )}

            <div className="grid md:grid-cols-2 gap-8">
              {/* Left Column - Inputs */}
              <div className="space-y-6">
                {/* Skills Selection */}
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <label className="flex items-center gap-2 text-gray-700 font-semibold">
                      <Sparkles size={20} className="text-blue-600" />
                      Skills (Select Multiple)
                    </label>
                    {selectedSkills.length > 0 && (
                      <button
                        onClick={() => setSelectedSkills([])}
                        className="text-sm text-red-600 hover:text-red-700 font-medium flex items-center gap-1"
                      >
                        <X size={16} />
                        Clear All
                      </button>
                    )}
                  </div>
                  
                  {/* Search Bar */}
                  <div className="relative mb-3">
                    <Search size={18} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search skills..."
                      value={skillSearchQuery}
                      onChange={(e) => setSkillSearchQuery(e.target.value)}
                      className="w-full pl-10 pr-10 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    {skillSearchQuery && (
                      <button
                        onClick={() => setSkillSearchQuery('')}
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                      >
                        <X size={18} />
                      </button>
                    )}
                  </div>
                  
                  {/* Skills List */}
                  <div className="border border-gray-300 rounded-lg p-4 max-h-64 overflow-y-auto bg-gray-50">
                    {filteredSkills.length > 0 ? (
                      <div className="space-y-2">
                        {filteredSkills.map(skill => (
                          <label key={skill} className="flex items-center gap-2 cursor-pointer hover:bg-blue-50 p-2 rounded">
                            <input
                              type="checkbox"
                              checked={selectedSkills.includes(skill)}
                              onChange={() => toggleSkill(skill)}
                              className="w-4 h-4 text-blue-600 rounded"
                            />
                            <span className="text-sm text-gray-700">{skill}</span>
                          </label>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center text-gray-500 py-4">
                        No skills found matching "{skillSearchQuery}"
                      </div>
                    )}
                  </div>
                  
                  <div className="mt-2 text-sm text-gray-600">
                    Selected: {selectedSkills.length} skill(s)
                  </div>
                </div>

                {/* Qualification */}
                <div>
                  <label className="flex items-center gap-2 text-gray-700 font-semibold mb-3">
                    <Award size={20} className="text-blue-600" />
                    Qualification
                  </label>
                  <select
                    value={selectedQualification}
                    onChange={(e) => setSelectedQualification(e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">Select qualification...</option>
                    {options.qualifications.map(qual => (
                      <option key={qual} value={qual}>{qual}</option>
                    ))}
                  </select>
                </div>

                {/* Experience Level */}
                <div>
                  <label className="flex items-center gap-2 text-gray-700 font-semibold mb-3">
                    <TrendingUp size={20} className="text-blue-600" />
                    Experience Level
                  </label>
                  <select
                    value={selectedExperience}
                    onChange={(e) => setSelectedExperience(e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">Select experience level...</option>
                    {options.experience_levels.map(exp => (
                      <option key={exp} value={exp}>{exp}</option>
                    ))}
                  </select>
                </div>

                {/* Target Job Role */}
                <div>
                  <label className="flex items-center gap-2 text-gray-700 font-semibold mb-3">
                    <User size={20} className="text-blue-600" />
                    Target Job Role
                  </label>
                  <select
                    value={selectedJobRole}
                    onChange={(e) => setSelectedJobRole(e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">Select job role...</option>
                    {options.job_roles.map(role => (
                      <option key={role} value={role}>{role}</option>
                    ))}
                  </select>
                </div>

                {/* Predict Button */}
                <button
                  onClick={predictMatch}
                  disabled={isLoading}
                  className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-4 rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                  {isLoading ? 'Analyzing...' : 'Calculate Match Score'}
                </button>
              </div>

              {/* Right Column - Results */}
              <div>
                {result ? (
                  <div className="space-y-6">
                    {/* Main Score */}
                    <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl border-2 border-gray-200 p-6">
                      <h2 className="text-lg font-semibold text-gray-700 mb-4 text-center">
                        Match Score for {result.job_role}
                      </h2>
                      <div className="text-center">
                        <div className={`text-6xl font-bold mb-2 ${getScoreColor(getScorePercentage(result.match_score))}`}>
                          {result.match_score.toFixed(2)}/5.0
                        </div>
                        <div className={`text-xl font-semibold mb-4 ${getScoreColor(getScorePercentage(result.match_score))}`}>
                          {getScoreLabel(getScorePercentage(result.match_score))}
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3 mb-6">
                          <div
                            className={`h-3 rounded-full transition-all ${
                              getScorePercentage(result.match_score) >= 75 ? 'bg-green-500' :
                              getScorePercentage(result.match_score) >= 50 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${getScorePercentage(result.match_score)}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>

                    {/* Detailed Breakdown */}
                    <div className="bg-white border-2 border-gray-200 rounded-xl p-6">
                      <h3 className="font-semibold text-gray-700 mb-4">Score Breakdown</h3>
                      <div className="space-y-4">
                        <div>
                          <div className="flex justify-between text-sm mb-2">
                            <span className="text-gray-600">Skills & Qualification Similarity</span>
                            <span className="font-semibold text-blue-600">
                              {(result.skill_qual_similarity * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-500 h-2 rounded-full"
                              style={{ width: `${result.skill_qual_similarity * 100}%` }}
                            ></div>
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between text-sm mb-2">
                            <span className="text-gray-600">Experience Match</span>
                            <span className="font-semibold text-blue-600">
                              {(result.experience_match_score * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-500 h-2 rounded-full"
                              style={{ width: `${result.experience_match_score * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Selected Profile Summary */}
                    <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
                      <h3 className="font-semibold text-gray-700 mb-3">Your Profile</h3>
                      <div className="space-y-2 text-sm">
                        <div>
                          <span className="text-gray-600">Skills:</span>
                          <span className="ml-2 text-gray-800">{selectedSkills.join(', ')}</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Qualification:</span>
                          <span className="ml-2 text-gray-800">{selectedQualification}</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Experience:</span>
                          <span className="ml-2 text-gray-800">{selectedExperience}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center border-2 border-dashed border-gray-300 rounded-xl p-8">
                    <div className="text-center text-gray-500">
                      <Briefcase size={48} className="mx-auto mb-4 opacity-50" />
                      <p>Fill in the candidate details and click</p>
                      <p className="font-semibold">Calculate Match Score</p>
                      <p className="mt-2">to see prediction results</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="bg-gray-50 px-8 py-4 text-center text-sm text-gray-500">
            Powered by TensorFlow Deep Learning Model • Flask API Backend
          </div>
        </div>
      </div>
    </div>
  );
};

export default JobMatcher;