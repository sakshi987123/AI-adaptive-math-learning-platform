// Adaptive Learning System with Machine Learning
// Implements: Performance Analysis, Difficulty Adjustment, Personalized Recommendations

class AdaptiveLearningSystem {
    constructor() {
        this.userPerformance = new Map();
        this.difficultyLevels = new Map();
        this.learningPatterns = new Map();
        this.recommendationEngine = new RecommendationEngine();
        this.performanceAnalyzer = new PerformanceAnalyzer();
        
        // Initialize ML models
        this.initializeModels();
    }

    initializeModels() {
        // Performance prediction model
        this.performanceModel = {
            weights: {
                completionRate: 0.4,
                timeSpent: 0.3,
                accuracy: 0.2,
                engagement: 0.1
            },
            bias: 0.1,
            learningRate: 0.01
        };

        // Difficulty adjustment model
        this.difficultyModel = {
            currentDifficulty: 0.5, // 0-1 scale
                adjustmentFactor: 0.1,
                minDifficulty: 0.1,
                maxDifficulty: 1.0,
                adaptationSpeed: 0.05
        };

        // Learning pattern recognition
        this.patternModel = {
            sequences: [],
                    patternWeights: new Map(),
                    recognitionThreshold: 0.7
        };
    }

    // Track user performance
    trackPerformance(userId, lessonId, performanceData) {
        const key = `${userId}_${lessonId}`;
        
        if (!this.userPerformance.has(key)) {
            this.userPerformance.set(key, []);
        }

        const userHistory = this.userPerformance.get(key);
        userHistory.push({
            timestamp: Date.now(),
            ...performanceData,
            difficulty: this.difficultyModel.currentDifficulty
        });

        // Analyze and adapt
        this.analyzeAndAdapt(userId, lessonId, userHistory);
    }

    // Machine Learning Analysis
    analyzeAndAdapt(userId, lessonId, history) {
        const analysis = this.performanceAnalyzer.analyze(history);
        
        // Update difficulty based on performance
        this.adjustDifficulty(analysis);
        
        // Update learning patterns
        this.updateLearningPatterns(analysis);
        
        // Generate recommendations
        const recommendations = this.recommendationEngine.generate(analysis, this.patternModel);
        
        // Store updated models
        this.saveModelUpdates(userId, lessonId, analysis, recommendations);
        
        return {
            analysis,
            recommendations,
            adjustedDifficulty: this.difficultyModel.currentDifficulty
        };
    }

    // Difficulty Adjustment Algorithm
    adjustDifficulty(analysis) {
        const { performance, trend } = analysis;
        
        // Calculate optimal difficulty using weighted scoring
        const performanceScore = this.calculatePerformanceScore(performance);
        
        // Apply adaptive adjustment
        let targetDifficulty = this.difficultyModel.currentDifficulty;
        
        if (performanceScore > 0.8) {
            // User performing well, increase difficulty
            targetDifficulty = Math.min(
                this.difficultyModel.maxDifficulty,
                targetDifficulty + this.difficultyModel.adjustmentFactor
            );
        } else if (performanceScore < 0.4) {
            // User struggling, decrease difficulty
            targetDifficulty = Math.max(
                this.difficultyModel.minDifficulty,
                targetDifficulty - this.difficultyModel.adjustmentFactor
            );
        }

        // Smooth transition using learning rate
        this.difficultyModel.currentDifficulty = 
            this.difficultyModel.currentDifficulty * (1 - this.difficultyModel.learningRate) +
            targetDifficulty * this.difficultyModel.learningRate;
    }

    calculatePerformanceScore(performance) {
        const weights = this.performanceModel.weights;
        
        return (
            performance.completionRate * weights.completionRate +
            performance.timeEfficiency * weights.timeSpent +
            performance.accuracy * weights.accuracy +
            performance.engagement * weights.engagement
        ) / Object.values(weights).reduce((a, b) => a + b, 0);
    }

    // Learning Pattern Recognition
    updateLearningPatterns(analysis) {
        const { sequence, performance } = analysis;
        
        // Store learning sequence
        this.patternModel.sequences.push({
            sequence,
            performance,
            timestamp: Date.now()
        });

        // Keep only recent patterns (last 50)
        if (this.patternModel.sequences.length > 50) {
            this.patternModel.sequences = this.patternModel.sequences.slice(-50);
        }

        // Update pattern weights based on success
        this.updatePatternWeights(sequence, performance);
    }

    updatePatternWeights(sequence, performance) {
        const sequenceKey = sequence.join('-');
        const currentWeight = this.patternModel.patternWeights.get(sequenceKey) || 0.5;
        
        // Adjust weight based on performance
        const adjustment = (performance > 0.5) ? 0.1 : -0.1;
        const newWeight = Math.max(0.1, Math.min(0.9, currentWeight + adjustment));
        
        this.patternModel.patternWeights.set(sequenceKey, newWeight);
    }

    // Save model updates
    saveModelUpdates(userId, lessonId, analysis, recommendations) {
        const modelUpdate = {
            userId,
            lessonId,
            timestamp: Date.now(),
            difficultyModel: { ...this.difficultyModel },
            patternModel: { ...this.patternModel },
            analysis,
            recommendations
        };

        // Save to localStorage (in production, would save to database)
        localStorage.setItem(`ml_model_${userId}`, JSON.stringify(modelUpdate));
    }

    // Get personalized content recommendations
    getRecommendations(userId) {
        const userHistory = this.getUserHistory(userId);
        if (userHistory.length === 0) {
            return this.getDefaultRecommendations();
        }

        return this.recommendationEngine.generatePersonalized(userHistory, this.patternModel);
    }

    getUserHistory(userId) {
        const modelData = localStorage.getItem(`ml_model_${userId}`);
        if (modelData) {
            const parsed = JSON.parse(modelData);
            return parsed.analysis?.history || [];
        }
        return [];
    }

    getDefaultRecommendations() {
        return [
            { type: 'lesson', priority: 'high', reason: 'Start with basics' },
            { type: 'practice', priority: 'medium', reason: 'Reinforce concepts' },
            { type: 'quiz', priority: 'low', reason: 'Test understanding' }
        ];
    }
}

// Performance Analyzer Class
class PerformanceAnalyzer {
    analyze(history) {
        if (history.length < 2) {
            return { performance: 0.5, trend: 'insufficient_data' };
        }

        const recent = history.slice(-10); // Analyze last 10 sessions
        const performance = this.calculateMetrics(recent);
        const trend = this.detectTrend(recent);

        return {
            performance,
            trend,
            metrics: this.getDetailedMetrics(recent)
        };
    }

    calculateMetrics(sessions) {
        const completionRate = sessions.filter(s => s.completed).length / sessions.length;
        const avgTimeSpent = sessions.reduce((sum, s) => sum + (s.timeSpent || 0), 0) / sessions.length;
        const avgAccuracy = sessions.reduce((sum, s) => sum + (s.accuracy || 0), 0) / sessions.length;
        const engagement = this.calculateEngagement(sessions);

        return {
            completionRate,
            timeEfficiency: this.normalizeTime(avgTimeSpent),
            accuracy: avgAccuracy,
            engagement
        };
    }

    normalizeTime(avgTime) {
        // Normalize time efficiency (lower is better for same content)
        const expectedTime = 300; // 5 minutes per lesson
        return Math.max(0, Math.min(1, expectedTime / avgTime));
    }

    calculateEngagement(sessions) {
        // Calculate engagement based on interactions
        const interactions = sessions.reduce((sum, s) => sum + (s.interactions || 0), 0);
        const maxPossible = sessions.length * 100; // Assume 100 interactions max per session
        return interactions / maxPossible;
    }

    detectTrend(sessions) {
        if (sessions.length < 3) return 'insufficient_data';
        
        const recent = sessions.slice(-5);
        const performance = recent.map(s => this.getOverallScore(s));
        
        let increasing = 0, decreasing = 0;
        for (let i = 1; i < performance.length; i++) {
            if (performance[i] > performance[i-1]) increasing++;
            else if (performance[i] < performance[i-1]) decreasing++;
        }

        if (increasing > decreasing) return 'improving';
        if (decreasing > increasing) return 'declining';
        return 'stable';
    }

    getOverallScore(session) {
        return (
            (session.completionRate || 0) * 0.4 +
            (session.timeEfficiency || 0) * 0.3 +
            (session.accuracy || 0) * 0.2 +
            (session.engagement || 0) * 0.1
        );
    }

    getDetailedMetrics(sessions) {
        return {
            averageSessionLength: sessions.reduce((sum, s) => sum + (s.duration || 0), 0) / sessions.length,
            peakPerformance: Math.max(...sessions.map(s => this.getOverallScore(s))),
            consistency: this.calculateConsistency(sessions),
            learningVelocity: this.calculateLearningVelocity(sessions)
        };
    }

    calculateConsistency(sessions) {
        const scores = sessions.map(s => this.getOverallScore(s));
        const mean = scores.reduce((sum, s) => sum + s, 0) / scores.length;
        const variance = scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / scores.length;
        return 1 / (1 + variance); // Lower variance = higher consistency
    }

    calculateLearningVelocity(sessions) {
        if (sessions.length < 2) return 0;
        
        const first = sessions[0];
        const last = sessions[sessions.length - 1];
        const timeDiff = last.timestamp - first.timestamp;
        const scoreDiff = this.getOverallScore(last) - this.getOverallScore(first);
        
        return timeDiff > 0 ? scoreDiff / (timeDiff / (1000 * 60 * 60 * 24)) : 0; // Score change per day
    }
}

// Recommendation Engine
class RecommendationEngine {
    generate(analysis, patternModel) {
        const recommendations = [];
        const { performance, trend, metrics } = analysis;

        // Performance-based recommendations
        if (performance < 0.4) {
            recommendations.push({
                type: 'review',
                priority: 'high',
                reason: 'Low performance detected',
                action: 'Review previous lessons'
            });
        }

        if (performance > 0.8) {
            recommendations.push({
                type: 'advance',
                priority: 'medium',
                reason: 'High performance achieved',
                action: 'Move to next difficulty level'
            });
        }

        // Trend-based recommendations
        if (trend === 'declining') {
            recommendations.push({
                type: 'intervention',
                priority: 'high',
                reason: 'Performance declining',
                action: 'Consider additional practice'
            });
        }

        // Pattern-based recommendations
        const patternRecommendations = this.generatePatternBased(patternModel);
        recommendations.push(...patternRecommendations);

        return recommendations.sort((a, b) => this.getPriorityScore(b) - this.getPriorityScore(a));
    }

    generatePatternBased(patternModel) {
        const recommendations = [];
        const topPatterns = this.getTopPatterns(patternModel);

        topPatterns.forEach(pattern => {
            if (pattern.weight > 0.7) {
                recommendations.push({
                    type: 'pattern_optimization',
                    priority: 'medium',
                    reason: 'Strong learning pattern detected',
                    action: `Follow ${pattern.sequence.join(' → ')} sequence`
                });
            }
        });

        return recommendations;
    }

    getTopPatterns(patternModel) {
        return Array.from(patternModel.patternWeights.entries())
            .map(([sequence, weight]) => ({ sequence, weight }))
            .sort((a, b) => b.weight - a.weight)
            .slice(0, 5);
    }

    getPriorityScore(recommendation) {
        const priorityScores = { high: 3, medium: 2, low: 1 };
        return priorityScores[recommendation.priority] || 0;
    }

    generatePersonalized(history, patternModel) {
        // Analyze user's learning style from history
        const learningStyle = this.analyzeLearningStyle(history);
        const recommendations = [];

        // Style-based recommendations
        if (learningStyle.visual) {
            recommendations.push({
                type: 'content',
                priority: 'high',
                reason: 'Visual learner detected',
                action: 'Increase visual content'
            });
        }

        if (learningStyle.kinesthetic) {
            recommendations.push({
                type: 'interactive',
                priority: 'high',
                reason: 'Hands-on learner detected',
                action: 'Add interactive exercises'
            });
        }

        return recommendations;
    }

    analyzeLearningStyle(history) {
        // Simple learning style detection based on interaction patterns
        const visualInteractions = history.filter(h => h.visualContent).length;
        const interactiveElements = history.filter(h => h.interactiveElements).length;
        const totalSessions = history.length;

        return {
            visual: visualInteractions / totalSessions > 0.6,
            kinesthetic: interactiveElements / totalSessions > 0.5,
            auditory: false // Would need audio interaction data
        };
    }
}

// Export for use in main application
window.AdaptiveLearningSystem = AdaptiveLearningSystem;
window.PerformanceAnalyzer = PerformanceAnalyzer;
window.RecommendationEngine = RecommendationEngine;
