// Explainable AI (XAI) for Model Interpretability
// Implements: Feature Importance, Model Explanation, Decision Visualization

class ExplainableAI {
    constructor() {
        this.explainers = new Map();
        this.featureImportance = new Map();
        this.decisionPaths = [];
        this.explanationHistory = [];
        
        // Initialize explainers
        this.initializeExplainers();
    }

    // Initialize different explanation methods
    initializeExplainers() {
        this.explainers.set('lime', new LIMEExplainer());
        this.explainers.set('shap', new SHAPExplainer());
        this.explainers.set('feature_importance', new FeatureImportanceExplainer());
        this.explainers.set('decision_tree', new DecisionTreeExplainer());
        this.explainers.set('attention', new AttentionExplainer());
    }

    // Main explanation interface
    explain(model, input, method = 'lime') {
        const explainer = this.explainers.get(method);
        if (!explainer) {
            throw new Error(`Unknown explanation method: ${method}`);
        }

        const explanation = explainer.explain(model, input);
        
        // Store explanation
        this.storeExplanation(model, input, explanation, method);
        
        return explanation;
    }

    // Generate comprehensive explanation
    generateComprehensiveExplanation(model, input, prediction) {
        const explanations = {};
        
        // Get explanations from different methods
        explanations.lime = this.explain(model, input, 'lime');
        explanations.shap = this.explain(model, input, 'shap');
        explanations.featureImportance = this.explain(model, input, 'feature_importance');
        explanations.decisionTree = this.explain(model, input, 'decision_tree');
        explanations.attention = this.explain(model, input, 'attention');
        
        // Combine explanations
        const comprehensive = {
            prediction,
            input,
            explanations,
            summary: this.generateExplanationSummary(explanations),
            confidence: this.calculateExplanationConfidence(explanations),
            visualization: this.generateVisualization(explanations)
        };
        
        return comprehensive;
    }

    // Generate explanation summary
    generateExplanationSummary(explanations) {
        const summary = {
            keyFeatures: this.identifyKeyFeatures(explanations),
            reasoning: this.generateReasoning(explanations),
            alternatives: this.generateAlternatives(explanations),
            limitations: this.identifyLimitations(explanations)
        };
        
        return summary;
    }

    // Identify key features across explanations
    identifyKeyFeatures(explanations) {
        const featureScores = new Map();
        
        // Collect feature importance from all explainers
        Object.values(explanations).forEach(explanation => {
            if (explanation.featureImportance) {
                explanation.featureImportance.forEach((score, feature) => {
                    const currentScore = featureScores.get(feature) || 0;
                    featureScores.set(feature, currentScore + score);
                });
            }
        });
        
        // Sort by importance
        const sortedFeatures = Array.from(featureScores.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10); // Top 10 features
        
        return sortedFeatures;
    }

    // Generate human-readable reasoning
    generateReasoning(explanations) {
        const reasons = [];
        
        // Extract reasoning patterns
        if (explanations.lime && explanations.lime.explanation) {
            reasons.push(`LIME analysis: ${explanations.lime.explanation}`);
        }
        
        if (explanations.shap && explanations.shap.explanation) {
            reasons.push(`SHAP analysis: ${explanations.shap.explanation}`);
        }
        
        if (explanations.featureImportance && explanations.featureImportance.topFeatures) {
            const topFeatures = explanations.featureImportance.topFeatures.slice(0, 3);
            const featureList = topFeatures.map(f => f.name).join(', ');
            reasons.push(`Most important features: ${featureList}`);
        }
        
        return reasons;
    }

    // Generate alternative explanations
    generateAlternatives(explanations) {
        const alternatives = [];
        
        // Suggest alternative approaches based on explanations
        if (explanations.decisionTree && explanations.decisionTree.paths) {
            alternatives.push('Alternative paths in decision tree available');
        }
        
        if (explanations.attention && explanations.attention.highAttention) {
            alternatives.push('Consider focusing on high-attention features');
        }
        
        return alternatives;
    }

    // Identify explanation limitations
    identifyLimitations(explanations) {
        const limitations = [];
        
        // Check for confidence issues
        const confidences = Object.values(explanations).map(exp => exp.confidence || 0);
        const avgConfidence = confidences.reduce((sum, conf) => sum + conf, 0) / confidences.length;
        
        if (avgConfidence < 0.7) {
            limitations.push('Low confidence in explanations');
        }
        
        // Check for feature importance consistency
        const featureConsistency = this.checkFeatureConsistency(explanations);
        if (!featureConsistency) {
            limitations.push('Inconsistent feature importance across methods');
        }
        
        return limitations;
    }

    // Calculate explanation confidence
    calculateExplanationConfidence(explanations) {
        const confidences = Object.values(explanations).map(exp => exp.confidence || 0);
        return confidences.reduce((sum, conf) => sum + conf, 0) / confidences.length;
    }

    // Generate visualization data
    generateVisualization(explanations) {
        return {
            featureImportance: this.createFeatureImportanceChart(explanations),
            decisionPaths: this.createDecisionPathDiagram(explanations),
            attentionWeights: this.createAttentionVisualization(explanations),
            contributionChart: this.createContributionChart(explanations)
        };
    }

    // Create feature importance chart data
    createFeatureImportanceChart(explanations) {
        const features = new Map();
        
        // Aggregate feature importance
        Object.values(explanations).forEach(explanation => {
            if (explanation.featureImportance) {
                explanation.featureImportance.forEach((score, feature) => {
                    const currentScore = features.get(feature) || 0;
                    features.set(feature, currentScore + score);
                });
            }
        });
        
        return {
            type: 'bar_chart',
            data: Array.from(features.entries()).map(([feature, score]) => ({
                feature,
                importance: score / Object.values(explanations).length
            })),
            title: 'Feature Importance Across All Methods'
        };
    }

    // Create decision path diagram
    createDecisionPathDiagram(explanations) {
        if (!explanations.decisionTree || !explanations.decisionTree.paths) {
            return null;
        }
        
        return {
            type: 'flowchart',
            data: explanations.decisionTree.paths,
            title: 'Decision Paths'
        };
    }

    // Create attention visualization
    createAttentionVisualization(explanations) {
        if (!explanations.attention || !explanations.attention.weights) {
            return null;
        }
        
        return {
            type: 'heatmap',
            data: explanations.attention.weights,
            title: 'Attention Weights'
        };
    }

    // Create contribution chart
    createContributionChart(explanations) {
        const contributions = new Map();
        
        Object.values(explanations).forEach(explanation => {
            if (explanation.contributions) {
                explanation.contributions.forEach((contrib, feature) => {
                    const currentContrib = contributions.get(feature) || 0;
                    contributions.set(feature, currentContrib + contrib);
                });
            }
        });
        
        return {
            type: 'pie_chart',
            data: Array.from(contributions.entries()).map(([feature, contrib]) => ({
                feature,
                contribution: contrib
            })),
            title: 'Feature Contributions'
        };
    }

    // Check feature consistency across methods
    checkFeatureConsistency(explanations) {
        const featureRanks = new Map();
        
        // Collect feature rankings from each method
        Object.entries(explanations).forEach(([method, explanation]) => {
            if (explanation.featureImportance) {
                explanation.featureImportance.forEach((score, feature) => {
                    if (!featureRanks.has(feature)) {
                        featureRanks.set(feature, []);
                    }
                    featureRanks.get(feature).push({ method, score });
                });
            }
        });
        
        // Check consistency
        let consistent = true;
        featureRanks.forEach((ranks, feature) => {
            const scores = ranks.map(r => r.score);
            const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
            const variance = scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
            
            if (variance > 0.1) { // High variance indicates inconsistency
                consistent = false;
            }
        });
        
        return consistent;
    }

    // Store explanation for later analysis
    storeExplanation(model, input, explanation, method) {
        const record = {
            model: model.name || 'unknown',
            input,
            explanation,
            method,
            timestamp: Date.now()
        };
        
        this.explanationHistory.push(record);
        
        // Keep history manageable
        if (this.explanationHistory.length > 1000) {
            this.explanationHistory = this.explanationHistory.slice(-500);
        }
        
        // Save to localStorage
        localStorage.setItem(`xai_explanation_${Date.now()}`, JSON.stringify(record));
    }

    // Get explanation analytics
    getAnalytics() {
        const analytics = {
            totalExplanations: this.explanationHistory.length,
            methodUsage: this.getMethodUsage(),
            averageConfidence: this.getAverageConfidence(),
            commonFeatures: this.getCommonFeatures(),
            explanationTrends: this.getExplanationTrends()
        };
        
        return analytics;
    }

    // Get method usage statistics
    getMethodUsage() {
        const usage = new Map();
        
        this.explanationHistory.forEach(record => {
            const method = record.method;
            usage.set(method, (usage.get(method) || 0) + 1);
        });
        
        return Object.fromEntries(usage);
    }

    // Get average confidence across explanations
    getAverageConfidence() {
        const confidences = this.explanationHistory
            .filter(record => record.explanation.confidence)
            .map(record => record.explanation.confidence);
        
        if (confidences.length === 0) return 0;
        
        return confidences.reduce((sum, conf) => sum + conf, 0) / confidences.length;
    }

    // Get most commonly explained features
    getCommonFeatures() {
        const featureCounts = new Map();
        
        this.explanationHistory.forEach(record => {
            if (record.explanation.featureImportance) {
                record.explanation.featureImportance.forEach((_, feature) => {
                    featureCounts.set(feature, (featureCounts.get(feature) || 0) + 1);
                });
            }
        });
        
        return Array.from(featureCounts.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10)
            .map(([feature, count]) => ({ feature, count }));
    }

    // Get explanation trends over time
    getExplanationTrends() {
        const trends = {
            confidenceTrend: this.calculateConfidenceTrend(),
            featureTrend: this.calculateFeatureTrend(),
            methodTrend: this.calculateMethodTrend()
        };
        
        return trends;
    }

    // Calculate confidence trend
    calculateConfidenceTrend() {
        const recent = this.explanationHistory.slice(-50);
        const confidences = recent
            .filter(record => record.explanation.confidence)
            .map(record => record.explanation.confidence);
        
        if (confidences.length < 10) return 'insufficient_data';
        
        const firstHalf = confidences.slice(0, Math.floor(confidences.length / 2));
        const secondHalf = confidences.slice(Math.floor(confidences.length / 2));
        
        const firstAvg = firstHalf.reduce((sum, conf) => sum + conf, 0) / firstHalf.length;
        const secondAvg = secondHalf.reduce((sum, conf) => sum + conf, 0) / secondHalf.length;
        
        return secondAvg > firstAvg ? 'improving' : 'declining';
    }

    // Calculate feature trend
    calculateFeatureTrend() {
        const recent = this.explanationHistory.slice(-50);
        const features = new Map();
        
        recent.forEach(record => {
            if (record.explanation.featureImportance) {
                record.explanation.featureImportance.forEach((_, feature) => {
                    features.set(feature, (features.get(feature) || 0) + 1);
                });
            }
        });
        
        const sortedFeatures = Array.from(features.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);
        
        return sortedFeatures.map(([feature, count]) => ({ feature, count }));
    }

    // Calculate method trend
    calculateMethodTrend() {
        const recent = this.explanationHistory.slice(-50);
        const methodCounts = new Map();
        
        recent.forEach(record => {
            const method = record.method;
            methodCounts.set(method, (methodCounts.get(method) || 0) + 1);
        });
        
        return Object.fromEntries(methodCounts);
    }

    // Export explanation data
    exportExplanations(format = 'json') {
        const exportData = {
            explanations: this.explanationHistory,
            analytics: this.getAnalytics(),
            timestamp: Date.now()
        };
        
        switch (format) {
            case 'json':
                return JSON.stringify(exportData, null, 2);
            case 'csv':
                return this.convertToCSV(exportData);
            default:
                return exportData;
        }
    }

    // Convert to CSV format
    convertToCSV(data) {
        const headers = ['timestamp', 'method', 'confidence', 'features', 'explanation'];
        const rows = data.explanations.map(record => [
            record.timestamp,
            record.method,
            record.explanation.confidence || '',
            JSON.stringify(record.explanation.featureImportance || {}),
            JSON.stringify(record.explanation)
        ]);
        
        return [headers, ...rows].map(row => row.join(',')).join('\n');
    }
}

// LIME Explainer
class LIMEExplainer {
    explain(model, input) {
        // Simplified LIME implementation
        const perturbations = this.generatePerturbations(input);
        const predictions = perturbations.map(p => model.predict(p));
        
        const explanation = {
            method: 'LIME',
            featureImportance: this.calculateFeatureImportance(perturbations, predictions),
            explanation: this.generateLIMEExplanation(perturbations, predictions),
            confidence: this.calculateLIMEConfidence(predictions),
            perturbations: perturbations
        };
        
        return explanation;
    }

    generatePerturbations(input) {
        const perturbations = [input]; // Include original
        
        // Generate perturbed versions
        for (let i = 0; i < 100; i++) {
            const perturbation = this.perturbInput(input);
            perturbations.push(perturbation);
        }
        
        return perturbations;
    }

    perturbInput(input) {
        // Simple perturbation: add noise to features
        const perturbed = { ...input };
        
        Object.keys(perturbed).forEach(key => {
            if (typeof perturbed[key] === 'number') {
                perturbed[key] += (Math.random() - 0.5) * 0.1;
            }
        });
        
        return perturbed;
    }

    calculateFeatureImportance(perturbations, predictions) {
        const importance = new Map();
        
        // Calculate feature importance based on prediction changes
        perturbations.forEach((perturbation, index) => {
            const prediction = predictions[index];
            const originalPrediction = predictions[0]; // Original input prediction
            
            Object.keys(perturbation).forEach(feature => {
                if (perturbation[feature] !== perturbations[0][feature]) {
                    const currentImportance = importance.get(feature) || 0;
                    const impact = Math.abs(prediction - originalPrediction);
                    importance.set(feature, currentImportance + impact);
                }
            });
        });
        
        return Array.from(importance.entries());
    }

    generateLIMEExplanation(perturbations, predictions) {
        const topFeatures = this.calculateFeatureImportance(perturbations, predictions)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3);
        
        return `Model prediction is primarily influenced by: ${topFeatures.map(f => f[0]).join(', ')}`;
    }

    calculateLIMEConfidence(predictions) {
        const variance = predictions.reduce((sum, pred) => {
            const mean = predictions.reduce((s, p) => s + p, 0) / predictions.length;
            return sum + Math.pow(pred - mean, 2);
        }, 0) / predictions.length;
        
        return Math.max(0, 1 - variance);
    }
}

// SHAP Explainer
class SHAPExplainer {
    explain(model, input) {
        // Simplified SHAP implementation
        const baseline = this.calculateBaseline(model);
        const contributions = this.calculateSHAPContributions(model, input, baseline);
        
        const explanation = {
            method: 'SHAP',
            contributions,
            explanation: this.generateSHAPExplanation(contributions),
            confidence: this.calculateSHAPConfidence(contributions),
            baseline
        };
        
        return explanation;
    }

    calculateBaseline(model) {
        // Create baseline input (all zeros or means)
        const baseline = {};
        
        // For simplicity, use zeros as baseline
        return baseline;
    }

    calculateSHAPContributions(model, input, baseline) {
        const contributions = new Map();
        
        // Calculate contribution for each feature
        Object.keys(input).forEach(feature => {
            const inputWithFeature = { ...baseline, [feature]: input[feature] };
            const predictionWithFeature = model.predict(inputWithFeature);
            const baselinePrediction = model.predict(baseline);
            
            const contribution = predictionWithFeature - baselinePrediction;
            contributions.set(feature, contribution);
        });
        
        return Array.from(contributions.entries());
    }

    generateSHAPExplanation(contributions) {
        const positiveContributions = contributions.filter(([_, contrib]) => contrib > 0);
        const negativeContributions = contributions.filter(([_, contrib]) => contrib < 0);
        
        let explanation = 'SHAP analysis shows: ';
        
        if (positiveContributions.length > 0) {
            const positiveFeatures = positiveContributions.map(([feature, _]) => feature).join(', ');
            explanation += `Positive contributions from: ${positiveFeatures}. `;
        }
        
        if (negativeContributions.length > 0) {
            const negativeFeatures = negativeContributions.map(([feature, _]) => feature).join(', ');
            explanation += `Negative contributions from: ${negativeFeatures}.`;
        }
        
        return explanation;
    }

    calculateSHAPConfidence(contributions) {
        const totalContribution = contributions.reduce((sum, [_, contrib]) => sum + Math.abs(contrib), 0);
        const avgContribution = totalContribution / contributions.length;
        
        return Math.min(1, avgContribution / 10); // Normalize to 0-1
    }
}

// Feature Importance Explainer
class FeatureImportanceExplainer {
    explain(model, input) {
        const importance = this.calculateGlobalFeatureImportance(model);
        const localImportance = this.calculateLocalFeatureImportance(model, input);
        
        const explanation = {
            method: 'Feature Importance',
            featureImportance: importance,
            localImportance,
            explanation: this.generateFeatureExplanation(importance),
            confidence: this.calculateFeatureConfidence(importance)
        };
        
        return explanation;
    }

    calculateGlobalFeatureImportance(model) {
        // Simplified: use random feature importance
        const features = ['accuracy', 'time_spent', 'interactions', 'difficulty', 'engagement'];
        const importance = features.map(feature => [
            feature,
            Math.random() // In real implementation, would calculate actual importance
        ]);
        
        return importance;
    }

    calculateLocalFeatureImportance(model, input) {
        // Calculate importance for this specific input
        const features = Object.keys(input);
        const importance = features.map(feature => [
            feature,
            Math.abs(input[feature]) * Math.random() // Simplified calculation
        ]);
        
        return importance;
    }

    generateFeatureExplanation(importance) {
        const topFeatures = importance
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);
        
        return `Most important features: ${topFeatures.map(f => f[0]).join(', ')}`;
    }

    calculateFeatureConfidence(importance) {
        const totalImportance = importance.reduce((sum, [_, imp]) => sum + imp, 0);
        const avgImportance = totalImportance / importance.length;
        
        return Math.min(1, avgImportance);
    }
}

// Decision Tree Explainer
class DecisionTreeExplainer {
    explain(model, input) {
        const paths = this.extractDecisionPaths(model, input);
        
        const explanation = {
            method: 'Decision Tree',
            paths,
            explanation: this.generateDecisionTreeExplanation(paths),
            confidence: this.calculateDecisionTreeConfidence(paths)
        };
        
        return explanation;
    }

    extractDecisionPaths(model, input) {
        // Simplified decision path extraction
        const paths = [
            {
                path: ['if accuracy > 0.8', 'if time_spent < 300'],
                outcome: 'Advanced content',
                confidence: 0.8
            },
            {
                path: ['if accuracy < 0.5', 'if interactions > 20'],
                outcome: 'Remedial content',
                confidence: 0.7
            }
        ];
        
        return paths;
    }

    generateDecisionTreeExplanation(paths) {
        const pathDescriptions = paths.map(p => 
            `Path: ${p.path.join(' → ')} → ${p.outcome} (confidence: ${p.confidence})`
        );
        
        return `Decision paths: ${pathDescriptions.join('; ')}`;
    }

    calculateDecisionTreeConfidence(paths) {
        const confidences = paths.map(p => p.confidence);
        return confidences.reduce((sum, conf) => sum + conf, 0) / confidences.length;
    }
}

// Attention Explainer
class AttentionExplainer {
    explain(model, input) {
        const attentionWeights = this.calculateAttentionWeights(model, input);
        
        const explanation = {
            method: 'Attention',
            weights: attentionWeights,
            explanation: this.generateAttentionExplanation(attentionWeights),
            confidence: this.calculateAttentionConfidence(attentionWeights)
        };
        
        return explanation;
    }

    calculateAttentionWeights(model, input) {
        // Simplified attention weight calculation
        const features = Object.keys(input);
        const weights = features.map(feature => [
            feature,
            Math.random() // In real implementation, would calculate actual attention
        ]);
        
        return weights;
    }

    generateAttentionExplanation(weights) {
        const highAttention = weights
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(w => w[0]);
        
        return `High attention on: ${highAttention.join(', ')}`;
    }

    calculateAttentionConfidence(weights) {
        const totalAttention = weights.reduce((sum, [_, weight]) => sum + weight, 0);
        const avgAttention = totalAttention / weights.length;
        
        return Math.min(1, avgAttention);
    }
}

// Export for use in main application
window.ExplainableAI = ExplainableAI;
window.LIMEExplainer = LIMEExplainer;
window.SHAPExplainer = SHAPExplainer;
window.FeatureImportanceExplainer = FeatureImportanceExplainer;
window.DecisionTreeExplainer = DecisionTreeExplainer;
window.AttentionExplainer = AttentionExplainer;
