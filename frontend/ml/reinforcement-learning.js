// Reinforcement Learning for Personalized Learning Paths
// Implements: Q-Learning, Policy Optimization, Adaptive Learning Sequences

class ReinforcementLearningSystem {
    constructor() {
        this.qTable = new Map();
        this.policy = new Map();
        this.rewards = new Map();
        this.experienceBuffer = [];
        this.learningRate = 0.1;
        this.discountFactor = 0.95;
        this.epsilon = 0.3; // Exploration rate
        this.epsilonDecay = 0.995;
        
        // State and action spaces
        this.stateSpace = this.defineStateSpace();
        this.actionSpace = this.defineActionSpace();
        
        // Training metrics
        this.episodeCount = 0;
        this.totalReward = 0;
        this.averageReward = 0;
        this.convergenceHistory = [];
        
        // Initialize Q-table
        this.initializeQTable();
    }

    // Define state space
    defineStateSpace() {
        return {
            userLevel: ['beginner', 'intermediate', 'advanced'],
            performance: ['low', 'medium', 'high'],
            engagement: ['low', 'medium', 'high'],
            difficulty: ['easy', 'medium', 'hard'],
            topic: ['pythagoras', 'triangles', 'angles', 'formulas'],
            learningStyle: ['visual', 'auditory', 'kinesthetic']
        };
    }

    // Define action space
    defineActionSpace() {
        return [
            'show_next_lesson',
            'provide_practice',
            'offer_hint',
            'adjust_difficulty',
            'change_content_type',
            'provide_review',
            'introduce_challenge',
            'offer_break',
            'suggest_alternative',
            'provide_feedback'
        ];
    }

    // Initialize Q-table
    initializeQTable() {
        const states = this.generateAllStates();
        
        states.forEach(state => {
            const stateKey = this.stateToString(state);
            const qValues = {};
            
            this.actionSpace.forEach(action => {
                qValues[action] = Math.random() * 0.1; // Small random initialization
            });
            
            this.qTable.set(stateKey, qValues);
        });
        
        console.log(`Q-table initialized with ${states.length} states and ${this.actionSpace.length} actions`);
    }

    // Generate all possible states
    generateAllStates() {
        const states = [];
        const { userLevel, performance, engagement, difficulty, topic, learningStyle } = this.stateSpace;
        
        // Generate all combinations (simplified for demonstration)
        userLevel.forEach(level => {
            performance.forEach(perf => {
                engagement.forEach(eng => {
                    difficulty.forEach(diff => {
                        topic.forEach(top => {
                            learningStyle.forEach(style => {
                                states.push({
                                    userLevel: level,
                                    performance: perf,
                                    engagement: eng,
                                    difficulty: diff,
                                    topic: top,
                                    learningStyle: style
                                });
                            });
                        });
                    });
                });
            });
        });
        
        return states;
    }

    // Convert state to string key
    stateToString(state) {
        return `${state.userLevel}_${state.performance}_${state.engagement}_${state.difficulty}_${state.topic}_${state.learningStyle}`;
    }

    // Main RL training loop
    train(userInteractions, maxEpisodes = 1000) {
        console.log('Starting reinforcement learning training...');
        
        for (let episode = 0; episode < maxEpisodes; episode++) {
            const episodeReward = this.runEpisode(userInteractions);
            this.updateMetrics(episodeReward, episode);
            
            // Decay exploration rate
            this.epsilon *= this.epsilonDecay;
            
            // Log progress
            if (episode % 100 === 0) {
                console.log(`Episode ${episode}: Average Reward = ${this.averageReward.toFixed(4)}`);
            }
            
            // Check for convergence
            if (this.checkConvergence()) {
                console.log('Convergence detected, stopping training');
                break;
            }
        }
        
        console.log('Reinforcement learning training completed');
        return this.getTrainingResults();
    }

    // Run single episode
    runEpisode(userInteractions) {
        let totalReward = 0;
        let currentState = this.getInitialState(userInteractions);
        let steps = 0;
        const maxSteps = 50;
        
        while (steps < maxSteps && !this.isTerminalState(currentState)) {
            // Choose action using epsilon-greedy policy
            const action = this.chooseAction(currentState);
            
            // Execute action and observe next state and reward
            const { nextState, reward } = this.executeAction(currentState, action, userInteractions);
            
            // Update Q-value
            this.updateQValue(currentState, action, reward, nextState);
            
            // Update state and cumulative reward
            currentState = nextState;
            totalReward += reward;
            steps++;
            
            // Store experience
            this.storeExperience(currentState, action, reward, nextState);
        }
        
        this.episodeCount++;
        return totalReward;
    }

    // Get initial state from user interactions
    getInitialState(userInteractions) {
        const recentInteractions = userInteractions.slice(-10);
        
        return {
            userLevel: this.assessUserLevel(recentInteractions),
            performance: this.assessPerformance(recentInteractions),
            engagement: this.assessEngagement(recentInteractions),
            difficulty: this.assessDifficulty(recentInteractions),
            topic: this.assessCurrentTopic(recentInteractions),
            learningStyle: this.assessLearningStyle(recentInteractions)
        };
    }

    // Choose action using epsilon-greedy policy
    chooseAction(state) {
        const stateKey = this.stateToString(state);
        const qValues = this.qTable.get(stateKey);
        
        if (Math.random() < this.epsilon) {
            // Exploration: random action
            return this.actionSpace[Math.floor(Math.random() * this.actionSpace.length)];
        } else {
            // Exploitation: best action
            return this.getBestAction(qValues);
        }
    }

    // Get best action from Q-values
    getBestAction(qValues) {
        let bestAction = null;
        let bestValue = -Infinity;
        
        Object.entries(qValues).forEach(([action, value]) => {
            if (value > bestValue) {
                bestValue = value;
                bestAction = action;
            }
        });
        
        return bestAction || this.actionSpace[0];
    }

    // Execute action and observe result
    executeAction(state, action, userInteractions) {
        const nextState = this.getNextState(state, action);
        const reward = this.calculateReward(state, action, nextState, userInteractions);
        
        return { nextState, reward };
    }

    // Get next state after action
    getNextState(currentState, action) {
        const nextState = { ...currentState };
        
        // Update state based on action
        switch (action) {
            case 'show_next_lesson':
                nextState.performance = this.updatePerformance(currentState.performance, 0.1);
                nextState.difficulty = this.increaseDifficulty(currentState.difficulty);
                break;
                
            case 'provide_practice':
                nextState.engagement = this.updateEngagement(currentState.engagement, 0.2);
                nextState.performance = this.updatePerformance(currentState.performance, 0.05);
                break;
                
            case 'adjust_difficulty':
                nextState.difficulty = this.optimizeDifficulty(currentState);
                break;
                
            case 'provide_review':
                nextState.performance = this.updatePerformance(currentState.performance, 0.15);
                break;
                
            case 'introduce_challenge':
                nextState.userLevel = this.advanceUserLevel(currentState.userLevel);
                nextState.difficulty = this.increaseDifficulty(currentState.difficulty);
                break;
                
            case 'offer_break':
                nextState.engagement = this.updateEngagement(currentState.engagement, 0.1);
                break;
                
            case 'suggest_alternative':
                nextState.engagement = this.updateEngagement(currentState.engagement, 0.05);
                break;
        }
        
        return nextState;
    }

    // Calculate reward for state-action pair
    calculateReward(state, action, nextState, userInteractions) {
        let reward = 0;
        
        // Performance improvement reward
        const performanceImprovement = this.getPerformanceImprovement(state, nextState);
        reward += performanceImprovement * 10;
        
        // Engagement reward
        const engagementImprovement = this.getEngagementImprovement(state, nextState);
        reward += engagementImprovement * 5;
        
        // Learning efficiency reward
        const learningEfficiency = this.calculateLearningEfficiency(state, action, userInteractions);
        reward += learningEfficiency * 8;
        
        // Appropriate difficulty reward
        const difficultyAppropriateness = this.assessDifficultyAppropriateness(state, nextState);
        reward += difficultyAppropriateness * 6;
        
        // Time efficiency penalty
        const timeEfficiency = this.assessTimeEfficiency(action, userInteractions);
        reward -= timeEfficiency * 2;
        
        // Exploration bonus
        if (this.isExploratoryAction(action)) {
            reward += 1;
        }
        
        return reward;
    }

    // Update Q-value using Q-learning formula
    updateQValue(state, action, reward, nextState) {
        const stateKey = this.stateToString(state);
        const nextStateKey = this.stateToString(nextState);
        
        const qValues = this.qTable.get(stateKey);
        const nextQValues = this.qTable.get(nextStateKey);
        const maxNextQ = Math.max(...Object.values(nextQValues));
        
        // Q-learning update rule
        const currentQ = qValues[action];
        const newQ = currentQ + this.learningRate * (reward + this.discountFactor * maxNextQ - currentQ);
        
        qValues[action] = newQ;
        this.qTable.set(stateKey, qValues);
    }

    // Store experience for replay
    storeExperience(state, action, reward, nextState) {
        const experience = {
            state,
            action,
            reward,
            nextState,
            timestamp: Date.now()
        };
        
        this.experienceBuffer.push(experience);
        
        // Keep buffer size manageable
        if (this.experienceBuffer.length > 10000) {
            this.experienceBuffer = this.experienceBuffer.slice(-5000);
        }
    }

    // Experience replay for better learning
    experienceReplay(batchSize = 32) {
        if (this.experienceBuffer.length < batchSize) {
            return;
        }
        
        // Sample random batch
        const batch = [];
        for (let i = 0; i < batchSize; i++) {
            const randomIndex = Math.floor(Math.random() * this.experienceBuffer.length);
            batch.push(this.experienceBuffer[randomIndex]);
        }
        
        // Update Q-values based on experience replay
        batch.forEach(experience => {
            this.updateQValue(experience.state, experience.action, experience.reward, experience.nextState);
        });
    }

    // Assessment methods
    assessUserLevel(interactions) {
        const performance = interactions.filter(i => i.type === 'performance').map(i => i.score);
        if (performance.length === 0) return 'beginner';
        
        const avgPerformance = performance.reduce((sum, p) => sum + p, 0) / performance.length;
        
        if (avgPerformance > 0.8) return 'advanced';
        if (avgPerformance > 0.5) return 'intermediate';
        return 'beginner';
    }

    assessPerformance(interactions) {
        const recentPerformance = interactions.slice(-5).filter(i => i.type === 'performance');
        if (recentPerformance.length === 0) return 'medium';
        
        const avgScore = recentPerformance.reduce((sum, i) => sum + i.score, 0) / recentPerformance.length;
        
        if (avgScore > 0.8) return 'high';
        if (avgScore > 0.5) return 'medium';
        return 'low';
    }

    assessEngagement(interactions) {
        const recentEngagement = interactions.slice(-5).filter(i => i.type === 'engagement');
        if (recentEngagement.length === 0) return 'medium';
        
        const avgEngagement = recentEngagement.reduce((sum, i) => sum + i.level, 0) / recentEngagement.length;
        
        if (avgEngagement > 0.7) return 'high';
        if (avgEngagement > 0.4) return 'medium';
        return 'low';
    }

    assessDifficulty(interactions) {
        const difficultyFeedback = interactions.filter(i => i.type === 'difficulty_feedback');
        if (difficultyFeedback.length === 0) return 'medium';
        
        const avgDifficulty = difficultyFeedback.reduce((sum, i) => sum + i.difficulty, 0) / difficultyFeedback.length;
        
        if (avgDifficulty > 0.7) return 'hard';
        if (avgDifficulty > 0.4) return 'medium';
        return 'easy';
    }

    assessCurrentTopic(interactions) {
        const topicInteractions = interactions.filter(i => i.type === 'topic');
        if (topicInteractions.length === 0) return 'pythagoras';
        
        // Return most recent topic
        return topicInteractions[topicInteractions.length - 1].topic;
    }

    assessLearningStyle(interactions) {
        const styleInteractions = interactions.filter(i => i.type === 'learning_style');
        if (styleInteractions.length === 0) return 'visual';
        
        // Return most common learning style
        const styleCounts = {};
        styleInteractions.forEach(i => {
            styleCounts[i.style] = (styleCounts[i.style] || 0) + 1;
        });
        
        return Object.entries(styleCounts).sort((a, b) => b[1] - a[1])[0][0];
    }

    // State update helpers
    updatePerformance(current, change) {
        const newValue = Math.max(0, Math.min(1, current + change));
        if (newValue > 0.8) return 'high';
        if (newValue > 0.5) return 'medium';
        return 'low';
    }

    updateEngagement(current, change) {
        const newValue = Math.max(0, Math.min(1, current + change));
        if (newValue > 0.7) return 'high';
        if (newValue > 0.4) return 'medium';
        return 'low';
    }

    increaseDifficulty(current) {
        const levels = ['easy', 'medium', 'hard'];
        const currentIndex = levels.indexOf(current);
        return levels[Math.min(currentIndex + 1, levels.length - 1)];
    }

    optimizeDifficulty(currentState) {
        // Optimize difficulty based on current performance and engagement
        if (currentState.performance === 'low' && currentState.engagement === 'low') {
            return 'easy';
        } else if (currentState.performance === 'high' && currentState.engagement === 'high') {
            return 'hard';
        }
        return 'medium';
    }

    advanceUserLevel(current) {
        const levels = ['beginner', 'intermediate', 'advanced'];
        const currentIndex = levels.indexOf(current);
        return levels[Math.min(currentIndex + 1, levels.length - 1)];
    }

    // Reward calculation helpers
    getPerformanceImprovement(state, nextState) {
        const levels = { low: 0, medium: 1, high: 2 };
        return levels[nextState.performance] - levels[state.performance];
    }

    getEngagementImprovement(state, nextState) {
        const levels = { low: 0, medium: 1, high: 2 };
        return levels[nextState.engagement] - levels[state.engagement];
    }

    calculateLearningEfficiency(state, action, userInteractions) {
        // Measure how efficiently the action led to learning
        const relevantInteractions = userInteractions.filter(i => 
            i.action === action && i.timestamp > Date.now() - 300000 // Last 5 minutes
        );
        
        if (relevantInteractions.length === 0) return 0;
        
        const avgTimeToComplete = relevantInteractions.reduce((sum, i) => sum + i.duration, 0) / relevantInteractions.length;
        const expectedTime = this.getExpectedTimeForAction(action);
        
        return Math.max(0, 1 - (avgTimeToComplete / expectedTime));
    }

    getExpectedTimeForAction(action) {
        const timeEstimates = {
            'show_next_lesson': 300,
            'provide_practice': 600,
            'offer_hint': 60,
            'adjust_difficulty': 120,
            'provide_review': 240,
            'introduce_challenge': 900,
            'offer_break': 180,
            'suggest_alternative': 150,
            'provide_feedback': 90
        };
        
        return timeEstimates[action] || 300;
    }

    assessDifficultyAppropriateness(state, nextState) {
        // Reward appropriate difficulty matching
        if (state.performance === 'low' && nextState.difficulty === 'easy') return 1;
        if (state.performance === 'medium' && nextState.difficulty === 'medium') return 1;
        if (state.performance === 'high' && nextState.difficulty === 'hard') return 1;
        return -0.5;
    }

    assessTimeEfficiency(action, userInteractions) {
        // Penalize actions that take too long
        const actionInteractions = userInteractions.filter(i => i.action === action);
        if (actionInteractions.length === 0) return 0;
        
        const avgDuration = actionInteractions.reduce((sum, i) => sum + i.duration, 0) / actionInteractions.length;
        const expectedDuration = this.getExpectedTimeForAction(action);
        
        return Math.max(0, (avgDuration - expectedDuration) / expectedDuration);
    }

    isExploratoryAction(action) {
        const exploratoryActions = ['suggest_alternative', 'introduce_challenge', 'offer_break'];
        return exploratoryActions.includes(action);
    }

    // Convergence checking
    checkConvergence() {
        if (this.convergenceHistory.length < 10) return false;
        
        const recentRewards = this.convergenceHistory.slice(-10);
        const avgRecentReward = recentRewards.reduce((sum, r) => sum + r, 0) / recentRewards.length;
        const variance = recentRewards.reduce((sum, r) => sum + Math.pow(r - avgRecentReward, 2), 0) / recentRewards.length;
        
        // Consider converged if variance is very low
        return variance < 0.01;
    }

    // Update training metrics
    updateMetrics(episodeReward, episode) {
        this.totalReward += episodeReward;
        this.averageReward = this.totalReward / (episode + 1);
        this.convergenceHistory.push(episodeReward);
        
        // Keep history manageable
        if (this.convergenceHistory.length > 100) {
            this.convergenceHistory = this.convergenceHistory.slice(-50);
        }
    }

    // Check if state is terminal
    isTerminalState(state) {
        // Terminal states: high performance + high engagement + appropriate difficulty
        return state.performance === 'high' && 
               state.engagement === 'high' && 
               (state.difficulty === 'hard' || state.userLevel === 'advanced');
    }

    // Get training results
    getTrainingResults() {
        return {
            episodes: this.episodeCount,
            totalReward: this.totalReward,
            averageReward: this.averageReward,
            convergence: this.checkConvergence(),
            qTableSize: this.qTable.size,
            finalEpsilon: this.epsilon,
            policy: this.extractPolicy()
        };
    }

    // Extract optimal policy
    extractPolicy() {
        const policy = {};
        
        this.qTable.forEach((qValues, stateKey) => {
            const bestAction = this.getBestAction(qValues);
            policy[stateKey] = bestAction;
        });
        
        return policy;
    }

    // Get recommended action for current state
    getRecommendedAction(currentState) {
        const stateKey = this.stateToString(currentState);
        const qValues = this.qTable.get(stateKey);
        
        if (!qValues) {
            // Unknown state, return default action
            return 'show_next_lesson';
        }
        
        return this.getBestAction(qValues);
    }

    // Save trained model
    saveModel() {
        const modelData = {
            qTable: Object.fromEntries(this.qTable),
            policy: this.extractPolicy(),
            metrics: {
                episodes: this.episodeCount,
                totalReward: this.totalReward,
                averageReward: this.averageReward,
                convergence: this.checkConvergence(),
                finalEpsilon: this.epsilon
            },
            stateSpace: this.stateSpace,
            actionSpace: this.actionSpace,
            hyperparameters: {
                learningRate: this.learningRate,
                discountFactor: this.discountFactor,
                epsilonDecay: this.epsilonDecay
            },
            timestamp: Date.now()
        };
        
        localStorage.setItem('rl_model', JSON.stringify(modelData));
        console.log('Reforcement learning model saved');
        return modelData;
    }

    // Load trained model
    loadModel() {
        const savedModel = localStorage.getItem('rl_model');
        if (savedModel) {
            const modelData = JSON.parse(savedModel);
            
            // Restore Q-table
            this.qTable = new Map(Object.entries(modelData.qTable));
            
            // Restore metrics
            this.episodeCount = modelData.metrics.episodes;
            this.totalReward = modelData.metrics.totalReward;
            this.averageReward = modelData.metrics.averageReward;
            this.epsilon = modelData.metrics.finalEpsilon;
            
            console.log('Reforcement learning model loaded');
            return true;
        }
        return false;
    }

    // Get learning analytics
    getAnalytics() {
        return {
            totalStates: this.qTable.size,
            totalEpisodes: this.episodeCount,
            averageReward: this.averageReward,
            convergenceStatus: this.checkConvergence(),
            policyCoverage: this.calculatePolicyCoverage(),
            actionDistribution: this.getActionDistribution(),
            learningProgress: this.assessLearningProgress()
        };
    }

    calculatePolicyCoverage() {
        const policy = this.extractPolicy();
        const totalStates = this.qTable.size;
        const coveredStates = Object.keys(policy).length;
        
        return coveredStates / totalStates;
    }

    getActionDistribution() {
        const policy = this.extractPolicy();
        const actionCounts = {};
        
        Object.values(policy).forEach(action => {
            actionCounts[action] = (actionCounts[action] || 0) + 1;
        });
        
        const totalActions = Object.values(actionCounts).reduce((sum, count) => sum + count, 0);
        
        return Object.entries(actionCounts).map(([action, count]) => ({
            action,
            count,
            percentage: (count / totalActions) * 100
        }));
    }

    assessLearningProgress() {
        if (this.convergenceHistory.length < 10) return 'insufficient_data';
        
        const recentRewards = this.convergenceHistory.slice(-10);
        const olderRewards = this.convergenceHistory.slice(-20, -10);
        
        if (olderRewards.length === 0) return 'in_progress';
        
        const recentAvg = recentRewards.reduce((sum, r) => sum + r, 0) / recentRewards.length;
        const olderAvg = olderRewards.reduce((sum, r) => sum + r, 0) / olderRewards.length;
        
        if (recentAvg > olderAvg + 0.1) return 'improving';
        if (recentAvg < olderAvg - 0.1) return 'declining';
        return 'stable';
    }
}

// Export for use in main application
window.ReinforcementLearningSystem = ReinforcementLearningSystem;
