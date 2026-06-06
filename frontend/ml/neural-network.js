// Deep Learning Neural Network for Pattern Recognition
// Implements: Feedforward Neural Network, Backpropagation, Pattern Recognition

class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        
        // Initialize weights and biases
        this.weights1 = this.initializeWeights(inputSize, hiddenSize);
        this.weights2 = this.initializeWeights(hiddenSize, outputSize);
        this.bias1 = this.initializeBias(hiddenSize);
        this.bias2 = this.initializeBias(outputSize);
        
        // Learning parameters
        this.learningRate = 0.01;
        this.epochs = 1000;
        this.batchSize = 32;
        
        // Activation functions
        this.activation = this.relu;
        this.activationDerivative = this.reluDerivative;
        
        // Training history
        this.trainingHistory = [];
        this.validationHistory = [];
    }

    // Initialize weights with Xavier initialization
    initializeWeights(rows, cols) {
        const weights = [];
        for (let i = 0; i < rows; i++) {
            weights[i] = [];
            for (let j = 0; j < cols; j++) {
                weights[i][j] = (Math.random() - 0.5) * 2 * Math.sqrt(2.0 / cols);
            }
        }
        return weights;
    }

    initializeBias(size) {
        return Array(size).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    }

    // ReLU activation function
    relu(x) {
        return Math.max(0, x);
    }

    reluDerivative(x) {
        return x > 0 ? 1 : 0;
    }

    // Softmax for output layer
    softmax(outputs) {
        const expOutputs = outputs.map(Math.exp);
        const sumExp = expOutputs.reduce((sum, val) => sum + val, 0);
        return expOutputs.map(val => val / sumExp);
    }

    // Forward propagation
    forward(input) {
        // Hidden layer
        const hidden = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let sum = 0;
            for (let j = 0; j < this.inputSize; j++) {
                sum += input[j] * this.weights1[i][j];
            }
            hidden[i] = this.activation(sum + this.bias1[i]);
        }

        // Output layer
        const output = [];
        for (let i = 0; i < this.outputSize; i++) {
            let sum = 0;
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += hidden[j] * this.weights2[i][j];
            }
            output[i] = sum + this.bias2[i];
        }

        return this.softmax(output);
    }

    // Backward propagation
    backward(input, target, output, hidden) {
        // Calculate output layer error
        const outputError = [];
        for (let i = 0; i < this.outputSize; i++) {
            outputError[i] = target[i] - output[i];
        }

        // Calculate hidden layer error
        const hiddenError = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let error = 0;
            for (let j = 0; j < this.outputSize; j++) {
                error += outputError[j] * this.weights2[j][i];
            }
            hiddenError[i] = error * this.activationDerivative(hidden[i]);
        }

        return { outputError, hiddenError };
    }

    // Update weights using gradient descent
    updateWeights(input, hidden, outputError, hiddenError) {
        // Update output layer weights
        for (let i = 0; i < this.outputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                this.weights2[i][j] += this.learningRate * outputError[i] * hidden[j];
            }
            this.bias2[i] += this.learningRate * outputError[i];
        }

        // Update hidden layer weights
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.inputSize; j++) {
                let error = 0;
                for (let k = 0; k < this.outputSize; k++) {
                    error += outputError[k] * this.weights2[k][i];
                }
                this.weights1[i][j] += this.learningRate * error * hiddenError[i] * input[j];
            }
            this.bias1[i] += this.learningRate * hiddenError[i];
        }
    }

    // Training the neural network
    train(trainingData, validationData = null) {
        console.log('Starting neural network training...');
        
        for (let epoch = 0; epoch < this.epochs; epoch++) {
            let totalError = 0;
            
            // Shuffle training data
            const shuffled = this.shuffleArray([...trainingData]);
            
            // Mini-batch training
            for (let i = 0; i < shuffled.length; i++) {
                const { input, target } = shuffled[i];
                
                // Forward pass
                const hidden = this.forwardHidden(input);
                const output = this.forwardOutput(hidden);
                
                // Backward pass
                const { outputError, hiddenError } = this.backward(input, target, output, hidden);
                
                // Update weights
                this.updateWeights(input, hidden, output, outputError, hiddenError);
                
                // Calculate error
                const error = outputError.reduce((sum, err) => sum + err * err, 0);
                totalError += error;
            }
            
            // Calculate average error
            const avgError = totalError / shuffled.length;
            this.trainingHistory.push({ epoch, error: avgError });
            
            // Validation
            if (validationData && epoch % 10 === 0) {
                const validationError = this.validate(validationData);
                this.validationHistory.push({ epoch, error: validationError });
                console.log(`Epoch ${epoch}: Training Error = ${avgError.toFixed(4)}, Validation Error = ${validationError.toFixed(4)}`);
            }
            
            // Early stopping
            if (avgError < 0.001) {
                console.log('Early stopping triggered');
                break;
            }
        }
        
        console.log('Training completed');
        return this.trainingHistory;
    }

    forwardHidden(input) {
        const hidden = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let sum = 0;
            for (let j = 0; j < this.inputSize; j++) {
                sum += input[j] * this.weights1[i][j];
            }
            hidden[i] = this.activation(sum + this.bias1[i]);
        }
        return hidden;
    }

    forwardOutput(hidden) {
        const output = [];
        for (let i = 0; i < this.outputSize; i++) {
            let sum = 0;
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += hidden[j] * this.weights2[i][j];
            }
            output[i] = sum + this.bias2[i];
        }
        return this.softmax(output);
    }

    backward(input, target, output, hidden) {
        // Calculate output layer error
        const outputError = [];
        for (let i = 0; i < this.outputSize; i++) {
            outputError[i] = target[i] - output[i];
        }

        // Calculate hidden layer error
        const hiddenError = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let error = 0;
            for (let j = 0; j < this.outputSize; j++) {
                error += outputError[j] * this.weights2[j][i];
            }
            hiddenError[i] = error * this.activationDerivative(hidden[i]);
        }

        return { outputError, hiddenError };
    }

    // Validate the network
    validate(validationData) {
        let totalError = 0;
        
        for (const { input, target } of validationData) {
            const output = this.forward(input);
            const error = output.reduce((sum, val, idx) => {
                return sum + Math.pow(val - target[idx], 2);
            }, 0);
            totalError += error;
        }
        
        return totalError / validationData.length;
    }

    // Predict using trained network
    predict(input) {
        return this.forward(input);
    }

    // Save model weights
    saveModel() {
        const model = {
            weights1: this.weights1,
            weights2: this.weights2,
            bias1: this.bias1,
            bias2: this.bias2,
            architecture: {
                inputSize: this.inputSize,
                hiddenSize: this.hiddenSize,
                outputSize: this.outputSize
            },
            trainingHistory: this.trainingHistory,
            validationHistory: this.validationHistory
        };
        
        localStorage.setItem('neural_network_model', JSON.stringify(model));
        console.log('Model saved to localStorage');
        return model;
    }

    // Load model weights
    loadModel() {
        const savedModel = localStorage.getItem('neural_network_model');
        if (savedModel) {
            const model = JSON.parse(savedModel);
            this.weights1 = model.weights1;
            this.weights2 = model.weights2;
            this.bias1 = model.bias1;
            this.bias2 = model.bias2;
            this.trainingHistory = model.trainingHistory || [];
            this.validationHistory = model.validationHistory || [];
            console.log('Model loaded from localStorage');
            return true;
        }
        return false;
    }

    // Utility function to shuffle array
    shuffleArray(array) {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }

    // Get model performance metrics
    getMetrics() {
        if (this.trainingHistory.length === 0) return null;
        
        const finalError = this.trainingHistory[this.trainingHistory.length - 1].error;
        const initialError = this.trainingHistory[0].error;
        const improvement = ((initialError - finalError) / initialError) * 100;
        
        return {
            finalError,
            initialError,
            improvement,
            epochs: this.trainingHistory.length,
            convergence: finalError < 0.01
        };
    }
}

// Pattern Recognition System for Learning Analytics
class PatternRecognitionSystem {
    constructor() {
        this.neuralNetwork = new NeuralNetwork(10, 20, 3); // 10 inputs, 20 hidden, 3 outputs
        this.patterns = new Map();
        this.sequences = [];
    }

    // Recognize learning patterns from user behavior
    recognizePattern(userSessions) {
        if (userSessions.length < 5) {
            return { confidence: 0, pattern: 'insufficient_data' };
        }

        // Extract features from sessions
        const features = this.extractFeatures(userSessions);
        
        // Use neural network to classify pattern
        const prediction = this.neuralNetwork.predict(features);
        const confidence = Math.max(...prediction);
        const patternIndex = prediction.indexOf(confidence);
        
        const patterns = ['beginner', 'intermediate', 'advanced', 'struggling', 'excelling'];
        const recognizedPattern = patterns[patternIndex];
        
        return {
            confidence,
            pattern: recognizedPattern,
            features: features
        };
    }

    // Extract features from user sessions
    extractFeatures(sessions) {
        const features = [];
        
        // Time-based features
        const avgTime = sessions.reduce((sum, s) => sum + (s.duration || 0), 0) / sessions.length;
        const timeVariation = this.calculateVariation(sessions.map(s => s.duration || 0));
        
        // Performance-based features
        const avgScore = sessions.reduce((sum, s) => sum + (s.score || 0), 0) / sessions.length;
        const scoreTrend = this.calculateTrend(sessions.map(s => s.score || 0));
        
        // Interaction-based features
        const avgInteractions = sessions.reduce((sum, s) => sum + (s.interactions || 0), 0) / sessions.length;
        const interactionPattern = this.analyzeInteractionPattern(sessions);
        
        // Normalize features
        features.push(
            this.normalize(avgTime, 0, 600), // 0-10 minutes
            this.normalize(timeVariation, 0, 200),
            this.normalize(avgScore, 0, 100),
            this.normalize(scoreTrend, -1, 1),
            this.normalize(avgInteractions, 0, 50),
            this.normalize(interactionPattern, 0, 1)
        );
        
        return features;
    }

    // Calculate variation in data
    calculateVariation(values) {
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        return Math.sqrt(variance);
    }

    // Calculate trend in data
    calculateTrend(values) {
        if (values.length < 2) return 0;
        
        let increasing = 0, decreasing = 0;
        for (let i = 1; i < values.length; i++) {
            if (values[i] > values[i-1]) increasing++;
            else if (values[i] < values[i-1]) decreasing++;
        }
        
        return (increasing - decreasing) / (values.length - 1);
    }

    // Analyze interaction patterns
    analyzeInteractionPattern(sessions) {
        // Simple pattern: ratio of different interaction types
        const interactionTypes = {
            clicks: 0,
            hover: 0,
            keyboard: 0,
            video: 0
        };
        
        sessions.forEach(session => {
            if (session.interactions) {
                session.interactions.forEach(interaction => {
                    interactionTypes[interaction.type]++;
                });
            }
        });
        
        const total = Object.values(interactionTypes).reduce((sum, val) => sum + val, 0);
        return total > 0 ? interactionTypes.clicks / total : 0.5;
    }

    // Normalize values to 0-1 range
    normalize(value, min, max) {
        return (value - min) / (max - min);
    }

    // Train the pattern recognition system
    train(trainingData) {
        console.log('Training pattern recognition system...');
        
        // Prepare training data
        const preparedData = trainingData.map(session => ({
            input: this.extractFeatures([session]),
            target: this.encodeLearningStyle(session.pattern)
        }));
        
        // Train neural network
        this.neuralNetwork.train(preparedData);
        
        console.log('Pattern recognition training completed');
    }

    // Encode learning style for neural network
    encodeLearningStyle(pattern) {
        const styles = ['beginner', 'intermediate', 'advanced', 'struggling', 'excelling'];
        const encoding = [0, 0, 0, 0, 0];
        encoding[styles.indexOf(pattern)] = 1;
        return encoding;
    }

    // Get personalized recommendations
    getRecommendations(userId) {
        const userSessions = this.getUserSessions(userId);
        if (userSessions.length < 5) {
            return this.getDefaultRecommendations();
        }

        const pattern = this.recognizePattern(userSessions);
        const recommendations = this.generateRecommendations(pattern);
        
        return {
            pattern,
            recommendations,
            confidence: pattern.confidence
        };
    }

    getUserSessions(userId) {
        // In production, would fetch from database
        const savedSessions = localStorage.getItem(`user_sessions_${userId}`);
        return savedSessions ? JSON.parse(savedSessions) : [];
    }

    generateRecommendations(pattern) {
        const recommendations = [];
        
        switch (pattern.pattern) {
            case 'beginner':
                recommendations.push(
                    { type: 'content', priority: 'high', action: 'Focus on basic concepts' },
                    { type: 'pace', priority: 'high', action: 'Slower pacing with more examples' }
                );
                break;
                
            case 'struggling':
                recommendations.push(
                    { type: 'review', priority: 'high', action: 'Review previous lessons' },
                    { type: 'practice', priority: 'high', action: 'Additional practice exercises' },
                    { type: 'support', priority: 'medium', action: 'Offer additional help resources' }
                );
                break;
                
            case 'advanced':
                recommendations.push(
                    { type: 'challenge', priority: 'medium', action: 'Introduce advanced problems' },
                    { type: 'explore', priority: 'low', action: 'Explore related topics' }
                );
                break;
                
            case 'excelling':
                recommendations.push(
                    { type: 'advance', priority: 'high', action: 'Ready for next level' },
                    { type: 'enrichment', priority: 'medium', action: 'Supplementary materials' }
                );
                break;
        }
        
        return recommendations;
    }

    getDefaultRecommendations() {
        return [
            { type: 'assessment', priority: 'high', action: 'Take placement test' },
            { type: 'foundation', priority: 'high', action: 'Start with basics' },
            { type: 'guided', priority: 'medium', action: 'Use guided learning mode' }
        ];
    }

    // Save pattern recognition model
    saveModel() {
        const model = {
            neuralNetwork: this.neuralNetwork.saveModel(),
            patterns: Array.from(this.patterns.entries()),
            sequences: this.sequences,
            timestamp: Date.now()
        };
        
        localStorage.setItem('pattern_recognition_model', JSON.stringify(model));
        console.log('Pattern recognition model saved');
    }

    // Load pattern recognition model
    loadModel() {
        const savedModel = localStorage.getItem('pattern_recognition_model');
        if (savedModel) {
            const model = JSON.parse(savedModel);
            this.neuralNetwork.loadModel();
            this.patterns = new Map(model.patterns);
            this.sequences = model.sequences || [];
            console.log('Pattern recognition model loaded');
            return true;
        }
        return false;
    }
}

// Export for use in main application
window.NeuralNetwork = NeuralNetwork;
window.PatternRecognitionSystem = PatternRecognitionSystem;
