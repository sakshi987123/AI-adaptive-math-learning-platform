// LSTM Networks for Sequential Learning
// Implements: Long Short-Term Memory, Sequential Data Processing, Time Series Analysis

class LSTMNetwork {
    constructor(inputSize, hiddenSize, outputSize, numLayers = 2) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.numLayers = numLayers;
        
        // Initialize LSTM parameters
        this.lstmLayers = this.initializeLSTMLayers();
        this.outputWeights = this.initializeWeights(hiddenSize, outputSize);
        this.outputBias = this.initializeBias(outputSize);
        
        // Learning parameters
        this.learningRate = 0.001;
        this.clipValue = 5.0;
        this.epochs = 100;
        this.batchSize = 32;
        
        // Training history
        this.trainingHistory = [];
        this.validationHistory = [];
    }

    // Initialize LSTM layers
    initializeLSTMLayers() {
        const layers = [];
        
        for (let l = 0; l < this.numLayers; l++) {
            const layer = {
                // Input gate weights
                Wf: this.initializeWeights(this.inputSize + this.hiddenSize, this.hiddenSize),
                bf: this.initializeBias(this.hiddenSize),
                
                // Forget gate weights
                Wi: this.initializeWeights(this.inputSize + this.hiddenSize, this.hiddenSize),
                bi: this.initializeBias(this.hiddenSize),
                
                // Output gate weights
                Wo: this.initializeWeights(this.hiddenSize, this.hiddenSize),
                bo: this.initializeBias(this.hiddenSize),
                
                // Cell state weights
                Wc: this.initializeWeights(this.inputSize + this.hiddenSize, this.hiddenSize),
                bc: this.initializeBias(this.hiddenSize)
            };
            
            layers.push(layer);
        }
        
        return layers;
    }

    // Initialize weights with Xavier initialization
    initializeWeights(rows, cols) {
        const weights = [];
        for (let i = 0; i < rows; i++) {
            weights[i] = [];
            for (let j = 0; j < cols; j++) {
                weights[i][j] = (Math.random() - 0.5) * 2 * Math.sqrt(6.0 / (rows + cols));
            }
        }
        return weights;
    }

    // Initialize bias
    initializeBias(size) {
        return Array(size).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    }

    // LSTM forward pass
    forward(input, previousHidden = null, previousCell = null) {
        let hidden = previousHidden || Array(this.hiddenSize).fill(0);
        let cell = previousCell || Array(this.hiddenSize).fill(0);
        
        for (let l = 0; l < this.numLayers; l++) {
            const layer = this.lstmLayers[l];
            const layerInput = l === 0 ? input : hidden;
            
            const lstmOutput = this.lstmCellForward(
                layerInput, hidden, cell,
                layer.Wf, layer.Wi, layer.Wo, layer.Wc,
                layer.bf, layer.bi, layer.bo, layer.bc
            );
            
            hidden = lstmOutput.hidden;
            cell = lstmOutput.cell;
        }
        
        // Final output layer
        const output = this.calculateOutput(hidden);
        
        return {
            output,
            hidden: hidden[this.numLayers - 1],
            cell: cell[this.numLayers - 1]
        };
    }

    // LSTM cell forward pass
    lstmCellForward(input, hidden, cell, Wf, Wi, Wo, Wc, bf, bi, bo, bc) {
        const concatenatedInput = this.concatenateInput(input, hidden);
        
        // Gate calculations
        const forgetGate = this.sigmoid(
            this.matrixVectorMultiply(concatenatedInput, Wf) + bf
        );
        
        const inputGate = this.sigmoid(
            this.matrixVectorMultiply(concatenatedInput, Wi) + bi
        );
        
        const outputGate = this.sigmoid(
            this.matrixVectorMultiply(concatenatedInput, Wo) + bo
        );
        
        // Cell state candidate
        const cellCandidate = this.tanh(
            this.matrixVectorMultiply(concatenatedInput, Wc) + bc
        );
        
        // Update cell state and hidden state
        const newCell = forgetGate * cell + inputGate * cellCandidate;
        const newHidden = outputGate * this.tanh(newCell);
        
        return {
            hidden: newHidden,
            cell: newCell
        };
    }

    // Concatenate input and hidden state
    concatenateInput(input, hidden) {
        return [...input, ...hidden];
    }

    // Matrix-vector multiplication
    matrixVectorMultiply(matrix, vector) {
        return matrix.map(row => 
            row.reduce((sum, weight, i) => sum + weight * vector[i], 0)
        );
    }

    // Vector-matrix multiplication
    matrixVectorMultiply(vector, matrix) {
        const result = [];
        for (let i = 0; i < matrix[0].length; i++) {
            let sum = 0;
            for (let j = 0; j < vector.length; j++) {
                sum += vector[j] * matrix[j][i];
            }
            result[i] = sum;
        }
        return result;
    }

    // Sigmoid activation
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Tanh activation
    tanh(x) {
        return Math.tanh(x);
    }

    // Calculate final output
    calculateOutput(hidden) {
        return this.matrixVectorMultiply(this.outputWeights, hidden) + this.outputBias;
    }

    // Backward pass through LSTM
    backward(input, previousHidden, previousCell, target, output, hidden, cell) {
        const gradients = this.calculateGradients(input, previousHidden, previousCell, target, output, hidden, cell);
        
        // Update weights using gradients
        this.updateWeights(gradients);
        
        return gradients.loss;
    }

    // Calculate gradients for LSTM
    calculateGradients(input, previousHidden, previousCell, target, output, hidden, cell) {
        const outputError = this.calculateOutputError(target, output);
        
        // Backpropagate through output layer
        const outputGradients = this.calculateOutputGradients(outputError, hidden);
        
        // Backpropagate through LSTM layers
        const lstmGradients = this.calculateLSTMGradients(
            input, previousHidden, previousCell, hidden, cell, outputGradients
        );
        
        return {
            outputError,
            outputGradients,
            lstmGradients,
            loss: this.calculateLoss(target, output)
        };
    }

    // Calculate output error
    calculateOutputError(target, output) {
        const errors = [];
        for (let i = 0; i < this.outputSize; i++) {
            errors[i] = target[i] - output[i];
        }
        return errors;
    }

    // Calculate output gradients
    calculateOutputGradients(outputError, hidden) {
        return outputError.map(error => 
            error * hidden.map(h => h >= 0 ? 1 : 0) // Derivative of tanh
        );
    }

    // Calculate LSTM gradients
    calculateLSTMGradients(input, previousHidden, previousCell, hidden, cell, outputGradients) {
        const gradients = [];
        
        for (let l = this.numLayers - 1; l >= 0; l--) {
            const layer = this.lstmLayers[l];
            const layerInput = l === 0 ? input : hidden[l - 1];
            
            const gradient = this.calculateLSTMCellGradients(
                layerInput, hidden[l - 1], cell[l - 1], hidden[l], cell[l], outputGradients, layer
            );
            
            gradients.unshift(gradient);
        }
        
        return gradients;
    }

    // Calculate gradients for single LSTM cell
    calculateLSTMCellGradients(input, previousHidden, previousCell, hidden, cell, outputGradients, layer) {
        const concatenatedInput = this.concatenateInput(input, previousHidden);
        
        // Calculate gradients for each gate
        const forgetGateGradient = this.calculateGateGradients(
            outputGradients, hidden, cell, layer.Wf, layer.bf
        );
        
        const inputGateGradient = this.calculateGateGradients(
            outputGradients, hidden, cell, layer.Wi, layer.bi
        );
        
        const outputGateGradient = this.calculateGateGradients(
            outputGradients, hidden, cell, layer.Wo, layer.bo
        );
        
        const cellCandidateGradient = this.calculateCellCandidateGradients(
            outputGradients, hidden, cell, layer.Wc, layer.bc, forgetGateGradient, inputGateGradient
        );
        
        return {
            Wf: forgetGateGradient.W,
            bf: forgetGateGradient.b,
            Wi: inputGateGradient.W,
            bi: inputGateGradient.b,
            Wo: outputGateGradient.W,
            bo: outputGateGradient.b,
            Wc: cellCandidateGradient.W,
            bc: cellCandidateGradient.b
        };
    }

    // Calculate gate gradients
    calculateGateGradients(outputGradients, hidden, cell, weights, bias) {
        const gateOutput = outputGradients.map(og => 
            og * hidden.map(h => h >= 0 ? 1 : 0) // Derivative of tanh
        );
        
        const inputGradient = this.matrixVectorMultiply(gateOutput, this.concatenateInput(hidden, cell));
        const weightGradient = this.matrixVectorMultiply(gateOutput, weights);
        
        return {
            W: weightGradient,
            b: gateOutput.reduce((sum, og, i) => sum + og, 0)
        };
    }

    // Calculate cell candidate gradients
    calculateCellCandidateGradients(outputGradients, hidden, cell, weights, bias, forgetGradient, inputGradient) {
        const cellGradient = outputGradients.map(og => 
            og * hidden.map(h => h >= 0 ? 1 : 0) // Derivative of tanh
        );
        
        const cellInputGradient = this.matrixVectorMultiply(cellGradient, this.concatenateInput(hidden, cell));
        const weightGradient = this.matrixVectorMultiply(cellGradient, weights);
        
        return {
            W: weightGradient,
            b: cellGradient.reduce((sum, cg, i) => sum + cg, 0)
        };
    }

    // Calculate loss
    calculateLoss(target, output) {
        return output.reduce((sum, val, i) => 
            sum + Math.pow(val - target[i], 2), 0
        ) / output.length;
    }

    // Update weights using gradients
    updateWeights(gradients) {
        const learningRate = this.learningRate;
        
        // Update output layer
        for (let i = 0; i < this.outputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                this.outputWeights[i][j] -= learningRate * gradients.outputGradients[i][j];
            }
            this.outputBias[i] -= learningRate * gradients.outputGradients[i].reduce((sum, og) => sum + og, 0);
        }
        
        // Update LSTM layers
        for (let l = 0; l < this.numLayers; l++) {
            const layer = this.lstmLayers[l];
            const gradient = gradients.lstmGradients[l];
            
            // Update forget gate
            for (let i = 0; i < layer.Wf.length; i++) {
                for (let j = 0; j < layer.Wf[i].length; j++) {
                    layer.Wf[i][j] -= learningRate * gradient.Wf[i][j];
                }
            }
            for (let i = 0; i < layer.bf.length; i++) {
                layer.bf[i] -= learningRate * gradient.bf[i];
            }
            
            // Update input gate
            for (let i = 0; i < layer.Wi.length; i++) {
                for (let j = 0; j < layer.Wi[i].length; j++) {
                    layer.Wi[i][j] -= learningRate * gradient.Wi[i][j];
                }
            }
            for (let i = 0; i < layer.bi.length; i++) {
                layer.bi[i] -= learningRate * gradient.bi[i];
            }
            
            // Update output gate
            for (let i = 0; i < layer.Wo.length; i++) {
                for (let j = 0; j < layer.Wo[i].length; j++) {
                    layer.Wo[i][j] -= learningRate * gradient.Wo[i][j];
                }
            }
            for (let i = 0; i < layer.bo.length; i++) {
                layer.bo[i] -= learningRate * gradient.bo[i];
            }
            
            // Update cell state
            for (let i = 0; i < layer.Wc.length; i++) {
                for (let j = 0; j < layer.Wc[i].length; j++) {
                    layer.Wc[i][j] -= learningRate * gradient.Wc[i][j];
                }
            }
            for (let i = 0; i < layer.bc.length; i++) {
                layer.bc[i] -= learningRate * gradient.bc[i];
            }
        }
    }

    // Clip gradients to prevent exploding
    clipGradients(gradients) {
        const maxNorm = this.clipValue;
        
        // Clip LSTM gradients
        for (let l = 0; l < gradients.lstmGradients.length; l++) {
            const gradient = gradients.lstmGradients[l];
            
            // Clip each weight matrix
            this.clipMatrix(gradient.Wf, maxNorm);
            this.clipMatrix(gradient.Wi, maxNorm);
            this.clipMatrix(gradient.Wo, maxNorm);
            this.clipMatrix(gradient.Wc, maxNorm);
            
            // Clip bias vectors
            this.clipVector(gradient.bf, maxNorm);
            this.clipVector(gradient.bi, maxNorm);
            this.clipVector(gradient.bo, maxNorm);
            this.clipVector(gradient.bc, maxNorm);
        }
        
        // Clip output gradients
        this.clipMatrix(gradients.outputGradients, maxNorm);
    }

    // Clip matrix values
    clipMatrix(matrix, maxNorm) {
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                const norm = Math.sqrt(matrix[i].reduce((sum, val) => sum + val * val, 0));
                if (norm > maxNorm) {
                    const scale = maxNorm / norm;
                    matrix[i][j] *= scale;
                }
            }
        }
    }

    // Clip vector values
    clipVector(vector, maxNorm) {
        const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
        if (norm > maxNorm) {
            const scale = maxNorm / norm;
            for (let i = 0; i < vector.length; i++) {
                vector[i] *= scale;
            }
        }
    }

    // Train LSTM network
    train(trainingData, validationData = null) {
        console.log('Starting LSTM training...');
        
        for (let epoch = 0; epoch < this.epochs; epoch++) {
            let totalLoss = 0;
            
            // Shuffle training data
            const shuffled = this.shuffleArray([...trainingData]);
            
            // Mini-batch training
            for (let i = 0; i < shuffled.length; i++) {
                const { input, target } = shuffled[i];
                
                // Forward pass
                const output = this.forward(input);
                
                // Backward pass
                const loss = this.backward(input, null, null, target, output.output, output.hidden, output.cell);
                
                totalLoss += loss;
            }
            
            // Calculate average loss
            const avgLoss = totalLoss / shuffled.length;
            this.trainingHistory.push({ epoch, loss: avgLoss });
            
            // Validation
            if (validationData && epoch % 10 === 0) {
                const validationLoss = this.validate(validationData);
                this.validationHistory.push({ epoch, error: validationLoss });
                console.log(`Epoch ${epoch}: Training Loss = ${avgLoss.toFixed(6)}, Validation Loss = ${validationLoss.toFixed(6)}`);
            }
            
            // Early stopping
            if (avgLoss < 0.01) {
                console.log('Early stopping triggered');
                break;
            }
        }
        
        console.log('LSTM training completed');
        return this.trainingHistory;
    }

    // Validate LSTM network
    validate(validationData) {
        let totalLoss = 0;
        
        for (const { input, target } of validationData) {
            const output = this.forward(input);
            const loss = this.calculateLoss(target, output.output);
            totalLoss += loss;
        }
        
        return totalLoss / validationData.length;
    }

    // Generate sequence
    generateSequence(input, length = 50) {
        const sequence = [...input];
        let hidden = null;
        let cell = null;
        
        for (let i = 0; i < length; i++) {
            const output = this.forward(sequence.slice(-this.inputSize), hidden, cell);
            sequence.push(output.output[0]); // Take first output token
            hidden = output.hidden;
            cell = output.cell;
        }
        
        return sequence;
    }

    // Save LSTM model
    saveModel() {
        const model = {
            inputSize: this.inputSize,
            hiddenSize: this.hiddenSize,
            outputSize: this.outputSize,
            numLayers: this.numLayers,
            lstmLayers: this.lstmLayers,
            outputWeights: this.outputWeights,
            outputBias: this.outputBias,
            trainingHistory: this.trainingHistory,
            validationHistory: this.validationHistory
        };
        
        localStorage.setItem('lstm_model', JSON.stringify(model));
        console.log('LSTM model saved');
        return model;
    }

    // Load LSTM model
    loadModel() {
        const savedModel = localStorage.getItem('lstm_model');
        if (savedModel) {
            const model = JSON.parse(savedModel);
            this.lstmLayers = model.lstmLayers;
            this.outputWeights = model.outputWeights;
            this.outputBias = model.outputBias;
            this.trainingHistory = model.trainingHistory || [];
            this.validationHistory = model.validationHistory || [];
            console.log('LSTM model loaded');
            return true;
        }
        return false;
    }

    // Get model metrics
    getMetrics() {
        if (this.trainingHistory.length === 0) return null;
        
        const finalLoss = this.trainingHistory[this.trainingHistory.length - 1].loss;
        const initialLoss = this.trainingHistory[0].loss;
        const improvement = ((initialLoss - finalLoss) / initialLoss) * 100;
        
        return {
            finalLoss,
            initialLoss,
            improvement,
            epochs: this.trainingHistory.length,
            convergence: finalLoss < 0.01
        };
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
}

// Sequential Learning System
class SequentialLearningSystem {
    constructor() {
        this.lstmNetwork = new LSTMNetwork(50, 128, 100, 3); // 50 inputs, 128 hidden, 100 outputs
        this.sequences = new Map();
        this.patterns = new Map();
        this.learningHistory = [];
    }

    // Learn from user interaction sequences
    learnFromSequence(userId, sequence) {
        const sequenceKey = sequence.join('-');
        
        if (!this.sequences.has(sequenceKey)) {
            this.sequences.set(sequenceKey, []);
        }
        
        const sequenceHistory = this.sequences.get(sequenceKey);
        sequenceHistory.push({
            timestamp: Date.now(),
            userId,
            sequence
        });
        
        // Train LSTM on sequence data
        if (sequenceHistory.length > 10) {
            this.trainLSTMOnSequences(sequenceHistory);
        }
    }

    // Train LSTM on collected sequences
    trainLSTMOnSequences(sequences) {
        console.log('Training LSTM on user sequences...');
        
        // Prepare training data
        const trainingData = sequences.map(seq => ({
            input: this.prepareSequenceInput(seq.slice(0, -1)),
            target: this.prepareSequenceTarget(seq.slice(1))
        }));
        
        // Train the LSTM
        this.lstmNetwork.train(trainingData);
        
        console.log('LSTM sequence training completed');
    }

    // Prepare sequence input for LSTM
    prepareSequenceInput(sequence) {
        // Convert sequence to numerical representation
        return sequence.map(item => this.encodeItem(item));
    }

    // Prepare sequence target for LSTM
    prepareSequenceTarget(sequence) {
        // Next item in sequence prediction
        return this.encodeItem(sequence[0]);
    }

    // Encode item for LSTM
    encodeItem(item) {
        // Simple encoding based on item properties
        const encoding = Array(50).fill(0);
        encoding[0] = item.type || 0;
        encoding[1] = item.difficulty || 0;
        encoding[2] = item.duration || 0;
        encoding[3] = item.success || 0;
        
        return encoding;
    }

    // Predict next item in sequence
    predictNextItem(sequence) {
        const input = this.prepareSequenceInput(sequence);
        const output = this.lstmNetwork.forward(input);
        
        // Decode output to item prediction
        const prediction = this.decodeOutput(output.output);
        
        return prediction;
    }

    // Decode LSTM output to item
    decodeOutput(output) {
        // Find the most likely next item
        const maxIndex = output.indexOf(Math.max(...output));
        
        return {
            type: ['lesson', 'practice', 'quiz', 'break'][maxIndex % 4],
            difficulty: Math.floor(Math.random() * 3),
            duration: Math.floor(Math.random() * 300) + 60,
            confidence: Math.max(...output)
        };
    }

    // Analyze learning patterns
    analyzeLearningPatterns(userId) {
        const userSequences = this.getUserSequences(userId);
        
        if (userSequences.length < 5) {
            return {
                pattern: 'insufficient_data',
                recommendations: ['Continue learning to gather more data']
            };
        }
        
        // Analyze sequence patterns
        const patterns = this.extractPatterns(userSequences);
        const recommendations = this.generateRecommendations(patterns);
        
        return {
            patterns,
            recommendations,
            totalSequences: userSequences.length
        };
    }

    // Extract patterns from sequences
    extractPatterns(sequences) {
        const patterns = [];
        
        // Find common sequences
        const sequenceCounts = new Map();
        sequences.forEach(seq => {
            const key = seq.sequence.join('-');
            sequenceCounts.set(key, (sequenceCounts.get(key) || 0) + 1);
        });
        
        // Get most common patterns
        const sortedPatterns = Array.from(sequenceCounts.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);
        
        sortedPatterns.forEach(([sequence, count]) => {
            patterns.push({
                sequence,
                frequency: count,
                confidence: count / sequences.length
            });
        });
        
        return patterns;
    }

    // Generate recommendations based on patterns
    generateRecommendations(patterns) {
        const recommendations = [];
        
        patterns.forEach(pattern => {
            if (pattern.confidence > 0.3) {
                recommendations.push({
                    type: 'content_suggestion',
                    priority: 'high',
                    suggestion: `Users often follow: ${pattern.sequence.join(' → ')}`
                });
            }
        });
        
        return recommendations;
    }

    // Get user sequences
    getUserSequences(userId) {
        const savedSequences = localStorage.getItem(`user_sequences_${userId}`);
        return savedSequences ? JSON.parse(savedSequences) : [];
    }

    // Save sequence data
    saveSequenceData(userId, sequence) {
        const sequenceKey = sequence.join('-');
        
        if (!this.sequences.has(sequenceKey)) {
            this.sequences.set(sequenceKey, []);
        }
        
        const sequenceHistory = this.sequences.get(sequenceKey);
        sequenceHistory.push({
            timestamp: Date.now(),
            userId,
            sequence
        });
        
        localStorage.setItem(`user_sequences_${userId}`, JSON.stringify(Array.from(this.sequences.entries())));
    }
}

// Export for use in main application
window.LSTMNetwork = LSTMNetwork;
window.SequentialLearningSystem = SequentialLearningSystem;
