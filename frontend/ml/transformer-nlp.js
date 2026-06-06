// Transformer Architecture for Natural Language Processing
// Implements: Multi-Head Attention, Positional Encoding, Transformer Blocks

class TransformerModel {
    constructor(vocabSize = 10000, dModel = 512, nHeads = 8, nLayers = 6) {
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.nHeads = nHeads;
        this.nLayers = nLayers;
        this.dff = 2048; // Feed-forward dimension
        this.dropout = 0.1;
        
        // Initialize model components
        this.tokenEmbedding = this.initializeEmbedding(vocabSize, dModel);
        this.positionalEncoding = new PositionalEncoding(dModel);
        this.encoderLayers = this.initializeEncoderLayers(nLayers);
        this.decoderLayers = this.initializeDecoderLayers(nLayers);
        this.outputProjection = this.initializeProjection(dModel, vocabSize);
        
        // Training state
        this.trainingHistory = [];
        this.lossHistory = [];
        this.optimizer = new AdamOptimizer(dModel);
    }

    // Initialize token embeddings
    initializeEmbedding(vocabSize, dModel) {
        const embedding = [];
        for (let i = 0; i < vocabSize; i++) {
            embedding[i] = [];
            for (let j = 0; j < dModel; j++) {
                embedding[i][j] = (Math.random() - 0.5) * 2 * Math.sqrt(2.0 / dModel);
            }
        }
        return embedding;
    }

    // Initialize encoder layers
    initializeEncoderLayers(nLayers) {
        const layers = [];
        for (let i = 0; i < nLayers; i++) {
            layers.push(new TransformerEncoderLayer(this.dModel, this.nHeads, this.dff, this.dropout));
        }
        return layers;
    }

    // Initialize decoder layers
    initializeDecoderLayers(nLayers) {
        const layers = [];
        for (let i = 0; i < nLayers; i++) {
            layers.push(new TransformerDecoderLayer(this.dModel, this.nHeads, this.dff, this.dropout));
        }
        return layers;
    }

    // Initialize output projection
    initializeProjection(dModel, vocabSize) {
        return (Math.random() - 0.5) * 2 * Math.sqrt(2.0 / dModel) * 
               Array(vocabSize).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    }

    // Forward pass through transformer
    forward(input, mask = null) {
        // Token embedding and positional encoding
        const embedded = this.embedAndEncode(input);
        
        // Pass through encoder layers
        let encoded = embedded;
        for (const layer of this.encoderLayers) {
            encoded = layer.forward(encoded, mask);
        }
        
        return encoded;
    }

    // Decode from encoded representation
    decode(encoded, targetMask = null) {
        let decoded = encoded;
        
        // Pass through decoder layers
        for (const layer of this.decoderLayers) {
            decoded = layer.forward(decoded, targetMask);
        }
        
        // Project to vocabulary
        const output = this.outputProjection.forward(decoded);
        
        return output;
    }

    // Embed tokens and add positional encoding
    embedAndEncode(input) {
        const embedded = this.tokenEmbedding[input];
        const positional = this.positionalEncoding.encode(embedded.length);
        
        return embedded.map((token, i) => 
            token.map((val, j) => val + positional[i][j])
        );
    }

    // Generate text sequence
    generate(inputText, maxLength = 100, temperature = 0.7) {
        // Tokenize input
        const tokens = this.tokenize(inputText);
        
        // Start with input tokens
        let currentTokens = [...tokens];
        const generatedTokens = [];
        
        for (let i = 0; i < maxLength; i++) {
            // Forward pass
            const mask = this.createCausalMask(currentTokens.length);
            const output = this.forward(currentTokens, mask);
            
            // Sample next token
            const nextToken = this.sampleToken(output[output.length - 1], temperature);
            generatedTokens.push(nextToken);
            currentTokens.push(nextToken);
            
            // Stop if EOS token
            if (nextToken === this.vocabSize - 1) break;
        }
        
        // Decode tokens back to text
        return this.detokenize(generatedTokens);
    }

    // Sample token from output distribution
    sampleToken(outputLogits, temperature) {
        // Apply temperature
        const scaledLogits = outputLogits.map(logit => logit / temperature);
        
        // Apply softmax
        const expLogits = scaledLogits.map(Math.exp);
        const sumExp = expLogits.reduce((sum, val) => sum + val, 0);
        const probabilities = expLogits.map(val => val / sumExp);
        
        // Sample based on probabilities
        const random = Math.random();
        let cumulative = 0;
        
        for (let i = 0; i < probabilities.length; i++) {
            cumulative += probabilities[i];
            if (random < cumulative) {
                return i;
            }
        }
        
        return probabilities.length - 1; // Default to last token
    }

    // Create causal attention mask
    createCausalMask(seqLength) {
        const mask = [];
        for (let i = 0; i < seqLength; i++) {
            const row = [];
            for (let j = 0; j < seqLength; j++) {
                row.push(j <= i ? 1 : 0);
            }
            mask.push(row);
        }
        return mask;
    }

    // Simple tokenization
    tokenize(text) {
        // In production, would use proper tokenizer
        return text.toLowerCase().split(/\s+/).map(word => 
            Math.min(this.vocabSize - 2, this.hashWord(word))
        );
    }

    // Simple detokenization
    detokenize(tokens) {
        // In production, would use proper detokenizer
        return tokens.map(token => {
            if (token === this.vocabSize - 1) return '[EOS]';
            return `token_${token}`;
        }).join(' ');
    }

    // Hash word to token ID
    hashWord(word) {
        let hash = 0;
        for (let i = 0; i < word.length; i++) {
            hash = ((hash << 5) - hash) + word.charCodeAt(i);
            hash = hash & hash;
            hash = ((hash << 5) - hash) + word.charCodeAt(i);
        }
        return Math.abs(hash) % (this.vocabSize - 2);
    }

    // Train the transformer
    train(trainingData, validationData, epochs = 10) {
        console.log('Starting transformer training...');
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
            const batchCount = Math.floor(trainingData.length / 32);
            
            // Mini-batch training
            for (let batch = 0; batch < batchCount; batch++) {
                const batch = trainingData.slice(batch * 32, (batch + 1) * 32);
                const batchLoss = this.trainBatch(batch);
                totalLoss += batchLoss;
            }
            
            const avgLoss = totalLoss / batchCount;
            this.lossHistory.push({ epoch, loss: avgLoss });
            
            // Validation
            if (epoch % 2 === 0) {
                const validationLoss = this.validate(validationData);
                console.log(`Epoch ${epoch}: Loss = ${avgLoss.toFixed(4)}, Val Loss = ${validationLoss.toFixed(4)}`);
            }
        }
        
        console.log('Transformer training completed');
        return this.lossHistory;
    }

    // Train on mini-batch
    trainBatch(batch) {
        let batchLoss = 0;
        
        for (const example of batch) {
            const { input, target } = example;
            
            // Forward pass
            const output = this.forward(input);
            
            // Calculate loss
            const loss = this.calculateLoss(output, target);
            batchLoss += loss;
            
            // Backward pass and update
            this.backward(loss);
        }
        
        return batchLoss / batch.length;
    }

    // Calculate cross-entropy loss
    calculateLoss(output, target) {
        const outputLogits = output[output.length - 1]; // Last token logits
        const targetToken = target[target.length - 1];
        
        // Cross-entropy loss
        const probs = this.softmax(outputLogits);
        const targetProb = probs[targetToken];
        const loss = -Math.log(Math.max(targetProb, 1e-8));
        
        return loss;
    }

    // Softmax function
    softmax(logits) {
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(logit => Math.exp(logit - maxLogit));
        const sumExp = expLogits.reduce((sum, val) => sum + val, 0);
        return expLogits.map(val => val / sumExp);
    }

    // Backward pass (simplified)
    backward(loss) {
        // In a real implementation, would compute gradients
        // and update all model parameters
        this.optimizer.step(loss);
    }

    // Validate model
    validate(validationData) {
        let totalLoss = 0;
        
        for (const example of validationData) {
            const { input, target } = example;
            const output = this.forward(input);
            const loss = this.calculateLoss(output, target);
            totalLoss += loss;
        }
        
        return totalLoss / validationData.length;
    }

    // Save model
    saveModel() {
        const model = {
            vocabSize: this.vocabSize,
            dModel: this.dModel,
            nHeads: this.nHeads,
            nLayers: this.nLayers,
            tokenEmbedding: this.tokenEmbedding,
            encoderLayers: this.encoderLayers.map(layer => layer.saveWeights()),
            decoderLayers: this.decoderLayers.map(layer => layer.saveWeights()),
            outputProjection: this.outputProjection,
            trainingHistory: this.trainingHistory,
            lossHistory: this.lossHistory
        };
        
        localStorage.setItem('transformer_model', JSON.stringify(model));
        console.log('Transformer model saved');
        return model;
    }

    // Load model
    loadModel() {
        const savedModel = localStorage.getItem('transformer_model');
        if (savedModel) {
            const model = JSON.parse(savedModel);
            // Restore model parameters
            console.log('Transformer model loaded');
            return true;
        }
        return false;
    }
}

// Transformer Encoder Layer
class TransformerEncoderLayer {
    constructor(dModel, nHeads, dff, dropout) {
        this.dModel = dModel;
        this.nHeads = nHeads;
        this.dff = dff;
        this.dropout = dropout;
        
        this.multiHeadAttention = new MultiHeadAttention(dModel, nHeads);
        this.norm1 = new LayerNorm(dModel);
        this.norm2 = new LayerNorm(dModel);
        this.feedForward = new FeedForward(dModel, dff);
    }

    forward(x, mask = null) {
        // Multi-head self-attention
        const attention = this.multiHeadAttention.forward(x, x, x, mask);
        
        // Add and norm
        const attentionOutput = this.dropout(attention);
        const norm1 = this.norm1.forward(attentionOutput);
        
        // Feed-forward network
        const ffOutput = this.feedForward.forward(norm1);
        
        // Add and norm
        const ffOutputWithDropout = this.dropout(ffOutput);
        const norm2 = this.norm2.forward(ffOutputWithDropout);
        
        // Residual connection
        return x.map((val, i) => val + norm2[i]);
    }

    saveWeights() {
        return {
            multiHeadAttention: this.multiHeadAttention.saveWeights(),
            feedForward: this.feedForward.saveWeights()
        };
    }
}

// Transformer Decoder Layer
class TransformerDecoderLayer {
    constructor(dModel, nHeads, dff, dropout) {
        this.dModel = dModel;
        this.nHeads = nHeads;
        this.dff = dff;
        this.dropout = dropout;
        
        this.multiHeadAttention = new MultiHeadAttention(dModel, nHeads);
        this.norm1 = new LayerNorm(dModel);
        this.norm2 = new LayerNorm(dModel);
        this.norm3 = new LayerNorm(dModel);
        this.feedForward = new FeedForward(dModel, dff);
    }

    forward(x, encoderOutput, targetMask = null) {
        // Multi-head attention with encoder output
        const attention = this.multiHeadAttention.forward(x, encoderOutput, x, targetMask);
        
        // Add and norm
        const attentionOutput = this.dropout(attention);
        const norm1 = this.norm1.forward(attentionOutput);
        
        // Feed-forward network
        const ffOutput = this.feedForward.forward(norm1);
        
        // Add and norm
        const ffOutputWithDropout = this.dropout(ffOutput);
        const norm2 = this.norm2.forward(ffOutputWithDropout);
        
        // Residual connection
        return x.map((val, i) => val + norm2[i]);
    }

    saveWeights() {
        return {
            multiHeadAttention: this.multiHeadAttention.saveWeights(),
            feedForward: this.feedForward.saveWeights()
        };
    }
}

// Multi-Head Attention Mechanism
class MultiHeadAttention {
    constructor(dModel, nHeads) {
        this.dModel = dModel;
        this.nHeads = nHeads;
        this.dK = dModel / nHeads;
        this.dV = dModel / nHeads;
        
        this.wQ = this.initializeWeights(dModel, dModel);
        this.wK = this.initializeWeights(dModel, dModel);
        this.wV = this.initializeWeights(dModel, dModel);
        this.wO = this.initializeWeights(dModel, dModel);
    }

    forward(Q, K, V, mask = null) {
        const batchSize = Q.length;
        const seqLength = Q[0].length;
        
        // Linear projections
        const Q_proj = this.linear(Q, this.wQ);
        const K_proj = this.linear(K, this.wK);
        const V_proj = this.linear(V, this.wV);
        
        // Reshape for multi-head attention
        const Q_heads = this.reshapeForHeads(Q_proj);
        const K_heads = this.reshapeForHeads(K_proj);
        const V_heads = this.reshapeForHeads(V_proj);
        
        // Scaled dot-product attention
        const scores = this.scaledDotProductAttention(Q_heads, K_heads);
        
        // Apply mask if provided
        if (mask) {
            this.applyMask(scores, mask);
        }
        
        // Softmax attention weights
        const attention = this.softmax(scores);
        
        // Apply attention to values
        const context = this.attentionWeights(attention, V_heads);
        
        // Reshape and project
        const context_reshaped = this.reshapeFromHeads(context);
        const output = this.linear(context_reshaped, this.wO);
        
        return output;
    }

    scaledDotProductAttention(Q, K) {
        const dK = K[K.length - 1].length;
        const scores = [];
        
        for (let i = 0; i < Q.length; i++) {
            const batchScores = [];
            for (let j = 0; j < Q[i].length; j++) {
                const score = Q[i][j] * K[i][j] / Math.sqrt(dK);
                batchScores.push(score);
            }
            scores.push(batchScores);
        }
        
        return scores;
    }

    reshapeForHeads(x) {
        const batchSize = x.length;
        const seqLength = x[0].length;
        const dHead = this.dK;
        
        const reshaped = [];
        for (let b = 0; b < batchSize; b++) {
            const batch = [];
            for (let s = 0; s < seqLength; s++) {
                const head = [];
                for (let h = 0; h < this.nHeads; h++) {
                    head.push(x[b][s][h * dHead + h]);
                }
                batch.push(head);
            }
            reshaped.push(batch);
        }
        
        return reshaped;
    }

    reshapeFromHeads(x) {
        const batchSize = x.length;
        const seqLength = x[0].length;
        const dHead = this.dK;
        
        const reshaped = [];
        for (let b = 0; b < batchSize; b++) {
            const batch = [];
            for (let s = 0; s < seqLength; s++) {
                let headVector = [];
                for (let h = 0; h < this.nHeads; h++) {
                    headVector.push(x[b][s][h][h * dHead + h]);
                }
                batch.push(headVector.reduce((sum, val) => sum + val, 0));
            }
            reshaped.push(batch);
        }
        
        return reshaped;
    }

    softmax(scores) {
        const maxScores = scores.map(batch => batch.map(Math.max));
        const expScores = scores.map((batch, b) => 
            batch.map(score => Math.exp(score - maxScores[b]))
        );
        const sumExp = expScores.map((batch, b) => 
            batch.reduce((sum, val) => sum + val, 0)
        );
        
        return expScores.map((batch, b) => 
            batch.map((val, i) => val / sumExp[b][i])
        );
    }

    attentionWeights(attention, V) {
        const batchSize = attention.length;
        const seqLength = attention[0].length;
        const dHead = this.dK;
        
        const weighted = [];
        for (let b = 0; b < batchSize; b++) {
            const batch = [];
            for (let s = 0; s < seqLength; s++) {
                let headVector = [];
                for (let h = 0; h < this.nHeads; h++) {
                    let headVal = 0;
                    for (let s2 = 0; s2 < seqLength; s2++) {
                        headVal += attention[b][s][h][s2] * V[b][s2][h * dHead + h];
                    }
                    headVector.push(headVal);
                }
                batch.push(headVector);
            }
            weighted.push(batch);
        }
        
        return weighted;
    }

    linear(x, weights) {
        return x.map(vector => 
            weights.map(weight => 
                vector.reduce((sum, val, i) => sum + val * weight[i], 0)
            )
        );
    }

    applyMask(scores, mask) {
        for (let b = 0; b < scores.length; b++) {
            for (let i = 0; i < scores[b].length; i++) {
                for (let j = 0; j < scores[b][i].length; j++) {
                    if (mask[b][i][j] === 0) {
                        scores[b][i][j] = -Infinity;
                    }
                }
            }
        }
    }

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

    saveWeights() {
        return {
            wQ: this.wQ,
            wK: this.wK,
            wV: this.wV,
            wO: this.wO
        };
    }
}

// Supporting classes
class PositionalEncoding {
    constructor(dModel) {
        this.dModel = dModel;
        this.maxLen = 5000;
        this.encoding = this.createPositionalEncoding();
    }

    createPositionalEncoding() {
        const encoding = [];
        for (let pos = 0; pos < this.maxLen; pos++) {
            const row = [];
            for (let i = 0; i < this.dModel; i++) {
                if (i % 2 === 0) {
                    row.push(Math.sin(pos / Math.pow(10000, 2 * i / this.dModel)));
                } else {
                    row.push(Math.cos(pos / Math.pow(10000, 2 * i / this.dModel)));
                }
            }
            encoding.push(row);
        }
        return encoding;
    }

    encode(seqLength) {
        return this.encoding.slice(0, seqLength);
    }
}

class LayerNorm {
    constructor(dModel) {
        this.dModel = dModel;
        this.gamma = Array(dModel).fill(1);
        this.beta = Array(dModel).fill(0);
    }

    forward(x) {
        const mean = x.reduce((sum, val) => sum + val, 0) / x.length;
        const variance = x.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / x.length;
        const std = Math.sqrt(variance + 1e-6);
        
        return x.map(val => 
            this.gamma.map((g, i) => g * (val[i] - mean) / std)
        ).map((normVal, i) => normVal[i] + this.beta[i])
        );
    }
}

class FeedForward {
    constructor(dModel, dff) {
        this.dModel = dModel;
        this.dff = dff;
        
        this.w1 = this.initializeWeights(dModel, dff);
        this.b1 = Array(dff).fill(0);
        this.w2 = this.initializeWeights(dff, dModel);
        this.b2 = Array(dModel).fill(0);
    }

    forward(x) {
        // First linear layer with ReLU
        const hidden = x.map(vector => 
            this.w1.map((weights, i) => 
                weights.reduce((sum, w, j) => sum + vector[j] * w[j], 0)
            )
        ).map(sum => Math.max(0, sum + this.b1))
        );
        
        // Second linear layer
        const output = hidden.map(h => 
            this.w2.map((weights, i) => 
                weights.reduce((sum, w, j) => sum + h[j] * w[j], 0)
            )
        ).map(sum => sum + this.b2)
        );
        
        return output;
    }

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

    saveWeights() {
        return {
            w1: this.w1,
            b1: this.b1,
            w2: this.w2,
            b2: this.b2
        };
    }
}

class AdamOptimizer {
    constructor(dModel) {
        this.dModel = dModel;
        this.learningRate = 0.001;
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.epsilon = 1e-8;
        
        this.m = {};
        this.v = {};
    }

    step(loss) {
        // In a real implementation, would compute gradients and update parameters
        console.log(`Adam optimizer step with loss: ${loss}`);
        
        // Simplified parameter update
        Object.keys(this.m).forEach(key => {
            this.m[key] = this.beta1 * this.m[key] + (1 - this.beta1) * loss;
            this.v[key] = this.beta2 * this.v[key] + (1 - this.beta2) * loss;
        });
    }
}

// Export for use in main application
window.TransformerModel = TransformerModel;
window.TransformerEncoderLayer = TransformerEncoderLayer;
window.TransformerDecoderLayer = TransformerDecoderLayer;
window.MultiHeadAttention = MultiHeadAttention;
window.PositionalEncoding = PositionalEncoding;
window.LayerNorm = LayerNorm;
window.FeedForward = FeedForward;
window.AdamOptimizer = AdamOptimizer;
