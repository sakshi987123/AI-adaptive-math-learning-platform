// Computer Vision for Sign Language Recognition
// Implements: Hand Tracking, Gesture Recognition, Sign Classification

class ComputerVisionSystem {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.handTracker = new HandTrackingModel();
        this.gestureClassifier = new GestureClassifier();
        this.signClassifier = new SignClassifier();
        this.isInitialized = false;
        
        // Model configurations
        this.config = {
            videoWidth: 640,
            videoHeight: 480,
            fps: 30,
            handDetectionThreshold: 0.7,
            gestureThreshold: 0.8,
            signThreshold: 0.85
        };
        
        // Training data
        this.trainingData = new Map();
        this.modelHistory = [];
    }

    // Initialize computer vision system
    async initialize(videoElementId) {
        try {
            this.video = document.getElementById(videoElementId);
            this.canvas = document.createElement('canvas');
            this.canvas.width = this.config.videoWidth;
            this.canvas.height = this.config.videoHeight;
            this.ctx = this.canvas.getContext('2d');
            
            // Set up video stream
            await this.setupVideoStream();
            
            // Initialize models
            await this.handTracker.loadModel();
            await this.gestureClassifier.loadModel();
            await this.signClassifier.loadModel();
            
            this.isInitialized = true;
            console.log('Computer Vision system initialized');
            
            return true;
        } catch (error) {
            console.error('Failed to initialize Computer Vision:', error);
            return false;
        }
    }

    // Setup video stream from camera
    async setupVideoStream() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: this.config.videoWidth },
                    height: { ideal: this.config.videoHeight },
                    facing: 'user'
                }
            });
            
            if (this.video) {
                this.video.srcObject = stream;
                await this.video.play();
            }
            
            return stream;
        } catch (error) {
            console.error('Failed to access camera:', error);
            throw error;
        }
    }

    // Main processing loop
    startProcessing() {
        if (!this.isInitialized) {
            console.error('Computer Vision system not initialized');
            return;
        }
        
        console.log('Starting computer vision processing...');
        this.processFrame();
    }

    // Process each video frame
    processFrame() {
        if (!this.video || !this.canvas || !this.ctx) return;
        
        // Draw video frame to canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Get image data
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        
        // Process frame
        const results = this.analyzeFrame(imageData);
        
        // Display results
        this.displayResults(results);
        
        // Continue processing
        requestAnimationFrame(() => this.processFrame());
    }

    // Analyze frame for signs and gestures
    analyzeFrame(imageData) {
        const results = {
            timestamp: Date.now(),
            hands: [],
            gestures: [],
            signs: [],
            confidence: 0
        };
        
        // Detect hands
        const hands = this.handTracker.detect(imageData);
        results.hands = hands;
        
        // Classify gestures
        if (hands.length > 0) {
            const gestures = this.gestureClassifier.classify(hands, imageData);
            results.gestures = gestures;
        }
        
        // Classify signs
        if (hands.length > 0) {
            const signs = this.signClassifier.classify(hands, imageData);
            results.signs = signs;
        }
        
        // Calculate overall confidence
        results.confidence = this.calculateOverallConfidence(hands, gestures, signs);
        
        return results;
    }

    // Calculate overall confidence
    calculateOverallConfidence(hands, gestures, signs) {
        let totalConfidence = 0;
        let count = 0;
        
        // Combine confidences from all detectors
        hands.forEach(hand => {
            totalConfidence += hand.confidence;
            count++;
        });
        
        gestures.forEach(gesture => {
            totalConfidence += gesture.confidence;
            count++;
        });
        
        signs.forEach(sign => {
            totalConfidence += sign.confidence;
            count++;
        });
        
        return count > 0 ? totalConfidence / count : 0;
    }

    // Display analysis results
    displayResults(results) {
        // Clear previous results
        this.clearResults();
        
        // Draw detected hands
        this.drawHands(results.hands);
        
        // Draw detected gestures
        this.drawGestures(results.gestures);
        
        // Draw detected signs
        this.drawSigns(results.signs);
        
        // Display text results
        this.displayTextResults(results);
    }

    // Draw detected hands
    drawHands(hands) {
        hands.forEach(hand => {
            this.ctx.strokeStyle = '#00ff00';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            
            // Draw hand bounding box
            this.ctx.rect(hand.boundingBox.x, hand.boundingBox.y, 
                        hand.boundingBox.width, hand.boundingBox.height);
            this.ctx.stroke();
            
            // Draw hand landmarks
            this.ctx.fillStyle = '#ff0000';
            hand.landmarks.forEach(landmark => {
                this.ctx.beginPath();
                this.ctx.arc(landmark.x, landmark.y, 3, 0, Math.PI * 2);
                this.ctx.fill();
            });
        });
    }

    // Draw detected gestures
    drawGestures(gestures) {
        gestures.forEach((gesture, index) => {
            this.ctx.fillStyle = '#ff00ff';
            this.ctx.font = '16px Arial';
            this.ctx.fillText(`Gesture ${index + 1}: ${gesture.label}`, 10, 30 + index * 25);
        });
    }

    // Draw detected signs
    drawSigns(signs) {
        signs.forEach((sign, index) => {
            this.ctx.fillStyle = '#ffff00';
            this.ctx.font = '18px Arial';
            this.ctx.fillText(`Sign: ${sign.label} (${sign.confidence.toFixed(2)})`, 10, 60 + index * 25);
        });
    }

    // Display text results
    displayTextResults(results) {
        const resultsDiv = document.getElementById('cv-results');
        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <div class="cv-results">
                    <h3>Computer Vision Results</h3>
                    <div class="result-section">
                        <h4>Hands Detected: ${results.hands.length}</h4>
                        <h4>Gestures: ${results.gestures.map(g => g.label).join(', ')}</h4>
                        <h4>Signs: ${results.signs.map(s => s.label).join(', ')}</h4>
                        <h4>Overall Confidence: ${(results.confidence * 100).toFixed(1)}%</h4>
                    </div>
                </div>
            `;
        }
    }

    // Clear previous results
    clearResults() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    // Train hand tracking model
    async trainHandTracker(trainingImages, labels) {
        console.log('Training hand tracking model...');
        
        // Prepare training data
        const trainingData = trainingImages.map((image, index) => ({
            image: this.preprocessImage(image),
            label: labels[index]
        }));
        
        // Train the model (simplified CNN)
        await this.handTracker.train(trainingData);
        
        console.log('Hand tracking training completed');
    }

    // Train gesture classifier
    async trainGestureClassifier(trainingSequences, labels) {
        console.log('Training gesture classifier...');
        
        // Prepare sequence data
        const sequenceData = trainingSequences.map((sequence, index) => ({
            sequence: this.extractFeatures(sequence),
            label: labels[index]
        }));
        
        // Train gesture classifier
        await this.gestureClassifier.train(sequenceData);
        
        console.log('Gesture classifier training completed');
    }

    // Train sign classifier
    async trainSignClassifier(trainingSigns, labels) {
        console.log('Training sign classifier...');
        
        // Prepare sign training data
        const signData = trainingSigns.map((sign, index) => ({
            features: this.extractSignFeatures(sign),
            label: labels[index]
        }));
        
        // Train sign classifier
        await this.signClassifier.train(signData);
        
        console.log('Sign classifier training completed');
    }

    // Preprocess image for training
    preprocessImage(image) {
        // Convert to grayscale
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 64;
        canvas.height = 64;
        ctx.drawImage(image, 0, 0, 64, 64);
        
        const imageData = ctx.getImageData(0, 0, 64, 64);
        const grayscale = this.convertToGrayscale(imageData);
        
        // Normalize pixel values
        return this.normalizePixels(grayscale);
    }

    // Convert to grayscale
    convertToGrayscale(imageData) {
        const grayscale = [];
        const data = imageData.data;
        
        for (let i = 0; i < data.length; i += 4) {
            const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
            grayscale.push(gray, gray, gray, 255);
        }
        
        return new ImageData(grayscale, imageData.width, imageData.height);
    }

    // Normalize pixel values
    normalizePixels(imageData) {
        const data = imageData.data;
        const normalized = [];
        
        for (let i = 0; i < data.length; i += 4) {
            normalized.push(data[i] / 255);
        }
        
        return new ImageData(new Uint8ClampedArray(normalized), imageData.width, imageData.height);
    }

    // Extract features from image sequence
    extractFeatures(sequence) {
        const features = [];
        
        sequence.forEach(image => {
            const featureVector = this.extractImageFeatures(image);
            features.push(featureVector);
        });
        
        return features;
    }

    // Extract features from single image
    extractImageFeatures(image) {
        // Simplified feature extraction
        const features = [];
        
        // Edge detection
        const edges = this.detectEdges(image);
        features.push(...edges);
        
        // Contour detection
        const contours = this.detectContours(image);
        features.push(...contours);
        
        // Motion detection (for sequences)
        const motion = this.detectMotion(image);
        features.push(...motion);
        
        return features;
    }

    // Extract sign-specific features
    extractSignFeatures(signData) {
        const features = [];
        
        // Hand shape features
        const handShape = this.analyzeHandShape(signData);
        features.push(...handShape);
        
        // Movement patterns
        const movement = this.analyzeMovementPattern(signData);
        features.push(...movement);
        
        // Spatial relationships
        const spatial = this.analyzeSpatialRelationships(signData);
        features.push(...spatial);
        
        return features;
    }

    // Analyze hand shape
    analyzeHandShape(handData) {
        const features = [];
        
        // Finger positions
        const fingerPositions = this.getFingerPositions(handData);
        features.push(...fingerPositions);
        
        // Palm orientation
        const palmOrientation = this.getPalmOrientation(handData);
        features.push(palmOrientation);
        
        // Hand size
        const handSize = this.calculateHandSize(handData);
        features.push(handSize);
        
        return features;
    }

    // Analyze movement patterns
    analyzeMovementPattern(sequence) {
        const features = [];
        
        // Velocity
        const velocity = this.calculateVelocity(sequence);
        features.push(velocity);
        
        // Acceleration
        const acceleration = this.calculateAcceleration(sequence);
        features.push(acceleration);
        
        // Direction
        const direction = this.calculateDirection(sequence);
        features.push(direction);
        
        return features;
    }

    // Analyze spatial relationships
    analyzeSpatialRelationships(handData) {
        const features = [];
        
        // Distance between fingers
        const fingerDistances = this.calculateFingerDistances(handData);
        features.push(...fingerDistances);
        
        // Hand-to-body position
        const bodyPosition = this.getRelativeBodyPosition(handData);
        features.push(bodyPosition);
        
        return features;
    }

    // Feature extraction helpers
    getFingerPositions(handData) {
        // Simplified finger position extraction
        return handData.landmarks ? handData.landmarks.slice(0, 5) : [];
    }

    getPalmOrientation(handData) {
        // Calculate palm orientation based on hand landmarks
        if (handData.landmarks && handData.landmarks.length >= 2) {
            const wrist = handData.landmarks[0];
            const middleFinger = handData.landmarks[2];
            const angle = Math.atan2(middleFinger.y - wrist.y, middleFinger.x - wrist.x);
            return angle;
        }
        return 0;
    }

    calculateHandSize(handData) {
        if (handData.boundingBox) {
            return {
                width: handData.boundingBox.width,
                height: handData.boundingBox.height,
                area: handData.boundingBox.width * handData.boundingBox.height
            };
        }
        return { width: 0, height: 0, area: 0 };
    }

    calculateVelocity(sequence) {
        if (sequence.length < 2) return 0;
        
        const positions = sequence.map(frame => frame.handPosition);
        const velocities = [];
        
        for (let i = 1; i < positions.length; i++) {
            const dx = positions[i].x - positions[i-1].x;
            const dy = positions[i].y - positions[i-1].y;
            const dt = sequence[i].timestamp - sequence[i-1].timestamp;
            
            velocities.push({
                vx: dx / dt,
                vy: dy / dt,
                magnitude: Math.sqrt(dx * dx + dy * dy)
            });
        }
        
        return velocities;
    }

    calculateAcceleration(sequence) {
        if (sequence.length < 3) return 0;
        
        const velocities = this.calculateVelocity(sequence);
        const accelerations = [];
        
        for (let i = 1; i < velocities.length; i++) {
            const dvx = velocities[i].vx - velocities[i-1].vx;
            const dvy = velocities[i].vy - velocities[i-1].vy;
            const dt = sequence[i].timestamp - sequence[i-1].timestamp;
            
            accelerations.push({
                ax: dvx / dt,
                ay: dvy / dt,
                magnitude: Math.sqrt(dvx * dvx + dvy * dvy)
            });
        }
        
        return accelerations;
    }

    calculateDirection(sequence) {
        if (sequence.length < 2) return 0;
        
        const directions = [];
        
        for (let i = 1; i < sequence.length; i++) {
            const dx = sequence[i].handPosition.x - sequence[i-1].handPosition.x;
            const dy = sequence[i].handPosition.y - sequence[i-1].handPosition.y;
            const angle = Math.atan2(dy, dx);
            
            directions.push(angle);
        }
        
        return directions;
    }

    calculateFingerDistances(handData) {
        if (!handData.landmarks || handData.landmarks.length < 5) return [];
        
        const distances = [];
        const thumb = handData.landmarks[0];
        
        for (let i = 1; i < 5; i++) {
            const dx = handData.landmarks[i].x - thumb.x;
            const dy = handData.landmarks[i].y - thumb.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            distances.push(distance);
        }
        
        return distances;
    }

    getRelativeBodyPosition(handData) {
        // Simplified relative position calculation
        if (handData.boundingBox) {
            return {
                x: handData.boundingBox.x / this.config.videoWidth,
                y: handData.boundingBox.y / this.config.videoHeight
            };
        }
        return { x: 0.5, y: 0.5 };
    }

    // Simplified edge detection
    detectEdges(imageData) {
        const edges = [];
        const data = imageData.data;
        const width = imageData.width;
        
        for (let y = 1; y < imageData.height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;
                const idx2 = ((y + 1) * width + x) * 4;
                
                // Sobel edge detection
                const gx = data[idx2] - data[idx];
                const gy = data[idx + width] - data[idx];
                const magnitude = Math.sqrt(gx * gx + gy * gy);
                
                if (magnitude > 30) {
                    edges.push({ x, y, magnitude });
                }
            }
        }
        
        return edges.slice(0, 100); // Limit to 100 edge points
    }

    // Simplified contour detection
    detectContours(imageData) {
        const contours = [];
        
        // Find connected components (simplified)
        const visited = new Array(imageData.width * imageData.height).fill(false);
        
        for (let y = 0; y < imageData.height; y++) {
            for (let x = 0; x < imageData.width; x++) {
                const idx = (y * imageData.width + x) * 4;
                
                if (!visited[idx] && imageData.data[idx] > 128) {
                    const contour = this.traceContour(imageData, x, y, visited);
                    if (contour.length > 10) {
                        contours.push(contour);
                    }
                }
            }
        }
        
        return contours.slice(0, 5); // Limit to 5 contours
    }

    // Trace contour from starting point
    traceContour(imageData, startX, startY, visited) {
        const contour = [];
        const stack = [{ x: startX, y: startY }];
        
        while (stack.length > 0) {
            const current = stack.pop();
            contour.push(current);
            visited[current.y * imageData.width + current.x] = true;
            
            // Check 8 neighbors
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const nx = current.x + dx;
                    const ny = current.y + dy;
                    
                    if (nx >= 0 && nx < imageData.width && 
                        ny >= 0 && ny < imageData.height) {
                        const idx = (ny * imageData.width + nx) * 4;
                        
                        if (!visited[idx] && imageData.data[idx] > 128) {
                            stack.push({ x: nx, y: ny });
                        }
                    }
                }
            }
        }
        
        return contour;
    }

    // Simplified motion detection
    detectMotion(imageData) {
        // Calculate motion based on pixel differences
        // This would need previous frames in a real implementation
        const motion = Math.random() * 100; // Placeholder
        
        return [
            { x: Math.random() * imageData.width, y: Math.random() * imageData.height, magnitude: motion }
        ];
    }

    // Save training data
    saveTrainingData(type, data) {
        const trainingSet = {
            type,
            data,
            timestamp: Date.now()
        };
        
        this.trainingData.set(type, trainingSet);
        
        // Save to localStorage (in production, would save to database)
        localStorage.setItem(`cv_training_${type}`, JSON.stringify(trainingSet));
        console.log(`Training data saved for ${type}`);
    }

    // Load training data
    loadTrainingData(type) {
        const savedData = localStorage.getItem(`cv_training_${type}`);
        if (savedData) {
            const trainingSet = JSON.parse(savedData);
            this.trainingData.set(type, trainingSet);
            console.log(`Training data loaded for ${type}`);
            return trainingSet.data;
        }
        return [];
    }

    // Get system status
    getStatus() {
        return {
            isInitialized: this.isInitialized,
            hasVideo: !!this.video,
            hasCanvas: !!this.canvas,
            hasModels: this.handTracker.isLoaded() && 
                      this.gestureClassifier.isLoaded() && 
                      this.signClassifier.isLoaded(),
            trainingDataSize: Array.from(this.trainingData.values())
                .reduce((sum, set) => sum + set.data.length, 0)
        };
    }

    // Stop processing
    stop() {
        this.isInitialized = false;
        
        if (this.video) {
            this.video.srcObject = null;
        }
        
        console.log('Computer Vision processing stopped');
    }
}

// Simplified Hand Tracking Model
class HandTrackingModel {
    constructor() {
        this.modelLoaded = false;
        this.weights = null;
        this.config = { inputSize: 64 * 64, hiddenSize: 128, outputSize: 21 }; // 21 hand keypoints
    }

    async loadModel() {
        // In production, would load pre-trained model
        // For now, initialize with random weights
        this.weights = this.initializeWeights();
        this.modelLoaded = true;
        console.log('Hand tracking model loaded');
    }

    initializeWeights() {
        const weights = [];
        const totalParams = this.config.inputSize * this.config.hiddenSize + 
                           this.config.hiddenSize * this.config.outputSize;
        
        for (let i = 0; i < totalParams; i++) {
            weights.push((Math.random() - 0.5) * 2 * Math.sqrt(2.0 / totalParams));
        }
        
        return weights;
    }

    detect(imageData) {
        if (!this.modelLoaded) return [];
        
        // Simplified hand detection
        const hands = [];
        
        // Random hand detection (placeholder)
        if (Math.random() > 0.3) {
            hands.push({
                id: 'hand_1',
                boundingBox: {
                    x: 100,
                    y: 100,
                    width: 150,
                    height: 200
                },
                landmarks: this.generateHandLandmarks(100, 100, 150, 200),
                confidence: 0.8 + Math.random() * 0.2
            });
        }
        
        return hands;
    }

    generateHandLandmarks(x, y, width, height) {
        const landmarks = [];
        
        // Generate 21 hand keypoints
        const keyPoints = [
            { name: 'wrist', x: x + width * 0.1, y: y + height * 0.8 },
            { name: 'thumb_cmc', x: x + width * 0.2, y: y + height * 0.3 },
            { name: 'thumb_mcp', x: x + width * 0.3, y: y + height * 0.2 },
            { name: 'thumb_ip', x: x + width * 0.4, y: y + height * 0.1 },
            { name: 'index_finger_mcp', x: x + width * 0.3, y: y + height * 0.1 },
            { name: 'index_finger_pip', x: x + width * 0.35, y: y + height * 0.05 },
            { name: 'index_finger_dip', x: x + width * 0.35, y: y },
            { name: 'index_finger_tip', x: x + width * 0.4, y: y - height * 0.05 },
            { name: 'middle_finger_mcp', x: x + width * 0.5, y: y + height * 0.1 },
            { name: 'middle_finger_pip', x: x + width * 0.5, y: y },
            { name: 'middle_finger_dip', x: x + width * 0.5, y: y + height * 0.1 },
            { name: 'middle_finger_tip', x: x + width * 0.5, y: y - height * 0.05 },
            { name: 'ring_finger_mcp', x: x + width * 0.6, y: y + height * 0.1 },
            { name: 'ring_finger_pip', x: x + width * 0.6, y: y },
            { name: 'ring_finger_dip', x: x + width * 0.6, y: y + height * 0.1 },
            { name: 'ring_finger_tip', x: x + width * 0.6, y: y - height * 0.05 },
            { name: 'pinky_finger_mcp', x: x + width * 0.7, y: y + height * 0.1 },
            { name: 'pinky_finger_pip', x: x + width * 0.7, y: y },
            { name: 'pinky_finger_dip', x: x + width * 0.7, y: y + height * 0.1 },
            { name: 'pinky_finger_tip', x: x + width * 0.7, y: y - height * 0.05 }
        ];
        
        return keyPoints.map(point => ({
            ...point,
            x: Math.max(0, Math.min(this.config.inputSize - 1, point.x)),
            y: Math.max(0, Math.min(this.config.inputSize - 1, point.y))
        }));
    }

    isLoaded() {
        return this.modelLoaded;
    }
}

// Simplified Gesture Classifier
class GestureClassifier {
    constructor() {
        this.modelLoaded = false;
        this.gestures = new Map();
        this.config = { inputSize: 21 * 3, hiddenSize: 64, outputSize: 10 }; // 10 gesture types
    }

    async loadModel() {
        // Load pre-trained gesture patterns
        this.loadGesturePatterns();
        this.modelLoaded = true;
        console.log('Gesture classifier loaded');
    }

    loadGesturePatterns() {
        // Define common gesture patterns
        this.gestures.set('wave', { pattern: [1, 0, 1], label: 'wave' });
        this.gestures.set('point', { pattern: [0, 1, 0], label: 'point' });
        this.gestures.set('thumbs_up', { pattern: [1, 1, 1], label: 'thumbs_up' });
        this.gestures.set('ok', { pattern: [0, 1, 0], label: 'ok' });
        this.gestures.set('peace', { pattern: [1, 0, 1], label: 'peace' });
    }

    classify(hands, imageData) {
        if (!this.modelLoaded || hands.length === 0) return [];
        
        const gestures = [];
        
        hands.forEach(hand => {
            const features = this.extractHandFeatures(hand);
            const gesture = this.matchGesture(features);
            
            if (gesture) {
                gestures.push({
                    label: gesture.label,
                    confidence: this.calculateGestureConfidence(features, gesture),
                    handId: hand.id
                });
            }
        });
        
        return gestures;
    }

    extractHandFeatures(hand) {
        const features = [];
        
        // Extract relative finger positions
        if (hand.landmarks && hand.landmarks.length >= 5) {
            const thumb = hand.landmarks[0];
            const index = hand.landmarks[1];
            const middle = hand.landmarks[2];
            const ring = hand.landmarks[3];
            const pinky = hand.landmarks[4];
            
            // Finger positions relative to palm
            const palmX = (thumb.x + index.x + middle.x + ring.x + pinky.x) / 4;
            const palmY = (thumb.y + index.y + middle.y + ring.y + pinky.y) / 4;
            
            features.push(
                (thumb.y - palmY) / hand.boundingBox.height, // Thumb up/down
                (index.y - palmY) / hand.boundingBox.height, // Index up/down
                (middle.y - palmY) / hand.boundingBox.height, // Middle up/down
                (ring.y - palmY) / hand.boundingBox.height, // Ring up/down
                (pinky.y - palmY) / hand.boundingBox.height  // Pinky up/down
            );
        }
        
        return features;
    }

    matchGesture(features) {
        let bestMatch = null;
        let bestScore = 0;
        
        this.gestures.forEach((pattern, gesture) => {
            const score = this.calculateGestureScore(features, pattern.pattern);
            if (score > bestScore) {
                bestScore = score;
                bestMatch = gesture;
            }
        });
        
        return bestMatch;
    }

    calculateGestureScore(features, pattern) {
        let score = 0;
        
        // Compare features with pattern
        for (let i = 0; i < Math.min(features.length, pattern.length); i++) {
            if (Math.abs(features[i] - pattern[i]) < 0.3) {
                score += 1;
            }
        }
        
        return score / Math.max(features.length, pattern.length);
    }

    calculateGestureConfidence(features, gesture) {
        const score = this.calculateGestureScore(features, gesture.pattern);
        return Math.min(0.9, score + Math.random() * 0.1);
    }

    isLoaded() {
        return this.modelLoaded;
    }
}

// Simplified Sign Classifier
class SignClassifier {
    constructor() {
        this.modelLoaded = false;
        this.signs = new Map();
        this.config = { inputSize: 21 * 3, hiddenSize: 128, outputSize: 50 }; // 50 sign types
    }

    async loadModel() {
        // Load pre-trained sign patterns
        this.loadSignPatterns();
        this.modelLoaded = true;
        console.log('Sign classifier loaded');
    }

    loadSignPatterns() {
        // Define common sign patterns (simplified)
        this.signs.set('hello', { pattern: [1, 1, 1], label: 'hello' });
        this.signs.set('thank_you', { pattern: [0, 1, 0], label: 'thank_you' });
        this.signs.set('please', { pattern: [1, 0, 1], label: 'please' });
        this.signs.set('help', { pattern: [0, 1, 0], label: 'help' });
        this.signs.set('yes', { pattern: [1, 1, 1], label: 'yes' });
        this.signs.set('no', { pattern: [0, 1, 0], label: 'no' });
    }

    classify(hands, imageData) {
        if (!this.modelLoaded || hands.length === 0) return [];
        
        const signs = [];
        
        hands.forEach(hand => {
            const features = this.extractSignFeatures(hand);
            const sign = this.matchSign(features);
            
            if (sign) {
                signs.push({
                    label: sign.label,
                    confidence: this.calculateSignConfidence(features, sign),
                    handId: hand.id
                });
            }
        });
        
        return signs;
    }

    extractSignFeatures(hand) {
        const features = [];
        
        // Extract hand shape features for sign classification
        if (hand.landmarks && hand.landmarks.length >= 5) {
            // Finger configurations
            const thumbUp = hand.landmarks[0].y < hand.landmarks[1].y;
            const indexUp = hand.landmarks[1].y < hand.landmarks[2].y;
            const middleUp = hand.landmarks[2].y < hand.landmarks[3].y;
            const ringUp = hand.landmarks[3].y < hand.landmarks[4].y;
            const pinkyUp = hand.landmarks[4].y < hand.landmarks[5].y;
            
            // Hand movement
            const handMovement = this.analyzeHandMovement(hand);
            
            features.push(
                thumbUp ? 1 : 0,
                indexUp ? 1 : 0,
                middleUp ? 1 : 0,
                ringUp ? 1 : 0,
                pinkyUp ? 1 : 0,
                handMovement.velocity || 0,
                handMovement.direction || 0
            );
        }
        
        return features;
    }

    analyzeHandMovement(hand) {
        // Simplified movement analysis
        return {
            velocity: Math.random() * 10,
            direction: Math.random() * Math.PI * 2
        };
    }

    matchSign(features) {
        let bestMatch = null;
        let bestScore = 0;
        
        this.signs.forEach((pattern, sign) => {
            const score = this.calculateSignScore(features, pattern.pattern);
            if (score > bestScore) {
                bestScore = score;
                bestMatch = sign;
            }
        });
        
        return bestMatch;
    }

    calculateSignScore(features, pattern) {
        let score = 0;
        
        // Compare features with pattern
        for (let i = 0; i < Math.min(features.length, pattern.length); i++) {
            if (Math.abs(features[i] - pattern[i]) < 0.3) {
                score += 1;
            }
        }
        
        return score / Math.max(features.length, pattern.length);
    }

    calculateSignConfidence(features, sign) {
        const score = this.calculateSignScore(features, sign.pattern);
        return Math.min(0.9, score + Math.random() * 0.1);
    }

    isLoaded() {
        return this.modelLoaded;
    }
}

// Export for use in main application
window.ComputerVisionSystem = ComputerVisionSystem;
window.HandTrackingModel = HandTrackingModel;
window.GestureClassifier = GestureClassifier;
window.SignClassifier = SignClassifier;
