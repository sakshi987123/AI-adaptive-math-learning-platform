// Natural Language Processing for Text Analysis
// Implements: Text Parsing, Sentiment Analysis, Entity Recognition

class NLPProcessor {
    constructor() {
        this.vocabulary = new Map();
        this.entities = new Map();
        this.sentimentModel = new SentimentAnalyzer();
        this.textClassifier = new TextClassifier();
        this.languageModel = new LanguageModel();
    }

    // Main text processing pipeline
    processText(text, options = {}) {
        const results = {
            original: text,
            tokens: this.tokenize(text),
            entities: this.extractEntities(text),
            sentiment: this.analyzeSentiment(text),
            classification: this.classifyText(text),
            language: this.detectLanguage(text),
            complexity: this.analyzeComplexity(text),
            keyPhrases: this.extractKeyPhrases(text)
        };
        
        return results;
    }

    // Tokenization
    tokenize(text) {
        return text.toLowerCase()
            .split(/\s+/)
            .filter(token => token.length > 0)
            .map(token => this.normalizeToken(token));
    }

    normalizeToken(token) {
        return token
            .replace(/[^\w\s]/g, '')
            .replace(/[^\w\s]/g, '')
            .toLowerCase();
    }

    // Entity extraction
    extractEntities(text) {
        const entities = {
            numbers: this.extractNumbers(text),
            mathematical: this.extractMathEntities(text),
            educational: this.extractEducationalEntities(text),
            temporal: this.extractTemporalEntities(text)
        };
        
        return entities;
    }

    extractNumbers(text) {
        const numbers = text.match(/\d+\.?\d*/g) || [];
        return numbers.map(num => parseFloat(num));
    }

    extractMathEntities(text) {
        const mathTerms = [
            'triangle', 'pythagoras', 'hypotenuse', 'angle', 'formula',
            'equation', 'variable', 'coefficient', 'exponent', 'root',
            'algebra', 'geometry', 'trigonometry', 'calculus', 'theorem',
            'proof', 'calculation', 'solution', 'graph', 'function', 'derivative'
        ];
        
        const found = [];
        mathTerms.forEach(term => {
            if (text.toLowerCase().includes(term)) {
                found.push({ type: 'mathematical', value: term });
            }
        });
        
        return found;
    }

    extractEducationalEntities(text) {
        const eduTerms = [
            'lesson', 'student', 'teacher', 'quiz', 'test', 'exam', 'homework',
            'practice', 'exercise', 'problem', 'solution', 'concept', 'theory',
            'principle', 'example', 'definition', 'explanation', 'instruction'
        ];
        
        const found = [];
        eduTerms.forEach(term => {
            if (text.toLowerCase().includes(term)) {
                found.push({ type: 'educational', value: term });
            }
        });
        
        return found;
    }

    extractTemporalEntities(text) {
        const temporal = [];
        
        // Time expressions
        const timePatterns = [
            { pattern: /\b(morning|afternoon|evening|night)\b/i, type: 'time_of_day' },
            { pattern: /\b(today|tomorrow|yesterday|now|current)\b/i, type: 'relative_time' },
            { pattern: /\b(january|february|march|april|may|june|july|august|september|october|november|december)\b/i, type: 'month' },
            { pattern: /\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/i, type: 'day_of_week' },
            { pattern: /\b(week|month|year|hour|minute|second)\b/i, type: 'time_unit' }
        ];
        
        timePatterns.forEach(({ pattern, type }) => {
            const match = text.match(pattern);
            if (match) {
                temporal.push({ type, value: match[0] });
            }
        });
        
        return temporal;
    }

    // Sentiment analysis
    analyzeSentiment(text) {
        const positive = ['good', 'excellent', 'great', 'amazing', 'helpful', 'clear', 'understand',
                        'learned', 'mastered', 'completed', 'successful', 'easy', 'fun'];
        const negative = ['difficult', 'confusing', 'hard', 'struggling', 'stuck', 'frustrated',
                        'boring', 'unclear', 'wrong', 'error', 'fail', 'bad'];
        
        const words = this.tokenize(text);
        let score = 0;
        
        words.forEach(word => {
            if (positive.includes(word)) score += 1;
            if (negative.includes(word)) score -= 1;
        });
        
        const normalizedScore = Math.max(-1, Math.min(1, score / words.length));
        
        return {
            score: normalizedScore,
            sentiment: this.classifySentiment(normalizedScore),
            confidence: Math.abs(normalizedScore)
        };
    }

    classifySentiment(score) {
        if (score > 0.3) return 'positive';
        if (score < -0.3) return 'negative';
        return 'neutral';
    }

    // Text classification
    classifyText(text) {
        const features = this.extractTextFeatures(text);
        const category = this.predictCategory(features);
        
        return {
            category,
            confidence: this.calculateClassificationConfidence(features, category)
        };
    }

    extractTextFeatures(text) {
        const words = this.tokenize(text);
        
        return {
            wordCount: words.length,
            avgWordLength: words.reduce((sum, word) => sum + word.length, 0) / words.length,
            uniqueWords: new Set(words).size,
            mathTerms: words.filter(word => this.isMathTerm(word)).length,
            questionWords: words.filter(word => this.isQuestionWord(word)).length,
            length: text.length,
            complexity: this.calculateTextComplexity(words)
        };
    }

    isMathTerm(word) {
        const mathTerms = ['triangle', 'angle', 'formula', 'equation', 'variable', 'solve',
                       'calculate', 'theorem', 'proof', 'geometry', 'algebra'];
        return mathTerms.includes(word.toLowerCase());
    }

    isQuestionWord(word) {
        const questionWords = ['what', 'how', 'why', 'when', 'where', 'who', 'which',
                           'can', 'could', 'would', 'should', 'is', 'are', 'do',
                           'does', 'did', 'will', 'may', 'might', 'explain'];
        return questionWords.includes(word.toLowerCase());
    }

    calculateTextComplexity(words) {
        const avgLength = words.reduce((sum, word) => sum + word.length, 0) / words.length;
        const longWords = words.filter(word => word.length > 8).length;
        const uniqueRatio = new Set(words).size / words.length;
        
        return {
            avgLength,
            longWordRatio: longWords / words.length,
            uniqueRatio,
            complexity: avgLength > 6 ? 'high' : avgLength > 4 ? 'medium' : 'low'
        };
    }

    predictCategory(features) {
        // Simple rule-based classification
        if (features.mathTerms > 2) return 'mathematical';
        if (features.questionWords > 1) return 'question';
        if (features.wordCount < 10) return 'simple';
        if (features.avgWordLength > 8) return 'complex';
        return 'general';
    }

    calculateClassificationConfidence(features, category) {
        let confidence = 0.5;
        
        if (category === 'mathematical' && features.mathTerms > 3) confidence += 0.3;
        if (category === 'question' && features.questionWords > 2) confidence += 0.2;
        if (category === 'complex' && features.complexity === 'high') confidence += 0.2;
        
        return Math.min(0.9, confidence);
    }

    // Language detection
    detectLanguage(text) {
        const englishWords = ['the', 'and', 'is', 'to', 'of', 'in', 'that', 'have', 'for'];
        const words = this.tokenize(text);
        
        const englishCount = words.filter(word => englishWords.includes(word)).length;
        const englishRatio = englishCount / words.length;
        
        return {
            language: englishRatio > 0.6 ? 'english' : 'unknown',
            confidence: englishRatio
        };
    }

    // Key phrase extraction
    extractKeyPhrases(text) {
        const phrases = [];
        
        // Common educational phrases
        const phrasePatterns = [
            { pattern: /how to\s+calculate/i, type: 'instruction' },
            { pattern: /what is\s+a\s+/i, type: 'definition' },
            { pattern: /step by step/i, type: 'process' },
            { pattern: /for example/i, type: 'example' },
            { pattern: /in other words/i, type: 'clarification' },
            { pattern: /the main goal is/i, type: 'objective' }
        ];
        
        phrasePatterns.forEach(({ pattern, type }) => {
            const match = text.match(pattern);
            if (match) {
                phrases.push({ type, phrase: match[0] });
            }
        });
        
        return phrases;
    }

    // Advanced text processing
    summarizeText(text, maxLength = 100) {
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        
        if (sentences.length <= 3) {
            return text.substring(0, maxLength);
        }
        
        // Extract key sentences
        const keySentences = sentences.slice(0, 3);
        const summary = keySentences.join('. ');
        
        return {
            summary,
            originalLength: text.length,
            compressionRatio: summary.length / text.length
        };
    }

    // Generate educational content
    generateExplanation(concept, difficulty = 'medium') {
        const templates = {
            easy: [
                `Let's learn about ${concept}. First, we'll look at the basics.`,
                `What is ${concept}? It's actually quite simple once you understand it.`
            ],
            medium: [
                `Building on our previous knowledge, let's explore ${concept} in more detail.`,
                `To understand ${concept}, we need to consider several key aspects.`
            ],
            hard: [
                `Now we'll dive deep into ${concept} with advanced concepts and applications.`,
                `Let's analyze ${concept} from multiple perspectives and derive complex relationships.`
            ]
        };
        
        const template = templates[difficulty] || templates.medium;
        const variations = this.generateVariations(template);
        
        return {
            main: template,
            variations,
            difficulty
        };
    }

    generateVariations(baseText) {
        const variations = [];
        
        // Create different versions of the explanation
        variations.push({
            type: 'visual',
            text: baseText + ' We can visualize this with diagrams.'
        });
        
        variations.push({
            type: 'step-by-step',
            text: baseText + ' Let's break this down into clear steps.'
        });
        
        variations.push({
            type: 'analogy',
            text: baseText + ' Think of it like building blocks.'
        });
        
        return variations;
    }
}

// Sentiment Analyzer
class SentimentAnalyzer {
    constructor() {
        this.positiveWords = new Set(['good', 'excellent', 'great', 'helpful', 'learned', 'understand']);
        this.negativeWords = new Set(['difficult', 'confusing', 'frustrated', 'stuck', 'boring']);
    }

    analyze(text) {
        const words = text.toLowerCase().split(/\s+/);
        let score = 0;
        
        words.forEach(word => {
            if (this.positiveWords.has(word)) score += 1;
            if (this.negativeWords.has(word)) score -= 1;
        });
        
        return {
            score: score / Math.max(1, words.length),
            sentiment: score > 0 ? 'positive' : score < 0 ? 'negative' : 'neutral'
        };
    }
}

// Text Classifier
class TextClassifier {
    constructor() {
        this.categories = {
            mathematical: ['triangle', 'formula', 'equation', 'calculate', 'theorem'],
            question: ['what', 'how', 'why', 'when', 'where', 'explain'],
            instructional: ['step', 'first', 'next', 'then', 'finally'],
            feedback: ['good', 'excellent', 'try', 'correct', 'wrong', 'improve']
        };
    }

    classify(text) {
        const words = text.toLowerCase().split(/\s+/);
        const scores = {};
        
        Object.entries(this.categories).forEach(([category, keywords]) => {
            scores[category] = keywords.filter(keyword => words.includes(keyword)).length;
        });
        
        const bestCategory = Object.entries(scores)
            .sort((a, b) => b[1] - a[1])[0][0];
        
        return bestCategory;
    }
}

// Language Model
class LanguageModel {
    constructor() {
        this.contextWindow = 5;
        this.model = this.initializeModel();
    }

    initializeModel() {
        return {
            embeddings: new Map(),
            context: new Map(),
            vocabulary: new Map()
        };
    }

    generateText(prompt, maxLength = 100) {
        const words = this.tokenize(prompt);
        const generated = [];
        
        for (let i = 0; i < maxLength; i++) {
            const word = this.selectNextWord(words, generated);
            generated.push(word);
        }
        
        return generated.join(' ');
    }

    selectNextWord(vocabulary, generated) {
        // Simplified next word selection
        const available = vocabulary.filter(word => !generated.includes(word));
        if (available.length === 0) {
            return 'the';
        }
        
        return available[Math.floor(Math.random() * available.length)];
    }
}

// Export for use in main application
window.NLPProcessor = NLPProcessor;
window.SentimentAnalyzer = SentimentAnalyzer;
window.TextClassifier = TextClassifier;
window.LanguageModel = LanguageModel;
