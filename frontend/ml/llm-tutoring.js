// Large Language Models for Intelligent Tutoring
// Implements: Conversational AI, Question Answering, Personalized Tutoring

class LLMTutoringSystem {
    constructor() {
        this.model = new LanguageModel();
        this.conversationHistory = new Map();
        this.knowledgeBase = new KnowledgeBase();
        this.tutoringEngine = new TutoringEngine();
        this.personalizationEngine = new PersonalizationEngine();
        
        // Model configuration
        this.config = {
            maxTokens: 2048,
            temperature: 0.7,
            topP: 0.9,
            presencePenalty: 0.1,
            frequencyPenalty: 0.1
        };
        
        // Tutoring modes
        this.tutoringModes = [
            'explainer', 'question_answerer', 'problem_solver', 
            'concept_connector', 'practice_generator', 'assessment_helper'
        ];
        
        // Initialize model
        this.initializeModel();
    }

    // Initialize LLM model
    async initializeModel() {
        try {
            await this.model.loadModel();
            await this.knowledgeBase.load();
            console.log('LLM Tutoring System initialized');
            return true;
        } catch (error) {
            console.error('Failed to initialize LLM:', error);
            return false;
        }
    }

    // Main tutoring interface
    async tutor(userId, userMessage, context = {}) {
        const conversation = this.getConversation(userId);
        
        // Process user message
        const processedMessage = this.preprocessMessage(userMessage, context);
        
        // Generate response
        const response = await this.generateResponse(userId, processedMessage, conversation, context);
        
        // Store conversation
        this.addToConversation(userId, userMessage, response);
        
        // Update user profile
        this.updateUserProfile(userId, userMessage, response);
        
        return response;
    }

    // Preprocess user message
    preprocessMessage(message, context) {
        const processed = {
            original: message,
            cleaned: this.cleanMessage(message),
            entities: this.extractEntities(message),
            intent: this.detectIntent(message),
            context: context
        };
        
        return processed;
    }

    // Clean message text
    cleanMessage(message) {
        return message
            .trim()
            .toLowerCase()
            .replace(/[^\w\s\?\!\.\,\;]/g, '')
            .replace(/\s+/g, ' ');
    }

    // Extract entities from message
    extractEntities(message) {
        const entities = {
            mathematical: this.extractMathEntities(message),
            educational: this.extractEducationalEntities(message),
            temporal: this.extractTemporalEntities(message),
            difficulty: this.extractDifficultyLevel(message)
        };
        
        return entities;
    }

    extractMathEntities(message) {
        const mathPatterns = [
            { pattern: /triangle/gi, entity: 'triangle' },
            { pattern: /pythagoras/gi, entity: 'pythagoras' },
            { pattern: /hypotenuse/gi, entity: 'hypotenuse' },
            { pattern: /angle/gi, entity: 'angle' },
            { pattern: /formula/gi, entity: 'formula' },
            { pattern: /equation/gi, entity: 'equation' },
            { pattern: /\d+\s*=\s*\d+/g, entity: 'equation' },
            { pattern: /a\^2\s*\+\s*b\^2\s*=\s*c\^2/gi, entity: 'pythagorean_theorem' }
        ];
        
        const entities = [];
        mathPatterns.forEach(({ pattern, entity }) => {
            if (pattern.test(message)) {
                entities.push({ type: 'mathematical', value: entity });
            }
        });
        
        return entities;
    }

    extractEducationalEntities(message) {
        const eduPatterns = [
            { pattern: /learn/gi, entity: 'learning' },
            { pattern: /teach/gi, entity: 'teaching' },
            { pattern: /explain/gi, entity: 'explanation' },
            { pattern: /help/gi, entity: 'help' },
            { pattern: /practice/gi, entity: 'practice' },
            { pattern: /quiz/gi, entity: 'quiz' },
            { pattern: /lesson/gi, entity: 'lesson' }
        ];
        
        const entities = [];
        eduPatterns.forEach(({ pattern, entity }) => {
            if (pattern.test(message)) {
                entities.push({ type: 'educational', value: entity });
            }
        });
        
        return entities;
    }

    extractTemporalEntities(message) {
        const temporalPatterns = [
            { pattern: /now/gi, entity: 'present' },
            { pattern: /today/gi, entity: 'today' },
            { pattern: /tomorrow/gi, entity: 'tomorrow' },
            { pattern: /next/gi, entity: 'future' },
            { pattern: /previous/gi, entity: 'past' }
        ];
        
        const entities = [];
        temporalPatterns.forEach(({ pattern, entity }) => {
            if (pattern.test(message)) {
                entities.push({ type: 'temporal', value: entity });
            }
        });
        
        return entities;
    }

    extractDifficultyLevel(message) {
        const difficultyPatterns = [
            { pattern: /easy|simple|basic/gi, level: 'beginner' },
            { pattern: /medium|intermediate/gi, level: 'intermediate' },
            { pattern: /hard|difficult|advanced/gi, level: 'advanced' }
        ];
        
        for (const { pattern, level } of difficultyPatterns) {
            if (pattern.test(message)) {
                return level;
            }
        }
        
        return null;
    }

    // Detect user intent
    detectIntent(message) {
        const intents = {
            question: this.isQuestion(message),
            explanation: this.isExplanationRequest(message),
            problem_solving: this.isProblemSolving(message),
            practice: this.isPracticeRequest(message),
            assessment: this.isAssessmentRequest(message),
            conversation: this.isConversation(message)
        };
        
        // Return the most likely intent
        const detectedIntents = Object.entries(intents)
            .filter(([_, detected]) => detected)
            .map(([intent]) => intent);
        
        return detectedIntents.length > 0 ? detectedIntents[0] : 'conversation';
    }

    isQuestion(message) {
        const questionWords = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 
                             'can', 'could', 'would', 'should', 'is', 'are', 'do'];
        const questionMarks = message.match(/\?/g) || [];
        
        return questionWords.some(word => message.includes(word)) || questionMarks.length > 0;
    }

    isExplanationRequest(message) {
        const explanationWords = ['explain', 'explain to me', 'tell me about', 'what is', 
                               'describe', 'clarify', 'elaborate'];
        return explanationWords.some(phrase => message.includes(phrase));
    }

    isProblemSolving(message) {
        const solvingWords = ['solve', 'calculate', 'find', 'determine', 'compute', 
                            'work out', 'figure out'];
        return solvingWords.some(word => message.includes(word));
    }

    isPracticeRequest(message) {
        const practiceWords = ['practice', 'exercise', 'drill', 'work through', 
                              'give me problems', 'more examples'];
        return practiceWords.some(word => message.includes(word));
    }

    isAssessmentRequest(message) {
        const assessmentWords = ['test', 'quiz', 'evaluate', 'check', 'measure', 
                               'assess', 'grade'];
        return assessmentWords.some(word => message.includes(word));
    }

    isConversation(message) {
        const conversationWords = ['hello', 'hi', 'thanks', 'goodbye', 'bye', 'ok', 
                                 'great', 'awesome', 'understand'];
        return conversationWords.some(word => message.includes(word));
    }

    // Generate response using LLM
    async generateResponse(userId, processedMessage, conversation, context) {
        const userProfile = this.getUserProfile(userId);
        const tutoringMode = this.selectTutoringMode(processedMessage.intent, userProfile);
        
        // Build prompt
        const prompt = this.buildPrompt(processedMessage, conversation, userProfile, tutoringMode);
        
        // Generate response
        const rawResponse = await this.model.generate(prompt, this.config);
        
        // Post-process response
        const response = this.postprocessResponse(rawResponse, tutoringMode, userProfile);
        
        return response;
    }

    // Select appropriate tutoring mode
    selectTutoringMode(intent, userProfile) {
        const modeMap = {
            question: 'question_answerer',
            explanation: 'explainer',
            problem_solving: 'problem_solver',
            practice: 'practice_generator',
            assessment: 'assessment_helper'
        };
        
        // Consider user preferences
        if (userProfile.preferredMode && modeMap[intent] === userProfile.preferredMode) {
            return userProfile.preferredMode;
        }
        
        return modeMap[intent] || 'explainer';
    }

    // Build comprehensive prompt
    buildPrompt(processedMessage, conversation, userProfile, tutoringMode) {
        const prompt = {
            system: this.buildSystemPrompt(tutoringMode, userProfile),
            context: this.buildContextPrompt(conversation, userProfile),
            user: this.buildUserPrompt(processedMessage, tutoringMode),
            constraints: this.buildConstraintsPrompt(userProfile)
        };
        
        return this.combinePrompts(prompt);
    }

    buildSystemPrompt(tutoringMode, userProfile) {
        const basePrompt = `You are an intelligent educational tutor specializing in mathematics, particularly Pythagoras theorem. 
Your goal is to provide clear, accurate, and personalized educational support.`;
        
        const modePrompts = {
            explainer: `Focus on providing clear, step-by-step explanations with visual descriptions.`,
            question_answerer: `Answer questions directly and accurately, providing context and examples.`,
            problem_solver: `Walk through problem-solving steps, explaining the reasoning behind each step.`,
            practice_generator: `Create appropriate practice problems with varying difficulty levels.`,
            assessment_helper: `Help assess understanding through targeted questions and feedback.`,
            concept_connector: `Connect new concepts to previously learned material.`
        };
        
        const personalization = userProfile.learningStyle ? 
            `Adapt your teaching style to ${userProfile.learningStyle} learners.` : '';
        
        return `${basePrompt} ${modePrompts[tutoringMode]} ${personalization}`;
    }

    buildContextPrompt(conversation, userProfile) {
        const recentMessages = conversation.slice(-5);
        const context = recentMessages.map(msg => `User: ${msg.user}\nTutor: ${msg.tutor}`).join('\n');
        
        const userProfileContext = userProfile.currentTopic ? 
            `Current topic: ${userProfile.currentTopic}` : '';
        
        return `${context}\n${userProfileContext}`;
    }

    buildUserPrompt(processedMessage, tutoringMode) {
        const promptMap = {
            explainer: `Please explain: ${processedMessage.cleaned}`,
            question_answerer: `Please answer: ${processedMessage.cleaned}`,
            problem_solver: `Please solve: ${processedMessage.cleaned}`,
            practice_generator: `Please create practice problems for: ${processedMessage.cleaned}`,
            assessment_helper: `Please assess understanding of: ${processedMessage.cleaned}`,
            concept_connector: `Please connect: ${processedMessage.cleaned} to related concepts`
        };
        
        return promptMap[tutoringMode] || `Please help with: ${processedMessage.cleaned}`;
    }

    buildConstraintsPrompt(userProfile) {
        const constraints = [];
        
        if (userProfile.difficulty) {
            constraints.push(`Difficulty level: ${userProfile.difficulty}`);
        }
        
        if (userProfile.maxResponseLength) {
            constraints.push(`Keep response under ${userProfile.maxResponseLength} words`);
        }
        
        if (userProfile.language) {
            constraints.push(`Respond in ${userProfile.language}`);
        }
        
        return constraints.join('. ');
    }

    combinePrompts(prompt) {
        return `${prompt.system}\n\nContext:\n${prompt.context}\n\nUser: ${prompt.user}\n\nConstraints: ${prompt.constraints}\n\nTutor:`;
    }

    // Post-process LLM response
    postprocessResponse(rawResponse, tutoringMode, userProfile) {
        const processed = {
            original: rawResponse,
            mode: tutoringMode,
            personalized: this.personalizeResponse(rawResponse, userProfile),
            enhanced: this.enhanceResponse(rawResponse, tutoringMode),
            metadata: {
                timestamp: Date.now(),
                mode: tutoringMode,
                personalization: userProfile.learningStyle || 'general'
            }
        };
        
        return processed;
    }

    personalizeResponse(response, userProfile) {
        let personalized = response;
        
        // Adapt to learning style
        if (userProfile.learningStyle === 'visual') {
            personalized += '\n\nVisual aids: Consider drawing diagrams to visualize these concepts.';
        } else if (userProfile.learningStyle === 'auditory') {
            personalized += '\n\nAudio tip: Try explaining this concept aloud to reinforce understanding.';
        } else if (userProfile.learningStyle === 'kinesthetic') {
            personalized += '\n\nHands-on: Try working through examples step-by-step with paper and pencil.';
        }
        
        // Adapt to difficulty level
        if (userProfile.difficulty === 'beginner') {
            personalized = this.simplifyForBeginners(personalized);
        } else if (userProfile.difficulty === 'advanced') {
            personalized = this.enhanceForAdvanced(personalized);
        }
        
        return personalized;
    }

    simplifyForBeginners(text) {
        // Add beginner-friendly explanations
        const simplified = text
            .replace(/complex/gi, 'detailed')
            .replace(/advanced/gi, 'more challenging')
            .replace(/theoretical/gi, 'practical');
        
        return simplified + '\n\nBeginner tip: Take your time and work through each step carefully.';
    }

    enhanceForAdvanced(text) {
        // Add advanced insights
        const enhanced = text + '\n\nAdvanced insight: Consider how this concept connects to higher-level mathematics.';
        return enhanced;
    }

    enhanceResponse(response, tutoringMode) {
        const enhancements = {
            explainer: this.addVisualAids(response),
            question_answerer: this.addExamples(response),
            problem_solver: this.addAlternativeMethods(response),
            practice_generator: this.addProgressiveDifficulty(response),
            assessment_helper: this.addFeedback(response),
            concept_connector: this.addConnections(response)
        };
        
        return enhancements[tutoringMode] || response;
    }

    addVisualAids(text) {
        return text + '\n\nVisual representation: [Diagram showing the concept]';
    }

    addExamples(text) {
        return text + '\n\nExample: [Practical application of the concept]';
    }

    addAlternativeMethods(text) {
        return text + '\n\nAlternative approach: [Different way to solve the problem]';
    }

    addProgressiveDifficulty(text) {
        return text + '\n\nDifficulty progression: Easy → Medium → Hard';
    }

    addFeedback(text) {
        return text + '\n\nSelf-check: Test your understanding with these questions';
    }

    addConnections(text) {
        return text + '\n\nRelated concepts: [Connected mathematical ideas]';
    }

    // Conversation management
    getConversation(userId) {
        if (!this.conversationHistory.has(userId)) {
            this.conversationHistory.set(userId, []);
        }
        return this.conversationHistory.get(userId);
    }

    addToConversation(userId, userMessage, tutorResponse) {
        const conversation = this.getConversation(userId);
        conversation.push({
            user: userMessage,
            tutor: tutorResponse.personalized || tutorResponse.original,
            timestamp: Date.now(),
            mode: tutorResponse.mode
        });
        
        // Keep conversation size manageable
        if (conversation.length > 20) {
            conversation.splice(0, 5); // Remove oldest 5 messages
        }
        
        this.conversationHistory.set(userId, conversation);
    }

    // User profile management
    getUserProfile(userId) {
        const savedProfile = localStorage.getItem(`llm_profile_${userId}`);
        return savedProfile ? JSON.parse(savedProfile) : this.createDefaultProfile();
    }

    createDefaultProfile() {
        return {
            learningStyle: null,
            difficulty: null,
            preferredMode: null,
            currentTopic: null,
            language: 'english',
            maxResponseLength: 500,
            interests: []
        };
    }

    updateUserProfile(userId, userMessage, tutorResponse) {
        const profile = this.getUserProfile(userId);
        
        // Update based on interaction
        if (!profile.learningStyle && userMessage.includes('visual')) {
            profile.learningStyle = 'visual';
        }
        
        if (!profile.difficulty && userMessage.includes('easy')) {
            profile.difficulty = 'beginner';
        }
        
        if (!profile.currentTopic && tutorResponse.entities?.mathematical) {
            profile.currentTopic = tutorResponse.entities.mathematical[0].value;
        }
        
        // Save updated profile
        localStorage.setItem(`llm_profile_${userId}`, JSON.stringify(profile));
    }

    // Knowledge base integration
    async searchKnowledgeBase(query) {
        return await this.knowledgeBase.search(query);
    }

    // Analytics and improvement
    getAnalytics(userId) {
        const conversation = this.getConversation(userId);
        
        return {
            totalInteractions: conversation.length,
            averageResponseLength: conversation.reduce((sum, msg) => 
                sum + msg.tutor.length, 0) / conversation.length,
            mostUsedMode: this.getMostUsedMode(conversation),
            topicFrequency: this.getTopicFrequency(conversation),
            improvementSuggestions: this.getImprovementSuggestions(conversation)
        };
    }

    getMostUsedMode(conversation) {
        const modeCounts = {};
        conversation.forEach(msg => {
            modeCounts[msg.mode] = (modeCounts[msg.mode] || 0) + 1;
        });
        
        return Object.entries(modeCounts)
            .sort((a, b) => b[1] - a[1])[0][0];
    }

    getTopicFrequency(conversation) {
        const topics = {};
        conversation.forEach(msg => {
            // Extract topics from messages (simplified)
            const mathTopics = msg.user.match(/triangle|pythagoras|angle|formula/gi) || [];
            mathTopics.forEach(topic => {
                topics[topic.toLowerCase()] = (topics[topic.toLowerCase()] || 0) + 1;
            });
        });
        
        return topics;
    }

    getImprovementSuggestions(conversation) {
        const suggestions = [];
        
        // Analyze conversation patterns
        const avgLength = conversation.reduce((sum, msg) => sum + msg.tutor.length, 0) / conversation.length;
        
        if (avgLength > 800) {
            suggestions.push('Consider shorter, more focused responses');
        }
        
        if (avgLength < 200) {
            suggestions.push('Consider more detailed explanations');
        }
        
        return suggestions;
    }

    // Save conversation history
    saveConversation(userId) {
        const conversation = this.getConversation(userId);
        const data = {
            userId,
            conversation,
            timestamp: Date.now()
        };
        
        localStorage.setItem(`llm_conversation_${userId}_${Date.now()}`, JSON.stringify(data));
    }

    // Export model and data
    exportModel() {
        const modelData = {
            config: this.config,
            tutoringModes: this.tutoringModes,
            conversationHistory: Object.fromEntries(this.conversationHistory),
            timestamp: Date.now()
        };
        
        localStorage.setItem('llm_model_export', JSON.stringify(modelData));
        console.log('LLM model exported');
        return modelData;
    }
}

// Language Model (simplified implementation)
class LanguageModel {
    constructor() {
        this.modelLoaded = false;
        this.vocabulary = new Map();
        this.embeddings = new Map();
    }

    async loadModel() {
        // In production, would load actual LLM
        // For now, initialize with basic vocabulary
        this.initializeVocabulary();
        this.modelLoaded = true;
        console.log('Language model loaded');
    }

    initializeVocabulary() {
        const words = [
            'pythagoras', 'triangle', 'hypotenuse', 'angle', 'formula',
            'calculate', 'solve', 'explain', 'learn', 'practice', 'understand',
            'mathematics', 'geometry', 'algebra', 'theorem', 'proof', 'solution'
        ];
        
        words.forEach(word => {
            this.vocabulary.set(word, this.generateEmbedding(word));
        });
    }

    generateEmbedding(word) {
        // Simple embedding generation
        const embedding = [];
        for (let i = 0; i < 128; i++) {
            embedding.push((word.charCodeAt(i % word.length) - 100) / 100);
        }
        return embedding;
    }

    async generate(prompt, config) {
        // Simplified text generation
        const response = this.generateResponse(prompt, config);
        return response;
    }

    generateResponse(prompt, config) {
        // Simple rule-based response generation
        if (prompt.includes('explain')) {
            return this.generateExplanation(prompt);
        } else if (prompt.includes('solve')) {
            return this.generateSolution(prompt);
        } else if (prompt.includes('what is')) {
            return this.generateDefinition(prompt);
        } else {
            return this.generateGeneralResponse(prompt);
        }
    }

    generateExplanation(prompt) {
        return `To understand this concept, let's break it down step by step:\n\n1. First, identify the key components\n2. Then, understand how they relate\n3. Finally, practice with examples\n\nThis approach helps build a solid foundation for learning.`;
    }

    generateSolution(prompt) {
        return `To solve this problem:\n\n1. Identify what you're looking for\n2. Write down the relevant formula\n3. Substitute the known values\n4. Solve for the unknown\n5. Check your answer\n\nLet me walk through this with a specific example.`;
    }

    generateDefinition(prompt) {
        return `This is a fundamental mathematical concept that describes the relationship between different elements. It's important because it helps us solve practical problems and understand deeper mathematical principles.`;
    }

    generateGeneralResponse(prompt) {
        return `I'm here to help you learn! Could you tell me more specifically what you'd like to understand or practice? I can explain concepts, solve problems, or create practice exercises tailored to your needs.`;
    }
}

// Knowledge Base
class KnowledgeBase {
    constructor() {
        this.knowledge = new Map();
        this.index = new Map();
    }

    async load() {
        // Load mathematical knowledge
        this.loadMathematicalKnowledge();
        console.log('Knowledge base loaded');
    }

    loadMathematicalKnowledge() {
        const concepts = [
            {
                topic: 'pythagoras_theorem',
                definition: 'a² + b² = c² for right triangles',
                explanation: 'The square of the hypotenuse equals the sum of squares of the other two sides',
                examples: ['3-4-5 triangle', '5-12-13 triangle'],
                applications: ['construction', 'navigation', 'physics']
            },
            {
                topic: 'right_triangle',
                definition: 'Triangle with one 90-degree angle',
                explanation: 'A triangle where one angle measures exactly 90 degrees',
                properties: ['hypotenuse is longest side', 'sum of angles is 180°'],
                examples: ['3-4-5', '6-8-10']
            }
        ];
        
        concepts.forEach(concept => {
            this.knowledge.set(concept.topic, concept);
            this.indexConcept(concept);
        });
    }

    indexConcept(concept) {
        const keywords = this.extractKeywords(concept);
        keywords.forEach(keyword => {
            if (!this.index.has(keyword)) {
                this.index.set(keyword, []);
            }
            this.index.get(keyword).push(concept.topic);
        });
    }

    extractKeywords(concept) {
        const text = `${concept.definition} ${concept.explanation} ${concept.topic}`;
        return text.toLowerCase().split(/\s+/).filter(word => word.length > 3);
    }

    async search(query) {
        const keywords = query.toLowerCase().split(/\s+/);
        const results = [];
        
        keywords.forEach(keyword => {
            if (this.index.has(keyword)) {
                const topics = this.index.get(keyword);
                topics.forEach(topic => {
                    if (this.knowledge.has(topic)) {
                        results.push(this.knowledge.get(topic));
                    }
                });
            }
        });
        
        return results;
    }
}

// Personalization Engine
class PersonalizationEngine {
    constructor() {
        this.userProfiles = new Map();
        this.learningStyles = ['visual', 'auditory', 'kinesthetic'];
        this.difficultyLevels = ['beginner', 'intermediate', 'advanced'];
    }

    personalize(content, userProfile) {
        let personalized = content;
        
        // Adapt to learning style
        if (userProfile.learningStyle) {
            personalized = this.adaptForLearningStyle(personalized, userProfile.learningStyle);
        }
        
        // Adapt to difficulty
        if (userProfile.difficulty) {
            personalized = this.adaptForDifficulty(personalized, userProfile.difficulty);
        }
        
        return personalized;
    }

    adaptForLearningStyle(content, style) {
        const adaptations = {
            visual: content + '\n\nVisual: Consider drawing diagrams to understand better.',
            auditory: content + '\n\nAuditory: Try explaining this aloud.',
            kinesthetic: content + '\n\nKinesthetic: Work through with hands-on examples.'
        };
        
        return adaptations[style] || content;
    }

    adaptForDifficulty(content, difficulty) {
        const adaptations = {
            beginner: content + '\n\nBeginner: Take it step by step.',
            intermediate: content + '\n\nIntermediate: Connect to previous concepts.',
            advanced: content + '\n\nAdvanced: Explore deeper implications.'
        };
        
        return adaptations[difficulty] || content;
    }
}

// Tutoring Engine
class TutoringEngine {
    constructor() {
        this.strategies = new Map();
        this.initializeStrategies();
    }

    initializeStrategies() {
        this.strategies.set('socratic', new SocraticStrategy());
        this.strategies.set('direct', new DirectStrategy());
        this.strategies.set('discovery', new DiscoveryStrategy());
    }

    getStrategy(strategyName) {
        return this.strategies.get(strategyName) || this.strategies.get('direct');
    }
}

// Socratic Strategy
class SocraticStrategy {
    generateResponse(query) {
        return `That's a great question! Let me ask you something to help you think through this: What do you already know about this topic?`;
    }
}

// Direct Strategy
class DirectStrategy {
    generateResponse(query) {
        return `Here's a clear explanation: [Direct explanation of the concept with examples]`;
    }
}

// Discovery Strategy
class DiscoveryStrategy {
    generateResponse(query) {
        return `Let's discover this together! Try this: [Guided discovery activity]`;
    }
}

// Export for use in main application
window.LLMTutoringSystem = LLMTutoringSystem;
window.LanguageModel = LanguageModel;
window.KnowledgeBase = KnowledgeBase;
window.PersonalizationEngine = PersonalizationEngine;
window.TutoringEngine = TutoringEngine;
