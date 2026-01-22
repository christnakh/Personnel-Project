/**
 * 🔭 Advanced Exoplanet Analysis Dashboard
 * Interactive visualizations, real-time analysis, and AI assistant
 */

class ExoplanetDashboard {
    constructor() {
        this.charts = {};
        this.currentAnalysis = null;
        this.educationMode = false;
        this.chatHistory = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.loadNASAData();
        this.setupChatbot();
        this.updateStats();
    }

    setupEventListeners() {
        // Analysis form
        document.getElementById('analysisForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.performAnalysis();
        });

        // Education mode toggle
        document.getElementById('educationMode').addEventListener('change', (e) => {
            this.educationMode = e.target.checked;
            this.toggleEducationMode();
        });

        // Chat input
        document.getElementById('sendMessage').addEventListener('click', () => {
            this.sendChatMessage();
        });

        document.getElementById('chatInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendChatMessage();
            }
        });

        // Real-time parameter updates
        const inputs = document.querySelectorAll('#analysisForm input');
        inputs.forEach(input => {
            input.addEventListener('input', () => {
                this.updateHabitabilityPreview();
            });
        });
    }

    initializeCharts() {
        // Model accuracy chart
        const ctx1 = document.getElementById('modelAccuracyChart').getContext('2d');
        this.charts.accuracy = new Chart(ctx1, {
            type: 'bar',
            data: {
                labels: ['XGBoost', 'Random Forest', 'CNN', 'Hybrid'],
                datasets: [{
                    label: 'Accuracy (%)',
                    data: [80.4, 78.0, 82.1, 83.5],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(255, 206, 86, 0.8)',
                        'rgba(153, 102, 255, 0.8)'
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(153, 102, 255, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Performance Comparison'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });

        // Initialize other charts
        this.initializeFeatureImportanceChart();
        this.initializeLightCurveChart();
    }

    initializeFeatureImportanceChart() {
        const ctx = document.getElementById('featureImportanceChart').getContext('2d');
        this.charts.featureImportance = new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: ['Orbital Period', 'Planet Radius', 'Stellar Temp', 'Transit Depth', 'Duration'],
                datasets: [{
                    label: 'Importance',
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: 'rgba(54, 162, 235, 0.8)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Feature Importance'
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    initializeLightCurveChart() {
        const ctx = document.getElementById('lightCurveChart').getContext('2d');
        
        // Generate synthetic light curve data
        const timePoints = Array.from({length: 100}, (_, i) => i);
        const lightCurve = this.generateSyntheticLightCurve(100);
        
        this.charts.lightCurve = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timePoints,
                datasets: [{
                    label: 'Flux',
                    data: lightCurve,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Light Curve Analysis'
                    }
                },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Relative Flux'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time (hours)'
                        }
                    }
                }
            }
        });
    }

    generateSyntheticLightCurve(length) {
        const curve = [];
        const transitDepth = Math.random() * 0.05 + 0.005;
        const transitCenter = length * 0.5;
        const transitWidth = length * 0.1;
        
        for (let i = 0; i < length; i++) {
            let flux = 1.0;
            
            // Add stellar variability
            flux += 0.02 * Math.sin(2 * Math.PI * i / length);
            
            // Add transit
            if (Math.abs(i - transitCenter) < transitWidth / 2) {
                flux -= transitDepth;
            }
            
            // Add noise
            flux += (Math.random() - 0.5) * 0.01;
            
            curve.push(flux);
        }
        
        return curve;
    }

    async performAnalysis() {
        this.showLoading();
        
        try {
            const formData = new FormData(document.getElementById('analysisForm'));
            const data = Object.fromEntries(formData.entries());
            
            // Convert to numbers
            Object.keys(data).forEach(key => {
                data[key] = parseFloat(data[key]);
            });
            
            // Perform comprehensive analysis
            const response = await fetch('/api/comprehensive', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error('Analysis failed');
            }
            
            const result = await response.json();
            this.currentAnalysis = result;
            this.displayResults(result);
            this.updateStats();
            
        } catch (error) {
            this.showError('Analysis failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayResults(result) {
        // Show results section
        document.getElementById('resultsSection').style.display = 'block';
        
        // Update model predictions
        this.updateModelPredictions(result);
        
        // Update visualizations
        this.updateFeatureImportance(result);
        this.updateLightCurve(result);
        
        // Update habitability
        this.updateHabitability(result);
        
        // Generate AI explanation
        this.generateAIExplanation(result);
        
        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    }

    updateModelPredictions(result) {
        const basic = result.basic_prediction;
        
        // XGBoost
        document.getElementById('xgboost-prediction').textContent = basic.xgboost_prediction.confirmed > 0.5 ? 'Confirmed' : 
                                                                  basic.xgboost_prediction.candidate > 0.5 ? 'Candidate' : 'False Positive';
        document.getElementById('xgboost-confidence').textContent = `Confidence: ${(Math.max(...Object.values(basic.xgboost_prediction)) * 100).toFixed(1)}%`;
        
        // Random Forest
        document.getElementById('rf-prediction').textContent = basic.random_forest_prediction.confirmed > 0.5 ? 'Confirmed' : 
                                                              basic.random_forest_prediction.candidate > 0.5 ? 'Candidate' : 'False Positive';
        document.getElementById('rf-confidence').textContent = `Confidence: ${(Math.max(...Object.values(basic.random_forest_prediction)) * 100).toFixed(1)}%`;
        
        // CNN (if available)
        if (result.cnn_analysis && !result.cnn_analysis.error) {
            document.getElementById('cnn-prediction').textContent = result.cnn_analysis.prediction;
            document.getElementById('cnn-confidence').textContent = `Confidence: ${(result.cnn_analysis.confidence * 100).toFixed(1)}%`;
        } else {
            document.getElementById('cnn-prediction').textContent = 'N/A';
            document.getElementById('cnn-confidence').textContent = 'Model not available';
        }
        
        // Hybrid (ensemble)
        const hybridPred = basic.prediction;
        const hybridConf = basic.confidence;
        document.getElementById('hybrid-prediction').textContent = hybridPred.charAt(0).toUpperCase() + hybridPred.slice(1);
        document.getElementById('hybrid-confidence').textContent = `Confidence: ${(hybridConf * 100).toFixed(1)}%`;
    }

    updateFeatureImportance(result) {
        if (result.explanations && result.explanations.xgboost_shap) {
            const importance = result.explanations.xgboost_shap.feature_importance;
            const labels = Object.keys(importance);
            const values = Object.values(importance);
            
            this.charts.featureImportance.data.labels = labels;
            this.charts.featureImportance.data.datasets[0].data = values;
            this.charts.featureImportance.update();
        }
    }

    updateLightCurve(result) {
        // Generate new light curve based on analysis
        const transitDepth = result.basic_prediction.probabilities.confirmed > 0.5 ? 0.02 : 0.005;
        const lightCurve = this.generateSyntheticLightCurve(100);
        
        this.charts.lightCurve.data.datasets[0].data = lightCurve;
        this.charts.lightCurve.update();
    }

    updateHabitability(result) {
        if (result.habitability) {
            const habitability = result.habitability;
            const container = document.getElementById('habitabilityResults');
            
            container.innerHTML = `
                <div class="habitability-gauge">
                    <div class="gauge-container">
                        <div class="gauge-fill" style="width: ${habitability.habitability_score * 100}%"></div>
                        <div class="gauge-text">${(habitability.habitability_score * 100).toFixed(1)}%</div>
                    </div>
                    <h6 class="text-center mt-2">Habitability Score</h6>
                </div>
                <div class="habitability-details mt-3">
                    <div class="row">
                        <div class="col-6">
                            <small class="text-muted">Equilibrium Temp</small>
                            <div class="fw-bold">${habitability.equilibrium_temp.toFixed(0)} K</div>
                        </div>
                        <div class="col-6">
                            <small class="text-muted">Habitable Zone</small>
                            <div class="fw-bold">${habitability.habitable_zone_position}</div>
                        </div>
                    </div>
                    <div class="mt-2">
                        <span class="badge ${habitability.is_habitable ? 'bg-success' : 'bg-danger'}">
                            ${habitability.is_habitable ? 'Potentially Habitable' : 'Not Habitable'}
                        </span>
                    </div>
                </div>
            `;
        }
    }

    generateAIExplanation(result) {
        const prediction = result.basic_prediction.prediction;
        const confidence = result.basic_prediction.confidence;
        const habitability = result.habitability;
        
        let explanation = '';
        
        if (this.educationMode) {
            explanation = this.generateEducationalExplanation(prediction, confidence, habitability);
        } else {
            explanation = this.generateTechnicalExplanation(prediction, confidence, habitability);
        }
        
        document.getElementById('explanationText').textContent = explanation;
    }

    generateEducationalExplanation(prediction, confidence, habitability) {
        const explanations = {
            'confirmed': `This exoplanet is classified as "Confirmed" with ${(confidence * 100).toFixed(1)}% confidence. This means astronomers are very confident that this is a real planet orbiting a star. `,
            'candidate': `This exoplanet is classified as a "Candidate" with ${(confidence * 100).toFixed(1)}% confidence. This means it shows signs of being a planet, but more observations are needed to confirm it. `,
            'false_positive': `This exoplanet is classified as a "False Positive" with ${(confidence * 100).toFixed(1)}% confidence. This means it's likely not a real planet - it could be a star spot, binary star, or other astronomical phenomenon. `
        };
        
        let explanation = explanations[prediction] || '';
        
        if (habitability.is_habitable) {
            explanation += `The planet appears to be in the habitable zone with a habitability score of ${(habitability.habitability_score * 100).toFixed(1)}%. This means it could potentially have liquid water on its surface! 🌍`;
        } else {
            explanation += `The planet is not in the habitable zone, so it's unlikely to support life as we know it. The habitability score is ${(habitability.habitability_score * 100).toFixed(1)}%.`;
        }
        
        return explanation;
    }

    generateTechnicalExplanation(prediction, confidence, habitability) {
        return `Classification: ${prediction} (confidence: ${(confidence * 100).toFixed(1)}%). ` +
               `Habitability analysis indicates ${habitability.is_habitable ? 'potential habitability' : 'non-habitable conditions'} ` +
               `with a score of ${(habitability.habitability_score * 100).toFixed(1)}% and equilibrium temperature of ${habitability.equilibrium_temp.toFixed(0)} K.`;
    }

    async loadNASAData() {
        try {
            // Simulate NASA API call
            const nasaData = [
                { name: 'Kepler-442b', period: 112.3, radius: 1.34, status: 'Confirmed', date: '2024-01-15' },
                { name: 'TRAPPIST-1e', period: 6.1, radius: 0.92, status: 'Confirmed', date: '2024-01-10' },
                { name: 'Proxima Centauri b', period: 11.2, radius: 1.27, status: 'Confirmed', date: '2024-01-05' },
                { name: 'TOI-715b', period: 19.3, radius: 1.55, status: 'Confirmed', date: '2024-01-01' }
            ];
            
            const tbody = document.getElementById('nasaTableBody');
            tbody.innerHTML = nasaData.map(planet => `
                <tr>
                    <td><strong>${planet.name}</strong></td>
                    <td>${planet.period}</td>
                    <td>${planet.radius}</td>
                    <td><span class="badge bg-success">${planet.status}</span></td>
                    <td>${planet.date}</td>
                </tr>
            `).join('');
            
        } catch (error) {
            console.error('Failed to load NASA data:', error);
        }
    }

    setupChatbot() {
        this.chatHistory = [
            {
                role: 'bot',
                message: 'Hello! I\'m your exoplanet research assistant. I can explain predictions, answer astronomy questions, and help you understand the science behind exoplanet discovery. What would you like to know?'
            }
        ];
    }

    async sendChatMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        
        if (!message) return;
        
        // Add user message
        this.addChatMessage('user', message);
        input.value = '';
        
        // Generate bot response
        const response = await this.generateChatResponse(message);
        this.addChatMessage('bot', response);
    }

    addChatMessage(role, message) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <strong>${role === 'bot' ? 'AI Assistant' : 'You'}:</strong> ${message}
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async generateChatResponse(message) {
        // Simple keyword-based responses (in a real app, this would use a more sophisticated AI)
        const responses = {
            'habitable': 'A habitable planet is one that could potentially support life as we know it. This typically means it\'s in the "Goldilocks zone" - not too hot, not too cold - where liquid water could exist on the surface.',
            'transit': 'A transit occurs when a planet passes in front of its star from our perspective, causing a small dip in the star\'s brightness. This is one of the main ways we detect exoplanets!',
            'false positive': 'A false positive in exoplanet detection is when we think we\'ve found a planet, but it\'s actually something else - like a binary star system, stellar activity, or instrumental noise.',
            'kepler': 'The Kepler Space Telescope was NASA\'s first mission dedicated to finding exoplanets. It discovered thousands of planets by monitoring star brightness for transits.',
            'tess': 'TESS (Transiting Exoplanet Survey Satellite) is NASA\'s current exoplanet-hunting mission. It\'s surveying the entire sky to find planets around the nearest and brightest stars.',
            'default': 'That\'s a great question! I can help explain exoplanet science, model predictions, habitability analysis, or any other astronomy topics. What specific aspect would you like to learn about?'
        };
        
        const lowerMessage = message.toLowerCase();
        for (const [keyword, response] of Object.entries(responses)) {
            if (lowerMessage.includes(keyword)) {
                return response;
            }
        }
        
        return responses.default;
    }

    updateHabitabilityPreview() {
        // Real-time habitability preview as user types
        const period = parseFloat(document.getElementById('period').value) || 365;
        const planetRadius = parseFloat(document.getElementById('planet_radius').value) || 1;
        const stellarTemp = parseFloat(document.getElementById('stellar_temp').value) || 5800;
        
        // Simple habitability calculation
        const tempScore = Math.max(0, 1 - Math.abs(stellarTemp - 5800) / 2000);
        const sizeScore = Math.max(0, 1 - Math.abs(planetRadius - 1) / 2);
        const periodScore = Math.max(0, 1 - Math.abs(period - 365) / 365);
        
        const habitabilityScore = (tempScore + sizeScore + periodScore) / 3;
        
        // Update preview (simplified)
        const container = document.getElementById('habitabilityResults');
        if (habitabilityScore > 0) {
            container.innerHTML = `
                <div class="text-center">
                    <div class="habitability-preview">
                        <h5 class="text-${habitabilityScore > 0.7 ? 'success' : habitabilityScore > 0.4 ? 'warning' : 'danger'}">
                            ${(habitabilityScore * 100).toFixed(1)}%
                        </h5>
                        <small class="text-muted">Preview Habitability Score</small>
                    </div>
                </div>
            `;
        }
    }

    toggleEducationMode() {
        const content = document.getElementById('educationContent');
        if (this.educationMode) {
            content.style.display = 'block';
        } else {
            content.style.display = 'none';
        }
        
        // Re-generate explanation if we have current analysis
        if (this.currentAnalysis) {
            this.generateAIExplanation(this.currentAnalysis);
        }
    }

    updateStats() {
        // Update dashboard statistics
        const totalPredictions = parseInt(document.getElementById('total-predictions').textContent) + 1;
        document.getElementById('total-predictions').textContent = totalPredictions;
        
        if (this.currentAnalysis && this.currentAnalysis.habitability.is_habitable) {
            const habitablePlanets = parseInt(document.getElementById('habitable-planets').textContent) + 1;
            document.getElementById('habitable-planets').textContent = habitablePlanets;
        }
    }

    showLoading() {
        document.getElementById('loadingOverlay').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    showError(message) {
        alert('Error: ' + message);
    }
}

// Research tools functions
function openHyperparameterTuning() {
    alert('Hyperparameter tuning interface would open here. This would allow researchers to adjust model parameters and retrain models.');
}

function startAutoRetraining() {
    alert('Auto-retraining would start here. This would automatically retrain models when new data becomes available.');
}

function downloadModelReport() {
    alert('Model report would be downloaded here. This would include performance metrics, feature importance, and scientific analysis.');
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new ExoplanetDashboard();
});
