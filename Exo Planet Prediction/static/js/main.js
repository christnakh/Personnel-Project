// Main JavaScript for Exoplanet Analysis System

// Global variables
let currentAnalysis = null;
let batchResults = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('🔭 Exoplanet Analysis System initialized');
    
    // Add smooth scrolling
    addSmoothScrolling();
    
    // Add loading states
    addLoadingStates();
    
    // Add form validation
    addFormValidation();
    
    // Add tooltips
    addTooltips();
});

// Smooth scrolling for navigation links
function addSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Add loading states to forms
function addLoadingStates() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            }
        });
    });
}

// Form validation
function addFormValidation() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!validateForm(this)) {
                e.preventDefault();
            }
        });
    });
}

function validateForm(form) {
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            showFieldError(field, 'This field is required');
            isValid = false;
        } else {
            clearFieldError(field);
        }
    });
    
    // Validate numeric fields
    const numericFields = form.querySelectorAll('input[type="number"]');
    numericFields.forEach(field => {
        const value = parseFloat(field.value);
        const min = parseFloat(field.getAttribute('min'));
        const max = parseFloat(field.getAttribute('max'));
        
        if (isNaN(value) || (min !== null && value < min) || (max !== null && value > max)) {
            showFieldError(field, `Value must be between ${min} and ${max}`);
            isValid = false;
        }
    });
    
    return isValid;
}

function showFieldError(field, message) {
    clearFieldError(field);
    field.classList.add('is-invalid');
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    field.parentNode.appendChild(errorDiv);
}

function clearFieldError(field) {
    field.classList.remove('is-invalid');
    const errorDiv = field.parentNode.querySelector('.invalid-feedback');
    if (errorDiv) {
        errorDiv.remove();
    }
}

// Add tooltips
function addTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// API Helper Functions
async function makeAPIRequest(endpoint, data) {
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Request failed:', error);
        throw error;
    }
}

// Analysis Functions
async function analyzeExoplanet(data) {
    return await makeAPIRequest('/api/analyze', data);
}

async function checkHabitability(data) {
    return await makeAPIRequest('/api/habitability', data);
}

async function detectAnomaly(data) {
    return await makeAPIRequest('/api/anomaly', data);
}

async function batchAnalyze(candidates) {
    return await makeAPIRequest('/api/batch', { candidates });
}

// Utility Functions
function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

function formatPercentage(num, decimals = 1) {
    return (parseFloat(num) * 100).toFixed(decimals) + '%';
}

function getPredictionColor(prediction) {
    switch(prediction.toLowerCase()) {
        case 'confirmed': return 'success';
        case 'candidate': return 'warning';
        case 'false_positive': return 'danger';
        default: return 'secondary';
    }
}

function getHabitabilityColor(isHabitable) {
    return isHabitable ? 'success' : 'danger';
}

function getAnomalyColor(isAnomaly) {
    return isAnomaly ? 'warning' : 'success';
}

// Chart Functions
function createProbabilitiesChart(canvasId, probabilities) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(probabilities).map(key => 
                key.charAt(0).toUpperCase() + key.slice(1).replace('_', ' ')
            ),
            datasets: [{
                data: Object.values(probabilities).map(val => val * 100),
                backgroundColor: [
                    '#28a745', // confirmed - green
                    '#ffc107', // candidate - yellow
                    '#dc3545'  // false_positive - red
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed.toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
}

function createHabitabilityChart(canvasId, habitability) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    const scores = [
        habitability.size_score * 100,
        habitability.stellar_score * 100,
        habitability.habitable_zone_score * 100,
        habitability.temperature_score * 100
    ];
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Size', 'Stellar', 'Habitable Zone', 'Temperature'],
            datasets: [{
                label: 'Habitability Scores',
                data: scores,
                backgroundColor: 'rgba(40, 167, 69, 0.2)',
                borderColor: 'rgba(40, 167, 69, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(40, 167, 69, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(40, 167, 69, 1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 20
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// File Handling
function readCSVFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const text = e.target.result;
                const lines = text.split('\n');
                const headers = lines[0].split(',').map(h => h.trim());
                const data = [];
                
                for (let i = 1; i < lines.length; i++) {
                    if (lines[i].trim()) {
                        const values = lines[i].split(',').map(v => v.trim());
                        const row = {};
                        headers.forEach((header, index) => {
                            row[header] = values[index] || '';
                        });
                        data.push(row);
                    }
                }
                
                resolve(data);
            } catch (error) {
                reject(error);
            }
        };
        reader.onerror = reject;
        reader.readAsText(file);
    });
}

function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
}

// Notification System
function showNotification(message, type = 'info') {
    const alertClass = `alert-${type}`;
    const icon = type === 'success' ? 'check-circle' : 
                 type === 'error' ? 'exclamation-triangle' : 
                 type === 'warning' ? 'exclamation-circle' : 'info-circle';
    
    const notification = document.createElement('div');
    notification.className = `alert ${alertClass} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        <i class="fas fa-${icon}"></i> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Error Handling
function handleError(error, context = '') {
    console.error(`Error in ${context}:`, error);
    showNotification(`Error: ${error.message}`, 'error');
}

// Animation Helpers
function fadeIn(element, duration = 300) {
    element.style.opacity = '0';
    element.style.display = 'block';
    
    let start = performance.now();
    
    function animate(timestamp) {
        const elapsed = timestamp - start;
        const progress = Math.min(elapsed / duration, 1);
        
        element.style.opacity = progress;
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }
    
    requestAnimationFrame(animate);
}

function fadeOut(element, duration = 300) {
    let start = performance.now();
    
    function animate(timestamp) {
        const elapsed = timestamp - start;
        const progress = Math.min(elapsed / duration, 1);
        
        element.style.opacity = 1 - progress;
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        } else {
            element.style.display = 'none';
        }
    }
    
    requestAnimationFrame(animate);
}

// Export functions for global use
window.ExoplanetAnalysis = {
    analyzeExoplanet,
    checkHabitability,
    detectAnomaly,
    batchAnalyze,
    createProbabilitiesChart,
    createHabitabilityChart,
    readCSVFile,
    downloadCSV,
    showNotification,
    handleError,
    formatNumber,
    formatPercentage,
    getPredictionColor,
    getHabitabilityColor,
    getAnomalyColor
};
