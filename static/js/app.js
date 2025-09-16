// Chicken Feed Management System - JavaScript

// Global variables
let isDispensing = false;

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize tooltips
    initializeTooltips();
    
    // Setup manual dispense form
    setupManualDispenseForm();
    
    // Setup real-time updates
    setupRealTimeUpdates();
    
    // Setup keyboard shortcuts
    setupKeyboardShortcuts();
}

// Initialize Bootstrap tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Setup manual dispense form
function setupManualDispenseForm() {
    const form = document.getElementById('manual-dispense-form');
    if (form) {
        form.addEventListener('submit', handleManualDispense);
    }
}

// Handle manual feed dispensing
async function handleManualDispense(event) {
    event.preventDefault();
    
    if (isDispensing) {
        return;
    }
    
    const amountInput = document.getElementById('amount');
    const amount = parseInt(amountInput.value);
    
    if (!amount || amount < 1 || amount > 1000) {
        showAlert('Please enter a valid amount (1-1000 grams)', 'danger');
        return;
    }
    
    await dispenseNow(amount, 'Manual dispense');
}

// Core dispense function
async function dispenseNow(amount, description = '') {
    if (isDispensing) {
        showAlert('Dispensing in progress, please wait...', 'warning');
        return;
    }
    
    const button = document.getElementById('dispense-btn');
    const statusDiv = document.getElementById('dispense-status');
    
    try {
        isDispensing = true;
        
        // Update UI
        if (button) {
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Dispensing...';
        }
        
        if (statusDiv) {
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-spinner fa-spin me-2"></i>
                    Dispensing ${amount}g of feed...
                </div>
            `;
        }
        
        // Make API call
        const response = await fetch('/dispense', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ amount: amount })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(`Successfully dispensed ${amount}g of feed!`, 'success');
            
            if (statusDiv) {
                statusDiv.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check me-2"></i>
                        Successfully dispensed ${amount}g of feed!
                    </div>
                `;
            }
            
            // Refresh stats
            refreshDashboardStats();
            
        } else {
            showAlert(`Dispense failed: ${data.error}`, 'danger');
            
            if (statusDiv) {
                statusDiv.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-times me-2"></i>
                        Dispense failed: ${data.error}
                    </div>
                `;
            }
        }
        
    } catch (error) {
        console.error('Dispense error:', error);
        showAlert('Network error occurred. Please try again.', 'danger');
        
        if (statusDiv) {
            statusDiv.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Network error occurred. Please try again.
                </div>
            `;
        }
    } finally {
        isDispensing = false;
        
        // Reset button
        if (button) {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-play me-2"></i>Dispense Feed';
        }
        
        // Hide status after delay
        if (statusDiv) {
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        }
    }
}

// Show alert messages
function showAlert(message, type = 'info', duration = 5000) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    
    const iconMap = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    
    alertDiv.innerHTML = `
        <i class="fas fa-${iconMap[type] || 'info-circle'} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, duration);
}

// Refresh dashboard statistics
async function refreshDashboardStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        // Update stats on dashboard
        const totalTodayEl = document.getElementById('total-today');
        const successfulTodayEl = document.getElementById('successful-today');
        const failedTodayEl = document.getElementById('failed-today');
        
        if (totalTodayEl) totalTodayEl.textContent = data.today.total_grams + 'g';
        if (successfulTodayEl) successfulTodayEl.textContent = data.today.successful_dispenses;
        if (failedTodayEl) failedTodayEl.textContent = data.today.failed_dispenses;
        
    } catch (error) {
        console.error('Error refreshing stats:', error);
    }
}

// Setup real-time updates
function setupRealTimeUpdates() {
    // Refresh dashboard stats every 30 seconds
    if (window.location.pathname === '/' || window.location.pathname === '/dashboard') {
        setInterval(refreshDashboardStats, 30000);
    }
}

// Setup keyboard shortcuts
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(event) {
        // Ctrl/Cmd + D for quick dispense (if on dashboard)
        if ((event.ctrlKey || event.metaKey) && event.key === 'd') {
            event.preventDefault();
            const amountInput = document.getElementById('amount');
            if (amountInput && !isDispensing) {
                const amount = parseInt(amountInput.value) || 50;
                dispenseNow(amount, 'Quick dispense (Ctrl+D)');
            }
        }
        
        // Escape to cancel/clear forms
        if (event.key === 'Escape') {
            const activeModal = document.querySelector('.modal.show');
            if (activeModal) {
                const modal = bootstrap.Modal.getInstance(activeModal);
                if (modal) modal.hide();
            }
        }
    });
}

// Schedule management functions
async function toggleSchedule(scheduleId) {
    try {
        const response = await fetch(`/schedules/${scheduleId}/toggle`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update UI elements
            const toggle = document.getElementById(`toggle-${scheduleId}`);
            const statusText = document.getElementById(`status-text-${scheduleId}`);
            
            if (toggle) toggle.checked = data.is_active;
            if (statusText) statusText.textContent = data.is_active ? 'Active' : 'Inactive';
            
            showAlert(
                `Schedule ${data.is_active ? 'activated' : 'deactivated'}`,
                'success'
            );
        } else {
            showAlert('Failed to update schedule', 'danger');
        }
    } catch (error) {
        console.error('Toggle schedule error:', error);
        showAlert('Error updating schedule', 'danger');
    }
}

// Utility functions
function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

function formatTime(timeString) {
    const [hours, minutes] = timeString.split(':');
    const date = new Date();
    date.setHours(parseInt(hours), parseInt(minutes));
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function validateFeedAmount(amount) {
    const num = parseInt(amount);
    return num >= 1 && num <= 1000;
}

// Form validation helpers
function setupFormValidation() {
    const forms = document.querySelectorAll('form[data-validate]');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                showAlert('Please fill in all required fields correctly', 'warning');
            }
            form.classList.add('was-validated');
        });
    });
}

// Loading state management
function setLoadingState(element, loading = true) {
    if (!element) return;
    
    if (loading) {
        element.disabled = true;
        const originalText = element.textContent;
        element.dataset.originalText = originalText;
        element.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Loading...';
    } else {
        element.disabled = false;
        element.innerHTML = element.dataset.originalText || 'Submit';
    }
}

// Network status monitoring
function setupNetworkMonitoring() {
    window.addEventListener('online', function() {
        showAlert('Connection restored', 'success', 3000);
    });
    
    window.addEventListener('offline', function() {
        showAlert('Connection lost. Some features may not work.', 'warning', 10000);
    });
}

// Initialize network monitoring
setupNetworkMonitoring();

// Export functions for use in templates
window.dispenseNow = dispenseNow;
window.toggleSchedule = toggleSchedule;
window.showAlert = showAlert;
window.refreshDashboardStats = refreshDashboardStats;
