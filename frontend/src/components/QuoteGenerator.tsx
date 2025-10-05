import React, { useState } from 'react';
import './QuoteGenerator.css';

interface ServiceOption {
  id: string;
  name: string;
  description: string;
  basePrice: number;
  processingTime: number;
  benefits: string[];
  icon: string;
}

interface CostBreakdown {
  collectionCost: number;
  processingCost: number;
  storageCost: number;
  operationalOverhead: number;
  totalCost: number;
  costPerSatellite: number;
  deltaVRequired: number;
  missionDuration: number;
}

interface QuoteData {
  quoteId: string;
  satellites: any[];
  selectedServices: string[];
  costBreakdown: CostBreakdown;
  timeline: {
    startDate: string;
    estimatedCompletion: string;
    missionDuration: number;
  };
  optimizationSuggestions: string[];
  validUntil: string;
}

interface QuoteGeneratorProps {
  serviceRequestData: any;
  onQuoteGenerated: (quote: QuoteData) => void;
  onBack: () => void;
}

const QuoteGenerator: React.FC<QuoteGeneratorProps> = ({ 
  serviceRequestData, 
  onQuoteGenerated, 
  onBack 
}) => {
  const [selectedServices, setSelectedServices] = useState<string[]>(['iss-recycling']);
  const [quote, setQuote] = useState<QuoteData | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showComparison, setShowComparison] = useState(false);

  const serviceOptions: ServiceOption[] = [
    {
      id: 'iss-recycling',
      name: 'ISS Recycling',
      description: 'Immediate material processing aboard the International Space Station',
      basePrice: 15000,
      processingTime: 10,
      benefits: [
        'Fastest processing time (7-14 days)',
        'Minimal transportation costs',
        '85-92% material recovery rate',
        'Immediate material availability'
      ],
      icon: '🏭'
    },
    {
      id: 'solar-forge',
      name: 'Deep Solar Forge',
      description: 'Advanced material refinement using concentrated solar energy',
      basePrice: 25000,
      processingTime: 35,
      benefits: [
        'Highest purity materials (99.9%+)',
        'Rare earth element extraction',
        'Advanced alloy production',
        'Premium material grades'
      ],
      icon: '☀️'
    },
    {
      id: 'heo-storage',
      name: 'HEO Storage',
      description: 'High Earth Orbit storage for processed materials',
      basePrice: 5000,
      processingTime: 0,
      benefits: [
        'Long-term material storage',
        'Strategic orbital positioning',
        'Future retrieval options',
        'Cost-effective warehousing'
      ],
      icon: '🌌'
    }
  ];

  const generateQuote = async () => {
    setIsGenerating(true);
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Calculate costs based on satellites and selected services
    const totalMass = serviceRequestData.satellites.reduce((sum: number, sat: any) => sum + sat.mass, 0);
    const satelliteCount = serviceRequestData.satellites.length;
    
    // Base collection cost calculation (simplified)
    const avgDeltaV = 150 + (satelliteCount - 1) * 75; // m/s per satellite
    const collectionCost = avgDeltaV * 1.27 * satelliteCount; // $1.27 per m/s
    
    // Processing costs based on selected services
    let processingCost = 0;
    let maxProcessingTime = 0;
    
    selectedServices.forEach(serviceId => {
      const service = serviceOptions.find(s => s.id === serviceId);
      if (service) {
        processingCost += service.basePrice * satelliteCount;
        maxProcessingTime = Math.max(maxProcessingTime, service.processingTime);
      }
    });
    
    // Storage cost (if HEO storage is selected)
    const storageCost = selectedServices.includes('heo-storage') ? 
      totalMass * 50 * 0.25 : 0; // $50/kg/year for 3 months
    
    const operationalOverhead = (collectionCost + processingCost) * 0.15;
    const totalCost = collectionCost + processingCost + storageCost + operationalOverhead;
    
    const costBreakdown: CostBreakdown = {
      collectionCost,
      processingCost,
      storageCost,
      operationalOverhead,
      totalCost,
      costPerSatellite: totalCost / satelliteCount,
      deltaVRequired: avgDeltaV,
      missionDuration: Math.max(14, maxProcessingTime)
    };
    
    const startDate = new Date(serviceRequestData.timelineConstraints.preferredStartDate);
    const completionDate = new Date(startDate);
    completionDate.setDate(completionDate.getDate() + costBreakdown.missionDuration);
    
    const validUntilDate = new Date();
    validUntilDate.setDate(validUntilDate.getDate() + 30);
    
    const newQuote: QuoteData = {
      quoteId: `ORB-${Date.now()}`,
      satellites: serviceRequestData.satellites,
      selectedServices,
      costBreakdown,
      timeline: {
        startDate: startDate.toISOString().split('T')[0],
        estimatedCompletion: completionDate.toISOString().split('T')[0],
        missionDuration: costBreakdown.missionDuration
      },
      optimizationSuggestions: generateOptimizationSuggestions(costBreakdown, selectedServices),
      validUntil: validUntilDate.toISOString().split('T')[0]
    };
    
    setQuote(newQuote);
    setIsGenerating(false);
    onQuoteGenerated(newQuote);
  };

  const generateOptimizationSuggestions = (costs: CostBreakdown, services: string[]): string[] => {
    const suggestions: string[] = [];
    
    if (costs.totalCost > 100000) {
      suggestions.push('Consider grouping satellites by orbital plane to reduce delta-v requirements');
    }
    
    if (services.includes('solar-forge') && services.includes('iss-recycling')) {
      suggestions.push('Combine ISS recycling for immediate needs with solar forge for premium materials');
    }
    
    if (!services.includes('heo-storage') && costs.totalCost > 50000) {
      suggestions.push('Add HEO storage to preserve processed materials for future use');
    }
    
    if (costs.missionDuration > 45) {
      suggestions.push('Consider phased approach to reduce overall mission timeline');
    }
    
    suggestions.push('Bundle multiple satellites in similar orbits for 15-25% cost savings');
    
    return suggestions;
  };

  const handleServiceToggle = (serviceId: string) => {
    setSelectedServices(prev => {
      if (prev.includes(serviceId)) {
        return prev.filter(id => id !== serviceId);
      } else {
        return [...prev, serviceId];
      }
    });
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  const renderServiceSelection = () => (
    <div className="service-selection">
      <h3>Select Processing Services</h3>
      <div className="service-options">
        {serviceOptions.map(service => (
          <div 
            key={service.id}
            className={`service-option ${selectedServices.includes(service.id) ? 'selected' : ''}`}
            onClick={() => handleServiceToggle(service.id)}
          >
            <div className="service-header">
              <div className="service-icon">{service.icon}</div>
              <div className="service-info">
                <h4>{service.name}</h4>
                <p className="service-price">{formatCurrency(service.basePrice)} per satellite</p>
              </div>
              <div className="service-checkbox">
                <input 
                  type="checkbox" 
                  checked={selectedServices.includes(service.id)}
                  onChange={() => handleServiceToggle(service.id)}
                />
              </div>
            </div>
            <p className="service-description">{service.description}</p>
            <div className="service-benefits">
              {service.benefits.map((benefit, index) => (
                <div key={index} className="benefit">
                  <span className="benefit-icon">✓</span>
                  <span>{benefit}</span>
                </div>
              ))}
            </div>
            <div className="service-timeline">
              Processing time: {service.processingTime > 0 ? `${service.processingTime} days` : 'Immediate'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderQuoteDisplay = () => {
    if (!quote) return null;

    return (
      <div className="quote-display">
        <div className="quote-header">
          <h3>Quote Generated</h3>
          <div className="quote-id">Quote ID: {quote.quoteId}</div>
          <div className="quote-validity">Valid until: {quote.validUntil}</div>
        </div>

        <div className="quote-content">
          <div className="cost-breakdown-section">
            <h4>Cost Breakdown</h4>
            <div className="cost-breakdown">
              <div className="cost-item">
                <span className="cost-label">Collection & Transport</span>
                <span className="cost-value">{formatCurrency(quote.costBreakdown.collectionCost)}</span>
              </div>
              <div className="cost-item">
                <span className="cost-label">Processing Services</span>
                <span className="cost-value">{formatCurrency(quote.costBreakdown.processingCost)}</span>
              </div>
              {quote.costBreakdown.storageCost > 0 && (
                <div className="cost-item">
                  <span className="cost-label">Storage (3 months)</span>
                  <span className="cost-value">{formatCurrency(quote.costBreakdown.storageCost)}</span>
                </div>
              )}
              <div className="cost-item">
                <span className="cost-label">Operational Overhead</span>
                <span className="cost-value">{formatCurrency(quote.costBreakdown.operationalOverhead)}</span>
              </div>
              <div className="cost-item total">
                <span className="cost-label">Total Cost</span>
                <span className="cost-value">{formatCurrency(quote.costBreakdown.totalCost)}</span>
              </div>
              <div className="cost-per-satellite">
                Average cost per satellite: {formatCurrency(quote.costBreakdown.costPerSatellite)}
              </div>
            </div>
          </div>

          <div className="mission-details">
            <h4>Mission Details</h4>
            <div className="detail-grid">
              <div className="detail-item">
                <span className="detail-label">Satellites</span>
                <span className="detail-value">{quote.satellites.length}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Total Δv Required</span>
                <span className="detail-value">{quote.costBreakdown.deltaVRequired.toFixed(1)} m/s</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Mission Duration</span>
                <span className="detail-value">{quote.timeline.missionDuration} days</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Start Date</span>
                <span className="detail-value">{quote.timeline.startDate}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Completion Date</span>
                <span className="detail-value">{quote.timeline.estimatedCompletion}</span>
              </div>
            </div>
          </div>

          <div className="selected-services">
            <h4>Selected Services</h4>
            <div className="service-list">
              {quote.selectedServices.map(serviceId => {
                const service = serviceOptions.find(s => s.id === serviceId);
                return service ? (
                  <div key={serviceId} className="selected-service">
                    <span className="service-icon">{service.icon}</span>
                    <span className="service-name">{service.name}</span>
                    <span className="service-cost">{formatCurrency(service.basePrice * quote.satellites.length)}</span>
                  </div>
                ) : null;
              })}
            </div>
          </div>

          <div className="optimization-suggestions">
            <h4>Optimization Suggestions</h4>
            <div className="suggestions-list">
              {quote.optimizationSuggestions.map((suggestion, index) => (
                <div key={index} className="suggestion">
                  <span className="suggestion-icon">💡</span>
                  <span>{suggestion}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="quote-actions">
          <button 
            className="btn-secondary"
            onClick={() => setShowComparison(!showComparison)}
          >
            {showComparison ? 'Hide' : 'Show'} Service Comparison
          </button>
          <button 
            className="btn-primary"
            onClick={() => alert('Quote accepted! We will contact you to finalize the mission details.')}
          >
            Accept Quote
          </button>
        </div>
      </div>
    );
  };

  const renderServiceComparison = () => {
    if (!showComparison) return null;

    return (
      <div className="service-comparison">
        <h4>Service Comparison</h4>
        <div className="comparison-table">
          <div className="comparison-header">
            <div className="comparison-cell">Service</div>
            <div className="comparison-cell">Cost per Satellite</div>
            <div className="comparison-cell">Processing Time</div>
            <div className="comparison-cell">Material Recovery</div>
            <div className="comparison-cell">Best For</div>
          </div>
          {serviceOptions.map(service => (
            <div key={service.id} className="comparison-row">
              <div className="comparison-cell">
                <span className="service-icon">{service.icon}</span>
                {service.name}
              </div>
              <div className="comparison-cell">{formatCurrency(service.basePrice)}</div>
              <div className="comparison-cell">
                {service.processingTime > 0 ? `${service.processingTime} days` : 'Immediate'}
              </div>
              <div className="comparison-cell">
                {service.id === 'iss-recycling' && '85-92%'}
                {service.id === 'solar-forge' && '99.9%+'}
                {service.id === 'heo-storage' && 'N/A'}
              </div>
              <div className="comparison-cell">
                {service.id === 'iss-recycling' && 'Quick turnaround'}
                {service.id === 'solar-forge' && 'Premium materials'}
                {service.id === 'heo-storage' && 'Future use'}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="quote-generator">
      <div className="quote-header">
        <h2>Generate Quote</h2>
        <p>Configure your satellite debris removal service and get an instant quote</p>
      </div>

      {!quote && (
        <>
          {renderServiceSelection()}
          
          <div className="quote-summary">
            <h4>Request Summary</h4>
            <div className="summary-grid">
              <div className="summary-item">
                <span className="summary-label">Satellites</span>
                <span className="summary-value">{serviceRequestData.satellites.length}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Total Mass</span>
                <span className="summary-value">
                  {serviceRequestData.satellites.reduce((sum: number, sat: any) => sum + sat.mass, 0).toFixed(1)} kg
                </span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Preferred Start</span>
                <span className="summary-value">{serviceRequestData.timelineConstraints.preferredStartDate}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Max Budget</span>
                <span className="summary-value">{formatCurrency(serviceRequestData.budgetConstraints.maxBudget)}</span>
              </div>
            </div>
          </div>

          <div className="quote-actions">
            <button className="btn-secondary" onClick={onBack}>
              Back to Request
            </button>
            <button 
              className="btn-primary"
              onClick={generateQuote}
              disabled={selectedServices.length === 0 || isGenerating}
            >
              {isGenerating ? 'Generating Quote...' : 'Generate Quote'}
            </button>
          </div>
        </>
      )}

      {isGenerating && (
        <div className="generating-overlay">
          <div className="generating-content">
            <div className="spinner"></div>
            <h3>Generating Your Quote</h3>
            <p>Optimizing routes and calculating costs...</p>
          </div>
        </div>
      )}

      {quote && (
        <>
          {renderQuoteDisplay()}
          {renderServiceComparison()}
        </>
      )}
    </div>
  );
};

export default QuoteGenerator;