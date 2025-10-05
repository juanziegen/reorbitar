import React, { useState } from 'react';
import './ServiceRequestForm.css';

interface SatelliteData {
  id: string;
  name: string;
  tleLine1: string;
  tleLine2: string;
  mass: number;
  decommissionDate: string;
}

interface TimelineConstraints {
  preferredStartDate: string;
  maxMissionDuration: number; // days
  urgency: 'low' | 'medium' | 'high';
}

interface BudgetConstraints {
  maxBudget: number;
  currency: string;
  paymentTerms: string;
}

interface ServiceRequestData {
  clientName: string;
  clientEmail: string;
  company: string;
  satellites: SatelliteData[];
  timelineConstraints: TimelineConstraints;
  budgetConstraints: BudgetConstraints;
  additionalRequirements: string;
}

interface ServiceRequestFormProps {
  onSubmit: (data: ServiceRequestData) => void;
  onCancel: () => void;
}

const ServiceRequestForm: React.FC<ServiceRequestFormProps> = ({ onSubmit, onCancel }) => {
  const [formData, setFormData] = useState<ServiceRequestData>({
    clientName: '',
    clientEmail: '',
    company: '',
    satellites: [{
      id: '',
      name: '',
      tleLine1: '',
      tleLine2: '',
      mass: 0,
      decommissionDate: ''
    }],
    timelineConstraints: {
      preferredStartDate: '',
      maxMissionDuration: 30,
      urgency: 'medium'
    },
    budgetConstraints: {
      maxBudget: 0,
      currency: 'USD',
      paymentTerms: 'net-30'
    },
    additionalRequirements: ''
  });

  const [currentStep, setCurrentStep] = useState(1);
  const totalSteps = 4;

  const handleInputChange = (field: string, value: any, index?: number) => {
    if (field.startsWith('satellite.') && index !== undefined) {
      const satelliteField = field.replace('satellite.', '');
      const updatedSatellites = [...formData.satellites];
      updatedSatellites[index] = {
        ...updatedSatellites[index],
        [satelliteField]: value
      };
      setFormData({ ...formData, satellites: updatedSatellites });
    } else if (field.startsWith('timeline.')) {
      const timelineField = field.replace('timeline.', '');
      setFormData({
        ...formData,
        timelineConstraints: {
          ...formData.timelineConstraints,
          [timelineField]: value
        }
      });
    } else if (field.startsWith('budget.')) {
      const budgetField = field.replace('budget.', '');
      setFormData({
        ...formData,
        budgetConstraints: {
          ...formData.budgetConstraints,
          [budgetField]: value
        }
      });
    } else {
      setFormData({ ...formData, [field]: value });
    }
  };

  const addSatellite = () => {
    setFormData({
      ...formData,
      satellites: [
        ...formData.satellites,
        {
          id: '',
          name: '',
          tleLine1: '',
          tleLine2: '',
          mass: 0,
          decommissionDate: ''
        }
      ]
    });
  };

  const removeSatellite = (index: number) => {
    if (formData.satellites.length > 1) {
      const updatedSatellites = formData.satellites.filter((_, i) => i !== index);
      setFormData({ ...formData, satellites: updatedSatellites });
    }
  };

  const validateStep = (step: number): boolean => {
    switch (step) {
      case 1:
        return formData.clientName.trim() !== '' && 
               formData.clientEmail.trim() !== '' && 
               formData.company.trim() !== '';
      case 2:
        return formData.satellites.every(sat => 
          sat.name.trim() !== '' && 
          sat.tleLine1.trim() !== '' && 
          sat.tleLine2.trim() !== '' && 
          sat.mass > 0
        );
      case 3:
        return formData.timelineConstraints.preferredStartDate !== '' &&
               formData.timelineConstraints.maxMissionDuration > 0;
      case 4:
        return formData.budgetConstraints.maxBudget > 0;
      default:
        return true;
    }
  };

  const nextStep = () => {
    if (validateStep(currentStep) && currentStep < totalSteps) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateStep(currentStep)) {
      onSubmit(formData);
    }
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="form-step">
            <h3>Client Information</h3>
            <div className="form-group">
              <label htmlFor="clientName">Full Name *</label>
              <input
                type="text"
                id="clientName"
                value={formData.clientName}
                onChange={(e) => handleInputChange('clientName', e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="clientEmail">Email Address *</label>
              <input
                type="email"
                id="clientEmail"
                value={formData.clientEmail}
                onChange={(e) => handleInputChange('clientEmail', e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="company">Company/Organization *</label>
              <input
                type="text"
                id="company"
                value={formData.company}
                onChange={(e) => handleInputChange('company', e.target.value)}
                required
              />
            </div>
          </div>
        );

      case 2:
        return (
          <div className="form-step">
            <h3>Satellite Information</h3>
            {formData.satellites.map((satellite, index) => (
              <div key={index} className="satellite-form">
                <div className="satellite-header">
                  <h4>Satellite {index + 1}</h4>
                  {formData.satellites.length > 1 && (
                    <button
                      type="button"
                      className="remove-satellite"
                      onClick={() => removeSatellite(index)}
                    >
                      Remove
                    </button>
                  )}
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label htmlFor={`satName-${index}`}>Satellite Name *</label>
                    <input
                      type="text"
                      id={`satName-${index}`}
                      value={satellite.name}
                      onChange={(e) => handleInputChange('satellite.name', e.target.value, index)}
                      required
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor={`satId-${index}`}>Satellite ID/NORAD</label>
                    <input
                      type="text"
                      id={`satId-${index}`}
                      value={satellite.id}
                      onChange={(e) => handleInputChange('satellite.id', e.target.value, index)}
                    />
                  </div>
                </div>
                <div className="form-group">
                  <label htmlFor={`tleLine1-${index}`}>TLE Line 1 *</label>
                  <input
                    type="text"
                    id={`tleLine1-${index}`}
                    value={satellite.tleLine1}
                    onChange={(e) => handleInputChange('satellite.tleLine1', e.target.value, index)}
                    placeholder="1 25544U 98067A   21001.00000000  .00002182  00000-0  40768-4 0  9990"
                    required
                  />
                </div>
                <div className="form-group">
                  <label htmlFor={`tleLine2-${index}`}>TLE Line 2 *</label>
                  <input
                    type="text"
                    id={`tleLine2-${index}`}
                    value={satellite.tleLine2}
                    onChange={(e) => handleInputChange('satellite.tleLine2', e.target.value, index)}
                    placeholder="2 25544  51.6461 339.2911 0002829  68.6102 291.5211 15.48919103265748"
                    required
                  />
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label htmlFor={`mass-${index}`}>Mass (kg) *</label>
                    <input
                      type="number"
                      id={`mass-${index}`}
                      value={satellite.mass}
                      onChange={(e) => handleInputChange('satellite.mass', parseFloat(e.target.value) || 0, index)}
                      min="0"
                      step="0.1"
                      required
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor={`decommissionDate-${index}`}>Decommission Date</label>
                    <input
                      type="date"
                      id={`decommissionDate-${index}`}
                      value={satellite.decommissionDate}
                      onChange={(e) => handleInputChange('satellite.decommissionDate', e.target.value, index)}
                    />
                  </div>
                </div>
              </div>
            ))}
            <button type="button" className="add-satellite" onClick={addSatellite}>
              Add Another Satellite
            </button>
          </div>
        );

      case 3:
        return (
          <div className="form-step">
            <h3>Timeline Requirements</h3>
            <div className="form-group">
              <label htmlFor="preferredStartDate">Preferred Start Date *</label>
              <input
                type="date"
                id="preferredStartDate"
                value={formData.timelineConstraints.preferredStartDate}
                onChange={(e) => handleInputChange('timeline.preferredStartDate', e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="maxMissionDuration">Maximum Mission Duration (days) *</label>
              <input
                type="number"
                id="maxMissionDuration"
                value={formData.timelineConstraints.maxMissionDuration}
                onChange={(e) => handleInputChange('timeline.maxMissionDuration', parseInt(e.target.value) || 0)}
                min="1"
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="urgency">Mission Urgency</label>
              <select
                id="urgency"
                value={formData.timelineConstraints.urgency}
                onChange={(e) => handleInputChange('timeline.urgency', e.target.value)}
              >
                <option value="low">Low - Flexible timeline</option>
                <option value="medium">Medium - Standard priority</option>
                <option value="high">High - Urgent removal required</option>
              </select>
            </div>
          </div>
        );

      case 4:
        return (
          <div className="form-step">
            <h3>Budget Constraints</h3>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="maxBudget">Maximum Budget *</label>
                <input
                  type="number"
                  id="maxBudget"
                  value={formData.budgetConstraints.maxBudget}
                  onChange={(e) => handleInputChange('budget.maxBudget', parseFloat(e.target.value) || 0)}
                  min="0"
                  step="1000"
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="currency">Currency</label>
                <select
                  id="currency"
                  value={formData.budgetConstraints.currency}
                  onChange={(e) => handleInputChange('budget.currency', e.target.value)}
                >
                  <option value="USD">USD</option>
                  <option value="EUR">EUR</option>
                  <option value="GBP">GBP</option>
                  <option value="JPY">JPY</option>
                </select>
              </div>
            </div>
            <div className="form-group">
              <label htmlFor="paymentTerms">Payment Terms</label>
              <select
                id="paymentTerms"
                value={formData.budgetConstraints.paymentTerms}
                onChange={(e) => handleInputChange('budget.paymentTerms', e.target.value)}
              >
                <option value="net-30">Net 30 days</option>
                <option value="net-60">Net 60 days</option>
                <option value="net-90">Net 90 days</option>
                <option value="upfront">50% upfront, 50% on completion</option>
                <option value="milestone">Milestone-based payments</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="additionalRequirements">Additional Requirements</label>
              <textarea
                id="additionalRequirements"
                value={formData.additionalRequirements}
                onChange={(e) => handleInputChange('additionalRequirements', e.target.value)}
                rows={4}
                placeholder="Any specific requirements, constraints, or preferences for the mission..."
              />
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="service-request-form">
      <div className="form-header">
        <h2>Request Satellite Debris Removal Service</h2>
        <div className="progress-bar">
          <div className="progress-steps">
            {Array.from({ length: totalSteps }, (_, i) => (
              <div
                key={i}
                className={`progress-step ${i + 1 <= currentStep ? 'active' : ''} ${i + 1 < currentStep ? 'completed' : ''}`}
              >
                <span className="step-number">{i + 1}</span>
                <span className="step-label">
                  {i === 0 && 'Client Info'}
                  {i === 1 && 'Satellites'}
                  {i === 2 && 'Timeline'}
                  {i === 3 && 'Budget'}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="form-content">
        {renderStepContent()}

        <div className="form-actions">
          <button
            type="button"
            className="btn-secondary"
            onClick={onCancel}
          >
            Cancel
          </button>
          
          {currentStep > 1 && (
            <button
              type="button"
              className="btn-secondary"
              onClick={prevStep}
            >
              Previous
            </button>
          )}

          {currentStep < totalSteps ? (
            <button
              type="button"
              className="btn-primary"
              onClick={nextStep}
              disabled={!validateStep(currentStep)}
            >
              Next
            </button>
          ) : (
            <button
              type="submit"
              className="btn-primary"
              disabled={!validateStep(currentStep)}
            >
              Submit Request
            </button>
          )}
        </div>
      </form>
    </div>
  );
};

export default ServiceRequestForm;