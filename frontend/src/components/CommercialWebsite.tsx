import React, { useState, useEffect } from 'react';
import ServiceRequestForm from './ServiceRequestForm';
import QuoteGenerator from './QuoteGenerator';
import './CommercialWebsite.css';

interface ServiceRequestData {
  clientName: string;
  clientEmail: string;
  company: string;
  satellites: any[];
  timelineConstraints: any;
  budgetConstraints: any;
  additionalRequirements: string;
}

interface CommercialWebsiteProps {
  onShowVisualization?: () => void;
}

const CommercialWebsite: React.FC<CommercialWebsiteProps> = ({ onShowVisualization }) => {
  const [showRequestForm, setShowRequestForm] = useState(false);
  const [showQuoteGenerator, setShowQuoteGenerator] = useState(false);
  const [serviceRequestData, setServiceRequestData] = useState<ServiceRequestData | null>(null);
  const [currentSection, setCurrentSection] = useState('home');
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleServiceRequest = (data: ServiceRequestData) => {
    console.log('Service request submitted:', data);
    setServiceRequestData(data);
    setShowRequestForm(false);
    setShowQuoteGenerator(true);
  };

  const handleCancelRequest = () => {
    setShowRequestForm(false);
  };

  const handleQuoteGenerated = (quote: any) => {
    console.log('Quote generated:', quote);
  };

  const handleBackToRequest = () => {
    setShowQuoteGenerator(false);
    setShowRequestForm(true);
  };

  const handleQuoteComplete = () => {
    setShowQuoteGenerator(false);
    setServiceRequestData(null);
  };

  const renderNavigation = () => (
    <nav className={`main-nav ${scrolled ? 'scrolled' : ''}`}>
      <div className="nav-container">
        <div className="nav-brand">
          <div className="brand-logo">
            <div className="logo-ring"></div>
            <div className="logo-core"></div>
          </div>
          <div className="brand-text">
            <h1 className="brand-name">
              <span className="brand-orbit">Re</span>
              <span className="brand-clean">OrbitAr</span>
            </h1>
            <span className="brand-tagline">Satellite Debris Removal</span>
          </div>
        </div>
        <div className="nav-links">
          <button
            className={`nav-link ${currentSection === 'home' ? 'active' : ''}`}
            onClick={() => setCurrentSection('home')}
          >
            <span className="nav-link-text">Home</span>
            <span className="nav-link-indicator"></span>
          </button>
          <button
            className={`nav-link ${currentSection === 'services' ? 'active' : ''}`}
            onClick={() => setCurrentSection('services')}
          >
            <span className="nav-link-text">Services</span>
            <span className="nav-link-indicator"></span>
          </button>
          <button
            className={`nav-link ${currentSection === 'capabilities' ? 'active' : ''}`}
            onClick={() => setCurrentSection('capabilities')}
          >
            <span className="nav-link-text">Capabilities</span>
            <span className="nav-link-indicator"></span>
          </button>
          <button
            className={`nav-link ${currentSection === 'about' ? 'active' : ''}`}
            onClick={() => setCurrentSection('about')}
          >
            <span className="nav-link-text">About</span>
            <span className="nav-link-indicator"></span>
          </button>
          {onShowVisualization && (
            <button
              className="nav-link nav-link-special"
              onClick={onShowVisualization}
            >
              <span className="nav-link-text">3D Visualization</span>
              <span className="nav-link-indicator"></span>
            </button>
          )}
          <button
            className="cta-button pulse-glow"
            onClick={() => setShowRequestForm(true)}
          >
            <span className="cta-icon">→</span>
            <span className="cta-text">Request Service</span>
          </button>
        </div>
      </div>
    </nav>
  );

  const renderHeroSection = () => (
    <section className="hero-section">
      <div className="hero-bg-overlay"></div>
      <div className="hero-grid">
        <div className="hero-content slide-up">
          <div className="hero-badge">
            <span className="badge-icon">✦</span>
            <span className="badge-text">Industry-Leading Space Debris Removal</span>
          </div>
          <h1 className="hero-title">
            The Future of
            <span className="hero-title-highlight"> Orbital Sustainability</span>
          </h1>
          <p className="hero-description">
            Advanced genetic algorithm optimization meets precision orbital mechanics.
            Our comprehensive platform delivers cost-effective satellite debris removal
            with transparent pricing and complete material processing capabilities.
          </p>

          <div className="hero-stats-grid">
            <div className="hero-stat glass-card">
              <div className="stat-icon">💎</div>
              <div className="stat-content">
                <div className="stat-value">$1.27</div>
                <div className="stat-label">per m/s Δv</div>
              </div>
            </div>
            <div className="hero-stat glass-card">
              <div className="stat-icon">🎯</div>
              <div className="stat-content">
                <div className="stat-value">99.2%</div>
                <div className="stat-label">Success Rate</div>
              </div>
            </div>
            <div className="hero-stat glass-card">
              <div className="stat-icon">🛰️</div>
              <div className="stat-content">
                <div className="stat-value">500+</div>
                <div className="stat-label">Processed</div>
              </div>
            </div>
          </div>

          <div className="hero-actions">
            <button
              className="btn-primary-hero"
              onClick={() => setShowRequestForm(true)}
            >
              <span className="btn-shimmer"></span>
              <span className="btn-content">
                <span className="btn-icon">→</span>
                <span className="btn-text">Get Your Quote</span>
              </span>
            </button>
            <button
              className="btn-secondary-hero"
              onClick={() => setCurrentSection('capabilities')}
            >
              <span className="btn-content">
                <span className="btn-text">Explore Capabilities</span>
                <span className="btn-icon">↗</span>
              </span>
            </button>
          </div>
        </div>

        <div className="hero-visual slide-up">
          <div className="orbital-scene">
            <div className="earth-glow"></div>
            <div className="earth"></div>
            <div className="orbit-ring ring-1"></div>
            <div className="orbit-ring ring-2"></div>
            <div className="orbit-ring ring-3"></div>
            <div className="satellite sat-1">
              <div className="sat-body"></div>
            </div>
            <div className="satellite sat-2">
              <div className="sat-body"></div>
            </div>
            <div className="satellite sat-3">
              <div className="sat-body"></div>
            </div>
            <div className="satellite sat-4">
              <div className="sat-body"></div>
            </div>
            <div className="connection-line line-1"></div>
            <div className="connection-line line-2"></div>
            <div className="connection-line line-3"></div>
          </div>
        </div>
      </div>
    </section>
  );

  const renderServicesSection = () => (
    <section className="services-section">
      <div className="container">
        <h2>Our Services</h2>
        <div className="services-grid">
          <div className="service-card">
            <div className="service-icon">🛰️</div>
            <h3>Satellite Collection</h3>
            <p>
              Optimized multi-satellite collection routes using advanced genetic algorithms. 
              Minimize delta-v requirements and mission costs while maximizing collection efficiency.
            </p>
            <ul>
              <li>Genetic algorithm route optimization</li>
              <li>Real-time orbital mechanics calculations</li>
              <li>Cost-effective mission planning</li>
              <li>3D visualization of collection routes</li>
            </ul>
          </div>
          
          <div className="service-card">
            <div className="service-icon">💰</div>
            <h3>Transparent Pricing</h3>
            <p>
              Accurate cost calculations based on proven biprop propulsion model. 
              Get detailed quotes with complete cost breakdowns for informed decision-making.
            </p>
            <ul>
              <li>$1.27 per m/s delta-v cost model</li>
              <li>Detailed cost breakdowns</li>
              <li>Multiple payment options</li>
              <li>No hidden fees</li>
            </ul>
          </div>
          
          <div className="service-card">
            <div className="service-icon">🎯</div>
            <h3>Mission Planning</h3>
            <p>
              Comprehensive mission planning with realistic operational constraints. 
              Timeline optimization, regulatory compliance, and risk assessment included.
            </p>
            <ul>
              <li>Operational constraint handling</li>
              <li>Regulatory compliance checks</li>
              <li>Timeline optimization</li>
              <li>Risk assessment and mitigation</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );

  const renderCapabilitiesSection = () => (
    <section className="capabilities-section">
      <div className="container">
        <h2>Complete Material Processing Ecosystem</h2>
        <p className="section-subtitle">
          From collection to reuse, we provide end-to-end satellite debris processing capabilities
        </p>
        
        <div className="capabilities-grid">
          <div className="capability-card featured">
            <div className="capability-header">
              <div className="capability-icon">🏭</div>
              <h3>ISS Recycling Operations</h3>
            </div>
            <div className="capability-content">
              <p>
                Our ISS-based recycling facility represents the cutting edge of space-based material processing. 
                Located in the microgravity environment of the International Space Station, our automated 
                systems can efficiently break down satellite components and recover valuable materials with 
                minimal energy requirements.
              </p>
              <div className="capability-features">
                <div className="feature">
                  <span className="feature-label">Processing Capacity:</span>
                  <span className="feature-value">2,500 kg/month</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Material Types:</span>
                  <span className="feature-value">Aluminum, Steel, Electronics, Composites</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Processing Time:</span>
                  <span className="feature-value">7-14 days</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Recovery Rate:</span>
                  <span className="feature-value">85-92%</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Energy Source:</span>
                  <span className="feature-value">Solar panels + ISS power</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Automation Level:</span>
                  <span className="feature-value">95% autonomous</span>
                </div>
              </div>
              <div className="capability-benefits">
                <h4>Key Benefits</h4>
                <ul>
                  <li>Fastest turnaround time in the industry</li>
                  <li>Zero atmospheric contamination</li>
                  <li>Immediate availability for space construction projects</li>
                  <li>Proven track record with 500+ processed satellites</li>
                  <li>Real-time quality monitoring and reporting</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="capability-card">
            <div className="capability-header">
              <div className="capability-icon">☀️</div>
              <h3>Deep Solar Forge Station</h3>
            </div>
            <div className="capability-content">
              <p>
                Our Deep Solar Forge represents the pinnacle of space-based metallurgy. Located beyond 
                Earth's magnetosphere, this facility harnesses concentrated solar energy to achieve 
                temperatures and processing conditions impossible on Earth, producing ultra-pure materials 
                and exotic alloys for advanced space applications.
              </p>
              <div className="capability-features">
                <div className="feature">
                  <span className="feature-label">Temperature Range:</span>
                  <span className="feature-value">Up to 3,000°C</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Purity Level:</span>
                  <span className="feature-value">99.9%+ (99.99% for premium)</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Processing Time:</span>
                  <span className="feature-value">21-45 days</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Output Materials:</span>
                  <span className="feature-value">Titanium, Rare Earth Elements, Superalloys</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Solar Concentrator:</span>
                  <span className="feature-value">1,000x concentration ratio</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Vacuum Level:</span>
                  <span className="feature-value">10⁻¹² Torr (perfect vacuum)</span>
                </div>
              </div>
              <div className="capability-benefits">
                <h4>Unique Advantages</h4>
                <ul>
                  <li>Ultra-high purity materials impossible to achieve on Earth</li>
                  <li>Zero atmospheric contamination during processing</li>
                  <li>Exotic alloy production in perfect vacuum conditions</li>
                  <li>Rare earth element extraction and refinement</li>
                  <li>Custom material specifications for advanced applications</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="capability-card">
            <div className="capability-header">
              <div className="capability-icon">🌌</div>
              <h3>HEO Storage Management</h3>
            </div>
            <div className="capability-content">
              <p>
                Our High Earth Orbit storage network provides secure, long-term warehousing for processed 
                materials in strategically positioned orbital depots. These facilities serve as material 
                banks for future space construction projects, offering cost-effective storage and rapid 
                deployment capabilities.
              </p>
              <div className="capability-features">
                <div className="feature">
                  <span className="feature-label">Storage Capacity:</span>
                  <span className="feature-value">50,000 kg total (expandable)</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Orbit Altitude:</span>
                  <span className="feature-value">35,786 km (GEO) + L4/L5 points</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Storage Cost:</span>
                  <span className="feature-value">$50/kg/year</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Retrieval Time:</span>
                  <span className="feature-value">3-7 days</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Storage Locations:</span>
                  <span className="feature-value">5 orbital depots</span>
                </div>
                <div className="feature">
                  <span className="feature-label">Security Level:</span>
                  <span className="feature-value">Military-grade tracking</span>
                </div>
              </div>
              <div className="capability-benefits">
                <h4>Strategic Benefits</h4>
                <ul>
                  <li>Materials available when and where you need them</li>
                  <li>Reduced launch costs for future space projects</li>
                  <li>Strategic positioning at Lagrange points</li>
                  <li>Inventory management and automated tracking</li>
                  <li>Insurance coverage for stored materials</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        <div className="process-flow">
          <h3>Complete Processing Workflow</h3>
          <div className="workflow-steps">
            <div className="workflow-step">
              <div className="step-number">1</div>
              <div className="step-content">
                <h4>Collection</h4>
                <p>Optimized multi-satellite collection using genetic algorithms</p>
              </div>
            </div>
            <div className="workflow-arrow">→</div>
            <div className="workflow-step">
              <div className="step-number">2</div>
              <div className="step-content">
                <h4>Processing</h4>
                <p>ISS recycling or Solar Forge refinement based on requirements</p>
              </div>
            </div>
            <div className="workflow-arrow">→</div>
            <div className="workflow-step">
              <div className="step-number">3</div>
              <div className="step-content">
                <h4>Storage/Delivery</h4>
                <p>HEO storage or direct delivery to your space operations</p>
              </div>
            </div>
          </div>
        </div>

        <div className="capability-stats">
          <div className="stat-card">
            <div className="stat-number">500+</div>
            <div className="stat-label">Satellites Processed</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">99.2%</div>
            <div className="stat-label">Mission Success Rate</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">15,000+</div>
            <div className="stat-label">Tons Materials Recovered</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">24/7</div>
            <div className="stat-label">Operations Monitoring</div>
          </div>
        </div>
      </div>
    </section>
  );

  const renderAboutSection = () => (
    <section className="about-section">
      <div className="container">
        <div className="about-content">
          <div className="about-text">
            <h2>Leading the Future of Space Sustainability</h2>
            <p>
              ReOrbitAr represents the next generation of space debris management, combining
              cutting-edge optimization algorithms with proven orbital mechanics to deliver
              cost-effective satellite removal services.
            </p>
            <p>
              Our proprietary genetic algorithm optimization engine finds the most efficient
              collection routes, while our transparent pricing model based on actual delta-v
              requirements ensures you get the best value for your investment.
            </p>
            <div className="about-highlights">
              <div className="highlight">
                <h4>Advanced Technology</h4>
                <p>Genetic algorithms and real-time orbital mechanics</p>
              </div>
              <div className="highlight">
                <h4>Proven Results</h4>
                <p>500+ successful satellite removals with 99.2% success rate</p>
              </div>
              <div className="highlight">
                <h4>Complete Ecosystem</h4>
                <p>From collection to processing to storage solutions</p>
              </div>
            </div>
          </div>
          <div className="about-visual">
            <div className="tech-showcase">
              <div className="tech-item">
                <span className="tech-label">Genetic Algorithm</span>
                <div className="tech-bar">
                  <div className="tech-progress" style={{width: '95%'}}></div>
                </div>
              </div>
              <div className="tech-item">
                <span className="tech-label">Orbital Mechanics</span>
                <div className="tech-bar">
                  <div className="tech-progress" style={{width: '98%'}}></div>
                </div>
              </div>
              <div className="tech-item">
                <span className="tech-label">Cost Optimization</span>
                <div className="tech-bar">
                  <div className="tech-progress" style={{width: '92%'}}></div>
                </div>
              </div>
              <div className="tech-item">
                <span className="tech-label">Material Processing</span>
                <div className="tech-bar">
                  <div className="tech-progress" style={{width: '89%'}}></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );

  const renderMainContent = () => {
    if (showRequestForm) {
      return (
        <div className="form-overlay">
          <ServiceRequestForm 
            onSubmit={handleServiceRequest}
            onCancel={handleCancelRequest}
          />
        </div>
      );
    }

    if (showQuoteGenerator && serviceRequestData) {
      return (
        <div className="form-overlay">
          <QuoteGenerator
            serviceRequestData={serviceRequestData}
            onQuoteGenerated={handleQuoteGenerated}
            onBack={handleBackToRequest}
          />
        </div>
      );
    }

    return (
      <main className="main-content">
        {currentSection === 'home' && (
          <>
            {renderHeroSection()}
            {renderServicesSection()}
          </>
        )}
        {currentSection === 'services' && renderServicesSection()}
        {currentSection === 'capabilities' && renderCapabilitiesSection()}
        {currentSection === 'about' && renderAboutSection()}
      </main>
    );
  };

  return (
    <div className="commercial-website">
      {renderNavigation()}
      {renderMainContent()}
      
      <footer className="main-footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-section">
              <h4>ReOrbitAr</h4>
              <p>Professional satellite debris removal services with advanced optimization and transparent pricing.</p>
            </div>
            <div className="footer-section">
              <h4>Services</h4>
              <ul>
                <li>Satellite Collection</li>
                <li>Route Optimization</li>
                <li>Cost Analysis</li>
                <li>Mission Planning</li>
              </ul>
            </div>
            <div className="footer-section">
              <h4>Processing</h4>
              <ul>
                <li>ISS Recycling</li>
                <li>Solar Forge Refinement</li>
                <li>HEO Storage</li>
                <li>Material Recovery</li>
              </ul>
            </div>
            <div className="footer-section">
              <h4>Contact</h4>
              <p>Email: info@reorbitar.space</p>
              <p>Phone: +1 (555) 123-ORBIT</p>
              <p>Emergency: +1 (555) 911-SPACE</p>
            </div>
          </div>
          <div className="footer-bottom">
            <p>&copy; 2024 ReOrbitAr. All rights reserved. | Space debris removal services worldwide.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default CommercialWebsite;