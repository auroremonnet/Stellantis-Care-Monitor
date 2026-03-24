import React from 'react';
import './TemperatureControl.css';

const TemperatureControl = ({ temperature, onTemperatureChange, disabled, onUserAdjust }) => {
  const handleSliderChange = (e) => {
    const newTemp = parseFloat(e.target.value);
    onTemperatureChange(newTemp);
    if (onUserAdjust) onUserAdjust(newTemp);
  };

  return (
    <div className="temperature-control">
      <div className="temperature-display">
        <div className="temperature-gauge-wrapper">
          <div className="temperature-gauge">
            <div 
              className="temperature-gauge-fill"
              style={{ height: `${(temperature / 50) * 100}%` }}
            />
          </div>
          <div className="temperature-value">{temperature.toFixed(1)}°C</div>
        </div>
      </div>
      
      <div className="slider-container">
        <span className="slider-label">Froid</span>
        <input
          type="range"
          min="0"
          max="50"
          step="0.5"
          value={temperature}
          onChange={handleSliderChange}
          disabled={disabled}
          className="temperature-slider"
        />
        <span className="slider-label">Chaud</span>
      </div>
    </div>
  );
};

export default TemperatureControl;
