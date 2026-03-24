import React from 'react';
import './ActionButtons.css';

const ActionButtons = ({ onResponse, disabled }) => {
  const handleClick = async (response) => {
    if (!disabled) {
      await onResponse(response);
    }
  };

  return (
    <div className="action-buttons">
      <button
        className="action-button"
        onClick={() => handleClick('oui')}
        disabled={disabled}
      >
        OUI
      </button>
      <button
        className="action-button"
        onClick={() => handleClick('non')}
        disabled={disabled}
      >
        NON
      </button>
    </div>
  );
};

export default ActionButtons;
