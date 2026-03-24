import React from 'react';
import './VLMResponse.css';

const VLMResponse = ({ question }) => {
  if (!question) return null;

  return (
    <div className="vlm-response">
      <p className="vlm-question">{question}</p>
    </div>
  );
};

export default VLMResponse;
