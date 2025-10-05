import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders satellite debris removal service title', () => {
  render(<App />);
  const titleElement = screen.getByText(/Satellite Debris Removal Service/i);
  expect(titleElement).toBeInTheDocument();
});

test('renders 3D orbital visualization subtitle', () => {
  render(<App />);
  const subtitleElement = screen.getByText(/3D Orbital Visualization/i);
  expect(subtitleElement).toBeInTheDocument();
});