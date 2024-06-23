import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { blackScholes } from './blackScholes';

const commonStocks = [
    { ticker: 'AAPL', name: 'Apple Inc.' },
    { ticker: 'MSFT', name: 'Microsoft Corp.' },
    { ticker: 'GOOGL', name: 'Alphabet Inc.' },
    { ticker: 'AMZN', name: 'Amazon.com Inc.' },
    { ticker: 'TSLA', name: 'Tesla Inc.' }
];

const OptionsPricing = () => {
    const [selectedStock, setSelectedStock] = useState('');
    const [options, setOptions] = useState([]);
    const [selectedOption, setSelectedOption] = useState(null);
    const [realPrice, setRealPrice] = useState(null);
    const [predictedPrice, setPredictedPrice] = useState(null);
    
    useEffect(() => {
        if (selectedStock) {
            const fetchOptions = async () => {
                const response = await axios.get(`https://api.polygon.io/v3/reference/options/contracts?underlying_ticker=${selectedStock}&apiKey=STQPhq8p9WHEeW24dWiNRkhPPjNjr2YL`);
                setOptions(response.data.results);
            };
            fetchOptions();
        }
    }, [selectedStock]);
    
    useEffect(() => {
        if (selectedOption) {
            const { strike_price, expiration_date, option_type } = selectedOption;
            const S = 150; // Current stock price, replace with actual data
            const K = strike_price;
            const T = (new Date(expiration_date) - new Date()) / (365 * 24 * 60 * 60 * 1000);
            const r = 0.05; // Risk-free rate, adjust as needed
            const sigma = 0.2; // Volatility, replace with actual data
            
            const predicted = blackScholes(S, K, T, r, sigma, option_type);
            setPredictedPrice(predicted);
            
            const fetchRealPrice = async () => {
                const response = await axios.get(`https://api.polygon.io/v3/reference/options/contracts/${selectedOption.ticker}?apiKey=${process.env.REACT_APP_POLYGON_API_KEY}`);
                setRealPrice(response.data.results.last_price);
            };
            fetchRealPrice();
        }
    }, [selectedOption]);
    
    return (
        <div>
            <h1>Options Pricing</h1>
            <select onChange={(e) => setSelectedStock(e.target.value)} value={selectedStock}>
                <option value="">Select a Stock</option>
                {commonStocks.map(stock => (
                    <option key={stock.ticker} value={stock.ticker}>
                        {stock.name} ({stock.ticker})
                    </option>
                ))}
            </select>
            {selectedStock && options.length > 0 && (
                <select onChange={(e) => setSelectedOption(options.find(option => option.ticker === e.target.value))}>
                    <option value="">Select an Option</option>
                    {options.map(option => (
                        <option key={option.ticker} value={option.ticker}>
                            {option.ticker} - {option.strike_price} - {option.expiration_date}
                        </option>
                    ))}
                </select>
            )}
            {selectedOption && (
                <div>
                    <h2>Selected Option: {selectedOption.ticker}</h2>
                    <p>Strike Price: {selectedOption.strike_price}</p>
                    <p>Expiration Date: {selectedOption.expiration_date}</p>
                    <p>Option Type: {selectedOption.option_type}</p>
                    {realPrice !== null && predictedPrice !== null && (
                        <>
                            <p>Real Price: {realPrice}</p>
                            <p>Predicted Price (Black-Scholes): {predictedPrice.toFixed(2)}</p>
                        </>
                    )}
                </div>
            )}
        </div>
    );
};

export default OptionsPricing;
