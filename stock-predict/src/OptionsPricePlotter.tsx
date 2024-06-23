import React, { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

const OptionsPricePlotter = () => {
    const [underlying, setUnderlying] = useState('');
    const [optionId, setOptionId] = useState('');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [data, setData] = useState([]);

    const fetchPrices = async () => {
        try {
            const response = await axios.get(`https://api.polygon.io/v2/aggs/ticker/${optionId}/range/1/day/${startDate}/${endDate}?apiKey=STQPhq8p9WHEeW24dWiNRkhPPjNjr2YL`);
            const prices = response.data.results.map(item => ({
                date: new Date(item.t).toLocaleDateString(),
                close: item.c
            }));
            setData(prices);
        } catch (error) {
            console.error('Error fetching prices:', error);
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        fetchPrices();
    };

    return (
        <div>
            <h1>Options Price Plotter</h1>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>Underlying Asset:</label>
                    <input type="text" value={underlying} onChange={(e) => setUnderlying(e.target.value)} />
                </div>
                <div>
                    <label>Option Identifier:</label>
                    <input type="text" value={optionId} onChange={(e) => setOptionId(e.target.value)} />
                </div>
                <div>
                    <label>Start Date:</label>
                    <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
                </div>
                <div>
                    <label>End Date:</label>
                    <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
                </div>
                <button type="submit">Fetch and Plot</button>
            </form>
            {data.length > 0 && (
                <Plot
                    data={[
                        {
                            x: data.map(point => point.date),
                            y: data.map(point => point.close),
                            type: 'scatter',
                            mode: 'lines+markers',
                            marker: { color: 'blue' },
                        },
                    ]}
                    layout={{ title: 'Option Closing Prices Over Time' }}
                />
            )}
        </div>
    );
};

export default OptionsPricePlotter;
