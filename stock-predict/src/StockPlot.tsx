import React, { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { Button } from '@chakra-ui/react';


const StockPlot = () => {
  const [stock, setStock] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [stockData, setStockData] = useState(null);
  console.log(startDate)
  const [result, setResult] = useState("");
  
  const fetchData = async () => {
    const apiKey = 'STQPhq8p9WHEeW24dWiNRkhPPjNjr2YL';
    const url = `https://api.polygon.io/v2/aggs/ticker/${stock}/range/1/day/${startDate}/${endDate}?adjusted=true&sort=asc&apiKey=${apiKey}`;
    
    try {
      const response = await axios.get(url);
      setStockData(response.data.results);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };


  

  const handleSubmit = (e) => {
    e.preventDefault();
    fetchData();
  };

  const formatStockData = (data) => {
    if (!data) return { x: [], y: [] };

    const dates = data.map(point => new Date(point.t).toISOString().split('T')[0]);
    const prices = data.map(point => point.c);

    return { x: dates, y: prices };
  };

  
  const createCSV = (data) => {
    if (!data) return;

    const headers = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Name'];
    const rows = data.map(point => {
      return [
        new Date(point.t ).toISOString().split('T')[0],
        point.o,
        point.h,
        point.l,
        point.c,
        point.v,
        stock,
      ].join(',');
    });

    const csvContent = [headers.join(','), ...rows].join('\n');
    return csvContent
  }
  const plotData = formatStockData(stockData);
  const csv = createCSV(stockData)
  console.log("CSV: ", csv)

  function handlePredictClick(csv: any) {
    const url = "http://localhost:5173/predict";
    var formData = ""
    const jsonData = JSON.stringify(csv);
    // Fetch request to the Flask backend
    fetch(url, {
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      method: "POST",
      body: jsonData,
    })
      .then((response) => response.json())
      .then((response) => {
        /* var res = response.Prediction;
        var stripped_res = [];
        for (let i = 0; i < res.length; ++i){
          stripped_res.push(res[i][0])
        } */
        setResult(response.Prediction);
      });
  };


  const handleClick = (e) => {
    handlePredictClick(csv)
    console.log("Result: ",result)
  
  }
  var x_preds = plotData.x.slice(result.length)
  console.log(x_preds)
  return (
    <div>
      <h1>Stock Data Plotter</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Stock Symbol:
          <input type="text" value={stock} onChange={(e) => setStock(e.target.value)} required />
        </label>
        <br />
        <label>
          Start Date:
          <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} required />
        </label>
        <br />
        <label>
          End Date:
          <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} required />
        </label>
        <br />
        <button type="submit">Fetch Data</button>
      </form>
      {stockData && (
        <Plot
          data={[
            {
              x: plotData.x,
              y: plotData.y,
              type: 'scatter',
              mode: 'lines+markers',
              marker: { color: 'red' },
            },
            {
              x: plotData.x.slice(-result.length),
              y: result,
              type: 'scatter',
              mode: 'lines+markers',
              marker: { color: 'blue' },
            }
          ]}
          layout={{ title: `${stock} Stock Prices` }}
        />
      )}
      <Button onClick ={handleClick}>Predict</Button>
    </div>
  );
};

export default StockPlot;
