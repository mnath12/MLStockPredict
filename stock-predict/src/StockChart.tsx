import 
{
    useState,
    useEffect
} from 'react';
import Plot 
    from 'react-plotly.js';
import axios from 'axios';
import { createObjectCsvWriter } from 'csv-writer';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import express from 'express';
import { Button } from '@chakra-ui/react';
 
function StockChart({stock, handlePredictClick}: any) {
    const arr =[1,2]
    console.log(arr)
    const [stockData, setStockData] = useState({});
    console.log("HI")
    console.log(stock)
    useEffect(() => {
        const fetchStockData = async () => {
            try {
                const API_KEY = 'W9RWKMBODLNUJ3UX';
                let StockSymbol = stock;
                const response = await axios.get(
`https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${StockSymbol}&apikey=${API_KEY}`
                );
                setStockData(
                    response.data['Time Series (Daily)']
                );
            } catch (error) {
                console.error('Error fetching stock data:', error);
            }
        };
 
        fetchStockData();
    }, []);

    console.log("stock data: ,", stockData)

    function convertToCSVString(stockData: any) {
        // WORKS!
        var csvString = "Date,Open,High,Low,Close,Volume,Name\n"
        var csvRows = []
        csvRows.push(csvString)
        var dates = Object.keys(stockData)
        for (let i = 0; i < dates.length; i++) {
            const date = dates[i]
            const infoObject = stockData[date]
            var csvRow = [date, infoObject["1. open"], infoObject["2. high"], infoObject["3. low"], infoObject["4. close"], infoObject["5. volume"]]           
            csvRows.push(csvRow.toString()+"\n")
        }
        return csvRows.join("")
      }
    var csvContent =  convertToCSVString(stockData)
    console.log(csvContent)
    
    // Send text file to backend
    // Check if stockData exists before accessing its keys
    const dates =
        stockData ? Object.keys(stockData) : [];
    const closingPrices =
        dates.map(
            date =>
                parseFloat(
                    (stockData as any)[date]['4. close'] || 0
                )
        );
    console.log(stockData)
 
    return (
        <center>
            <h2>Stock Chart</h2>
            <Plot
                data={[
                    {
                        x: dates,
                        y: closingPrices,
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: { color: 'blue' },
                    }
                ]}
                layout={
                    {
                        title: 'Stock Market Prices'
                    }
                }
            />
            <Button colorScheme='blue' onChange={handlePredictClick(csvContent)}>predict</Button>

        </center>
    );
}
 
export default StockChart;