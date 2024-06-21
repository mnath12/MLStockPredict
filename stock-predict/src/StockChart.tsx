import 
{
    useState,
    useEffect
} from 'react';
import Plot 
    from 'react-plotly.js';
import axios from 'axios';
import { restClient } from '@polygon.io/client-js';
//const rest = restClient('STQPhq8p9WHEeW24dWiNRkhPPjNjr2YL');
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
    /*rest.stocks.aggregates("AAPL", 1, "day", "2023-01-01", "2023-04-14").then((data) => {
        console.log(data);
    }).catch(e => {
        console.error('An error happened:', e);
    });*/
    useEffect(() => {
        const fetchStockData = async () => {
            try {
                const API_KEY = 'W9RWKMBODLNUJ3UX';
                let StockSymbol = "AMZN";
                const response = await axios.get(
`https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${StockSymbol}&apikey=${API_KEY}`
                );
                //const response = await axios.get(`https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-02-10?adjusted=true&sort=asc&apiKey=STQPhq8p9WHEeW24dWiNRkhPPjNjr2YL`);
                //console.log("Response: ", response)
                setStockData(
                    response.data['Time Series (Daily)'] 
                    //response.data['results'] 
                );
            } catch (error) {
                console.error('Error fetching stock data:', error);
            }
        };
 
        fetchStockData();
    }, []);
    console.log("stock data: ,", stockData)
    function createDateArray(date1:any, date2:any, x:any) {
        // Helper function to format date as YYYY-MM-DD
        function formatDate(date:any) {
            let year = date.getFullYear();
            let month = String(date.getMonth() + 1).padStart(2, '0');
            let day = String(date.getDate()).padStart(2, '0');
            return `${year}-${month}-${day}`;
        }
    
        // Helper function to check if the date is a weekend
        function isWeekend(date:any) {
            let day = date.getDay();
            return (day === 0 || day === 6); // Sunday is 0 and Saturday is 6
        }
    
        // Convert input dates to Date objects
        let startDate = new Date(date1);
        let endDate = new Date(date2);
        let stepSize = x * 24 * 60 * 60 * 1000; // Convert step size from days to milliseconds
    
        let dateArray = [];
        for (let date = startDate; date <= endDate; date = new Date(date.getTime() + stepSize)) {
            if (!isWeekend(date)) {
                dateArray.push(formatDate(date));
            } else {
                // Adjust date to skip the weekend
                while (isWeekend(date) && date <= endDate) {
                    date = new Date(date.getTime() + stepSize);
                }
                if (date <= endDate && !isWeekend(date)) {
                    dateArray.push(formatDate(date));
                }
            }
        }
    
        return dateArray;
    }

    function convertToCSVString(stockData: any) {
        // WORKS!
        var csvString = "Date,Open,High,Low,Close,Volume,Name\n"
        var csvRows = []
        csvRows.push(csvString)
        var dates = Object.keys(stockData)
        for (let i = 0; i < dates.length; i++) {
            const date = dates[i]
            const infoObject = stockData[date]
            var csvRow = [date, infoObject["1. open"], infoObject["2. high"], infoObject["3. low"], infoObject["4. close"], infoObject["5. volume"], stock]           
            csvRows.push(csvRow.toString()+"\n")
        }
        return csvRows.join("")
      }

    /*function convertToCSVString(arr: any, dates: any) {
        var csvString = "Date,Open,High,Low,Close,Volume,Name\n"
        var csvRows = []
        csvRows.push(csvString)
        for (let i = 0; i < dates.length; i++) {
            const date = dates[i]
            
            var csvRow = [date, arr[i]["o"], arr[i]["h"], arr[i]["l"], arr[i]["c"], arr[i]["v"], stock]           
            csvRows.push(csvRow.toString()+"\n")
        }
        return csvRows.join("")
    }*/
    //var dates = createDateArray("2023-01-09", "2023-02-10", 1)
    //console.log(dates)
    //var csvContent =  convertToCSVString(stockData)
    //var csvContent =  convertToCSVString(stockData, dates)
    // console.log(csvContent)
    
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
    /*const closingPrices = []
    for (let i = 0; i < dates.length; i++) {
        closingPrices.push(stockData[i]['c'])
    }*/
    
    //console.log(stockData)
 
    return (
        <center>
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
            {/*<Button colorScheme='blue' onClick={handlePredictClick(csvContent)}>predict</Button>*/}

        </center>
    );
}
 
export default StockChart;