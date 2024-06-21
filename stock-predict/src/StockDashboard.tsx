import { Flex, Heading, Icon, Link, Text, Image, Input, Button, HStack } from "@chakra-ui/react";
import {useEffect, useState } from 'react'
import {
    FiHome,
    FiPieChart,
    FiDollarSign,
    FiBox,
    FiCalendar,
    FiChevronDown,
    FiChevronUp,
    FiPlus,
    FiCreditCard,
    FiSearch,
    FiBell
} from "react-icons/fi"
import StockChart from "./StockChart.tsx";
import StockPlot from "./StockPlot.tsx";

import axios from "axios";
export default function StockDashboard() {
    const [stockData, setStockData] = useState({});
    console.log("HI")
    

    const [stock, setStock] = useState("AMZN")
    const [value, setValue] = useState("")
    console.log(stock)
    const handleStockChange = (event:any) => {
        setValue(event.target.value)
    }
    function handleClick() {
        setStock(value)
        setValue("")
    }
    const [result, setResult] = useState("");
    function handlePredictClick(stockString: any) {
        const url = "http://localhost:5173/predict";
        var formData = ""
        const jsonData = JSON.stringify(stockString);
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
            setResult(response.Prediction);
          });
      };
      console.log("Result: ",result)


    return (
        <Flex
            h = "100vh"
            flexDir = "row"
            overflow = "hidden"
            maxW = "2000px"
        >
            {/* First column of stock dashbaord*/}
            <Flex
                w = "15%"
                flexDir="column"
                alignItems="center"
                backgroundColor="#020202"
                color="#fff"
            >
                <Flex
                    flexDir="column"
                    justifyContent="space-between"
                >
                    <Flex
                        flexDir="column"
                        as="nav"
                    >
                        <Heading
                            mt={50}
                            mb={100}
                            fontSize="4xl"
                            alignSelf="center"
                            letterSpacing="tight"
                        >
                            Stocks
                        </Heading>
                        <Flex 
                            flexDir="column" 
                            align="flex-start"
                            justifyContent="center"
                        >
                            <Flex className="sidebar-items">
                                <Link>
                                    <Icon as ={FiHome} fontSize="2xl" className="active-icon"/>
                                </Link>
                                <Link _hover = {{ textDecor: 'none' }}>
                                    <Text className = "active">Home</Text>
                                </Link>

                            </Flex>

                            

                            <Flex className="sidebar-items">
                                <Link>
                                    <Icon as ={FiHome} fontSize="2xl"/>
                                </Link>
                                <Link _hover = {{ textDecor: 'none' }}>
                                    <Text> Predictions </Text>
                                </Link>
                            </Flex>

                            <Flex className="sidebar-items">
                                <Link>
                                    <Icon as ={FiHome} fontSize="2xl"/>
                                </Link>
                                <Link _hover = {{ textDecor: 'none' }}>
                                    <Text>Research</Text>
                                </Link>
                            </Flex>

                        </Flex>

                    </Flex>

                </Flex>

            </Flex>

            {/* column 2 of stock dashbaord, contains chart and loss metrics below*/}
            <Flex
                w="55%"
                p="3%"
                flexDir="column"
                minH="100vh"
            >
                <Heading fontWeight="normal" mb={4} letterSpacing="tight">Daily Prices, <Flex fontWeight="bold" display="inline-flex">{stock}</Flex></Heading>
                {/*<Image src = '/download.png'
                               alt = 'sample image from ML model trained on IBM stock'
                               />*/}
                {/*<StockChart stock ={stock} handlePredictClick ={handlePredictClick}/>*/}
                <StockPlot/>

                <HStack> <Input placeholder="Enter Stock" onChange={handleStockChange}/> <Button onClick = {handleClick}>Change</Button></HStack>
                
            </Flex>

            {/* column 3 of stock dashbaord, 
             IDEAS:
             - put technical details about Black-Scholes here
            */}
            <Flex 
                w="35%"
                bgColor = "#F5F5F5"
                p = "3%"
                flexDir="column"
                overflow="auto"
            
            >

            </Flex>
        </Flex>
    )
}