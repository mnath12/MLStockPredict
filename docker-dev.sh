#!/bin/bash

# Docker development helper script for MLStockPredict

set -e

case "$1" in
    "build")
        echo "🔨 Building development container..."
        docker-compose build mlstockpredict-dev
        ;;
    
    "up")
        echo "🚀 Starting development container..."
        docker-compose up -d mlstockpredict-dev
        ;;
    
    "down")
        echo "🛑 Stopping development container..."
        docker-compose down
        ;;
    
    "shell")
        echo "🐚 Opening shell in development container..."
        docker-compose exec mlstockpredict-dev bash
        ;;
    
    "jupyter")
        echo "📓 Starting Jupyter notebook service..."
        docker-compose up -d jupyter
        echo "🌐 Jupyter available at: http://localhost:8888"
        ;;
    
    "logs")
        echo "📋 Showing container logs..."
        docker-compose logs -f mlstockpredict-dev
        ;;
    
    "clean")
        echo "🧹 Cleaning up containers and images..."
        docker-compose down --rmi all --volumes --remove-orphans
        docker system prune -f
        ;;
    
    "test")
        echo "🧪 Running tests in container..."
        docker-compose exec mlstockpredict-dev python -m pytest backtesting_module/tests/ -v
        ;;
    
    "train")
        echo "🤖 Running local training in container..."
        docker-compose exec mlstockpredict-dev python backtesting_module/local_training.py
        ;;
    
    "backtest")
        echo "📊 Running backtest in container..."
        docker-compose exec mlstockpredict-dev python backtesting_module/main.py
        ;;
    
    *)
        echo "Usage: $0 {build|up|down|shell|jupyter|logs|clean|test|train|backtest}"
        echo ""
        echo "Commands:"
        echo "  build     - Build the development container"
        echo "  up        - Start the development container"
        echo "  down      - Stop the development container"
        echo "  shell     - Open bash shell in container"
        echo "  jupyter   - Start Jupyter notebook service"
        echo "  logs      - Show container logs"
        echo "  clean     - Clean up containers and images"
        echo "  test      - Run tests in container"
        echo "  train     - Run local training script"
        echo "  backtest  - Run main backtesting script"
        echo ""
        echo "Example workflow:"
        echo "  $0 build    # Build container"
        echo "  $0 up       # Start container"
        echo "  $0 shell    # Open shell for development"
        echo "  $0 down     # Stop when done"
        exit 1
        ;;
esac 