# Stock Price Prediction and Risk Analysis

This project implements a **Hybrid CNN-LSTM model** for stock price forecasting and the **VaR-ECF method** for risk estimation.  
It provides an interactive web-based dashboard built with Flask (backend) and Svelte (frontend).

## Features
- Upload stock dataset (CSV)
- Predict stock prices (Open, High, Low, Close)
- Estimate Value at Risk (VaR) using Cornish-Fisher Expansion
- Visualize predictions and risks with interactive charts

## Tech Stack
- Python, TensorFlow, Pandas, Numpy, Flask
- Svelte, Vite
- PostgreSQL (for data storage)

## Installation
Clone this repository:
```bash
git clone https://github.com/alviyuana15/stock-price-prediction-cnn-lstm.git
cd stock-price-prediction-cnn-lstm

# sv

Everything you need to build a Svelte project, powered by [`sv`](https://github.com/sveltejs/cli).

## Creating a project

If you're seeing this, you've probably already done this step. Congrats!

```bash
# create a new project in the current directory
npx sv create

# create a new project in my-app
npx sv create my-app
```

## Developing

Once you've created a project and installed dependencies with `npm install` (or `pnpm install` or `yarn`), start a development server:

```bash
npm run dev

# or start the server and open the app in a new browser tab
npm run dev -- --open
```

## Building

To create a production version of your app:

```bash
npm run build
```

You can preview the production build with `npm run preview`.

> To deploy your app, you may need to install an [adapter](https://svelte.dev/docs/kit/adapters) for your target environment.
